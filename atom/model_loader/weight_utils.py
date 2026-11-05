# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# This code is adapted from https://github.com/ROCm/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py
# 【译】这段代码改编自 https://github.com/ROCm/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py
#
# 【导读】weight_utils.py：权重加载用到的几个小工具（全流程见 loader.py 文件头）
#   download_weights_from_hf            HF hub id → 本地目录（必要时下载）；调用方：weight_iterator.safetensors_weights_iterator
#   filter_duplicate_safetensors_files  有 index.json 时只保留索引里出现的分片；调用方：同上
#   local_model_dir                     模型名 → 本地目录（只解析、不下载）；调用方：models/utils.py、models/deepseek_v4.py、
#                                       models/deepseek_v41/weights.py、attentions/deepseek_v41/backend.py、
#                                       entrypoints/openai/chat_encoders.py（都在权重加载之后、按文件名找附带文件时用）
#   set_weight_attrs                    给参数挂属性（weight_loader 等）；调用方：model_ops/moe.py
#   get_lock / DisabledTqdm / enable_hf_transfer   下载用的跨进程文件锁、静音的进度条、自动启用 hf_transfer
# 对比nano-vllm：nano-vllm 只接受本地目录，没有下载和这些工具

import fnmatch
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import filelock
import huggingface_hub.constants
import torch
from huggingface_hub import HfFileSystem, snapshot_download
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
# 【译】文件锁用系统级的临时目录，这样多个用户可以共用同一个锁而不出错。临时目录里的锁文件在系统重启时会被自动删除，
#   用户也就不会抱怨烦人的锁文件
temp_dir = tempfile.gettempdir()


def enable_hf_transfer():
    """automatically activates hf_transfer"""
    # 【译】自动启用 hf_transfer
    # hf_transfer：HF 官方的 Rust 多连接下载器，比默认下载快；用户没显式设置 HF_HUB_ENABLE_HF_TRANSFER、并且装了这个包时才开
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            # 【译】可用时启用 hf hub transfer
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


# 模块导入时就执行一次
enable_hf_transfer()


# 永远禁用的进度条：传给 snapshot_download，让多个 rank 同时下载时不刷屏
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


# 每个模型一把跨进程文件锁（filelock）：多个 rank / 进程同时启动时只有一个真正下载，其余等它完成后直接用缓存
#   调用方：download_weights_from_hf
def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    # 【译】加上 hash，避免和老用户的锁文件冲突
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    # 【译】要让文件锁能跨用户共享，需要权限 0o666（所有用户可读写）
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


# 模型名（本地路径或 hub id）→ 本地目录，只解析、不下载；解析不了原样返回。调用方见文件头
#   例："deepseek-ai/DeepSeek-V4.1-Flash" → ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/<hash>
def local_model_dir(model: str) -> str:
    """The directory a model's files are in, whether it was named by id or path.

    `Config.model` is whatever the caller typed. Anything that reaches a file
    by joining onto it needs this first: joining onto a hub id yields
    `org/name/file`, a relative path that does not exist, and its absence reads
    as a missing checkpoint rather than as an unresolved id -- which is how
    `deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json` reached a
    `FileNotFoundError` in CI.

    Resolution only, never a download: every caller runs after the weights have
    loaded, so the snapshot is already in the hub cache and `allow_patterns=[]`
    asks for no files.

    A name that resolves to nothing comes back unchanged, because every caller
    already has an answer for "there is no such directory" and none of them is
    improved by a second one. The strict readers say so themselves --
    `CheckpointReader` refuses a non-directory and names this function -- and
    the probes treat it the way they treat a checkpoint without the file they
    were looking for.
    """
    # 【译】不管模型是用 id 还是用路径指定的，返回它的文件所在的目录。
    #   `Config.model` 是调用方输入的任何东西。任何通过拼接它来访问文件的地方都需要先调这个：拼接到一个 hub id 上得到的是
    #   `org/name/file`，一个不存在的相对路径，而它的不存在读起来像是缺了 checkpoint，而不是 id 没解析——
    #   `deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json` 就是这样在 CI 里撞上 `FileNotFoundError` 的。
    #   只解析、从不下载：每个调用方都在权重加载完之后运行，所以快照已经在 hub 缓存里，`allow_patterns=[]` 表示一个文件都不要。
    #   解析不出任何东西的名字原样返回，因为每个调用方本来就有对"没有这个目录"的处理，再多一个也不会更好。严格的读取方会自己
    #   说明——`CheckpointReader` 拒绝非目录并点名这个函数——而探测方会像对待"没有它要找的那个文件的 checkpoint"一样对待它
    if not model or os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True, allow_patterns=[])
    except Exception:  # noqa: BLE001 -- any failure to resolve means "not local"
        return model


# hub id → 本地快照目录。调用方：weight_iterator.safetensors_weights_iterator（路径不是目录时），传 allow_patterns=["*.safetensors"]、
#   ignore_patterns=["original/*"]
#   在线时先列出仓库文件，取 allow_patterns 里第一个能匹配到文件的模式（vLLM 原本用它在 safetensors / bin 等格式间选一种）；
#   HF_HUB_OFFLINE 时只用本地缓存。已在缓存里的文件不会重复下载
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    # 【译】从 Hugging Face Hub 下载模型权重。
    #   参数：
    #     model_name_or_path (str)：模型名或路径。
    #     cache_dir (Optional[str])：存放模型权重的缓存目录；为 None 时用 HF 的默认值。
    #     allow_patterns (list[str])：允许的权重文件模式，匹配其中任一模式的文件会被下载。
    #     revision (Optional[str])：模型的版本。
    #     ignore_patterns (Optional[Union[str, list[str]]])：要过滤掉的权重文件模式，匹配其中任一模式的文件会被忽略。
    #   返回：
    #     str：下载好的模型权重的路径
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Before we download we look at that is available:
        # 【译】下载之前先看看有哪些文件可用：
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        # 【译】根据可用的文件决定下载哪些
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    # 【译】用文件锁防止多个进程同时下载同一个模型的权重
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=local_only,
        )
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder


# 给参数（张量）挂上自定义属性，例如 weight_loader、is_transposed；已存在的属性不许覆盖（assert）
#   Python 张量对象可以随意加属性；ATOM 的加载就靠参数上的 weight_loader 属性找到"怎么写这个参数"（weight_dispatch.py）
#   调用方：model_ops/moe.py（FusedMoE 的各 create_weights 给专家参数挂 weight_loader 等属性）
def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    # 【译】在权重张量上设置属性。
    #   这个方法用来在权重张量上设置属性，不会覆盖已有的属性。
    #   参数：
    #     weight：权重张量。
    #     weight_attrs：要设置到权重张量上的属性字典
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        # 【译】NOTE(woosuk)：加载权重时我们经常这样做：
        #     narrowed_tensor = param.data.narrow(0, offset, len)
        #     narrowed_tensor.copy_(real_weight)
        #   并期望 narrowed_tensor 和 param.data 共享同一块存储。然而在 TPU 上 narrowed_tensor 会惰性地传播到基张量 param.data，
        #   造成多余的内存占用，有时会导致加载模型时 OOM。为避免这一点，我们在调用参数的 weight loader 之后同步参数张量。
        #   TODO(woosuk)：有更好的办法后去掉这个 hack
        # 注：这段说明是从 vLLM 原样带过来的，描述的 TPU 同步逻辑在这里并不存在——下面只有一句 setattr
        setattr(weight, key, value)


# 有 index.json 时只保留它的 weight_map 里出现过的分片文件；没有 index（单文件 checkpoint）就原样返回
#   例：目录里有 model-00001-of-00002.safetensors、model-00002-of-00002.safetensors，以及一份不在索引里的合并版 model.safetensors
#     → 只保留前两个，避免同一批权重被读两遍。调用方：weight_iterator.safetensors_weights_iterator
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    # 【译】model.safetensors.index.json 是一个映射：torch state_dict 里的 key → 存放该权重的 safetensors 文件
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    # 【译】遍历 weight_map（权重名: safetensors 文件），确定应该使用哪些文件
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    # 【译】过滤掉索引文件里找不到的文件
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files
