# SPDX-License-Identifier: MIT
"""Lifetime of V4.1's host-owned Engram tables.

Weights go through `atom.model_loader.loader.load_model` like every other
model's. These tables do not: they are mmap'd host embedding tables that the
Engram prefetcher reads row-wise, not parameters, so nothing in the parameter
path ever sees them and the mapping has to stay alive as long as the host does.
"""

# 【译】V4.1 主机持有的 Engram 表的生命周期。
#   权重和其他模型一样走 `atom.model_loader.loader.load_model`。这些表不走那条路：它们是 mmap 出来的主机 embedding 表，
#   由 Engram 预取器按行读取，不是参数，所以参数路径里没有任何东西会看到它们，而这个映射必须和宿主活得一样久
#
# 【导读】deepseek_v41.py（C 级）：DeepSeek-V4.1 的 Engram（按 n-gram 的 hash 查表取 embedding 的模块，第 8 批）所用的主机表，
#   直接从 checkpoint mmap 出来，放在 CPU 内存里，不进 GPU 参数
#   调用方：model_ops/engram/device/runtime.py 的 EngramInputPreparer.from_checkpoint（用 ExitStack 把它的生命周期挂到宿主上）
#   与主干的关系：参数路径一侧，deepseek_v41/model.py 的 weights_mapper 把 ".engram.embed." 映射到 None，让 load_model 直接丢掉
#     这些张量（不会出现在"被丢弃"告警里）；读表用的是 deepseek_v41/weights.py 的 CheckpointReader，不是本目录的 weight_iterator

from contextlib import contextmanager

from atom.models.deepseek_v41.weights import CheckpointReader, checkpoint_schema


# 打开 checkpoint，产出 {Engram 层号: HostEmbeddingTable}；with 块结束时 CheckpointReader 关闭（解除 mmap）
#   @contextmanager：把一个带 yield 的生成器函数变成 with 语句可用的上下文管理器，yield 出去的值就是 as 后面的变量
@contextmanager
def engram_tables(directory, config):
    with CheckpointReader(directory, checkpoint_schema(config)) as reader:
        yield reader.engram_tables(config)
