# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Kimi-K2.5 Model Configuration.

This configuration supports video-chunk as an internal modality type.
A video-chunk is the smallest independently processable unit of video.
"""

# 【译】Kimi-K2.5模型配置。
#   该配置把video-chunk作为一种内部模态类型来支持。
#   video-chunk是视频中可以独立处理的最小单元。
#
# 本文件给Kimi-K2.5定义HF配置类（改编自vLLM，见文件头版权行）
#   Kimi-K2.5 = 视觉塔（MoonViT）+ 多模态投影器 + 文本模型；文本模型沿用DeepSeek-V3结构，所以text_config是DeepseekV3Config
# 为什么要自己写：Kimi-K2.5官方仓库的配置类是remote code（下面KimiK25VisionConfig里的vt_*字段就来自它），
#   这里写一份本地版本，同时兼容HF标准字段名和官方仓库的字段名
# 调用方：vLLM插件 plugin/vllm/models/kimi_k25.py，用hf_config里的vision_config、text_config重建KimiK25Config，
#   再按它构造视觉塔（MoonViT3dPretrainedModel）和文本模型
# 原生ATOM路径不用这个类：config.get_hf_config对model_type="kimi_k25"只抽出text_config，
#   按text_config的model_type经_CONFIG_REGISTRY映射（如kimi_k2 -> deepseek_v3）后解析，见本批config.py；
#   原生的models/kimi_k25.py只跑文本主干，加载时跳过视觉塔和投影器的权重（与Qwen3.5不同，见qwen3_5.py科普①）
# 【分支】vLLM插件见第10批；DeepSeek-V3结构（MLA、MoE）见第7批（"第N批"指仓库根目录ANNOTATION_PLAN.md里的阅读批次）
#
# =====================================================================================
# 【科普】文本部分：DeepSeek-V3结构（text_config是DeepseekV3Config）
# =====================================================================================
# 和你熟悉的"MHA + dense FFN"Transformer相比，DeepSeek-V3改了两处：注意力换成MLA，FFN换成MoE。
#
# ① MLA（Multi-head Latent Attention，多头潜在注意力）——目的：把KV cache压到极小
#   MHA每层每个token要缓存 num_heads 份K和V。DeepSeek-V3有128个头，K每头192维、V每头128维，
#     一个token一层就是 128 x (192 + 128) = 40960 个数，长上下文时KV cache会非常大。
#   MLA的做法：不直接存K、V，而是先把hidden压缩成一个很短的"潜在向量"c（kv_lora_rank=512维），只缓存它；
#     用的时候再用两个上投影矩阵从c还原出每个头的K和V（"低秩分解"，lora_rank即低秩维度）。
#   RoPE的麻烦：RoPE要对K按位置旋转，旋转后就没法再和上投影矩阵"合并"了。所以MLA把K拆成两部分：
#     - 不带位置的部分（qk_nope_head_dim=128）：从c还原
#     - 带位置的部分（qk_rope_head_dim=64）：所有头共享一份，单独算、单独缓存
#   于是每个token每层只缓存 512 + 64 = 576 个数，约为MHA的1/71。
#   图示（某一层、一个token要往KV cache里存什么；数字是DeepseekV3Config的默认值）：
#     MHA： h ──W_K──> K：128头 × 192维  ┐
#           h ──W_V──> V：128头 × 128维  ┘ 全存：128 × (192+128) = 40960 个数
#     GQA： 同样的K、V，只是头数变少（对比用的假设：8个kv头 -> 8 × 320 = 2560 个数），每个头仍是完整的K、V
#     MLA： h ──kv_a_proj_with_mqa──> [ c（512维） ; k_rope（64维，所有头共用一份，带RoPE） ]   只存这576个数
#           算注意力时：c ──kv_b_proj（W_UK）──> 128个头各自的k_nope（128维），k = [k_nope ; k_rope]
#                       c ──kv_b_proj（W_UV）──> 128个头各自的v（128维）
#   GQA是"少存几个头"，MLA是"存一个压缩后的向量、用时再展开"，所以能比GQA省得更多
#   decode时还有个技巧（"权重吸收"）：q·k_nope = q·(W_UK c) = (W_UKᵀ q)·c，即把K的上投影矩阵合并进Q、把V的上投影合并进输出投影，
#     注意力直接在512维的c上算，不必真的把每个头的K、V还原出来。ATOM的实现见第7批 attention_mla.py。
#   Q也做了类似的压缩（q_lora_rank），主要为了省参数和计算，与KV cache无关。
#
# ② MoE（混合专家）——目的：参数多、算得少
#   通用原理和图示见 qwen3_next.py 文件头科普⑤（路由专家 + 共享专家）。DeepSeek-V3的数字（DeepseekV3Config默认值）：
#     256个路由专家里每个token挑8个，另有1个共享专家；开头 first_k_dense_replace=3 层仍是普通的dense FFN。
#     Kimi-K2沿用这个结构，只是专家更多（384个）。
#   ATOM的实现见第7批 moe.py、topK.py。
#
# 顺带一提（本文件用不到，但DeepSeek家族常见）：
#   DSA（DeepSeek Sparse Attention，V3.2 / GLM-5 用）在MLA之上再加一个轻量的"索引器"（indexer）：
#     每个query先用很便宜的打分从全部历史token里挑出top-k个（例如2048个），真正的注意力只和这k个token算。
#     这样超长上下文时注意力的计算量不再随长度增长。Kimi-K2 / K2.5没有DSA。
#     从MHA/GQA出发的图解见 config.py 里 glm5_kpool_block_size 上方的【科普】①②；
#     代码见第7批 models/deepseek_v2.py 的 Indexer 类（V3.2 / GLM-5 共用这个文件）。
#
# =====================================================================================
# 【科普】视觉部分：MoonViT + 投影器（vision_config）
# =====================================================================================
#   和其他多模态模型一样（可参考 qwen3_5.py 文件头科普①）：图片切成14x14像素的patch -> 过ViT（视觉Transformer）
#     -> 相邻2x2个patch合并（merge_kernel_size）-> 投影器（mm_projector）映射到文本hidden维度 ->
#     替换prompt里的占位token（media_placeholder_token_id）。
#   视频按"video-chunk"处理：几帧连续画面作为一个单元，ViT里同时做空间和时间维的注意力（video_attn_type），
#     合并时除了2x2空间合并，还把同一位置不同帧的特征取平均（merge_type="sd2_tpool"，tpool = temporal pooling，
#     时间维池化），所以一段视频不会因为帧多而产生成倍的视觉token。
# =====================================================================================
from transformers import DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig


# 返回第一个不是None的参数；用来在"HF标准字段名"和"官方仓库的vt_*字段名"之间取值，最后一个参数充当默认值
def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


# 视觉部分的配置：视觉塔（MoonViT，支持图片和视频）+ 多模态投影器（mm_projector，把视觉特征映射到文本hidden维度）
# 字段速查：
#   patch_size=14：每个patch 14x14像素
#   init_pos_emb_height / width / time、pos_emb_type="divided_fixed"：ViT的位置编码表按"高、宽、时间"
#     分开存（64 x 64 x 4），图片尺寸不同时按实际网格大小取用/插值
#   num_attention_heads / num_hidden_layers / hidden_size / intermediate_size：ViT本身的大小
#   merge_kernel_size=(2,2)、merge_type="sd2_tpool"：2x2空间合并 + 时间维平均（见文件头科普）
#   video_attn_type="spatial_temporal"：视频帧之间也做注意力
#   mm_projector_type="patchmerger"：投影器是一个把合并后的patch特征映射到文本维度的小MLP
#   mm_hidden_size / text_hidden_size：投影器输出维度，必须等于文本模型的hidden_size
# 输入：config.json里vision_config的各字段（两种命名都接受）；调用方：KimiK25Config.__init__经from_remote_config构造
class KimiK25VisionConfig(PretrainedConfig):
    model_type = "kimi_k25_vision"

    def __init__(
        self,
        # Vision Tower
        # 【译】视觉塔
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int | None = None,
        num_hidden_layers: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        # MM Projector
        # 【译】多模态投影器
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # moonshotai/Kimi-K2.5 remote-code field names
        # 【译】moonshotai/Kimi-K2.5远程代码使用的字段名
        # 它们与上面的num_attention_heads等同义，config.json里出现哪一套都行
        vt_num_attention_heads: int | None = None,
        vt_num_hidden_layers: int | None = None,
        vt_hidden_size: int | None = None,
        vt_intermediate_size: int | None = None,
        text_hidden_size: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 取值优先级：标准字段名 > vt_*别名 > 默认值（16头、27层、hidden 1152、FFN 4304）
        num_attention_heads = _first_non_none(
            num_attention_heads, vt_num_attention_heads, 16
        )
        num_hidden_layers = _first_non_none(num_hidden_layers, vt_num_hidden_layers, 27)
        hidden_size = _first_non_none(hidden_size, vt_hidden_size, 1152)
        intermediate_size = _first_non_none(
            intermediate_size, vt_intermediate_size, 4304
        )
        # Vision Tower
        # 【译】视觉塔
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Preserve the repo-specific aliases so either schema can be consumed.
        # 【译】保留仓库特有的别名字段，这样两种字段命名都能被使用方读到。
        self.vt_num_attention_heads = num_attention_heads
        self.vt_num_hidden_layers = num_hidden_layers
        self.vt_hidden_size = hidden_size
        self.vt_intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        # MM Projector
        # 【译】多模态投影器
        self.mm_projector_type = mm_projector_type
        # 没显式给mm_hidden_size时先填成视觉hidden_size；KimiK25Config.__init__看到"仍等于视觉hidden_size"，会把它改成文本hidden_size
        if mm_hidden_size is not None:
            self.mm_hidden_size = mm_hidden_size
        else:
            self.mm_hidden_size = hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        # text_hidden_size只有显式给出才设置；没给时由KimiK25Config.__init__补成文本模型的hidden_size
        if text_hidden_size is not None:
            self.text_hidden_size = text_hidden_size

    # 把各种形式的vision_config统一成KimiK25VisionConfig
    #   None -> 默认配置；已经是本类 -> 原样返回；dict / 带to_dict()的HF配置对象 / 普通对象 -> 先转成dict再构造
    # @classmethod：第一个参数cls是类本身，用来构造实例（cls(**config_dict)）
    @classmethod
    def from_remote_config(
        cls,
        vision_config: dict | PretrainedConfig | None,
    ) -> "KimiK25VisionConfig":
        if vision_config is None:
            return cls()
        if isinstance(vision_config, cls):
            return vision_config
        if isinstance(vision_config, dict):
            config_dict = dict(vision_config)
        elif hasattr(vision_config, "to_dict"):
            config_dict = dict(vision_config.to_dict())
        else:
            config_dict = dict(vars(vision_config))
        return cls(**config_dict)


# 顶层配置：vision_config（KimiK25VisionConfig）+ text_config（DeepseekV3Config）+ 媒体占位token等
# 调用方：plugin/vllm/models/kimi_k25.py里的KimiK25Config(vision_config, text_config)
class KimiK25Config(PretrainedConfig):
    """Kimi-K2.5 model configuration.

    Kimi-K2.5 extends Kimi-K2 with vision support using video-chunks.
    A video-chunk consists of multiple consecutive frames
    that are processed together with temporal pooling.

    Args:
        vision_config: Configuration for the vision tower and projector.
        text_config: Configuration for the text model (DeepseekV3).
        ignore_index: The ignore index for the loss function.
        media_placeholder_token_id: The token ID for media placeholders.
        pad_token_id: The token ID for padding.
    """

    # 【译】Kimi-K2.5模型配置。
    #   Kimi-K2.5在Kimi-K2的基础上，用video-chunk加入了视觉支持。
    #   一个video-chunk由多帧连续画面组成，这些帧经过时间维池化后一起处理。
    #   参数：
    #     vision_config：视觉塔和投影器的配置。
    #     text_config：文本模型（DeepseekV3）的配置。
    #     ignore_index：损失函数的忽略索引。
    #     media_placeholder_token_id：媒体占位符的token ID。
    #     pad_token_id：padding用的token ID。
    model_type = "kimi_k25"

    def __init__(
        self,
        vision_config: dict | KimiK25VisionConfig | None = None,
        text_config: dict | DeepseekV3Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = False,
        video_placeholder: str = "<|kimi_k25_video_placeholder|>",
        **kwargs,
    ):
        # Vision config
        # 【译】视觉配置
        self.vision_config = KimiK25VisionConfig.from_remote_config(vision_config)

        # Text config
        # 【译】文本配置
        # text_config可以是None（用默认DeepseekV3Config）、dict（config.json里的原始字典）或已构造好的对象（原样使用）
        # 文本模型的关键字段（都在DeepseekV3Config里，含义见文件头科普）：
        #   MLA：kv_lora_rank、q_lora_rank、qk_nope_head_dim、qk_rope_head_dim、v_head_dim
        #   MoE：n_routed_experts、num_experts_per_tok、n_shared_experts、moe_intermediate_size、first_k_dense_replace
        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config: DeepseekV3Config = text_config

        # Set mm_hidden_size to text hidden size if not explicitly set
        # 【译】如果没有显式设置mm_hidden_size，就设为文本模型的hidden size
        # "没显式设置"的判断方式：它仍等于视觉hidden_size（KimiK25VisionConfig里填的占位值）
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size
        if getattr(self.vision_config, "text_hidden_size", None) is None:
            self.vision_config.text_hidden_size = self.text_config.hidden_size

        # Other config
        # 【译】其他配置
        self.ignore_index = ignore_index
        # media_placeholder_token_id：prompt里图片/视频占位token的id，推理时这些位置换成视觉特征
        # use_unified_vision_chunk / video_placeholder：是否把图片也当成"只有1帧的video-chunk"统一处理，
        #   以及视频在文本里的占位字符串（由预处理展开成若干占位token）
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder

        # Propagate quantization config from text model
        # 【译】从文本模型传递量化配置
        # 量化配置写在text_config里；复制到顶层后，按顶层quantization_config读取量化方式的代码也能看到它
        #   （ATOM的QuantizationConfig读的是 hf_config.quantization_config，见config.py）
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        # 最后调用父类构造：PretrainedConfig.__init__处理pad_token_id和kwargs里其余的通用字段
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    # 下面两个只读属性把hidden_size、vocab_size转发给text_config，让只认顶层字段的代码也能直接用
    @property
    def hidden_size(self) -> int:
        """Get hidden size from text config for compatibility."""
        # 【译】为兼容起见，从文本配置中取hidden size。
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocab size from text config for compatibility."""
        # 【译】为兼容起见，从文本配置中取词表大小。
        return self.text_config.vocab_size


__all__ = ["KimiK25Config", "KimiK25VisionConfig"]
