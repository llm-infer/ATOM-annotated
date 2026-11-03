# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-MoE model configuration"""

# 【译】Qwen3.5-MoE模型配置
#
# 本文件是qwen3_5.py的MoE版本：Qwen3_5MoeTextConfig、Qwen3_5MoeVisionConfig、Qwen3_5MoeConfig，
#   与dense版逐字段对应，只是文本配置把dense FFN（intermediate_size）换成了MoE的一组字段
#   共有字段（layer_types、linear_*、RoPE、v4覆盖问题等）的解释见qwen3_5.py，这里只注释差异
# 调用方：models/qwen3_5.py用Qwen3_5MoeTextConfig做类型标注并再导出Qwen3_5MoeConfig；
#   与dense版一样，仓库里没有地方直接实例化这些类，运行时的hf_config来自config.get_hf_config
# 【分支】MoE层（FusedMoE、专家并行）见第7批；Qwen3.5模型实现见第8批
#
# 【科普】MoE（混合专家）一句话：把每层的一个大FFN换成很多小FFN（专家），每个token由路由器挑top-k个来算，
#   参数总量大但每个token的计算量小。详细说明见 qwen3_next.py 文件头科普⑤；
#   多模态、M-RoPE见 qwen3_5.py 文件头；GDN线性注意力见 qwen3_next.py 文件头科普②。
# 和dense版的对比（默认值）：
#   dense：hidden 4096、32层，每层一个 4096 -> 12288 -> 4096 的FFN
#   MoE：  hidden 2048、40层，每层256个 2048 -> 512 -> 2048 的小专家（每个token用8个）+ 1个共享专家
from transformers.configuration_utils import PretrainedConfig, layer_type_validation


# MoE文本模型配置（model_type="qwen3_5_moe_text"），对应config.json里的text_config；默认值只是占位
class Qwen3_5MoeTextConfig(PretrainedConfig):
    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    # 科普：gate_up_proj是什么——现代FFN是SwiGLU：FFN(x) = down( silu(gate(x)) * up(x) )，
    #   gate和up是两个同形状的投影，常把它们拼成一个矩阵gate_up_proj一次算完（ATOM也这样融合，第6批）
    # HF的张量并行方案（ATOM运行时不读）：experts.gate_up_proj是"packed_colwise"，
    #   说明HF把所有专家的gate/up权重打包成一个张量；ATOM加载时用detect_fused_expert_format区分
    #   这种打包格式（Qwen3.5 BF16）和逐专家存储的格式（Qwen3.5 FP8），见models/qwen3_5.py
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        # 与qwen3_5.py相同：允许rope_parameters里出现M-RoPE的mrope_section / mrope_interleaved
        kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # 与qwen3_5.py相同：按full_attention_interval（默认4）生成"3层GDN + 1层全注意力"的层类型表
        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                (
                    "linear_attention"
                    if bool((i + 1) % interval_pattern)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # linear attention part
        # 【译】线性注意力部分
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        # MoE字段：
        #   moe_intermediate_size：每个路由专家FFN的中间维度
        #   shared_expert_intermediate_size：共享专家（每个token都会经过）的中间维度
        #   num_experts_per_tok：每个token激活的专家数（top-k，默认8）；num_experts：路由专家总数（默认256）
        #   output_router_logits / router_aux_loss_coef：训练时的路由辅助损失，推理用不到
        # 注意：没有intermediate_size（dense FFN的维度）和decoder_sparse_step——Qwen3.5-MoE每一层都是MoE
        # ATOM读取方：models/qwen3_5.py 按这些字段建FusedMoE和共享专家；config.py的_normalize_moe_config_fields
        #   把num_experts抄成n_routed_experts，权重加载代码统一读后者
        # 例：一个token -> router对256个专家打分 -> 取前8个 -> 8个专家输出按分数加权 + 共享专家输出
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        super().__init__(**kwargs)
        # Set these AFTER super().__init__() because transformers v4's
        # PretrainedConfig.__init__ has these as explicit params with different
        # defaults (e.g. tie_word_embeddings=True) that would overwrite our values.
        # 【译】这些字段要在super().__init__()之后设置，因为transformers v4的PretrainedConfig.__init__
        #   把它们作为显式参数，且默认值不同（例如tie_word_embeddings=True），会覆盖我们的值。
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


# 视觉编码器配置，字段与Qwen3_5VisionConfig完全相同（见qwen3_5.py），只是model_type为"qwen3_5_moe"
class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


# 顶层多模态配置（model_type="qwen3_5_moe"），结构与Qwen3_5Config相同
#   同样只处理dict和None两种vision_config / text_config输入
class Qwen3_5MoeConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid v4 PretrainedConfig overwrite
        # 【译】在super().__init__()之后设置，避免被transformers v4的PretrainedConfig覆盖
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig"]
