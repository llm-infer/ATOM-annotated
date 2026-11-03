# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
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
"""Qwen3-Next model configuration"""

# 【译】Qwen3-Next模型配置
#
# 本文件给Qwen3-Next定义HF配置类Qwen3NextConfig（改编自transformers / vLLM，见文件头版权行）
# Qwen3-Next = 混合注意力 + MoE：默认每4层里前3层是Gated DeltaNet线性注意力、第4层是全注意力（默认带输出门attn_output_gate），
#   FFN是"路由专家 + 共享专家"的MoE；它也是Qwen3.5（qwen3_5.py）的结构基础
# 调用方：
#   models/qwen3_next.py、models/qwen3_next_mtp.py：用它标注atom_config.hf_config的类型
#   plugin/vllm/models/qwen3_next.py：同样只做类型标注
# 运行时的hf_config：config.get_hf_config里_CONFIG_REGISTRY把"qwen3_next"映射到transformers自带的同名配置类，
#   仓库里没有地方直接实例化本类；这里的类主要说明有哪些字段、默认值是什么
# 对比nano-vllm：nano-vllm的Qwen3只有全注意力和dense FFN；这里的linear_*、MoE字段都是它没有的
# 【分支】GDN线性注意力和Qwen3-Next模型实现见第8批；MoE见第7批
#
# =====================================================================================
# 【科普】读懂这个配置需要的几个概念（假设你只熟悉标准Transformer + MHA）
# =====================================================================================
#
# ① GQA / MQA：MHA的"省KV"变体（字段 num_attention_heads、num_key_value_heads）
#   MHA里每个query头都有自己的一份K、V头。推理时K、V要缓存起来（KV cache），头越多缓存越大。
#   GQA让多个query头"共用"一份K/V头：例如16个q头、2个kv头，就是每8个q头共享一组K/V，
#   KV cache缩小到MHA的1/8，效果损失很小。kv头数=1时叫MQA（极端情况），=q头数时就退化回MHA。
#   本模型默认 16 q头 / 2 kv头。
#
# ② 线性注意力与 Gated DeltaNet（GDN）（字段 linear_*）
#   标准（softmax）注意力：o_t = Σ_i softmax(q_t·k_i) v_i，要和"所有"历史token的k、v算，
#     所以必须保存全部历史K、V（KV cache随长度线性增长），每生成一个token的计算量也随长度增长。
#   线性注意力的想法：去掉softmax，o_t = Σ_i (q_t·k_i) v_i = (Σ_i v_i k_iᵀ) q_t。
#     括号里的 Σ v_i k_iᵀ 是一个固定大小的矩阵，记作状态 S（形状 [v_dim, k_dim]），
#     每来一个token只需 S += v_t k_tᵀ，再算 o_t = S q_t。于是：
#       - 不需要KV cache，每层每条序列只存一个固定大小的S（不管上下文多长）
#       - decode每步的计算量是常数，像RNN一样"循环"更新
#   代价：S的容量有限，信息会被"压缩"，精确回忆远处细节的能力不如softmax注意力。
#   GDN是线性注意力的改进版，更新规则（gated delta rule）大致是：
#       S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t · v_t k_tᵀ
#     - α_t（门控/衰减，0~1）：每步把旧记忆按比例"遗忘"一点，由模型根据输入决定
#       （权重里的 A_log、dt_bias 和投影出的 a 一起算出来）
#     - β_t（写入强度，0~1，由投影出的 b 过sigmoid得到）配合 (I − β k kᵀ)：先把S里和k_t相关的旧值擦掉一部分
#       再写入新值，这就是"delta rule"——像覆盖写，而不是一味累加，所以记忆更不容易"糊"
#     - 输出还会乘一个门 z（o = norm(o) · silu(z)），让模型控制每个通道输出多少
#   推理实现：prefill时一次处理很多token，用"chunk"算法（分块并行）；decode时一个token，用"recurrent"算法
#     （直接按上面的公式更新S）。见第8批 fla_ops/ 和 attention_gdn.py。
#   短卷积（linear_conv_kernel_dim=4）：q/k/v进GDN之前先过一个长度4的因果1D卷积，让每个token先混入前3个token的信息，
#     弥补线性注意力"局部细节"弱的问题。所以除了S，还要缓存最近3个token的卷积输入（conv state）。
#   ATOM里一层GDN每条序列的状态（gdn_attn.py）：
#     temporal_state [num_v_heads, v_dim, k_dim] = 32 x 128 x 128 ≈ 52万个数
#     conv_state     [kernel-1, conv_dim]        = 3 x (128*16*2 + 128*32) = 3 x 8192
#   对比：一层全注意力每个token的KV = 2(k,v) x 2头 x 256维 = 1024个数，约512个token就和GDN状态一样大；
#     上下文越长，GDN越划算。
#
# ③ 混合架构（字段 layer_types）
#   纯线性注意力回忆能力弱，纯全注意力长上下文太贵，于是交替排布：每4层里3层GDN + 1层全注意力。
#   全注意力层照常有KV cache（走第4批的block manager），GDN层只有固定大小的状态（走第4批的状态缓存）。
#   因此同一个模型里有两种"缓存"，这正是ATOM调度、KV管理比nano-vllm复杂的原因之一。
#
# ④ 带输出门的全注意力（attn_output_gate，config.json里的字段，本类没有显式声明，默认True）
#   全注意力层的q_proj多投影出一份"门"g，注意力输出 o 变成 o · sigmoid(g) 再进o_proj。
#   作用：让模型能把某些头的输出"关掉"，训练更稳定。ATOM因此用QKVGParallelLinear一次投影出q、k、v、g（第6批）。
#
# ⑤ MoE（混合专家，字段 num_experts、num_experts_per_tok、moe_intermediate_size 等）
#   标准Transformer每层一个FFN；MoE层换成很多个小FFN（"专家"），再加一个路由器（router/gate，一个线性层）：
#     router给每个token对每个专家打分 -> 取分数最高的k个（top-k，这里k=10）-> 只算这k个专家 ->
#     按分数加权求和。总参数很大（512个专家），但每个token只用到一小部分，所以"参数多、算得少"。
#     "80B-A3B"就是总参数80B、每个token激活约3B。
#   共享专家（shared expert）：一个所有token都必经的FFN，学通用知识，路由专家学专门知识，两者输出相加。
#     Qwen3-Next的共享专家输出还会乘一个sigmoid门（shared_expert_gate）。
#   norm_topk_prob=True：选出的k个专家的权重重新归一化，使它们加起来等于1。
#   ATOM里MoE的执行（把token分发给专家、多卡专家并行）见第7批 moe.py。
#
# ⑥ 部分RoPE（partial_rotary_factor=0.25）
#   标准RoPE对q/k的全部维度做旋转；这里只旋转head_dim的前25%（256维里的64维），其余维度不带位置信息。
#   经验上对长上下文外推更友好。
# =====================================================================================
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Qwen3-Next的配置类；docstring是HF风格的完整参数说明，中文译文在docstring之后
class Qwen3NextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3NextModel`]. It is used to instantiate a
    Qwen3-Next model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3-Next-80B-A3B-Instruct [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_theta` (`float`): The base period of the RoPE embeddings.
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
                `partial_rotary_factor` (`float`, *optional*, defaults to 0.25):
                    Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 10):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 512):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3NextMLP rather than Qwen3NextSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).

    ```python
    >>> from transformers import Qwen3NextModel, Qwen3NextConfig

    >>> # Initializing a Qwen3Next style configuration
    >>> configuration =  Qwen3NextConfig()

    >>> # Initializing a model from the Qwen3-Next-80B-A3B style configuration
    >>> model = Qwen3NextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """  # noqa: E501

    # 【译】这是用来保存[`Qwen3NextModel`]配置的配置类，用于按指定参数实例化Qwen3-Next模型、定义模型架构。
    #   用默认值实例化，会得到与Qwen3-Next-80B-A3B-Instruct
    #   [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)相近的配置。
    #
    #   配置对象继承自[`PretrainedConfig`]，可用来控制模型输出。更多信息请阅读[`PretrainedConfig`]的文档。
    #
    #   参数：
    #     vocab_size（`int`，可选，默认151936）：模型词表大小，决定`inputs_ids`能表示多少种不同的token。
    #     hidden_size（`int`，可选，默认2048）：隐藏表示的维度。
    #     intermediate_size（`int`，可选，默认5632）：MLP表示的维度。
    #     num_hidden_layers（`int`，可选，默认48）：Transformer编码器中的隐藏层数。
    #     num_attention_heads（`int`，可选，默认16）：Transformer编码器中每个注意力层的注意力头数。
    #     num_key_value_heads（`int`，可选，默认2）：实现分组查询注意力（GQA）所用的key_value头数。
    #       若`num_key_value_heads=num_attention_heads`，模型使用多头注意力（MHA）；若`num_key_value_heads=1`，
    #       使用多查询注意力（MQA）；否则使用GQA。把多头checkpoint转换为GQA checkpoint时，每组的key和value头
    #       应由该组内所有原始头做均值池化得到。详见[这篇论文](https://arxiv.org/pdf/2305.13245.pdf)。
    #       未指定时默认为`32`。
    #     hidden_act（`str`，可选，默认`"silu"`）：解码器中的非线性激活函数。
    #     max_position_embeddings（`int`，可选，默认32768）：模型可能用到的最大序列长度。
    #     initializer_range（`float`，可选，默认0.02）：初始化所有权重矩阵所用截断正态分布的标准差。
    #     rms_norm_eps（`float`，可选，默认1e-06）：RMS归一化层使用的epsilon。
    #     use_cache（`bool`，可选，默认`True`）：模型是否返回最后的key/value注意力（并非所有模型都使用）。
    #       仅在`config.is_decoder=True`时有意义。
    #     tie_word_embeddings（`bool`，可选，默认`False`）：输入和输出词嵌入是否共享权重。
    #     rope_parameters（`dict`，可选）：RoPE嵌入的缩放配置字典。注意：如果采用新的rope类型，并希望模型能处理
    #       更长的`max_position_embeddings`，建议相应地更新该值。期望的内容：
    #         `rope_theta`（`float`）：RoPE嵌入的基础周期。
    #         `rope_type`（`str`）：要使用的RoPE子变体，可取['default', 'linear', 'dynamic', 'yarn', 'longrope',
    #           'llama3']之一，'default'是原始RoPE实现。
    #         `factor`（`float`，可选）：除'default'外的所有rope类型都会用到，是施加在RoPE嵌入上的缩放因子。
    #           多数缩放类型中，因子x能让模型处理长度为x * 原始预训练最大长度的序列。
    #         `original_max_position_embeddings`（`int`，可选）：'dynamic'、'longrope'和'llama3'使用，
    #           表示预训练时的原始最大位置嵌入数。
    #         `attention_factor`（`float`，可选）：'yarn'和'longrope'使用，是施加在注意力计算上的缩放因子。
    #           未指定时默认取实现推荐的值，用`factor`字段推断建议值。
    #         `beta_fast`（`float`，可选）：仅'yarn'使用。线性斜坡函数中设定外推（仅外推）边界的参数。未指定时默认32。
    #         `beta_slow`（`float`，可选）：仅'yarn'使用。线性斜坡函数中设定内插（仅内插）边界的参数。未指定时默认1。
    #         `short_factor`（`List[float]`，可选）：仅'longrope'使用。施加在短上下文
    #           （< `original_max_position_embeddings`）上的缩放因子。必须是数字列表，长度等于hidden size除以注意力头数再除以2
    #         `long_factor`（`List[float]`，可选）：仅'longrope'使用。施加在长上下文
    #           （< `original_max_position_embeddings`）上的缩放因子。必须是数字列表，长度等于hidden size除以注意力头数再除以2
    #         `low_freq_factor`（`float`，可选）：仅'llama3'使用。施加在RoPE低频分量上的缩放因子
    #         `high_freq_factor`（`float`，可选）：仅'llama3'使用。施加在RoPE高频分量上的缩放因子
    #         `partial_rotary_factor`（`float`，可选，默认0.25）：query和key中施加旋转嵌入的部分所占比例。
    #     attention_bias（`bool`，可选，默认`False`）：自注意力的query、key、value和输出投影层是否使用偏置。
    #     attention_dropout（`float`，可选，默认0.0）：注意力概率的dropout比例。
    #     head_dim（`int`，可选，默认256）：多头注意力中投影权重的维度。
    #     linear_conv_kernel_dim（`int`，可选，默认4）：线性注意力层中所用卷积的核大小。
    #     linear_key_head_dim（`int`，可选，默认128）：线性注意力中每个key头的维度。
    #     linear_value_head_dim（`int`，可选，默认128）：线性注意力中每个value头的维度。
    #     linear_num_key_heads（`int`，可选，默认16）：线性注意力层使用的key头数。
    #     linear_num_value_heads（`int`，可选，默认32）：线性注意力层使用的value头数。
    #     decoder_sparse_step（`int`，可选，默认1）：MoE层出现的频率。
    #     moe_intermediate_size（`int`，可选，默认512）：路由专家的中间维度。
    #     shared_expert_intermediate_size（`int`，可选，默认512）：共享专家的中间维度。
    #     num_experts_per_tok（`int`，可选，默认10）：每个token选中的专家数。
    #     num_experts（`int`，可选，默认512）：路由专家的数量。
    #     norm_topk_prob（`bool`，可选，默认`True`）：是否对topk概率做归一化。
    #     output_router_logits（`bool`，可选，默认`False`）：模型是否返回路由器logits。开启后模型还可以输出
    #       辅助损失，包括负载均衡损失和路由器z-loss。
    #     router_aux_loss_coef（`float`，可选，默认0.001）：辅助损失在总损失中的系数。
    #     mlp_only_layers（`list[int]`，可选，默认`[]`）：指定哪些层使用Qwen3NextMLP而不是Qwen3NextSparseMoeBlock。
    #       列表里是层下标，有num_layers层时取值0到num_layers-1。若`mlp_only_layers`为空，则用`decoder_sparse_step`决定稀疏性。
    #     layer_types（`list[str]`，可选）：每一层的类型（注意力或线性注意力）。
    #
    #   示例（transformers中的用法）：
    #     from transformers import Qwen3NextModel, Qwen3NextConfig
    #     configuration = Qwen3NextConfig()        # 初始化一个Qwen3Next风格的配置
    #     model = Qwen3NextModel(configuration)    # 用Qwen3-Next-80B-A3B风格的配置初始化模型
    #     configuration = model.config             # 读取模型的配置
    #
    # 补充：上文"Transformer编码器"是HF文档模板的原话，Qwen3-Next实际是decoder-only模型
    # 字段速查（按上面的科普①~⑥分组）：
    #   ① GQA：num_attention_heads=16个q头，num_key_value_heads=2个kv头，head_dim=256（全注意力层）
    #   ② GDN：linear_num_key_heads=16、linear_num_value_heads=32、linear_key_head_dim=linear_value_head_dim=128、
    #     linear_conv_kernel_dim=4（这些只作用于线性注意力层，和①的字段互不相干）
    #   ③ 层类型：layer_types
    #   ⑤ MoE：num_experts=512、num_experts_per_tok=10、moe_intermediate_size=512（每个专家FFN的中间维度，
    #     比dense FFN的intermediate_size=5632小很多，因为每个token会过10个专家）、shared_expert_intermediate_size、
    #     norm_topk_prob、decoder_sparse_step、mlp_only_layers
    #   ⑥ RoPE：rope_parameters、partial_rotary_factor
    #   训练专用、推理不用：initializer_range、attention_dropout、output_router_logits、router_aux_loss_coef
    #     （router_aux_loss是训练时防止所有token都挤到少数几个专家上的"负载均衡"损失）
    # ATOM怎么用这些MoE字段（models/qwen3_next.py）：第i层不在mlp_only_layers里、num_experts>0、
    #   且(i+1) % decoder_sparse_step == 0时用Qwen3NextSparseMoeBlock，否则用dense的Qwen3NextMLP
    model_type = "qwen3_next"
    # keys_to_ignore_at_inference、base_model_tp_plan、base_model_pp_plan是HF transformers的约定写法，ATOM运行时不读取
    #   ATOM的张量并行由自己的并行Linear实现（第6批），流水线并行见第9批
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
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
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        layer_types=None,
        **kwargs,
    ):
        # 与Qwen3.5不同，这里先调用super().__init__再设置各字段，所以不存在"被父类默认值覆盖"的问题
        #   tie_word_embeddings显式传给父类；其余通用字段在kwargs里，由父类设成属性
        if mlp_only_layers is None:
            mlp_only_layers = []
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        # 【译】如果有`rope_scaling`就用它，否则使用`rope_parameters`
        # 背景：旧版config.json把RoPE配置写在rope_scaling字段，新版transformers改名为rope_parameters；两种都兼容
        # 注意：super().__init__(**kwargs)已经先看过kwargs（会把rope_theta等设成属性）；这里再从同一个dict里pop只是为了读值
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        # 把rope_theta和partial_rotary_factor补进rope_parameters：models/qwen3_next.py的Qwen3NextAttention
        #   只从rope_parameters里读它们（rotary_dim = head_dim * partial_rotary_factor，默认只旋转前1/4维）
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # layer_types没给时：第i层满足(i+1)%4==0时是全注意力，其余是线性注意力（48层时第3、7、...、47层是全注意力）
        # 即48层里36层GDN、12层全注意力：KV cache只有12层要存，比纯全注意力模型少3/4
        # ATOM读取方：models/qwen3_next.py 逐层决定建GDN层还是全注意力层；ModelRunner据此分别分配
        #   KV cache（全注意力层）和GDN状态（线性注意力层）
        # 这里调用layer_type_validation没传num_hidden_layers；Qwen3.5版本传了，会额外检查列表长度
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        # linear attention part
        # 【译】线性注意力部分
        # 线性注意力（GDN）层的形状：q/k/v先过长度为linear_conv_kernel_dim的短因果卷积；
        #   默认16个k头、32个v头，即每个q/k头被2个v头共享
        # 科普：这和①的GQA方向正好相反——GQA是"多个q共享一个kv"，这里是"一个q/k配多个v"。
        #   GDN的状态S按v头存（每个v头一个 [v_dim, k_dim] 矩阵），v头多 = 记忆容量大，而q/k少一些省计算
        # ATOM读取方：models/qwen3_next.py 的GDN层按这些字段建 in_proj_qkvz / in_proj_ba / conv1d 投影，
        #   gdn_attn.py 按它们算每层状态的形状（见文件头科普②里的数字）
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        # 【译】MoE参数
        # ATOM读取方：models/qwen3_next.py 建 FusedMoE（num_experts个专家、每token取num_experts_per_tok个，
        #   renormalize=norm_topk_prob）和共享专家；config.py 的 _normalize_moe_config_fields 还会把
        #   num_experts 抄一份成 n_routed_experts，供权重加载代码统一读取
        # 例（默认值）：一个token进MoE层 -> router对512个专家打分 -> 取前10个 -> 10个专家各算一个
        #   hidden 2048 -> 512 -> 2048 的小FFN -> 按分数加权相加 -> 再加上共享专家的输出
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers


__all__ = ["Qwen3NextConfig"]
