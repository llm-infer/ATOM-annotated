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
"""Qwen3.5 model configuration"""

# 【译】Qwen3.5模型配置
#
# 本文件给Qwen3.5（dense版，多模态）定义HF配置类：Qwen3_5TextConfig（文本）、Qwen3_5VisionConfig（视觉）、
#   Qwen3_5Config（顶层），改编自transformers / vLLM的同名配置（见文件头版权行）
# Qwen3.5的文本部分是混合架构：默认每4层里前3层是线性注意力（Gated DeltaNet，GDN），第4层是全注意力，
#   所以除了常规注意力字段，还多了一组linear_*字段描述线性注意力层；结构与Qwen3-Next（qwen3_next.py）相同
#   （GDN是什么、为什么3层GDN配1层全注意力，见qwen3_next.py文件头科普②③）
# 调用方：
#   models/qwen3_5.py：用Qwen3_5TextConfig做类型标注，并把Qwen3_5Config再导出给vLLM插件
#   plugin/vllm/models/qwen3_5.py：从models/qwen3_5.py导入Qwen3_5Config做类型标注
# 注意：仓库里没有地方直接实例化这些类。原生路径的hf_config来自config.get_hf_config：
#   model_type="qwen3_5"时抽出config.json里的text_config，交给transformers的AutoConfig.for_model构造（见本批config.py）
#   所以这里的类主要充当类型标注和"有哪些字段"的说明
# 对比nano-vllm：nano-vllm只支持Qwen3，直接用AutoConfig；ATOM要支持transformers未必内置的新模型，才有model_config/这个目录
# 【分支】Qwen3.5模型实现与GDN线性注意力见第8批（"第N批"指仓库根目录ANNOTATION_PLAN.md里的阅读批次）
# 下面的科普①②被qwen3_5_moe.py、kimi_k25.py和model_engine/sequence.py按编号引用
#
# =====================================================================================
# 【科普】读这个文件需要的概念（GQA、GDN线性注意力、混合层、MoE、部分RoPE已在 qwen3_next.py 文件头讲过，先读那里）
# =====================================================================================
#
# ① 多模态模型 = 视觉编码器 + 投影 + 语言模型
#   语言模型只认token序列，图片要先变成"一串向量"才能塞进去：
#     图片 -> 切成小方块（patch，这里16x16像素）-> 每个patch展平成向量 -> 过一个ViT（视觉Transformer，
#     本质就是你熟悉的Transformer编码器，双向注意力）-> 相邻2x2个patch合并成1个（spatial_merge_size=2，
#     减少4倍token数）-> 投影到语言模型的hidden维度 -> 替换掉prompt里的图片占位token。
#   例：一张448x448的图 -> 28x28=784个patch -> 合并后196个"视觉token"，在prompt里占196个位置。
#     图片 448x448 ─切16x16─> 784个patch ─ViT─> 784个向量 ─2x2合并─> 196个向量 ─投影─> 196个hidden向量
#                                                                                         │
#     prompt：    [文][文] <vision_start> <image> x 196 <vision_end> [文] ...                │
#     embedding： 文本位置照常查词表；196个<image>位置的embedding换成上面这196个向量 <───────┘
#     替换之后，视觉token和文本token一样进语言模型：一样做注意力、一样占KV cache
#   视频再多一个时间维：每2帧一组（temporal_patch_size=2）切成3D的patch。
#   本文件里：Qwen3_5VisionConfig描述视觉编码器，Qwen3_5TextConfig描述语言模型，Qwen3_5Config把两者包起来，
#     再加上占位token的id（image_token_id等）。
#   ATOM原生模式也跑视觉部分：get_hf_config把text_config当作hf_config，另把完整config存进Config.multimodal_config；
#     ModelRunner按架构名Qwen3_5ForConditionalGeneration加载models/qwen3_5.py的Qwen3_5MultimodalModel
#     （它构造models/qwen3_5_vl.py的Qwen3VisionTransformer；用法见recipes/Qwen3.5_multimodel.md、atom/examples/multimodal_inference.py）。
#     vLLM插件模式则不抽text_config，多模态交给vLLM处理（第10批）。
#
# ② M-RoPE（多模态RoPE，rope_parameters里的 mrope_section / mrope_interleaved）
#   普通RoPE里每个token只有一个位置编号（第几个token）。图片是二维的，视频是三维的，
#     只用一个序号会丢掉"上下左右"的空间关系。
#   M-RoPE把每个token的位置拆成三个编号 (t, h, w)：时间、行、列。文本token三个编号相同（退化成普通RoPE），
#     图片token的h、w取它在图上的行列。RoPE的旋转维度也按 mrope_section 分成三段，
#     分别用t、h、w的位置去旋转；mrope_interleaved表示三段在维度上是交错排布还是连续排布。
#   例（为了好读，假设图片合并后只有2x2=4个视觉token；排法与HF transformers的Qwen3-VL get_rope_index相同，
#     ATOM经models/qwen3_5.py的build_qwen3_5_mrope_input_positions调用aiter的MRotaryEmbedding.get_input_positions计算）：
#       token：    文A      文B      <vs>     图(0,0)  图(0,1)  图(1,0)  图(1,1)  <ve>     文C
#       普通RoPE： 0        1        2        3        4        5        6        7        8
#       M-RoPE：   (0,0,0)  (1,1,1)  (2,2,2)  (3,3,3)  (3,3,4)  (3,4,3)  (3,4,4)  (5,5,5)  (6,6,6)
#     - 文本token三个编号相同；图片token的t取起点3，h=3+行号、w=3+列号
#     - 图片之后的文本从"已用的最大编号+1"=5接着排，所以9个token只用到0~6这7个编号
#   mrope_position_delta = 最大编号 + 1 − token数 = 7 − 9 = −2：之后生成的都是文本token，位置 = 线性位置 + delta，
#     第1个生成token的线性位置是9 -> M-RoPE位置7，正好接在6后面（Sequence.mrope_position_delta；
#     decode时在model_ops/attentions/backends.py里加上）
#   一张448x448的图是14x14=196个视觉token，却只占14个编号，所以每张这样的图让delta减少196−14=182——delta通常是负数
#   ATOM读取方：ModelRunner看rope_parameters里有没有mrope_section决定是否按M-RoPE准备位置（每个token 3个位置号），
#     triton_mrope.py、triton_fused_qkv_norm_rope_cache.py 按 mrope_section 分段旋转（第6批）。
#
# ③ dense 与 MoE
#   本文件是dense版：每层的FFN是一个普通MLP（intermediate_size）。qwen3_5_moe.py 是MoE版，
#     把FFN换成路由专家 + 共享专家（见qwen3_next.py科普⑤），其余（注意力、GDN、视觉部分）完全一样。
# =====================================================================================
from transformers.configuration_utils import PretrainedConfig, layer_type_validation


# 文本模型配置（model_type="qwen3_5_text"），对应config.json里的text_config
#   下面的默认值只是占位，实际值都从checkpoint的config.json读入
class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"
    # 以下几个类属性是HF transformers的约定写法，ATOM运行时不读取它们：
    #   keys_to_ignore_at_inference：HF推理时从模型输出里忽略的键
    #   base_model_tp_plan：HF自带的张量并行切分方案（colwise按列切、rowwise按行切）；ATOM用自己的并行Linear（第6批）
    #   base_model_pp_plan：HF的流水线并行方案；ATOM的PP见第9批
    #   base_config_key：说明本类在顶层配置里对应哪个键（"text_config"）
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    # 字段速查：
    #   全注意力层：num_attention_heads、num_key_value_heads（GQA）、head_dim、attention_bias、rope_parameters
    #   GDN线性注意力层：linear_*（含义见qwen3_next.py科普②）
    #   dense FFN：intermediate_size、hidden_act（SwiGLU里gate分支的激活函数silu；SwiGLU见qwen3_next.py科普⑤）
    #     例（默认值）：每层 gate、up 各 4096->12288，down 12288->4096，共 3 × 4096 × 12288 ≈ 1.51亿参数，每个token全部用到
    #   层排布：layer_types（或config.json里的full_attention_interval）
    #   训练专用：initializer_range、attention_dropout
    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
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
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        # 告诉transformers的RoPE参数校验：rope_parameters里的mrope_section / mrope_interleaved是允许的额外键
        #   M-RoPE（多模态RoPE）把旋转维度分给时间、高、宽三种位置，图文混排的输入要用它（见文件头科普②）
        #   不加这句，transformers会把这两个"不认识"的键当成错误配置报警
        kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        # partial_rotary_factor=0.25：只对head_dim的前1/4做RoPE；用setdefault写进kwargs，在super().__init__(**kwargs)里成为配置属性
        #   例：head_dim=256 -> 只有64维带位置信息；用了M-RoPE时，mrope_section切分的也是这64维
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # layer_types：每层是"linear_attention"还是"full_attention"；config.json没给时按full_attention_interval（默认4）生成
        #   第i层满足(i+1)%4==0时是全注意力。例：32层时第3、7、11、...、31层是全注意力，其余是GDN线性注意力
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
        # layer_type_validation（transformers提供）：检查每项取值合法，并检查列表长度等于num_hidden_layers
        # ATOM读取方：models/qwen3_5.py 按 layer_types[i] 决定第i层建GDN层还是全注意力层
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # linear attention part
        # 【译】线性注意力部分
        # linear_conv_kernel_dim：GDN层里q/k/v先过一个短因果卷积，这是卷积核长度
        # linear_num_key_heads / linear_num_value_heads：默认16个k头、32个v头，即每个q/k头被2个v头共享
        # linear_key_head_dim / linear_value_head_dim：线性注意力每个头的维度
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
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


# 视觉编码器（ViT）配置；model_type与顶层相同（"qwen3_5"），靠base_config_key="vision_config"表明自己是子配置
#   depth=27层、hidden_size=1152、num_heads=16：ViT本身的大小（一个标准的Transformer编码器，双向注意力、无KV cache）
#   in_channels=3：RGB三通道；num_position_embeddings=2304：ViT的可学习位置编码表大小（48x48个patch位置），
#     图片更大/更小时会对它插值
#   patch_size=16：图片切成16x16的patch；spatial_merge_size=2：相邻2x2个patch合并成1个视觉token
#   temporal_patch_size=2：视频按2帧一组切patch；out_hidden_size：视觉特征输出维度，需与文本hidden_size对齐
class Qwen3_5VisionConfig(PretrainedConfig):
    model_type = "qwen3_5"
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


# 顶层多模态配置（model_type="qwen3_5"），对应checkpoint的整个config.json
#   sub_configs：HF的子配置声明，告诉transformers vision_config / text_config分别用哪个类解析
class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Qwen3_5TextConfig,
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
        # vision_config / text_config：是dict时用对应的子配置类构造；是None时用默认值
        #   代码本身只处理这两种情况：传入已构造好的配置对象时，这里不会给self.vision_config / self.text_config赋值
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        # image_token_id / video_token_id：prompt里图片、视频占位token；vision_start / vision_end：视觉段的起止标记token
        # 例：一张图片在prompt里展开成 <vision_start> <image> <image> ... <image> <vision_end>，
        #   中间<image>的个数等于这张图的视觉token数（见文件头科普①）；推理时这些位置的embedding
        #   会被替换成视觉编码器的输出
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid v4 PretrainedConfig overwrite
        # 【译】在super().__init__()之后设置，避免被transformers v4的PretrainedConfig覆盖
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig"]
