from argparse import Namespace

from .augmented_encoder import PharmHGT
from .cmpn import CMPN
import torchvision
from chemprop.nn_utils import get_activation_function, initialize_weights,index_select_ND
import pdb
from functools import partial
import logging
from mimetypes import init
from turtle import forward, hideturtle, up
import torch
import torch.nn as nn
from typing import NamedTuple, Union, Callable
import torch.nn.functional as F
import math
import copy
import numpy as np
from dgl import function as fn
from chemprop.features.featurization import get_atom_fdim,get_bond_fdim,get_pharm_fdim

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, pretrain: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

    def create_encoder(self, args: Namespace, encoder_name):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        if encoder_name == 'CMPNN':
            self.encoder = CMPN(args)
        elif encoder_name == 'PharmHGT':
            self.encoder = PharmHGT(args)
    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size),
                activation,
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        if not self.pretrain:
            output = self.encoder(*input)
            output = self.ffn(output)

            # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
            if self.classification and not self.training:
                output = self.sigmoid(output)
            if self.multiclass:
                output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
                if not self.training:
                    output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        else:
            output = self.ffn(self.encoder(*input))

        return output


class CrossAttentionLayer(nn.Module):
    """
    跨分子注意力：用一个药物的全局向量 Query 另一个药物的官能团序列（Key/Value）
    输出：该药物"看到"的对方关键官能团上下文 + 注意力权重图（可解释性热力图）
    """
    def __init__(self, args: Namespace):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = getattr(args, 'num_attention_heads', 4)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query_global, key_fgs, key_padding_mask=None):
        """
        Args:
            query_global: (B, H)           - 对方药物的全局表征
            key_fgs:      (B, L, H)         - 本药物的官能团序列（含 padding）
            key_padding_mask: (B, L) bool   - True 表示该位置是 padding
        Returns:
            ctx:          (B, H)            - 交互上下文向量
            attn_weights: (B, 1, L)         - 注意力分布（哪个官能团被关注）
        """
        B = query_global.size(0)
        query = query_global.unsqueeze(1)  # (B, 1, H)

        if key_padding_mask is not None:
            # key_padding_mask 应该是 (B, L) 的布尔张量，True 表示需要mask
            attn_mask = key_padding_mask  # PyTorch的MultiheadAttention会自动处理
        else:
            attn_mask = None

        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key_fgs,
            value=key_fgs,
            key_padding_mask=attn_mask,  # 使用 key_padding_mask 参数
            need_weights=True,
        )

        ctx = attn_output.squeeze(1)  # (B, H)
        return ctx, attn_weights  # attn_weights 可用于可视化


# =============================================================================
# 2. 门控融合层（防止信息过平滑，保留原始语义）
# =============================================================================
class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_z = nn.Linear(hidden_size * 2, hidden_size)  # update gate
        self.gate_r = nn.Linear(hidden_size * 2, hidden_size)  # candidate gate
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_original, h_interaction):
        combined = torch.cat([h_original, h_interaction], dim=-1)
        z = self.sigmoid(self.gate_z(combined))   # (B, H) 决定保留多少交互信息
        r = self.sigmoid(self.gate_r(combined))      # 候选更新向量
        h_new = (1 - z) * h_original + z * r
        return h_new


# =============================================================================
# 3. 主模型：DDIInteractionModel（支持二分类 / 多分类 / 多标签）
# =============================================================================
class DDIInteractionModel(nn.Module):
    def __init__(self, classification: bool, multiclass: bool, pretrain: bool, args: Namespace):
        super().__init__()
        self.args = args
        self.classification = classification
        self.multiclass = multiclass

        # 激活函数
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        if self.multiclass:
            self.softmax = nn.Softmax(dim=1)
            self.num_classes = args.multiclass_num_classes

        # ====================== 1. 共享编码器（Siamese） ======================
        self.encoder = self._create_encoder(args)

        # ====================== 2. 跨分子交互模块 ======================
        self.cross_attn_A_views_B = CrossAttentionLayer(args)  # A 查询 B 的官能团
        self.cross_attn_B_views_A = CrossAttentionLayer(args)  # B 查询 A 的官能团

        # ====================== 2.5 DDI类型嵌入表（新设计） ======================
        # 可学习的DDI类型嵌入表，用于提供DDI交互类型的先验知识
        self.ddi_type_embedding = nn.Parameter(
            torch.randn(86, args.hidden_size),
            requires_grad=True
        )

        # 可学习的缩放因子，用于平衡全局结构信息与DDI类型嵌入
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.alpha.data.fill_(0.1)

        # ====================== 3. 门控融合 ======================
        self.gate_A = GatedFusion(args.hidden_size)
        self.gate_B = GatedFusion(args.hidden_size)

        # ====================== 4. 预测头 FFN ======================
        self.create_ffn(args)

    def _create_encoder(self, args):
        encoder_name = args.encoder_name
        if encoder_name == 'CMPNN':
            return CMPN(args)
        elif encoder_name == 'PharmHGT':
            return PharmHGT(args)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def create_ffn(self, args):
        """特征组合方式：concat + DDI类型嵌入"""
        input_dim = args.hidden_size * 2  # [hA, hB, α·P_DDI]

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        layers = [
            dropout,
            nn.Linear(input_dim, args.ffn_hidden_size),
            activation,
        ]

        # 多分类需要更大输出
        output_size = args.output_size
        if self.multiclass:
            output_size = args.output_size * self.num_classes

        layers.extend([
            dropout,
            nn.Linear(args.ffn_hidden_size, output_size)
        ])

        self.ffn = nn.Sequential(*layers)

    def forward(self, step, pretrain, batch_graph_A, batch_graph_B, pooling_type, ddi_type=None):
        """
        前向传播

        Args:
            step: 训练步骤
            pretrain: 是否预训练
            batch_graph_A: 药物A的图数据
            batch_graph_B: 药物B的图数据
            ddi_type: DDI类型标签（可选，用于训练时查询DDI类型嵌入）

        Returns:
            logits: 预测logits
            attn_weights: 注意力权重（用于可视化）
        """
        # ====================== Step 1: 独立编码 ======================
        fg_A, h_global_A, h_global_A_fp = self.encoder(step, pretrain, batch_graph_A)   # fg: (B, max_fg, H)
        fg_B, h_global_B, h_global_B_fp = self.encoder(step, pretrain, batch_graph_B)
        fg_A = fg_A[:, 1:, :]
        fg_B = fg_B[:, 1:, :]

        # # ====================== Step 2: 跨分子注意力 ======================
        # # B 看 A：B 的全局向量查询 A 的哪些官能团重要
        mask_A = (fg_A.sum(dim=-1) == 0)
        ctx_for_A, attn_to_A = self.cross_attn_B_views_A(
            query_global=h_global_B,
            key_fgs=fg_A,
            key_padding_mask=mask_A
        )
        # # A 看 B
        mask_B = (fg_B.sum(dim=-1) == 0)
        ctx_for_B, attn_to_B = self.cross_attn_A_views_B(
            query_global=h_global_A,
            key_fgs=fg_B,
            key_padding_mask=mask_B
        )
        # ratio_A = (ctx_for_A - h_global_A).norm(dim=1).mean() / (h_global_A.norm(dim=1).mean() + 1e-9)
        # ratio_B = (ctx_for_B - h_global_B).norm(dim=1).mean() / (h_global_B.norm(dim=1).mean() + 1e-9)
        # print("delta ratio A:", ratio_A.item(), "delta ratio B:", ratio_B.item())
        # # # ====================== Step 3: 门控融合 ======================
        h_A_final = self.gate_A(h_global_A_fp, ctx_for_A)
        h_B_final = self.gate_B(h_global_B_fp, ctx_for_B)

        # # # ====================== Step 4: 特征组合 ======================
        # diff = torch.abs(h_A_final - h_B_final)
        # prod = h_A_final * h_B_final
        #combined = torch.cat([h_A_final, h_B_final, diff, prod], dim=1)
        combined = torch.cat([h_A_final, h_B_final], dim=1)  # (B, 4*H)
        #combined = torch.cat([h_global_A_fp, h_global_B_fp], dim=1)  # (B, 4*H)
        # ====================== Step 4: 获取DDI类型嵌入 ======================
        # 如果提供了DDI类型标签，查询对应的嵌入
        # if ddi_type is not None and self.training:
        #     # ddi_type 应该是 [0, 85] 之间的整数
        #     P_DDI = self.ddi_type_embedding[ddi_type.long()]  # (B, H)
        # else:
        #     # 推理时不使用DDI类型嵌入
        #     P_DDI = torch.zeros(h_global_A.size(0), h_global_A.size(1), device=h_global_A.device)
        #P_DDI = (ctx_for_A + ctx_for_B) / 2

        # ====================== Step 5: 特征组合（新设计） ======================
        # h_final = Concat(h_G_A, h_G_B, α · P_DDI)
        # combined = torch.cat([
        #    h_global_A,
        #    h_global_B,
        #    self.alpha * P_DDI
        # ], dim=1)  # (B, 3*H)

        # ====================== Step 6: 预测 ======================
        logits = self.ffn(combined)

        # 输出激活
        # if self.multiclass:
        #     if not self.training:
        #         logits = self.softmax(logits)
        # elif self.classification and not self.training:
        #         logits = self.sigmoid(logits)

        #return logits, (attn_to_A, attn_to_B)  # attn 可用于画交互热力图
        return logits


# =============================================================================
# 构建函数
# =============================================================================
def build_ddi_model(args: Namespace) -> nn.Module:
    """
    统一入口函数
    """
    # 设置输出维度
    output_size = args.num_tasks  # 一般是 1（二分类）或多个类型数
    args.output_size = output_size

    model = DDIInteractionModel(
        classification=args.dataset_type == 'classification',
        multiclass=args.dataset_type == 'multiclass',
        pretrain=False,
        args=args
    )

    # 参数初始化（非常重要！）
    initialize_weights(model)
    return model


def build_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', pretrain=args.pretrain)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model


def build_pretrain_model(args: Namespace, encoder_name) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.ffn_hidden_size = args.hidden_size//2
    args.output_size = args.hidden_size

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', pretrain=True)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)

    initialize_weights(model)

    return model


def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(self.hidden_size, 32)
        self.w_k = nn.Linear(self.hidden_size, 32)
        self.w_v = nn.Linear(self.hidden_size, 32)
        self.args = args
        self.dense = nn.Linear(32, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)
        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)

        return hidden_states, attn


class Prompt_generator(nn.Module):
    def __init__(self, args:Namespace):
        super(Prompt_generator, self).__init__()
        self.brics_fdim = args.pharm_fdim
        self.react_fdim = args.react_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Activation
        self.act_func = get_activation_function(args.activation)
        # add frage attention
        self.fg = nn.Parameter(torch.randn(1,self.hidden_size*3), requires_grad=True)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        self.atten_layers = nn.ModuleList([AttentionLayer(args) for _ in range(args.num_attention)])
        self.linear = nn.Linear(self.hidden_size,self.hidden_size)
        self.lr = nn.Linear(self.hidden_size*3,self.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size)
    def forward(self, f_atom, atom_scope, f_group, group_scope, mapping, mapping_scope):
        max_frag_size = max([g_size for _,g_size in group_scope])+1 # 加上一行填充位置
        f_frag_list = []
        padding_zero = torch.zeros((1,self.hidden_size)).cuda()
        for i,(g_start, g_size) in enumerate(group_scope):
            a_start,a_size = atom_scope[i]
            m_start,m_size = mapping_scope[i]
            cur_a = f_atom.narrow(0,a_start,a_size)
            cur_g = f_group.narrow(0,g_start,g_size)
            cur_m = mapping.narrow(0,m_start,m_size) #  
            cur_a = torch.cat([padding_zero,cur_a],dim=0) #
            cur_a = cur_a[cur_m]
            cur_g = torch.cat([cur_a.sum(dim=1),cur_a.max(dim=1)[0],cur_g],dim=1)
            cur_brics = torch.cat([self.fg,cur_g],dim=0)
            cur_frage = torch.nn.ZeroPad2d((0,0,0,max_frag_size-cur_brics.shape[0]))(cur_brics)
            f_frag_list.append(cur_frage.unsqueeze(0))
        f_frag_list = torch.cat(f_frag_list, 0)
        f_frag_list = self.act_func(self.lr(f_frag_list))
        hidden_states,self_att = self.atten_layers[0](f_frag_list,f_frag_list)
        for k,att in enumerate(self.atten_layers[1:]):
            hidden_states,self_att = att(hidden_states,f_frag_list)
        f_out = self.linear(hidden_states)
        f_out = self.norm(f_out)* self.alpha
        return f_out,self_att


class PromptGeneratorOutput(nn.Module):
    def __init__(self, args, self_output):
        super(PromptGeneratorOutput, self).__init__()
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.self_out(hidden_states)
        return hidden_states


def prompt_generator_output(args):
    return lambda self_output : PromptGeneratorOutput(args, self_output)


def add_FUNC_prompt(model: nn.Module, args: Namespace = None):
    model.encoder.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.encoder.W_i_atom)
    return model
