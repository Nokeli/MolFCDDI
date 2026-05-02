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


class GroupCoAttentionLayer(nn.Module):
    """
    Group-to-Group Co-Attention: 两个药物的官能团序列互为 Query/Key/Value。
    A 的每个官能团去看 B 的哪些官能团重要，反之亦然。
    """
    def __init__(self, args: Namespace):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = getattr(args, 'num_attention_heads', 4)

        self.attn_AB = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        self.attn_BA = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        self.layer_norm_A = nn.LayerNorm(self.hidden_size)
        self.layer_norm_B = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, fg_A, fg_B, mask_A=None, mask_B=None):
        """
        Args:
            fg_A: (B, L_A, H)
            fg_B: (B, L_B, H)
            mask_A: (B, L_A) bool, True for padding
            mask_B: (B, L_B) bool, True for padding
        Returns:
            updated_A: (B, L_A, H)  – A 看过 B 后的更新序列
            updated_B: (B, L_B, H)  – B 看过 A 后的更新序列
            attn_AB:   (B, L_A, L_B) – A→B 注意力权重
            attn_BA:   (B, L_B, L_A) – B→A 注意力权重
        """
        # A queries B
        co_A, attn_AB = self.attn_AB(
            query=fg_A, key=fg_B, value=fg_B,
            key_padding_mask=mask_B, need_weights=True
        )
        co_A = self.layer_norm_A(self.dropout(co_A) + fg_A)

        # B queries A
        co_B, attn_BA = self.attn_BA(
            query=fg_B, key=fg_A, value=fg_A,
            key_padding_mask=mask_A, need_weights=True
        )
        co_B = self.layer_norm_B(self.dropout(co_B) + fg_B)

        return co_A, co_B, attn_AB, attn_BA


# =============================================================================
# 2.1 序列注意力池化（替代粗暴 mean pooling）
# =============================================================================
class SeqAttentionPool(nn.Module):
    """
    对序列做 attention-based pooling：学一个可学习的 query，
    对序列内各位置加权求和，自动抑制 padding 和无关位置。
    """
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # 可学习的全局 query 向量
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.query)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        """
        Args:
            seq:  (B, L, H)
            mask: (B, L) bool, True for padding
        Returns:
            out:  (B, H)
        """
        B = seq.size(0)
        query = self.query.expand(B, 1, -1)  # (B, 1, H)

        attn_output, attn_weights = self.attn(
            query=query,
            key=seq,
            value=seq,
            key_padding_mask=mask,
            need_weights=True
        )
        out = attn_output.squeeze(1)  # (B, H)
        out = self.layer_norm(self.dropout(out) + query.squeeze(1))
        return out


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
        self.group_co_attn = GroupCoAttentionLayer(args)  # group-to-group co-attention

        # ====================== 2.5 DDI类型嵌入表（新设计） ======================
        # 可学习的DDI类型嵌入表，用于提供DDI交互类型的先验知识
        self.ddi_type_embedding = nn.Parameter(
            torch.randn(86, args.hidden_size),
            requires_grad=True
        )

        # 可学习的缩放因子，用于平衡全局结构信息与DDI类型嵌入
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.alpha.data.fill_(0.1)

        # ====================== 2.6 序列注意力池化（替代 mean pooling）======================
        self.pool_A = SeqAttentionPool(args.hidden_size, num_heads=getattr(args, 'num_attention_heads', 4), dropout=args.dropout)
        self.pool_B = SeqAttentionPool(args.hidden_size, num_heads=getattr(args, 'num_attention_heads', 4), dropout=args.dropout)

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

        # BPR binary: 强制输出 1-dim score
        layers.extend([
            dropout,
            nn.Linear(args.ffn_hidden_size, 1)
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

        # ====================== Step 2: Group-to-Group Co-Attention ======================
        mask_A = (fg_A.sum(dim=-1) == 0)
        mask_B = (fg_B.sum(dim=-1) == 0)
        co_A, co_B, attn_AB, attn_BA = self.group_co_attn(fg_A, fg_B, mask_A, mask_B)

        # 将更新后的 co-attention 序列池化为全局交互上下文
        ctx_for_A = self.pool_A(co_A, mask=mask_A)   # (B, H)
        ctx_for_B = self.pool_B(co_B, mask=mask_B)   # (B, H)

        # ====================== Step 3: 门控融合 ======================
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
        score = self.ffn(combined).squeeze(-1)  # (B, 1) -> (B,)
        return score


# =============================================================================
# 构建函数
# =============================================================================
def build_ddi_model(args: Namespace) -> nn.Module:
    """
    统一入口函数
    """
    # BPR binary: 固定输出 1-dim score
    args.output_size = 1

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
