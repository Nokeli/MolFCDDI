import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import math


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


class CrossAttentionLayer(nn.Module):
    """交叉注意力层，用于计算两个药物之间的交互"""

    def __init__(self, args):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = getattr(args, 'num_attention_heads', 4)
        self.head_dim = self.hidden_size // self.num_heads

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Query, Key, Value projections
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def forward(self, query_input, key_value_input, mask=None):
        """
        Args:
            query_input: [batch, seq_len_q, hidden_size]
            key_value_input: [batch, seq_len_kv, hidden_size]
            mask: [batch, seq_len_q, seq_len_kv] or None
        Returns:
            output: [batch, seq_len_q, hidden_size]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_kv]
        """
        batch_size = query_input.size(0)

        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key_value_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(key_value_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_kv]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len_q, head_dim]

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_proj(context)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.LayerNorm(output + query_input)

        return output, attn_weights


class DDI_Prompt_Generator(nn.Module):
    """
    用于DDI任务的Prompt生成器，支持双药物交互建模
    """

    def __init__(self, args: Namespace):
        super(DDI_Prompt_Generator, self).__init__()
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
        self.act_func = nn.ReLU() if not hasattr(args, 'activation') else self._get_activation(args.activation)

        # 单药物内部的自注意力层
        self.fg = nn.Parameter(torch.randn(1, self.hidden_size * 3), requires_grad=True)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)

        num_attention = getattr(args, 'num_attention', 2)
        self.self_atten_layers = nn.ModuleList([AttentionLayer(args) for _ in range(num_attention)])

        # DDI交互相关的组件
        # 可学习的交互查询向量
        self.fg_inter = nn.Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.1)

        # 双向交叉注意力层
        self.cross_attn_A2B = CrossAttentionLayer(args)
        self.cross_attn_B2A = CrossAttentionLayer(args)

        # 投影层
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size)

        # 交互融合层
        self.interaction_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        # 最终的交互表征层
        self.final_interaction = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )

    def _get_activation(self, activation_name):
        if activation_name == 'ReLU':
            return nn.ReLU()
        elif activation_name == 'LeakyReLU':
            return nn.LeakyReLU(0.1)
        elif activation_name == 'GELU':
            return nn.GELU()
        else:
            return nn.ReLU()

    def _process_single_drug(self, f_atom, atom_scope, f_group, group_scope, mapping, mapping_scope):
        """
        处理单个药物的官能团表征

        Args:
            f_atom: 原子特征 [num_atoms, hidden_size]
            atom_scope: 原子范围 [(start, size), ...]
            f_group: 官能团embedding [num_groups, hidden_size]
            group_scope: 官能团范围 [(start, size), ...]
            mapping: 映射索引
            mapping_scope: 映射范围

        Returns:
            frag_features: 片段特征（未经自注意力） [batch, max_frag_size, hidden_size]
        """
        max_frag_size = max([g_size for _, g_size in group_scope]) + 1
        f_frag_list = []
        padding_zero = torch.zeros((1, self.hidden_size), device=f_atom.device)

        for i, (g_start, g_size) in enumerate(group_scope):
            a_start, a_size = atom_scope[i]
            m_start, m_size = mapping_scope[i]

            cur_a = f_atom.narrow(0, a_start, a_size)
            cur_g = f_group.narrow(0, g_start, g_size)
            cur_m = mapping.narrow(0, m_start, m_size)

            cur_a = torch.cat([padding_zero, cur_a], dim=0)
            cur_a = cur_a[cur_m]

            # 聚合原子特征: [sum, max, group_embedding]
            cur_g = torch.cat([cur_a.sum(dim=1), cur_a.max(dim=1)[0], cur_g], dim=1)
            cur_brics = torch.cat([self.fg, cur_g], dim=0)
            cur_frage = torch.nn.ZeroPad2d((0, 0, 0, max_frag_size - cur_brics.shape[0]))(cur_brics)
            f_frag_list.append(cur_frage.unsqueeze(0))

        f_frag_list = torch.cat(f_frag_list, 0)  # [batch, max_frag_size, hidden_size*3]
        f_frag_list = self.act_func(self.lr(f_frag_list))  # [batch, max_frag_size, hidden_size]

        return f_frag_list

    def compute_interaction(self, drug_A_features, drug_B_features):
        """
        计算两个药物之间的交互

        Args:
            drug_A_features: Drug A的片段特征 [batch, seq_len_A, hidden_size]
            drug_B_features: Drug B的片段特征 [batch, seq_len_B, hidden_size]

        Returns:
            interaction_repr: 交互表征 [batch, hidden_size]
            attn_A2B: A对B的注意力权重 [batch, num_heads, seq_len_A, seq_len_B]
            attn_B2A: B对A的注意力权重 [batch, num_heads, seq_len_B, seq_len_A]
        """
        # 创建padding mask
        mask_A = (drug_A_features.sum(dim=-1) != 0).float()  # [batch, seq_len_A]
        mask_B = (drug_B_features.sum(dim=-1) != 0).float()  # [batch, seq_len_B]

        # A对B的交叉注意力: Q来自A, K和V来自B
        mask_A2B = torch.matmul(mask_A.unsqueeze(-1), mask_B.unsqueeze(1))  # [batch, seq_len_A, seq_len_B]
        inter_A2B, attn_A2B = self.cross_attn_A2B(drug_A_features, drug_B_features, mask_A2B)

        # B对A的交叉注意力: Q来自B, K和V来自A
        mask_B2A = torch.matmul(mask_B.unsqueeze(-1), mask_A.unsqueeze(1))  # [batch, seq_len_B, seq_len_A]
        inter_B2A, attn_B2A = self.cross_attn_B2A(drug_B_features, drug_A_features, mask_B2A)

        # 聚合交互信息（使用第0个位置的全局表征）
        inter_A2B_global = inter_A2B[:, 0, :]  # [batch, hidden_size]
        inter_B2A_global = inter_B2A[:, 0, :]  # [batch, hidden_size]

        # 获取原始药物表征
        drug_A_global = drug_A_features[:, 0, :]
        drug_B_global = drug_B_features[:, 0, :]

        # 融合多种交互信息
        # 1. 直接拼接
        concat_repr = torch.cat([inter_A2B_global, inter_B2A_global,
                                 drug_A_global * drug_B_global], dim=-1)
        fused_repr = self.interaction_fusion(concat_repr)

        # 2. 最终交互表征
        final_concat = torch.cat([drug_A_global + drug_B_global, fused_repr], dim=-1)
        interaction_repr = self.final_interaction(final_concat) * self.beta

        return interaction_repr, attn_A2B, attn_B2A

    def forward(self, drug_A_data, drug_B_data):
        """
        前向传播，处理DDI任务

        核心流程:
        1. 提取Drug A和Drug B的片段特征（未经自注意力）
        2. 在片段级别计算双向交叉注意力（A的片段看B的片段，B的片段看A的片段）
        3. 对交互后的特征进行自注意力处理
        4. 融合得到最终的DDI表征

        Args:
            drug_A_data: dict包含Drug A的数据
                - f_atom: 原子特征
                - atom_scope: 原子范围
                - f_group: 官能团embedding
                - group_scope: 官能团范围
                - mapping: 映射索引
                - mapping_scope: 映射范围
            drug_B_data: dict包含Drug B的数据（格式同上）

        Returns:
            interaction_repr: DDI交互表征 [batch, hidden_size]
            attention_maps: dict包含各种注意力权重
            drug_A_repr: Drug A单独表征 [batch, hidden_size]
            drug_B_repr: Drug B单独表征 [batch, hidden_size]
        """
        # Step 1: 提取Drug A和Drug B的原始片段特征（未经自注意力）
        drug_A_frags = self._process_single_drug(
            drug_A_data['f_atom'], drug_A_data['atom_scope'],
            drug_A_data['f_group'], drug_A_data['group_scope'],
            drug_A_data['mapping'], drug_A_data['mapping_scope']
        )  # [batch, max_frag_A, hidden_size]

        drug_B_frags = self._process_single_drug(
            drug_B_data['f_atom'], drug_B_data['atom_scope'],
            drug_B_data['f_group'], drug_B_data['group_scope'],
            drug_B_data['mapping'], drug_B_data['mapping_scope']
        )  # [batch, max_frag_B, hidden_size]

        # Step 2: 计算片段级别的双向交叉注意力
        # 这里的交叉注意力是在片段级别进行的
        interaction_repr, attn_A2B, attn_B2A = self.compute_interaction(
            drug_A_frags, drug_B_frags
        )

        # Step 3: 对原始片段特征进行自注意力处理（用于单药物表征）
        # Drug A的自注意力
        hidden_A, self_att_A = self.self_atten_layers[0](drug_A_frags, drug_A_frags)
        for att_layer in self.self_atten_layers[1:]:
            hidden_A, self_att_A = att_layer(hidden_A, drug_A_frags)

        f_out_A = self.linear(hidden_A)
        f_out_A = self.norm(f_out_A) * self.alpha
        drug_A_repr = f_out_A[:, 0, :]  # [batch, hidden_size]

        # Drug B的自注意力
        hidden_B, self_att_B = self.self_atten_layers[0](drug_B_frags, drug_B_frags)
        for att_layer in self.self_atten_layers[1:]:
            hidden_B, self_att_B = att_layer(hidden_B, drug_B_frags)

        f_out_B = self.linear(hidden_B)
        f_out_B = self.norm(f_out_B) * self.alpha
        drug_B_repr = f_out_B[:, 0, :]  # [batch, hidden_size]

        # 返回结果和注意力权重（用于可视化）
        attention_maps = {
            'self_att_A': self_att_A,
            'self_att_B': self_att_B,
            'cross_att_A2B': attn_A2B,  # [batch, num_heads, max_frag_A, max_frag_B]
            'cross_att_B2A': attn_B2A  # [batch, num_heads, max_frag_B, max_frag_A]
        }

        return interaction_repr, attention_maps, drug_A_repr, drug_B_repr


# 使用示例
if __name__ == "__main__":
    # 创建模拟的args
    class Args:
        pharm_fdim = 182
        react_fdim = 34
        hidden_size = 300
        bias = False
        depth = 3
        dropout = 0.1
        undirected = False
        atom_messages = True
        features_only = False
        use_input_features = False
        activation = 'ReLU'
        num_attention = 2
        num_attention_heads = 4


    args = Args()
    model = DDI_Prompt_Generator(args)

    # 模拟输入数据
    batch_size = 4
    num_atoms_A = 20
    num_groups_A = 5
    num_atoms_B = 15
    num_groups_B = 4

    drug_A_data = {
        'f_atom': torch.randn(num_atoms_A, args.hidden_size),
        'atom_scope': [(0, num_atoms_A)],
        'f_group': torch.randn(num_groups_A, args.hidden_size),
        'group_scope': [(0, num_groups_A)],
        'mapping': torch.randint(0, num_atoms_A, (num_groups_A, 3)),
        'mapping_scope': [(0, num_groups_A)]
    }

    drug_B_data = {
        'f_atom': torch.randn(num_atoms_B, args.hidden_size),
        'atom_scope': [(0, num_atoms_B)],
        'f_group': torch.randn(num_groups_B, args.hidden_size),
        'group_scope': [(0, num_groups_B)],
        'mapping': torch.randint(0, num_atoms_B, (num_groups_B, 3)),
        'mapping_scope': [(0, num_groups_B)]
    }

    # 前向传播
    interaction_repr, attention_maps, drug_A_repr, drug_B_repr = model(drug_A_data, drug_B_data)

    print(f"Interaction representation shape: {interaction_repr.shape}")
    print(f"Drug A representation shape: {drug_A_repr.shape}")
    print(f"Drug B representation shape: {drug_B_repr.shape}")
    print(f"Cross attention A2B shape: {attention_maps['cross_att_A2B'].shape}")
    print(f"Cross attention B2A shape: {attention_maps['cross_att_B2A'].shape}")