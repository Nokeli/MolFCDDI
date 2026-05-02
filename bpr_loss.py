"""
Sigmoid BCE Loss for Cold Start DDI Training
对齐参考代码 inductive_train.py 的 custom_loss.SigmoidLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Sigmoid BCE Loss (对齐参考代码的 SigmoidLoss)
    pos_score -> +∞ (sigmoid -> 1), neg_score -> -∞ (sigmoid -> 0)
    返回 (total_loss, pos_loss, neg_loss) 兼容原接口
    """
    def __init__(self):
        super().__init__()

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        p_loss = -F.logsigmoid(pos_score).mean()      # BCE(pos, 1)
        n_loss = -F.logsigmoid(-neg_score).mean()     # BCE(neg, 0)
        loss = (p_loss + n_loss) / 2
        return loss, p_loss, n_loss
