"""
BPR Loss for Cold Start DDI Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    优化 pos_score > neg_score
    兼容 inductive_train.py 接口: forward 返回 (loss, loss_p, loss_n)
    """
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        """
        Args:
            pos_score: (B,) 正样本对 score
            neg_score: (B,) 负样本对 score
        Returns:
            loss, loss_p, loss_n  （三个相同值，兼容对方 unpack）
        """
        diff = pos_score - neg_score - self.margin
        loss = -F.logsigmoid(diff).mean()
        return loss, loss, loss
