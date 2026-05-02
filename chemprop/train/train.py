from argparse import Namespace
import logging
from typing import Callable, List, Union

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from torch.utils.data import DataLoader
from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
import pdb
from tqdm import tqdm

def train(model: nn.Module,
          pretrain: bool,
          data,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: bool = False):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param pretrain: Whether in pretrain mode.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :return: Tuple of (average loss, total number of iterations trained).
    """

    model.train()
    #data.shuffle()

    loss_sum, iter_count = 0, 0
    num_iters = len(data)
    iter_size = args.batch_size
    train_pbar = tqdm(data)
    for batch_idx, batch_data in enumerate(train_pbar):
            # batch_data是一个字典,包含batch_graph1, batch_graph2, labels等
        batch1 = batch_data['batch_graph1']
        batch2 = batch_data['batch_graph2']
        targets = batch_data['labels'].long()
        features = batch_data['features']

        if next(model.parameters()).is_cuda:
            targets = targets.cuda()

        # Run model
        model.zero_grad()

        step = 'finetune'
        # 使用注意力聚合（默认使用attention pooling）
        pooling_type = getattr(args, 'pooling_type', 'attention')
        preds = model(step, pretrain, batch1, batch2, pooling_type=pooling_type)
        # Calculate loss for multiclass classification
        loss = loss_func(preds, targets)

        # 添加L2正则化项（可选）
        # gamma = getattr(args, 'gamma', 0.0)
        # if gamma > 0:
        #     reg_loss = gamma * torch.norm(model.alpha, p=2) ** 2
        #     loss = loss + reg_loss
        
        loss_sum += loss.item()
        iter_count += 1

        # Backward pass
        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        # n_iter += len(mol_batch)
    
    avg_loss = loss_sum / iter_count if iter_count > 0 else 0
    return avg_loss

