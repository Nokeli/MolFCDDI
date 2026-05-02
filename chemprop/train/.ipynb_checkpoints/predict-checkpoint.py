from typing import List

import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, StandardScaler
import numpy as np


def predict(model: nn.Module,
            pretrain: bool,
            data,
            batch_size: int,
            scaler: StandardScaler = None,
            pooling_type: str = 'attention') -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param pooling_type: 注意力聚合类型 ('sum', 'max', 'attention')
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    true_label = []
    num_iters, iter_step = len(data), batch_size

    for batch_idx, batch_data in enumerate(data):
            # batch_data是一个字典,包含batch_graph1, batch_graph2, labels等
        batch1 = batch_data['batch_graph1']
        batch2 = batch_data['batch_graph2']
        labels = batch_data['labels']
        features = batch_data['features']

        step = 'finetune'
        with torch.no_grad():
            batch_preds = model(step, pretrain, batch1, batch2, pooling_type=pooling_type)
        batch_preds = batch_preds.cpu().numpy()
        labels = labels
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        true_label.extend(labels)

    return preds,true_label


def get_emb(model: nn.Module,
            prompt: bool,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        step = 'pretrain'
        with torch.no_grad():
            batch_embs = model.encoder(step, prompt, batch, features_batch)

        batch_embs = batch_embs.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_embs = scaler.inverse_transform(batch_embs)
        
        # Collect vectors
        # batch_embs = batch_embs.tolist()
        # embs.extend(batch_embs)
        if i == 0:
            embs = batch_embs
        else:
            embs = np.vstack((embs, batch_embs)) 

    return embs