import logging
from typing import Callable, List

import torch.nn as nn

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import numpy as np
import torch


# def evaluate_predictions(preds: List[float],
#                          targets: List[float],
#                          metric_func: Callable,
#                          dataset_type: str,
#                          logger: logging.Logger = None) -> float:
#     """
#     Evaluates predictions using a metric function and filtering out invalid targets.
#
#     :param preds: A list of predictions (shape: data_size).
#     :param targets: A list of targets (shape: data_size).
#     :param metric_func: Metric function which takes in a list of targets and a list of predictions.
#     :param dataset_type: Dataset type ('classification' or 'regression').
#     :param logger: Logger.
#     :return: The score based on `metric_func`.
#     """
#     info = logger.info if logger is not None else print
#
#     if len(preds) == 0:
#         return float('nan')
#
#     # Filter out invalid targets (e.g., targets that are None)
#     valid_preds = []
#     valid_targets = []
#     for i in range(len(preds)):
#         if targets[i] is not None:  # Skip those without targets
#             valid_preds.append(preds[i])
#             valid_targets.append(targets[i])
#
#     # Compute metric
#     if len(valid_targets) == 0:
#         return float('nan')
#
#     # Skip if all targets or preds are identical, otherwise we'll crash during classification
#     if dataset_type == 'classification':
#         nan = False
#         if all(target == 0 for target in valid_targets) or all(target == 1 for target in valid_targets):
#             nan = True
#             # info('Warning: Found a task with targets all 0s or all 1s')
#         if all(pred == 0 for pred in valid_preds) or all(pred == 1 for pred in valid_preds):
#             nan = True
#             # info('Warning: Found a task with predictions all 0s or all 1s')
#
#         if nan:
#             return float('nan')
#
#     # For multiclass classification, ensure labels range is provided for metric function
#     if dataset_type == 'multiclass':
#         # return metric_func(valid_targets, valid_preds, labels=list(range(len(set(valid_targets)))))
#         return metric_func(valid_targets, valid_preds,labels=list(range(86)))
#     else:
#         return metric_func(valid_targets, valid_preds)

def evaluate_predictions(preds: List[float],
                         targets: List[float],
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None) -> dict:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.
    Returns a dictionary with loss and multiple evaluation metrics for classification tasks.

    :param preds: A list of predictions (shape: data_size x num_classes for multiclass).
    :param targets: A list of targets (shape: data_size).
    :param metric_func: Loss function to calculate the loss (e.g., cross-entropy).
    :param dataset_type: Dataset type ('classification', 'multiclass').
    :param logger: Logger.
    :return: A dictionary with loss and evaluation metrics (Accuracy, AUC, AUPR, F1-score).
    """
    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return {"loss": float('nan'), "accuracy": float('nan'), "auc": float('nan'), "aupr": float('nan'), "f1": float('nan')}

    # Convert to tensors for loss calculation
    preds_tensor = torch.tensor(preds)
    targets_tensor = torch.tensor(targets).long()
    preds_tensor = preds_tensor.cuda()
    targets_tensor = targets_tensor.cuda()
    # Calculate loss
    loss = metric_func(preds_tensor, targets_tensor)
    preds_tensor = preds_tensor.cpu()
    targets_tensor = targets_tensor.cpu()
    if loss.dim() > 0:
        loss = loss.mean()
    loss_value = loss.item()

    # Initialize dictionary to store results
    metrics = {"loss": loss_value}

    # Classification metrics: Accuracy, F1, AUC, and AUPR
    if dataset_type == 'classification' or dataset_type == 'multiclass':
        # For multiclass, get the predicted class labels and probabilities
        if dataset_type == 'multiclass':
            # Get probabilities using softmax
            pred_probs = torch.softmax(preds_tensor, dim=1).numpy()
            
            # Get predicted class labels
            pred_labels = np.argmax(pred_probs, axis=1)
            print(f"Predicted labels: {pred_labels}")
            print(f"Targets: {targets_tensor}")
        else:
            # Binary classification
            pred_probs = np.array(preds)
            pred_labels = np.round(pred_probs)

        targets_array = np.array(targets)

        # Calculate accuracy
        accuracy = accuracy_score(targets_array, pred_labels)
        metrics["accuracy"] = accuracy

        # Calculate AUC (area under ROC curve) - uses PROBABILITIES
        try:
            if dataset_type == 'multiclass':
                # For multiclass, need to binarize targets
                from sklearn.preprocessing import label_binarize
                n_classes = preds_tensor.shape[1]
                targets_binarized = label_binarize(targets_array, classes=range(n_classes))
                
                auc = roc_auc_score(targets_binarized, preds_tensor, average='macro', multi_class='ovr')
            else:
                # For binary classification
                auc = roc_auc_score(targets_array, preds_tensor)
        except ValueError as e:
            info(f"Could not calculate AUC: {e}")
            auc = float('nan')
        metrics["auc"] = auc

        # Calculate AUPR (Area under Precision-Recall curve) - uses PROBABILITIES
        try:
            if dataset_type == 'multiclass':
                # For multiclass, need to binarize targets
                from sklearn.preprocessing import label_binarize
                n_classes = preds_tensor.shape[1]
                targets_binarized = label_binarize(targets_array, classes=range(n_classes))

                aupr = average_precision_score(targets_binarized, preds_tensor, average='macro')
            else:
                # For binary classification
                aupr = average_precision_score(targets_array, preds_tensor)
        except ValueError as e:
            info(f"Could not calculate AUPR: {e}")
            aupr = float('nan')
        metrics["aupr"] = aupr

        # Calculate F1-score - uses PREDICTED LABELS
        try:
            f1 = f1_score(targets_array, pred_labels, average='macro')
        except ValueError as e:
            info(f"Could not calculate F1-score: {e}")
            f1 = float('nan')
        metrics["f1"] = f1

    return metrics

def evaluate(model: nn.Module,
             pretrain: bool,
             data,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds,tragets = predict(
        model=model,
        pretrain=pretrain,
        data=dataloader,
        batch_size=batch_size,
        scaler=scaler
    )

    # targets = data.targets()
    # targets = [int(x) for x in targets]
    targets = torch.tensor(targets)

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results
