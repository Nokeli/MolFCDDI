from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import Counter
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_ddi_data, get_task_names, split_data, load_data
from chemprop.models import build_model, build_pretrain_model, add_FUNC_prompt,build_ddi_model,DDIInteractionModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint, Early_stop
from chemprop.data import MoleculeDataset,CachedMolGraphCollator
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.data.scaffold import scaffold_to_smiles
from collections import defaultdict
import pickle
def create_ddi_dataloader(dataset: MoleculeDataset,
                          args: Namespace,
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 4,
                          pretrain: bool = False,
                          cache_size: int = 10000,
                          pin_memory: bool = True):
    """
    创建带缓存的DDI DataLoader
    
    :param dataset: MoleculeDataset_DDI实例
    :param args: 参数
    :param batch_size: 批大小
    :param shuffle: 是否打乱
    :param num_workers: 工作进程数(建议设为0,因为缓存在主进程)
    :param pretrain: 是否预训练模式
    :param cache_size: 缓存大小
    :param pin_memory: 是否使用pin_memory
    :return: (DataLoader, CachedMolGraphCollator)元组
    """
    # 检查dataset是否为空
    if dataset is None or len(dataset) == 0:
        raise ValueError("Dataset is None or empty!")
    
    # 创建带缓存的collate函数
    collate_fn = CachedMolGraphCollator(
        args=args,
        pretrain=pretrain,
        cache_size=cache_size
    )
    
    # 注意: 使用缓存时,建议num_workers=0,因为缓存在主进程中
    # 如果需要多进程,需要使用共享内存或其他机制
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 缓存在主进程,多进程会导致重复计算
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return dataloader, collate_fn
def get_class_weights(train_data, n_classes=86):
        """计算类别权重，处理不平衡"""
        labels = [int(item) for item in train_data.targets()]  # 假设标签在targets的第三个位置
        label_counts = Counter(labels)
        
        # 计算权重（样本少的类别权重大）
        weights = torch.zeros(n_classes)
        total_samples = len(labels)
        
        for label in range(n_classes):
            count = label_counts.get(label, 1)  # 避免除零
            # 使用平方根缓解极端权重
            weights[label] = (total_samples / count) ** 0.5
        
        # 归一化
        weights = weights / weights.mean()
    
        # 打印权重统计
        print(f"类别权重统计:")
        print(f"  最大权重: {weights.max():.2f} (最稀有类别)")
        print(f"  最小权重: {weights.min():.2f} (最常见类别)")
        print(f"  平均权重: {weights.mean():.2f}")
    
        weights = weights.cuda() 
        return weights

def ddi_run_training(args: Namespace, pretrain: bool, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    info('Loading data')
    # args.vocab = Vocab(args)
    args.task_names = get_task_names(args.data_path)
    data = get_ddi_data(path=args.data_path, args=args, logger=logger)
    train_data = data['train']
    val_data = data['valid']
    print(len(val_data))
    test_data = data['test']
    args.num_tasks = train_data.num_tasks()
    args.features_size = train_data.features_size()
    info(f'Number of tasks = {args.num_tasks}')
    print(train_data)
    train_loader, train_collator = create_ddi_dataloader(
        train_data,
        args=args,
        batch_size=256,
        shuffle=True,
        num_workers=0,  # 使用缓存时建议为0
        pretrain=False,
        cache_size=10000  # 缓存10000个不同的SMILES
    )
    valid_loader, valid_collator = create_ddi_dataloader(
        val_data,
        args=args,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pretrain=False,
        cache_size=5000
    )
    test_loader, test_collator = create_ddi_dataloader(
        test_data,
        args=args,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pretrain=False,
        cache_size=5000
    )
    # Split data
    debug(f'Load data from {args.exp_id} for Scaffold-{args.runs}')
    # if 0 < args.runs < 3:
    #     train_data, val_data, test_data = load_data(data, args, logger)
    # else:
    #     print('=' * 100)
    #     train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
    #                                                  seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None

    # Get loss and metric functions
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    # test_smiles, test_targets = test_data.smiles(), test_data.targets()
    val_targets = val_data.targets()
    val_targets = [int(x) for x in val_targets]
    val_targets = torch.tensor(val_targets)
    test_targets = test_data.targets()
    test_targets = [int(x) for x in test_targets]
    test_targets = torch.tensor(test_targets)
    test_len = len(test_data)
    n_classes = args.multiclass_num_classes
    train_class_weights = get_class_weights(train_data, n_classes=n_classes)
    valid_class_weights = get_class_weights(val_data, n_classes=n_classes)
    test_class_weights = get_class_weights(test_data, n_classes=n_classes)
    train_class_weights = train_class_weights.cuda()
    train_loss_func = nn.CrossEntropyLoss(weight=train_class_weights)
    val_loss_func = nn.CrossEntropyLoss(weight=valid_class_weights)
    test_loss_func = nn.CrossEntropyLoss(weight=test_class_weights)
    loss_func = get_loss_func(args)
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((test_len, args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((test_len, args.num_tasks))
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if args.checkpoint_path not in (None, ""):
            debug(f'Loading model from {args.checkpoint_path}')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model = build_ddi_model(args)
            model_state_dict = model.encoder.state_dict() if args.encoder else model.state_dict()
            pretrained_state_dict = {}
            for param_name in checkpoint.keys():
                if param_name not in model_state_dict:
                    print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
                elif model_state_dict[param_name].shape != checkpoint[param_name].shape:
                    print(f'Pretrained parameter "{param_name}" '
                          f'of shape {checkpoint[param_name].shape} does not match corresponding '
                          f'model parameter of shape {model_state_dict[param_name].shape}.')
                else:
                    pretrained_state_dict[param_name] = checkpoint[param_name]
            model_state_dict.update(pretrained_state_dict)
            if args.encoder:
                model.encoder.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
        else:
            debug(f'Building CrossFrag DDI model {model_idx}')
            model = build_ddi_model(args)

        if args.step == 'func_prompt':
            add_FUNC_prompt(model, args)
        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Early_stop
        early_stop = False
        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        if args.early_stop:
            stopper = Early_stop(patience=args.patience, minimize_score=args.minimize_score)
        for epoch in range(args.epochs):
            avg_loss = train(
                model=model,
                pretrain=pretrain,
                data=train_loader,
                loss_func=train_loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            # val_scores = evaluate(
            #     model=model,
            #     pretrain=pretrain,
            #     data=val_data,
            #     num_tasks=args.num_tasks,
            #     metric_func=loss_func,
            #     batch_size=args.batch_size,
            #     dataset_type=args.dataset_type,
            #     scaler=scaler,
            #     logger=logger
            # )
            # 使用注意力聚合（默认使用attention pooling）
            pooling_type = getattr(args, 'pooling_type', 'attention')
            val_preds,val_targets = predict(
                model=model,
                pretrain=pretrain,
                data=valid_loader,
                batch_size=args.batch_size,
                scaler=scaler,
                pooling_type=pooling_type
            )
            print(f"Preds shape: {np.array(val_preds).shape}")
            val_scores = evaluate_predictions(
                preds=val_preds,
                targets=val_targets,
                metric_func=train_loss_func,
                dataset_type=args.dataset_type,
                logger=logger
            )
            # Average validation score
            avg_val_score = np.nanmean(val_scores['loss'])
            test_preds,test_targets = predict(
                model=model,
                pretrain=pretrain,
                data=test_loader,
                batch_size=args.batch_size,
                scaler=scaler,
                pooling_type=pooling_type
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                metric_func=train_loss_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Average test score
            avg_test_score = np.nanmean(test_scores['loss'])

            improved = (
                (args.minimize_score and avg_val_score < best_score) or
                (not args.minimize_score and avg_val_score > best_score)
            )
            if improved:
                best_score = avg_val_score
                best_epoch = epoch + 1
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            # if args.early_stop and epoch >= args.last_early_stop:
            #     early_stop = stopper.step(avg_val_score)
                # info(
                #     f'Epoch{epoch + 1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
                #     best_epoch = {best_epoch + 1},patience = {stopper.counter}')
            info(
                f'Epoch {epoch + 1}/{args.epochs}, '
                f'train loss: {avg_loss:.4f}, '
                f'valid_{args.metric}: {avg_val_score:.6f}, '
                f'test_{args.metric}: {avg_test_score:.6f}, '
                f'best_epoch = {best_epoch}, '
                f'valid_accuracy = {val_scores["accuracy"]:.6f}, '
                f'valid_auc = {val_scores["auc"]:.6f}, '
                f'valid_aupr = {val_scores["aupr"]:.6f}, '
                f'valid_f1 = {val_scores["f1"]:.6f}'
                f'test_accuracy = {test_scores["accuracy"]:.6f}, '
                f'test_auc = {test_scores["auc"]:.6f}, '
                f'test_aupr = {test_scores["aupr"]:.6f}, '
                f'test_f1 = {test_scores["f1"]:.6f}')
            # else:
            #     # info(
            #     #     f'Epoch{epoch + 1}/{args.epochs},train loss:{avg_loss:.4f},valid_{args.metric} = {avg_val_score:.6f},test_{args.metric} = {avg_test_score:.6},\
            #     #     best_epoch = {best_epoch + 1}')
            #     info(
            #         f'Epoch {epoch + 1}/{args.epochs}, '
            #         f'train loss: {avg_loss:.4f}, '
            #         f'valid_{args.metric}: {avg_val_score:.6f}, '
            #         f'test_{args.metric}: {avg_test_score:.6f}, '
            #         f'best_epoch = {best_epoch + 1}, '
            #         f'valid_accuracy = {val_scores["accuracy"]:.6f}, '
            #         f'valid_auc = {val_scores["auc"]:.6f}, '
            #         f'valid_aupr = {val_scores["aupr"]:.6f}, '
            #         f'train_f1 = {val_scores["f1"]:.6f}'
            #         f'test_accuracy = {test_scores["accuracy"]:.6f}, '
            #         f'test_auc = {test_scores["auc"]:.6f}, '
            #         f'test_aupr = {test_scores["aupr"]:.6f}, '
            #         f'test_f1 = {test_scores["f1"]:.6f}')
            # if args.early_stop and early_stop:
            #     break
        # Evaluate on test set using model with best validation score
        #info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    #     model = load_checkpoint(os.path.join(save_dir, 'model.pt'), current_args=args, cuda=args.cuda, logger=logger)
    #     print(model)
    #     test_preds = predict(
    #         model=model,
    #         pretrain=pretrain,
    #         data=test_data,
    #         batch_size=args.batch_size,
    #         scaler=scaler
    #     )
    #     test_scores = evaluate_predictions(
    #         preds=test_preds,
    #         targets=test_targets,
    #         metric_func=metric_func,
    #         dataset_type=args.dataset_type,
    #         logger=logger
    #     )
    #     if len(test_preds) != 0:
    #         sum_test_preds += np.array(test_preds)
    #     # Average test score
    #     avg_test_score = np.nanmean(test_scores)
    #     info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
    #
    #     if args.show_individual_scores:
    #         # Individual test scores
    #         for task_name, test_score in zip(args.task_names, test_scores):
    #             info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
    # # Evaluate ensemble on test set
    # avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    #
    # ensemble_scores = evaluate_predictions(
    #     preds=avg_test_preds,
    #     targets=test_targets,
    #     num_tasks=args.num_tasks,
    #     metric_func=metric_func,
    #     dataset_type=args.dataset_type,
    #     logger=logger
    # )
    #
    # # Average ensemble score
    # avg_ensemble_test_score = np.nanmean(ensemble_scores)
    # info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    #
    # # Individual ensemble scores
    # if args.show_individual_scores:
    #     for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
    #         info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
    #
    # return avg_ensemble_test_score
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), current_args=args, cuda=args.cuda, logger=logger)
        print(model)

        # Predict on the test data
        test_preds,test_targets = predict(
            model=model,
            pretrain=pretrain,
            data=test_loader,
            batch_size=args.batch_size,
            scaler=scaler,
            pooling_type=pooling_type
        )

        # Evaluate the model predictions
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            metric_func=train_loss_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        # Calculate the average test score
        avg_test_score = np.nanmean(test_scores['loss'])
        info(
            f'Epoch {epoch + 1}/{args.epochs}, '
            f'test loss: {avg_loss:.4f}, '
            f'test_accuracy = {test_scores["accuracy"]:.6f}, '
            f'test_auc = {test_scores["auc"]:.6f}, '
            f'test_aupr = {test_scores["aupr"]:.6f}, '
            f'test_f1 = {test_scores["f1"]:.6f}')

        # Display individual task scores (if requested)
        # NOTE: This loop is designed for multi-task settings where test_scores is a list of per-task scores.
        # For DDI, test_scores is a dict of metrics (loss, accuracy, auc, etc.) for a single task,
        # so this loop is not applicable and has been disabled.
        if args.show_individual_scores and isinstance(test_scores, list):
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model test {task_name} {args.metric} = {test_score:.6f}')

        return avg_test_score
