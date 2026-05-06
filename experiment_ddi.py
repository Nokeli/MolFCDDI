"""
MolFCDDI Experiment Runner
=========================
支持三种切分场景（S1/S2/S3）×两种分类模式（binary/multiclass）

用法示例:
  # S1 Random + Binary
  python experiment_ddi.py --split_name S1_random --mode binary --gpu 0 --epochs 50

  # S2 One-unseen + Multiclass
  python experiment_ddi.py --split_name S2_one_unseen --mode multiclass --gpu 0 --epochs 50

  # S3 Both-unseen + Binary
  python experiment_ddi.py --split_name S3_both_unseen --mode binary --gpu 0 --epochs 50
"""

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from argparse import ArgumentParser
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

from chemprop.data import MoleculeDataset, CachedMolGraphCollator
from chemprop.data.utils import get_ddi_data, get_task_names
from chemprop.models import build_model, build_ddi_model, add_FUNC_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import (
    build_optimizer, build_lr_scheduler, get_loss_func,
    load_checkpoint, save_checkpoint, makedirs
)
from chemprop.train import train
from chemprop.train.predict import predict
from chemprop.train.evaluate import evaluate_predictions
from chemprop.torchlight import initialize_exp


def get_class_weights(train_dataset, n_classes):
    """计算类别权重，处理不平衡"""
    labels = [int(item) for item in train_dataset.targets()]
    label_counts = Counter(labels)
    weights = torch.zeros(n_classes)
    total_samples = len(labels)
    for label in range(n_classes):
        count = label_counts.get(label, 1)
        weights[label] = (total_samples / count) ** 0.5
    weights = weights / weights.mean()
    return weights


def create_dataloader(dataset, args, batch_size=256, shuffle=True, cache_size=10000):
    if dataset is None or len(dataset) == 0:
        raise ValueError("Dataset is None or empty!")
    collate_fn = CachedMolGraphCollator(args=args, pretrain=False, cache_size=cache_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=True, drop_last=False
    )


def run_experiment(args):
    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # Determine number of classes
    n_classes = 2 if args.mode == 'binary' else 86
    print(f"\n{'='*60}")
    print(f"  Experiment: {args.split_name} / {args.mode}")
    print(f"  GPU: {args.gpu} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"  Classes: {n_classes} | Split dir: {args.split_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    args.task_names = ['label']
    data = get_ddi_data(path="./data/drugbank.csv", args=args)
    train_data = data['train']
    val_data = data['valid']
    test_data = data['test']

    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    # Label distribution
    train_labels = Counter([int(t) for t in train_data.targets()])
    test_labels = Counter([int(t) for t in test_data.targets()])
    print(f"  Train label dist: {dict(sorted(train_labels.items()))}")
    print(f"  Test  label dist: {dict(sorted(test_labels.items()))}")

    # Create dataloaders
    train_loader = create_dataloader(train_data, args, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, args, batch_size=args.batch_size, shuffle=False, cache_size=5000)
    test_loader = create_dataloader(test_data, args, batch_size=args.batch_size, shuffle=False, cache_size=5000)

    # Class weights
    train_class_weights = get_class_weights(train_data, n_classes=n_classes).cuda()
    val_class_weights = get_class_weights(val_data, n_classes=n_classes).cuda()
    test_class_weights = get_class_weights(test_data, n_classes=n_classes).cuda()

    print(f"  Class weights computed for {n_classes} classes")
    print(f"    Train max weight: {train_class_weights.max():.2f}, min: {train_class_weights.min():.2f}")

    loss_func = nn.CrossEntropyLoss(weight=train_class_weights)

    # Build model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading pretrained encoder from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model = build_ddi_model(args)
        model_state_dict = model.encoder.state_dict() if args.encoder else model.state_dict()
        pretrained_state_dict = {}
        for param_name in checkpoint.keys():
            if param_name not in model_state_dict:
                print(f'  [WARN] Pretrained param "{param_name}" not in model')
            elif model_state_dict[param_name].shape != checkpoint[param_name].shape:
                print(f'  [WARN] Shape mismatch for "{param_name}"')
            else:
                pretrained_state_dict[param_name] = checkpoint[param_name]
        model_state_dict.update(pretrained_state_dict)
        if args.encoder:
            model.encoder.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    else:
        print("Building CrossFrag DDI model from scratch")
        model = build_ddi_model(args)

    if args.step == 'func_prompt':
        add_FUNC_prompt(model, args)

    print(f"\nModel parameters: {param_count(model):,}")
    model = model.cuda()
    print(model)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)

    # Training loop
    pooling_type = getattr(args, 'pooling_type', 'attention')
    best_val_score = float('inf')
    best_epoch = 0
    best_test_score = 0.0

    for epoch in range(args.epochs):
        avg_loss = train(
            model=model, pretrain=False, data=train_loader,
            loss_func=loss_func, optimizer=optimizer,
            scheduler=scheduler, args=args, n_iter=0, logger=None
        )

        if hasattr(scheduler, 'step'):
            scheduler.step()

        # Validation
        val_preds, val_targets = predict(
            model=model, pretrain=False, data=val_loader,
            batch_size=args.batch_size, scaler=None, pooling_type=pooling_type
        )
        val_targets_np = np.array([int(t) for t in val_targets])
        val_preds_np = np.array(val_preds)
        val_acc = (val_preds_np.argmax(1) == val_targets_np).mean()

        # Test
        test_preds, test_targets = predict(
            model=model, pretrain=False, data=test_loader,
            batch_size=args.batch_size, scaler=None, pooling_type=pooling_type
        )
        test_targets_np = np.array([int(t) for t in test_targets])
        test_preds_np = np.array(test_preds)
        test_acc = (test_preds_np.argmax(1) == test_targets_np).mean()

        # AUROC/AUPR for multiclass
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
        try:
            val_auc = roc_auc_score(
                np.eye(n_classes)[val_targets_np], val_preds_np, multi_class='ovr', average='macro'
            )
            val_aupr = average_precision_score(
                np.eye(n_classes)[val_targets_np], val_preds_np, average='macro'
            )
        except Exception:
            val_auc = 0.0
            val_aupr = 0.0
        try:
            test_auc = roc_auc_score(
                np.eye(n_classes)[test_targets_np], test_preds_np, multi_class='ovr', average='macro'
            )
            test_aupr = average_precision_score(
                np.eye(n_classes)[test_targets_np], test_preds_np, average='macro'
            )
        except Exception:
            test_auc = 0.0
            test_aupr = 0.0

        val_f1 = f1_score(val_targets_np, val_preds_np.argmax(1), average='macro', zero_division=0)
        test_f1 = f1_score(test_targets_np, test_preds_np.argmax(1), average='macro', zero_division=0)

        # Cross-entropy loss
        from torch.nn.functional import cross_entropy
        val_ce = cross_entropy(torch.tensor(val_preds_np).cuda(), torch.tensor(val_targets_np).cuda()).item()
        test_ce = cross_entropy(torch.tensor(test_preds_np).cuda(), torch.tensor(test_targets_np).cuda()).item()

        # Track best
        if val_ce < best_val_score:
            best_val_score = val_ce
            best_epoch = epoch + 1
            best_test_score = test_ce
            # Save best model
            save_dir = os.path.join(args.save_dir, 'best_model')
            makedirs(save_dir)
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, None, None, args)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val CE: {val_ce:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Val AUPR: {val_aupr:.4f} | Val F1: {val_f1:.4f} | "
            f"Test CE: {test_ce:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f} | "
            f"Test AUPR: {test_aupr:.4f} | Test F1: {test_f1:.4f} | "
            f"[Best ep={best_epoch}, best_val_ce={best_val_score:.4f}]"
        )

    print(f"\n{'='*60}")
    print(f"  Best Val CE: {best_val_score:.6f} (epoch {best_epoch})")
    print(f"  Corresponding Test CE: {best_test_score:.6f}")
    print(f"{'='*60}")

    return best_val_score


def main():
    parser = ArgumentParser(description="MolFCDDI Experiment Runner")

    # Data / Split
    parser.add_argument('--split_name', type=str, default='warm_start',
                        choices=['S1_random', 'S2_one_unseen', 'S3_both_unseen', 'warm_start'],
                        help='Split scenario')
    parser.add_argument('--mode', type=str, default='multiclass',
                        choices=['binary', 'multiclass'],
                        help='Classification mode')
    parser.add_argument('--split_dir', type=str,
                        default='./data/ddi_warm_start',
                        help='Directory containing split .npy files')

    # Model
    parser.add_argument('--checkpoint_path', type=str,
                        default='./ckpt/original_MoleculeModel.pkl',
                        help='Pretrained checkpoint path')
    parser.add_argument('--encoder', action='store_true', default=False,
                        help='Load only encoder from checkpoint')
    parser.add_argument('--encoder_name', type=str, default='CMPNN')
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--ffn_hidden_size', type=int, default=300)
    parser.add_argument('--ffn_num_layers', type=int, default=2)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--num_attention', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--atom_messages', action='store_true', default=True)
    parser.add_argument('--features_only', action='store_true', default=False)
    parser.add_argument('--use_input_features', action='store_true', default=False)
    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--undirected', action='store_true', default=False)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--increase_parm', type=int, default=1)

    # Data loading
    parser.add_argument('--features_path', type=str, default=None)
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--use_compound_names', action='store_true', default=False)

    # Prompt
    parser.add_argument('--add_step', type=str, default='concat_mol_frag_attention')
    parser.add_argument('--step', type=str, default='func_prompt')
    parser.add_argument('--pooling_type', type=str, default='attention')
    parser.add_argument('--gamma', type=float, default=0.01)

    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--final_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--l2_norm', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ensemble_size', type=int, default=1)

    # Loss / metric
    parser.add_argument('--dataset_type', type=str, default='multiclass')
    parser.add_argument('--multiclass_num_classes', type=int, default=86)
    parser.add_argument('--metric', type=str, default='cross_entropy')
    parser.add_argument('--minimize_score', action='store_true', default=True)
    parser.add_argument('--features_scaling', action='store_true', default=False)

    # Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{parser.parse_args().split_name}_{parser.parse_args().mode}_{timestamp}"
    default_save_dir = f'./dumped/{exp_name}'
    parser.add_argument('--save_dir', type=str, default=default_save_dir)
    parser.add_argument('--exp_name', type=str, default=exp_name)
    parser.add_argument('--exp_id', type=str, default=exp_name)

    # Parse args TWICE: first for defaults, second for actual values
    # (argparse doesn't let us reference another arg's default easily)
    args = parser.parse_args()

    # Override multiclass_num_classes based on mode
    if args.mode == 'binary':
        args.multiclass_num_classes = 2

    # Validate split files exist
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(args.split_dir, f"{args.split_name}_{args.mode}_{split}.npy")
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    makedirs(args.save_dir)
    print(f"Saving results to: {args.save_dir}")

    run_experiment(args)


if __name__ == '__main__':
    main()
