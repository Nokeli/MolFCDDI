"""
Cold Start DDI Training with BPR Loss
3-Fold Cross Validation
"""
import os
import sys
import time
import warnings
from datetime import datetime
from collections import defaultdict  # KEEP

import numpy as np
import pandas as pd
import torch

# Module-level logger — initialized in main() once output_dir is known
_log_fh = None

def log(msg=''):
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + '\n')
        _log_fh.flush()
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn import metrics
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop.models import build_ddi_model, add_FUNC_prompt
from chemprop.nn_utils import param_count
from bpr_loss import BPRLoss
from cold_data_loader import ColdStartDataset, ColdStartCollator


######################### Parameters ######################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--min_delta', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--encoder_name', type=str, default='CMPNN')
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--ffn_hidden_size', type=int, default=300)
parser.add_argument('--ffn_num_layers', type=int, default=2)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--atom_messages', action='store_true', default=True)
parser.add_argument('--checkpoint_path', type=str,
                    default='./ckpt/original_MoleculeModel.pkl')
parser.add_argument('--data_dir', type=str,
                    default='./data/cold_data/cold_start_processed')
parser.add_argument('--drug_smiles', type=str,
                    default='./data/cold_data/cold_start_processed/drug_smiles.csv')
parser.add_argument('--output_dir', type=str, default='./dumped/cold_start')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# 补充 chemprop 框架需要的默认参数
args.num_tasks = 1
args.dataset_type = 'classification'  # BPR binary
args.multiclass_num_classes = 2
args.features_only = False
args.features_size = 0
args.use_input_features = False
args.features_dim = 0
args.activation = 'ReLU'
args.bias = False
args.undirected = False
args.num_attention = 2
args.num_attention_heads = 4
args.add_step = 'concat_mol_frag_attention'
args.step = 'func_prompt'
args.pooling_type = 'attention'
args.gamma = 0.01
args.increase_parm = 1
args.encoder_name = 'CMPNN'
args.cuda = torch.cuda.is_available() and args.gpu >= 0
args.device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')

device = args.device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_pretrained_encoder(model, checkpoint_path):
    """只加载 encoder 权重"""
    if not os.path.exists(checkpoint_path):
        log(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_state = model.encoder.state_dict()
    pretrained_state_dict = {}
    skipped = []

    for param_name, param_value in checkpoint.items():
        # 兼容两种格式
        if param_name.startswith('encoder.'):
            encoder_key = param_name[len('encoder.'):]
        else:
            encoder_key = param_name

        if encoder_key not in encoder_state:
            skipped.append(param_name)
            continue
        if encoder_state[encoder_key].shape != param_value.shape:
            log(f'  [WARN] Shape mismatch for "{encoder_key}"')
            skipped.append(param_name)
            continue
        pretrained_state_dict[encoder_key] = param_value

    model.encoder.load_state_dict(pretrained_state_dict, strict=False)
    log(f"  Loaded {len(pretrained_state_dict)}/{len(encoder_state)} encoder params")
    if skipped:
        log(f"  Skipped {len(skipped)} params")


def do_compute(batch, model, device):
    """计算 batch 的 pos/neg score"""
    pos_graphs, neg_graphs = batch['pos'], batch['neg']
    rel_types = batch['rel_types']

    pos_score = model('finetune', False, pos_graphs[0], pos_graphs[1], 'attention')
    neg_score = model('finetune', False, neg_graphs[0], neg_graphs[1], 'attention')

    probas_pred = torch.cat([
        torch.sigmoid(pos_score.detach()),
        torch.sigmoid(neg_score.detach())
    ]).cpu().numpy()

    ground_truth = np.concatenate([
        np.ones(len(pos_score)),
        np.zeros(len(neg_score))
    ])

    rel_types_np = np.array(rel_types + rel_types)

    return pos_score, neg_score, probas_pred, ground_truth, rel_types_np


def do_compute_metrics(probas_pred, target, rel_types=None, per_rel=False):
    """纯二分类指标"""
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1 = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, _ = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    if per_rel and rel_types is not None:
        rel_metrics = {}
        for rel in np.unique(rel_types):
            mask = rel_types == rel
            t_mask = target[mask]
            if len(t_mask) > 0 and sum(t_mask) > 0 and sum(1 - t_mask) > 0:
                try:
                    rel_pred = pred[mask]
                    rel_probas = probas_pred[mask]
                    rel_metrics[int(rel)] = {
                        'acc': metrics.accuracy_score(t_mask, rel_pred),
                        'auroc': metrics.roc_auc_score(t_mask, rel_probas),
                        'f1': metrics.f1_score(t_mask, rel_pred),
                        'precision': metrics.precision_score(t_mask, rel_pred),
                        'recall': metrics.recall_score(t_mask, rel_pred),
                        'int_ap': metrics.auc(*metrics.precision_recall_curve(t_mask, rel_probas)[:2]),
                        'ap': metrics.average_precision_score(t_mask, rel_probas),
                    }
                except Exception as e:
                    rel_metrics[int(rel)] = {'error': str(e)}
            else:
                rel_metrics[int(rel)] = {'error': 'insufficient samples'}
        return acc, auroc, f1, precision, recall, int_ap, ap, rel_metrics

    return acc, auroc, f1, precision, recall, int_ap, ap


def eval_dataset(model, data_loader, device, dataset_name, per_rel=False):
    """评估一个数据集"""
    probas_pred = []
    ground_truth = []
    rel_types = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            _, _, batch_probas, batch_gt, batch_rel = do_compute(batch, model, device)
            probas_pred.append(batch_probas)
            ground_truth.append(batch_gt)
            rel_types.append(batch_rel)

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    rel_types = np.concatenate(rel_types)

    if per_rel:
        acc, auroc, f1, precision, recall, int_ap, ap, rel_metrics = \
            do_compute_metrics(probas_pred, ground_truth, rel_types, per_rel=True)
        return {
            'acc': acc, 'auroc': auroc, 'f1': f1,
            'precision': precision, 'recall': recall,
            'int_ap': int_ap, 'ap': ap,
            'rel_metrics': rel_metrics,
            'probas_pred': probas_pred, 'ground_truth': ground_truth,
            'rel_types': rel_types
        }

    acc, auroc, f1, precision, recall, int_ap, ap = \
        do_compute_metrics(probas_pred, ground_truth, rel_types, per_rel=False)
    return {
        'acc': acc, 'auroc': auroc, 'f1': f1,
        'precision': precision, 'recall': recall,
        'int_ap': int_ap, 'ap': ap
    }


def train_fold(fold_idx, args, device):
    """训练单个 fold"""
    log(f"\n{'='*60}")
    log(f"  Fold {fold_idx}")
    log(f"{'='*60}")

    fold_dir = os.path.join(args.data_dir, f'fold{fold_idx}')

    # 加载数据
    train_ds = ColdStartDataset(os.path.join(fold_dir, 'train.csv'), args.drug_smiles)
    s1_ds = ColdStartDataset(os.path.join(fold_dir, 's1.csv'), args.drug_smiles)
    s2_ds = ColdStartDataset(os.path.join(fold_dir, 's2.csv'), args.drug_smiles)

    collator = ColdStartCollator(args)
    # 减小 batch_size 避免 OOM；用 gradient accumulation 补偿
    train_bs = 128
    accum_steps = args.batch_size // train_bs  # 等效 batch = 512
    train_loader = DataLoader(train_ds, batch_size=train_bs,
                              shuffle=True, collate_fn=collator, num_workers=0)
    s1_loader = DataLoader(s1_ds, batch_size=256,
                           shuffle=False, collate_fn=collator, num_workers=0)
    s2_loader = DataLoader(s2_ds, batch_size=256,
                           shuffle=False, collate_fn=collator, num_workers=0)

    log(f"Train: {len(train_ds)}, S1: {len(s1_ds)}, S2: {len(s2_ds)}")

    # 模型
    model = build_ddi_model(args)
    if args.step == 'func_prompt':
        add_FUNC_prompt(model, args)
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        load_pretrained_encoder(model, args.checkpoint_path)
    model.to(device)
    log(f"Parameters: {param_count(model):,}")

    # 优化器
    loss_fn = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    best_s1_acc = 0.0
    best_s2_acc = 0.0
    best_s1_epoch = 0
    best_s2_epoch = 0
    patience_counter = 0

    # 训练循环
    log(f"Starting training loop: {args.n_epochs} epochs, {len(train_loader)} batches/epoch")
    for epoch in range(1, args.n_epochs + 1):
        start = time.time()

        # Train (with gradient accumulation)
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            pos_score, neg_score, _, _, _ = do_compute(batch, model, device)
            loss, _, _ = loss_fn(pos_score, neg_score)
            loss = loss / accum_steps  # scale loss
            loss.backward()
            train_loss += loss.item() * accum_steps * len(pos_score)
            
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            if batch_idx % 100 == 0:
                log(f"  [Fold {fold_idx} Ep {epoch}] batch {batch_idx}/{len(train_loader)}, loss={loss.item() * accum_steps:.4f}")
        train_loss /= len(train_ds)

        # Eval
        s1_scores = eval_dataset(model, s1_loader, device, 'S1')
        s2_scores = eval_dataset(model, s2_loader, device, 'S2')

        # Save best
        s1_improved = s2_improved = False
        if s1_scores['acc'] > best_s1_acc + args.min_delta:
            best_s1_acc = s1_scores['acc']
            best_s1_epoch = epoch
            s1_improved = True
            torch.save(model, os.path.join(args.output_dir, f'best_s1_fold{fold_idx}.pt'))
            log(f"  *** New best S1: acc={best_s1_acc:.4f} (epoch {epoch}) ***")

        if s2_scores['acc'] > best_s2_acc + args.min_delta:
            best_s2_acc = s2_scores['acc']
            best_s2_epoch = epoch
            s2_improved = True
            patience_counter = 0
            torch.save(model, os.path.join(args.output_dir, f'best_s2_fold{fold_idx}.pt'))
            log(f"  *** New best S2: acc={best_s2_acc:.4f} (epoch {epoch}) ***")
        else:
            patience_counter += 1

        scheduler.step()

        # Log
        log(f"Epoch {epoch:3d}/{args.n_epochs} ({time.time()-start:.1f}s) | "
              f"train_loss: {train_loss:.4f} | "
              f"S1_acc: {s1_scores['acc']:.4f} S1_auc: {s1_scores['auroc']:.4f} | "
              f"S2_acc: {s2_scores['acc']:.4f} S2_auc: {s2_scores['auroc']:.4f} | "
              f"Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            log(f"\nEarly stopping at epoch {epoch}")
            break

    log(f"\nFold {fold_idx} Best: S1@ep{best_s1_epoch} acc={best_s1_acc:.4f}, "
          f"S2@ep{best_s2_epoch} acc={best_s2_acc:.4f}")

    # Final test with best models
    s1_best = torch.load(os.path.join(args.output_dir, f'best_s1_fold{fold_idx}.pt'))
    s2_best = torch.load(os.path.join(args.output_dir, f'best_s2_fold{fold_idx}.pt'))

    final_s1 = eval_dataset(s1_best, s1_loader, device, 'S1', per_rel=True)
    final_s2 = eval_dataset(s2_best, s2_loader, device, 'S2', per_rel=True)

    return final_s1, final_s2


def average_metrics(results_list):
    """多个 fold 的结果取平均"""
    avg = {}
    keys = ['acc', 'auroc', 'f1', 'precision', 'recall', 'int_ap', 'ap']
    for k in keys:
        vals = [r[k] for r in results_list if k in r]
        avg[k] = np.mean(vals) if vals else 0.0
    return avg


def aggregate_per_rel(fold_results):
    """聚合 per-relation 指标"""
    all_rel_metrics = defaultdict(list)
    for fold_res in fold_results:
        if 'rel_metrics' not in fold_res:
            continue
        for rel, metrics in fold_res['rel_metrics'].items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                all_rel_metrics[rel].append(metrics)

    avg_rel = {}
    for rel, metrics_list in all_rel_metrics.items():
        avg_rel[rel] = {}
        for k in ['acc', 'auroc', 'f1', 'precision', 'recall', 'int_ap', 'ap']:
            vals = [m[k] for m in metrics_list if k in m]
            avg_rel[rel][k] = np.mean(vals) if vals else 0.0
    return avg_rel


def save_results(fold_results, output_path):
    """保存 per-relation CSV"""
    rows = []

    # S1
    s1_results = [r['s1'] for r in fold_results]
    avg_s1 = average_metrics(s1_results)
    rows.append({
        'Dataset': 'S1', 'ADR_Type': 'Overall',
        'Accuracy': avg_s1['acc'], 'AUROC': avg_s1['auroc'], 'F1': avg_s1['f1'],
        'Precision': avg_s1['precision'], 'Recall': avg_s1['recall'],
        'Int_AP': avg_s1['int_ap'], 'AP': avg_s1['ap']
    })

    # S1 per-rel
    s1_rel = aggregate_per_rel(s1_results)
    for rel in sorted(s1_rel.keys()):
        m = s1_rel[rel]
        rows.append({
            'Dataset': 'S1', 'ADR_Type': rel,
            'Accuracy': m['acc'], 'AUROC': m['auroc'], 'F1': m['f1'],
            'Precision': m['precision'], 'Recall': m['recall'],
            'Int_AP': m['int_ap'], 'AP': m['ap']
        })

    # S2
    s2_results = [r['s2'] for r in fold_results]
    avg_s2 = average_metrics(s2_results)
    rows.append({
        'Dataset': 'S2', 'ADR_Type': 'Overall',
        'Accuracy': avg_s2['acc'], 'AUROC': avg_s2['auroc'], 'F1': avg_s2['f1'],
        'Precision': avg_s2['precision'], 'Recall': avg_s2['recall'],
        'Int_AP': avg_s2['int_ap'], 'AP': avg_s2['ap']
    })

    # S2 per-rel
    s2_rel = aggregate_per_rel(s2_results)
    for rel in sorted(s2_rel.keys()):
        m = s2_rel[rel]
        rows.append({
            'Dataset': 'S2', 'ADR_Type': rel,
            'Accuracy': m['acc'], 'AUROC': m['auroc'], 'F1': m['f1'],
            'Precision': m['precision'], 'Recall': m['recall'],
            'Int_AP': m['int_ap'], 'AP': m['ap']
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    log(f"\nResults saved to {output_path}")
    return df


def main():
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化日志文件
    global _log_fh
    log_path = os.path.join(args.output_dir, 'train.log')
    _log_fh = open(log_path, 'w', buffering=1)

    log(f"{'='*60}")
    log(f"  Cold Start DDI Training (BPR)")
    log(f"  Device: {device}")
    log(f"  3-Fold CV | lr={args.lr} | batch={args.batch_size}")
    log(f"{'='*60}")

    fold_results = []
    for fold in [0, 1, 2]:
        s1_res, s2_res = train_fold(fold, args, device)
        fold_results.append({'s1': s1_res, 's2': s2_res})

    # Summary
    log(f"\n{'='*60}")
    log(f"  3-Fold Average Results")
    log(f"{'='*60}")

    avg_s1 = average_metrics([r['s1'] for r in fold_results])
    avg_s2 = average_metrics([r['s2'] for r in fold_results])

    log(f"  S1: Acc={avg_s1['acc']:.4f} AUROC={avg_s1['auroc']:.4f} "
          f"F1={avg_s1['f1']:.4f} AUPR={avg_s1['ap']:.4f}")
    log(f"  S2: Acc={avg_s2['acc']:.4f} AUROC={avg_s2['auroc']:.4f} "
          f"F1={avg_s2['f1']:.4f} AUPR={avg_s2['ap']:.4f}")

    # Save
    df = save_results(fold_results, os.path.join(args.output_dir, 'per_relation_results.csv'))
    log("\nFinal Results Table:")
    log(df.to_string())


if __name__ == '__main__':
    main()
