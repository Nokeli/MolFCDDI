"""
DDI Training Script with S1/S2/S3 Split Support
==============================================

Usage:
  python ddi_train.py --split_name S1_random --mode multiclass --gpu 0 --epochs 50
"""

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from argparse import ArgumentParser, Namespace
from logging import Logger
import os
from typing import Tuple
import numpy as np
from datetime import datetime, date

from chemprop.train.ddi_run_training import ddi_run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import modify_train_args


def run_stat(args, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print
    save_dir = args.save_dir
    args.save_dir = os.path.join(save_dir, f'run_{args.seed}')
    makedirs(args.save_dir)
    model_scores = ddi_run_training(args, args.pretrain, logger)
    info(f'{args.runs}-times runs')
    info(f'Scaffold {args.runs} ==> test {args.metric} = {model_scores:.6f}')
    return model_scores


def create_args():
    parser = ArgumentParser()

    # Data / Split
    parser.add_argument('--split_name', type=str, default='warm_start',
                        choices=['S1_random', 'S2_one_unseen', 'S3_both_unseen', 'warm_start', 'cold_semi', 'cold_strict'])
    parser.add_argument('--mode', type=str, default='multiclass',
                        choices=['binary', 'multiclass'])
    parser.add_argument('--split_dir', type=str,
                        default='./data/ddi_warm_start')

    # Model
    parser.add_argument('--checkpoint_path', type=str,
                        default='./ckpt/original_MoleculeModel.pkl')
    parser.add_argument('--encoder', action='store_true', default=False)
    parser.add_argument('--encoder_name', type=str, default='CMPNN',
                        choices=['CMPNN', 'MPNN', 'PharmHGT', 'CMPNDGL'])
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
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU'])
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--increase_parm', type=int, default=1)
    parser.add_argument('--add_reactive', action='store_true', default=False)

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
    parser.add_argument('--log_frequency', type=int, default=10)

    # Dataset
    parser.add_argument('--data_path', type=str, default='./data/drugbank.csv')
    parser.add_argument('--root_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='drugbank')
    parser.add_argument('--dataset_type', type=str, default='multiclass',
                        choices=['classification', 'regression', 'multiclass'])
    parser.add_argument('--split_type', type=str, default='scaffold_balanced')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--runs', type=int, default=0)
    parser.add_argument('--max_data_size', type=int, default=None)
    parser.add_argument('--use_compound_names', action='store_true', default=False)
    parser.add_argument('--features_scaling', action='store_true', default=False)
    parser.add_argument('--features_generator', type=str, nargs='*', default=None)
    parser.add_argument('--features_path', type=str, nargs='*', default=None)

    # Metric
    parser.add_argument('--multiclass_num_classes', type=int, default=86)
    parser.add_argument('--metric', type=str, default='cross_entropy',
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'])
    parser.add_argument('--minimize_score', action='store_true', default=True)
    parser.add_argument('--show_individual_scores', action='store_true', default=True)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--last_early_stop', type=int, default=0)

    # Output
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--dump_path', type=str, default='./dumped')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--exp_name', type=str, default='finetune')
    parser.add_argument('--exp_id', type=str, default='ddi')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--save_smiles_splits', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--crossval_index_sets', type=str, default=None)
    parser.add_argument('--crossval_index_dir', type=str, default=None)
    parser.add_argument('--crossval_index_file', type=str, default=None)
    parser.add_argument('--separate_train_path', type=str, default=None)
    parser.add_argument('--separate_val_path', type=str, default=None)
    parser.add_argument('--separate_test_path', type=str, default=None)
    parser.add_argument('--separate_val_features_path', type=str, nargs='*', default=None)
    parser.add_argument('--separate_test_features_path', type=str, nargs='*', default=None)
    parser.add_argument('--folds_file', type=str, default=None)
    parser.add_argument('--val_fold_index', type=int, default=None)
    parser.add_argument('--test_fold_index', type=int, default=None)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--no_features_scaling', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    # Override multiclass_num_classes based on mode
    if args.mode == 'binary':
        args.multiclass_num_classes = 2
        args.dataset_type = 'multiclass'  # still use CrossEntropy for binary
    else:
        args.multiclass_num_classes = 86
        args.dataset_type = 'multiclass'

    # Generate save_dir if not provided
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'./dumped/{args.split_name}_{args.mode}_{timestamp}'
    args.exp_name = f'{args.split_name}_{args.mode}'
    args.exp_id = f'{args.split_name}_{args.mode}'

    # Validate split files exist
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(args.split_dir, f'{args.split_name}_{args.mode}_{split}.npy')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f'Split file not found: {fpath}')

    return args


def main():
    args = create_args()
    args.pretrain = False

    import torch
    if args.gpu is not None and not args.no_cuda:
        torch.cuda.set_device(args.gpu)
        args.cuda = True
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.cuda = False
        args.device = torch.device('cpu')

    import numpy as np
    import torch as T
    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(args.seed)

    modify_train_args(args)

    print(f"\n{'='*60}")
    print(f"  Split: {args.split_name} | Mode: {args.mode}")
    print(f"  Classes: {args.multiclass_num_classes}")
    print(f"  GPU: {args.gpu} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"  Split dir: {args.split_dir}")
    print(f"  Dump path: {args.dump_path}")
    print(f"{'='*60}\n")

    # Pre-create dump directory structure (Windows compatibility)
    exp_folder = os.path.join(
        args.dump_path,
        date.today().strftime('%m%d-') + args.exp_name,
        args.exp_id
    )
    os.makedirs(exp_folder, exist_ok=True)
    args.save_dir = exp_folder
    print(f"  Save dir: {args.save_dir}\n")

    # Create a simple logger (compatible with torchlight's Logger interface)
    import logging
    log_file = os.path.join(exp_folder, 'train.log')
    _logger = logging.getLogger('train')
    _logger.setLevel(logging.INFO)
    _logger.handlers = []  # clear any existing handlers
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    _logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    _logger.addHandler(sh)

    class SimpleLogger:
        def __init__(self, logger):
            self._logger = logger
        def info(self, msg): self._logger.info(msg)
        def debug(self, msg): self._logger.debug(msg)
        def warning(self, msg): self._logger.warning(msg)

    logger = SimpleLogger(_logger)
    model_scores = run_stat(args, logger)
    print(f'\nFinal Result: {model_scores:.5f}')


if __name__ == '__main__':
    main()
