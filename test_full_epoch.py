import torch
import sys
import time
sys.path.insert(0, '.')

from torch.utils.data import DataLoader
from cold_data_loader import ColdStartDataset, ColdStartCollator
from chemprop.models import build_ddi_model, add_FUNC_prompt
from chemprop.nn_utils import param_count
from bpr_loss import BPRLoss

class Args:
    hidden_size = 300; ffn_hidden_size = 300; ffn_num_layers = 2
    depth = 3; dropout = 0.1; atom_messages = True
    dataset_type = 'classification'; multiclass_num_classes = 2
    features_only = False; features_size = 0
    use_input_features = False; features_dim = 0
    activation = 'ReLU'; bias = False; undirected = False
    num_attention = 2; num_attention_heads = 4
    add_step = 'concat_mol_frag_attention'; step = 'func_prompt'
    pooling_type = 'attention'; gamma = 0.01
    increase_parm = 1; encoder_name = 'CMPNN'
    num_tasks = 1; cuda = True
    device = torch.device('cuda:0')

args = Args()

print("Loading datasets...")
t0 = time.time()
train_ds = ColdStartDataset("data/cold_data/cold_start_processed/fold0/train.csv", "data/cold_data/cold_start_processed/drug_smiles.csv")
s1_ds = ColdStartDataset("data/cold_data/cold_start_processed/fold0/s1.csv", "data/cold_data/cold_start_processed/drug_smiles.csv")
s2_ds = ColdStartDataset("data/cold_data/cold_start_processed/fold0/s2.csv", "data/cold_data/cold_start_processed/drug_smiles.csv")
print(f"Loaded: train={len(train_ds)} s1={len(s1_ds)} s2={len(s2_ds)} ({time.time()-t0:.1f}s)")

collator = ColdStartCollator(args)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collator, num_workers=0)
s1_loader = DataLoader(s1_ds, batch_size=256, shuffle=False, collate_fn=collator, num_workers=0)
s2_loader = DataLoader(s2_ds, batch_size=256, shuffle=False, collate_fn=collator, num_workers=0)

print("Building model...")
model = build_ddi_model(args)
add_FUNC_prompt(model, args)
model.to(args.device)
print(f"Parameters: {param_count(model):,}")

loss_fn = BPRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

print(f"\nTraining first epoch ({len(train_loader)} batches)...")
model.train()
train_loss = 0.0
accum_steps = 4

for batch_idx, batch in enumerate(train_loader):
    t0 = time.time()
    pos_score = model('finetune', False, batch['pos'][0], batch['pos'][1], 'attention')
    neg_score = model('finetune', False, batch['neg'][0], batch['neg'][1], 'attention')
    loss, _, _ = loss_fn(pos_score, neg_score)
    loss = loss / accum_steps
    loss.backward()
    
    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss += loss.item() * accum_steps * len(pos_score)
    
    if (batch_idx + 1) % 50 == 0:
        print(f"  Batch {batch_idx+1}/{len(train_loader)}: {time.time()-t0:.2f}s/batch  loss={loss.item()*accum_steps:.4f}")

train_loss /= len(train_ds)
print(f"Epoch done! train_loss={train_loss:.4f}")

# Quick eval
print("\nEvaluating S1...")
s1_scores = {}
model.eval()
with torch.no_grad():
    import numpy as np
    from sklearn import metrics
    probas_pred = []
    ground_truth = []
    for batch in s1_loader:
        pos_score = model('finetune', False, batch['pos'][0], batch['pos'][1], 'attention')
        neg_score = model('finetune', False, batch['neg'][0], batch['neg'][1], 'attention')
        probas_pred.extend(torch.sigmoid(pos_score).cpu().numpy().tolist())
        probas_pred.extend(torch.sigmoid(neg_score).cpu().numpy().tolist())
        ground_truth.extend([1]*len(pos_score))
        ground_truth.extend([0]*len(neg_score))
    
    probas_pred = np.array(probas_pred)
    ground_truth = np.array(ground_truth)
    pred = (probas_pred >= 0.5).astype(int)
    s1_scores['acc'] = metrics.accuracy_score(ground_truth, pred)
    s1_scores['auroc'] = metrics.roc_auc_score(ground_truth, probas_pred)
    s1_scores['f1'] = metrics.f1_score(ground_truth, pred)
    print(f"S1: acc={s1_scores['acc']:.4f} auroc={s1_scores['auroc']:.4f} f1={s1_scores['f1']:.4f}")

print("\nEvaluating S2...")
with torch.no_grad():
    probas_pred = []
    ground_truth = []
    for batch in s2_loader:
        pos_score = model('finetune', False, batch['pos'][0], batch['pos'][1], 'attention')
        neg_score = model('finetune', False, batch['neg'][0], batch['neg'][1], 'attention')
        probas_pred.extend(torch.sigmoid(pos_score).cpu().numpy().tolist())
        probas_pred.extend(torch.sigmoid(neg_score).cpu().numpy().tolist())
        ground_truth.extend([1]*len(pos_score))
        ground_truth.extend([0]*len(neg_score))
    
    probas_pred = np.array(probas_pred)
    ground_truth = np.array(ground_truth)
    pred = (probas_pred >= 0.5).astype(int)
    s2_scores['acc'] = metrics.accuracy_score(ground_truth, pred)
    s2_scores['auroc'] = metrics.roc_auc_score(ground_truth, probas_pred)
    s2_scores['f1'] = metrics.f1_score(ground_truth, pred)
    print(f"S2: acc={s2_scores['acc']:.4f} auroc={s2_scores['auroc']:.4f} f1={s2_scores['f1']:.4f}")

print("\nSUCCESS!")
