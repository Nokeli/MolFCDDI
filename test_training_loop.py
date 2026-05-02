import torch
import sys
import time
sys.path.insert(0, '.')

from torch.utils.data import DataLoader
from cold_data_loader import ColdStartDataset, ColdStartCollator
from chemprop.models import build_ddi_model, add_FUNC_prompt
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

print("Loading dataset...")
ds = ColdStartDataset(
    "data/cold_data/cold_start_processed/fold0/train.csv",
    "data/cold_data/cold_start_processed/drug_smiles.csv"
)
collator = ColdStartCollator(args)
loader = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collator)

print(f"Dataset: {len(ds)} samples, {len(loader)} batches")

model = build_ddi_model(args)
add_FUNC_prompt(model, args)
model.to(args.device)
loss_fn = BPRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTesting first 10 batches...")
model.train()
for i, batch in enumerate(loader):
    t0 = time.time()
    pos_score = model('finetune', False, batch['pos'][0], batch['pos'][1], 'attention')
    neg_score = model('finetune', False, batch['neg'][0], batch['neg'][1], 'attention')
    loss, _, _ = loss_fn(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dt = time.time() - t0
    print(f"  Batch {i+1}/{len(loader)}: {dt:.2f}s  loss={loss.item():.4f}  "
          f"pos={pos_score.mean().item():.3f} neg={neg_score.mean().item():.3f}")
    if i >= 9:
        break

print("\nSUCCESS - training loop works!")
