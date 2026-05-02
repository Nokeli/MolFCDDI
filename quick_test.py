import torch
import sys
import time
sys.path.insert(0, '.')

from cold_data_loader import ColdStartDataset, ColdStartCollator
from chemprop.models import build_ddi_model, add_FUNC_prompt

class Args:
    hidden_size = 300
    ffn_hidden_size = 300
    ffn_num_layers = 2
    depth = 3
    dropout = 0.1
    atom_messages = True
    dataset_type = 'classification'
    multiclass_num_classes = 2
    features_only = False
    features_size = 0
    use_input_features = False
    features_dim = 0
    activation = 'ReLU'
    bias = False
    undirected = False
    num_attention = 2
    num_attention_heads = 4
    add_step = 'concat_mol_frag_attention'
    step = 'func_prompt'
    pooling_type = 'attention'
    gamma = 0.01
    increase_parm = 1
    encoder_name = 'CMPNN'
    num_tasks = 1
    cuda = True
    device = torch.device('cuda:0')

args = Args()

print("="*60)
print("Quick Test: Cold Start Data Loading & Model Forward")
print("="*60)

# 1. Load dataset
print("\n[1/5] Loading dataset...")
t0 = time.time()
ds = ColdStartDataset(
    "data/cold_data/cold_start_processed/fold0/train.csv",
    "data/cold_data/cold_start_processed/drug_smiles.csv"
)
print(f"  Dataset: {len(ds)} samples ({time.time()-t0:.1f}s)")

# 2. Check first sample
print("\n[2/5] Checking first sample...")
sample = ds[0]
print(f"  Pos: {sample['pos']['drug1']} - {sample['pos']['drug2']} (type={sample['pos']['rel_type']})")
print(f"  Neg: {sample['neg']['drug1']} - {sample['neg']['drug2']}")

# 3. Collate single sample
print("\n[3/5] Collating single sample...")
t0 = time.time()
collator = ColdStartCollator(args)
try:
    batch = collator([sample])
    print(f"  Batch created ({time.time()-t0:.1f}s)")
    print(f"  Keys: {list(batch.keys())}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Build model
print("\n[4/5] Building model...")
t0 = time.time()
model = build_ddi_model(args)
add_FUNC_prompt(model, args)
model.to(args.device)
print(f"  Model built ({time.time()-t0:.1f}s)")

# 5. Forward pass
print("\n[5/5] Forward pass...")
t0 = time.time()
try:
    with torch.no_grad():
        pos_score = model('finetune', False, batch['pos'][0], batch['pos'][1], 'attention')
        neg_score = model('finetune', False, batch['neg'][0], batch['neg'][1], 'attention')
    print(f"  Pos score: {pos_score.item():.4f}")
    print(f"  Neg score: {neg_score.item():.4f}")
    print(f"  Forward time: {time.time()-t0:.1f}s")
    print("\n" + "="*60)
    print("  SUCCESS!")
    print("="*60)
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test mini-batch
print("\n[Bonus] Testing mini-batch of 4...")
batch4 = collator([ds[i] for i in range(4)])
t0 = time.time()
with torch.no_grad():
    pos_score = model('finetune', False, batch4['pos'][0], batch4['pos'][1], 'attention')
    neg_score = model('finetune', False, batch4['neg'][0], batch4['neg'][1], 'attention')
print(f"  Batch-4 forward: {time.time()-t0:.1f}s")
print(f"  Pos scores: {pos_score.cpu().numpy()}")
print(f"  Neg scores: {neg_score.cpu().numpy()}")
