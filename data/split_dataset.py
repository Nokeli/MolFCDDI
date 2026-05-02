"""
DrugBank Dataset Splitting — Binary & Multi-class
=================================================

Split ratio: 80% train / 10% val / 10% test
Output format: .npy files

Binary classification:
  - Positive = all DrugBank pairs (including label=0)
  - Negative = sampled pairs NOT in DrugBank (1:1 balanced)
  - Apply S1/S2/S3 splits on combined pos+neg dataset

Multi-class classification:
  - Only positive pairs (label != 0), original classes
  - Apply S1/S2/S3 splits

Three evaluation scenarios:
  S1 (Random Split):      Both drugs seen in training; only pair is held out.
  S2 (One-unseen):         One known drug + one completely new drug.
  S3 (Both-unseen):        Both drugs are new (cold start).
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# === Paths ===
DATA_DIR = Path("E:/学习/molfcddi/data/drugbank")
OUTPUT_DIR = Path("E:/学习/molfcddi/data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load raw data ===
raw_df = pd.read_csv(DATA_DIR / "drugbank.csv")
print(f"DrugBank entries: {len(raw_df):,}")

def canon(d1, d2):
    return (min(d1, d2, key=str), max(d1, d2, key=str))

all_drugs = sorted(set(raw_df['smiles1']) | set(raw_df['smiles2']))
n_drugs = len(all_drugs)
drug_list = all_drugs
print(f"Unique drugs: {n_drugs}")

# Positive pairs
positive_pairs = set()
pos_records = []
for _, row in raw_df.iterrows():
    d1, d2 = canon(row['smiles1'], row['smiles2'])
    positive_pairs.add((d1, d2))
    pos_records.append({'smiles1': d1, 'smiles2': d2, 'label': row['label']})

positive_df = pd.DataFrame(pos_records).drop_duplicates(subset=['smiles1', 'smiles2']).reset_index(drop=True)
print(f"Unique positive pairs: {len(positive_df):,}")

# Multi-class: only label != 0
multiclass_df = positive_df[positive_df['label'] != 0].copy().reset_index(drop=True)
print(f"Multi-class pairs: {len(multiclass_df):,}")

# Binary positive
binary_positive_df = positive_df.copy().reset_index(drop=True)

# ============================================================
# Negative sampling (1:1 balanced)
# ============================================================
total_possible = n_drugs * (n_drugs - 1) // 2
neg_candidates = total_possible - len(positive_pairs)
print(f"Negative candidates: {neg_candidates:,}")
print("Sampling negative pairs...")

neg_set = set()
pos_count = len(positive_df)
max_attempts = neg_candidates * 10
attempts = 0
while len(neg_set) < pos_count:
    if attempts > max_attempts:
        break
    attempts += 1
    i, j = np.random.randint(0, n_drugs, 2)
    if i == j:
        continue
    d1, d2 = drug_list[i], drug_list[j]
    pair = canon(d1, d2)
    if pair not in positive_pairs and pair not in neg_set:
        neg_set.add(pair)

negative_df = pd.DataFrame(
    [{'smiles1': p[0], 'smiles2': p[1], 'label': 0} for p in neg_set]
)
print(f"Negative samples: {len(negative_df):,}")

binary_df = pd.concat([binary_positive_df, negative_df], ignore_index=True)
binary_df = binary_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Binary total: {len(binary_df):,}  (pos={int((binary_df['label']!=0).sum()):,}, neg={int((binary_df['label']==0).sum()):,})")

# ============================================================
# Split functions
# ============================================================

def three_way_split(df, train_ratio=0.8, val_ratio=0.1):
    """Random split into train/val/test (80/10/10)."""
    indices = np.random.permutation(len(df))
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()


def one_unseen_split(df, all_drugs, test_drug_ratio=0.2):
    """
    Split drugs: unseen pool (test-only) vs seen pool (train/val).
    Test = pairs where ONE drug is unseen.
    Train/Val = pairs where BOTH drugs are seen (then 80/10/10 within).
    """
    n_test = int(len(all_drugs) * test_drug_ratio)
    drugs_perm = np.random.permutation(all_drugs)
    test_drugs = set(drugs_perm[:n_test])
    train_drugs = set(drugs_perm[n_test:])

    # Pairs where both drugs are seen -> split into train/val/test
    seen_mask = df['smiles1'].isin(train_drugs) & df['smiles2'].isin(train_drugs)
    seen_df = df[seen_mask].copy()

    # Pairs where one drug is unseen -> goes entirely to test
    one_unseen_mask = (
        (df['smiles1'].isin(test_drugs) & df['smiles2'].isin(train_drugs)) |
        (df['smiles1'].isin(train_drugs) & df['smiles2'].isin(test_drugs))
    )
    one_unseen_df = df[one_unseen_mask].copy()

    # Random split of seen pairs into train/val/test (80/10/10 of seen)
    seen_train, seen_val, seen_test = three_way_split(seen_df, 0.8, 0.1)

    # Test = seen_test + all one-unseen pairs
    test_df = pd.concat([seen_test, one_unseen_df], ignore_index=True)

    return seen_train, seen_val, test_df


def both_unseen_split(df, all_drugs, test_drug_ratio=0.2):
    """
    Split drugs: unseen pool vs seen pool.
    Test = pairs where BOTH drugs are unseen.
    Train/Val = all remaining pairs (split 80/10/10).
    """
    n_test = int(len(all_drugs) * test_drug_ratio)
    drugs_perm = np.random.permutation(all_drugs)
    test_drugs = set(drugs_perm[:n_test])

    both_unseen_mask = df['smiles1'].isin(test_drugs) & df['smiles2'].isin(test_drugs)
    both_unseen_df = df[both_unseen_mask].copy()
    remaining_df = df[~both_unseen_mask].copy()

    train_df, val_df, seen_test_df = three_way_split(remaining_df, 0.8, 0.1)
    test_df = pd.concat([seen_test_df, both_unseen_df], ignore_index=True)

    return train_df, val_df, test_df


def save_npy(name, task, train_df, val_df, test_df):
    """Save as {name}_{task}_{split}.npy with columns smiles1, smiles2, label."""
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        out = split_df[['smiles1', 'smiles2', 'label']].copy()
        out.to_csv(f"{OUTPUT_DIR / name}_{task}_{split_name}_raw.csv", index=False)
        out.to_numpy()
        np.save(OUTPUT_DIR / f"{name}_{task}_{split_name}.npy", out.to_numpy())
        # Also save column names
        np.save(OUTPUT_DIR / f"{name}_{task}_{split_name}_cols.npy", np.array(['smiles1', 'smiles2', 'label'], dtype=object))


def print_stats(name, task, train_df, val_df, test_df):
    def stats(df):
        n = len(df)
        drugs = len(set(df['smiles1']) | set(df['smiles2']))
        pos = int((df['label'] != 0).sum())
        neg = int((df['label'] == 0).sum())
        n_cls = df['label'].nunique()
        return n, drugs, pos, neg, n_cls

    tr_n, tr_d, tr_p, tr_nneg, tr_c = stats(train_df)
    va_n, va_d, va_p, va_nneg, va_c = stats(val_df)
    te_n, te_d, te_p, te_nneg, te_c = stats(test_df)

    print(f"\n{'='*55}")
    print(f"  {name} / {task}")
    print(f"{'='*55}")
    print(f"  Split    Pairs       Drugs   Pos      Neg   Classes")
    print(f"  Train  {tr_n:>8,}   {tr_d:>5}  {tr_p:>6,}  {tr_nneg:>6,}   {tr_c:>5}")
    print(f"  Val    {va_n:>8,}   {va_d:>5}  {va_p:>6,}  {va_nneg:>6,}   {va_c:>5}")
    print(f"  Test   {te_n:>8,}   {te_d:>5}  {te_p:>6,}  {te_nneg:>6,}   {te_c:>5}")
    print(f"  Total  {tr_n+va_n+te_n:>8,}")

    # Verify distribution
    test_drugs = set(test_df['smiles1']) | set(test_df['smiles2'])
    train_drugs = set(train_df['smiles1']) | set(train_df['smiles2'])
    if name == "S1":
        print(f"  [OK] S1: train drugs cover {len(train_drugs & test_drugs)}/{len(test_drugs)} test drugs")
    elif name == "S2":
        unseen = test_drugs - train_drugs
        print(f"  [OK] S2: {len(unseen)}/{len(test_drugs)} test drugs are unseen")
    elif name == "S3":
        unseen = test_drugs - train_drugs
        print(f"  [OK] S3: {len(unseen)}/{len(test_drugs)} test drugs are unseen")


# ============================================================
# Run all splits
# ============================================================
def run_split(split_fn, df_input, all_drugs):
    """Unified interface for all split functions."""
    if split_fn is three_way_split:
        return split_fn(df_input)
    else:
        return split_fn(df_input, all_drugs)

splits_config = [
    ("S1_random", "binary", three_way_split, binary_df.copy()),
    ("S2_one_unseen", "binary", one_unseen_split, binary_df.copy()),
    ("S3_both_unseen", "binary", both_unseen_split, binary_df.copy()),
    ("S1_random", "multiclass", three_way_split, multiclass_df.copy()),
    ("S2_one_unseen", "multiclass", one_unseen_split, multiclass_df.copy()),
    ("S3_both_unseen", "multiclass", both_unseen_split, multiclass_df.copy()),
]

for name, task, split_fn, df_input in splits_config:
    train_df, val_df, test_df = run_split(split_fn, df_input, all_drugs)
    save_npy(name, task, train_df, val_df, test_df)
    print_stats(name, task, train_df, val_df, test_df)

# ============================================================
# File listing
# ============================================================
print(f"\n{'='*55}")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*55}")
npy_files = sorted(OUTPUT_DIR.glob("*.npy"))
for f in npy_files:
    print(f"  {f.name}  ({f.stat().st_size/1024/1024:.2f} MB)")

print(f"\nTotal: {len(npy_files)} .npy files (+ {len(list(OUTPUT_DIR.glob('*_raw.csv')))} raw CSVs)")
