"""
DrugBank Dataset Splitter for MoIFCL Robustness Evaluation

Three split scenarios:
- S1 (Random Split): Both drugs in test pairs seen in training (interpolation).
- S2 (One-unseen): One drug is completely new (new drug risk assessment).
- S3 (Both-unseen): Both drugs are unseen (cold-start).

Multi-class: original 86-class labels (0=negative interaction type, 1-85=positive types).
Binary: ALL original pairs = positive (label=1).
        Negative = random drug pairs NOT in original dataset, sampled 1:1.
Split ratio: 8:1:1 (train/val/test).
"""

import pandas as pd
import numpy as np
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "E:\\学习\\molfcddi\\data\\drugbank\\drugbank.csv"
OUTPUT_DIR = "E:\\学习\\molfcddi\\data\\splits"


def load_data():
    df = pd.read_csv(DATA_PATH)
    n_drugs = len(set(df['smiles1']) | set(df['smiles2']))
    print(f"Loaded {len(df)} pairs, {df['label'].nunique()} classes, {n_drugs} drugs")
    return df


def _shuffle_split(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    idx = np.random.permutation(n)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return df.iloc[idx[:t]].copy(), df.iloc[idx[t:t+v]].copy(), df.iloc[idx[t+v:]].copy()


def _stratified_split(df, train_ratio=0.8, val_ratio=0.1):
    """
    Stratified shuffle split: preserve label distribution across train/val/test.
    Uses StratifiedShuffleSplit, but classes with <2 samples (after train split)
    are forced into train since they can't be further split.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = df['label'].values
    n = len(df)
    t = int(n * train_ratio)
    v = int(n * val_ratio)

    # First stratified split: train vs rest
    sss = StratifiedShuffleSplit(n_splits=1, train_size=t, random_state=RANDOM_SEED)
    train_idx = next(sss.split(df, labels))[0]

    # Identify remaining set
    remaining_mask = np.ones(n, bool)
    remaining_mask[train_idx] = False
    remaining_labels = labels[remaining_mask]

    # Second stratified split: val vs test from remaining
    # Handle rare classes (<2 samples) by moving them to train
    rare_mask = pd.Series(remaining_labels).value_counts()
    rare_labels = set(rare_mask[rare_mask < 2].index)

    if rare_labels:
        rare_idx_global = np.where(pd.Series(remaining_labels).isin(rare_labels))[0]
        rare_idx_original = np.where(remaining_mask)[0][rare_idx_global]
        train_idx = np.concatenate([train_idx, rare_idx_original])
        remaining_mask[rare_idx_original] = False
        remaining_labels = labels[remaining_mask]

    n_rem = remaining_labels.shape[0]
    v_actual = min(v, n_rem - 1)  # need at least 2 samples for stratified split
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=v_actual, random_state=RANDOM_SEED)
    val_idx_local = next(sss2.split(np.zeros(n_rem), remaining_labels))[0]
    val_idx_global = np.where(remaining_mask)[0][val_idx_local]

    val_mask_arr = np.zeros(n, bool)
    val_mask_arr[val_idx_global] = True
    test_mask_arr = remaining_mask & ~val_mask_arr

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df[val_mask_arr].reset_index(drop=True)
    test_df = df[test_mask_arr].reset_index(drop=True)
    return train_df, val_df, test_df


def _fix_unseen(train, test):
    """Move pairs with drugs not in train from test to train."""
    train_drugs = set(train['smiles1']) | set(train['smiles2'])
    unseen = (set(test['smiles1']) | set(test['smiles2'])) - train_drugs
    if not unseen:
        return train, test
    move = test['smiles1'].isin(unseen) | test['smiles2'].isin(unseen)
    train = pd.concat([train, test[move]], ignore_index=True)
    test = test[~move].reset_index(drop=True)
    print(f"    Moved {move.sum()} pairs (unseen drugs) → train")
    return train, test


def generate_negative_pairs(n_pos, all_drugs_list, pos_existing_pairs):
    """
    Generate n_pos negative pairs (drug pairs NOT in pos_existing_pairs).
    Rejection sampling: random pair, skip if already exists.
    """
    n = len(all_drugs_list)
    neg_pairs = []
    attempts = 0
    max_attempts = n_pos * 200

    while len(neg_pairs) < n_pos and attempts < max_attempts:
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        attempts += 1
        if i == j:
            continue
        pair = (all_drugs_list[i], all_drugs_list[j])
        if pair not in pos_existing_pairs:
            neg_pairs.append(pair)

    if len(neg_pairs) < n_pos:
        print(f"    Warning: only got {len(neg_pairs)}/{n_pos} negatives")
    return neg_pairs


def make_binary(df_mc, all_drugs_list, pos_existing_pairs):
    """
    Binary: ALL original pairs = positive (label=1).
            Negatives = random non-existing pairs, sampled 1:1.
    """
    pos = df_mc.copy()
    n_pos = len(pos)
    pos['label'] = 1   # ← ALL original pairs become positive

    # Generate 1:1 negatives
    neg_pairs = generate_negative_pairs(n_pos, all_drugs_list, pos_existing_pairs)

    neg = pd.DataFrame(neg_pairs, columns=['smiles1', 'smiles2'])
    neg['label'] = 0

    out = pd.concat([pos, neg], ignore_index=True)
    out['label'] = out['label'].astype(int)
    out = out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return out


# ─── S1: Random Split ───────────────────────────────────────────
def split_s1(df):
    print("\n=== S1: Random Split ===")
    all_drugs_list = list(set(df['smiles1']) | set(df['smiles2']))
    pos_pairs = set(zip(df[df['label'] > 0]['smiles1'], df[df['label'] > 0]['smiles2']))

    # Multiclass: stratified split to preserve 86-class distribution
    train, val, test = _stratified_split(df, 0.8, 0.1)
    train, val = _fix_unseen(train, val)
    train, test = _fix_unseen(train, test)

    print(f"    MC  — Train:{len(train)}  Val:{len(val)}  Test:{len(test)}")

    # Binary: use regular shuffle (balanced 1:1, no stratification needed)
    train_bin = make_binary(train, all_drugs_list, pos_pairs)
    val_bin   = make_binary(val,   all_drugs_list, pos_pairs)
    test_bin  = make_binary(test,  all_drugs_list, pos_pairs)

    print(f"    Bin — Train:{len(train_bin)}  Val:{len(val_bin)}  Test:{len(test_bin)}")
    return train, val, test, train_bin, val_bin, test_bin


# ─── S2: One-unseen ─────────────────────────────────────────────
def split_s2(df, unseen_drugs):
    print("\n=== S2: One-unseen ===")
    all_drugs = set(df['smiles1']) | set(df['smiles2'])
    seen_drugs = all_drugs - unseen_drugs
    all_drugs_list = list(all_drugs)
    pos_pairs = set(zip(df[df['label'] > 0]['smiles1'], df[df['label'] > 0]['smiles2']))

    print(f"    Unseen:{len(unseen_drugs)}  Seen:{len(seen_drugs)}")

    # Type-A: both seen → train/val source
    mask_a = ~df['smiles1'].isin(unseen_drugs) & ~df['smiles2'].isin(unseen_drugs)
    df_a = df[mask_a].copy()
    print(f"    Type-A (both seen): {len(df_a)}")

    # Type-B: one seen + one unseen → S2 test
    mask_b = (df['smiles1'].isin(seen_drugs) & df['smiles2'].isin(unseen_drugs)) | \
             (df['smiles1'].isin(unseen_drugs) & df['smiles2'].isin(seen_drugs))
    df_b = df[mask_b].copy()
    print(f"    Type-B (one seen + one unseen): {len(df_b)}")

    # Verify S2 BEFORE stratified split (split may move rare samples into train,
    # which could change seen_tv)
    one_u = both_s = both_u = 0
    for _, row in df_b.iterrows():
        d1 = row['smiles1'] in seen_drugs
        d2 = row['smiles2'] in seen_drugs
        if d1 and d2: both_s += 1
        elif not d1 and not d2: both_u += 1
        else: one_u += 1
    print(f"    S2 verify — one_unseen={one_u}, both_seen={both_s}, both_unseen={both_u}")

    train_a, val_a, _ = _stratified_split(df_a, 0.8, 0.1)
    train, val, test = train_a, val_a, df_b

    train_bin = make_binary(train, all_drugs_list, pos_pairs)
    val_bin   = make_binary(val,   all_drugs_list, pos_pairs)
    test_bin  = make_binary(test,  all_drugs_list, pos_pairs)

    print(f"    MC  — Train:{len(train)}  Val:{len(val)}  Test:{len(test)}")
    print(f"    Bin — Train:{len(train_bin)}  Val:{len(val_bin)}  Test:{len(test_bin)}")
    return train, val, test, train_bin, val_bin, test_bin


# ─── S3: Both-unseen ────────────────────────────────────────────
def split_s3(df, unseen_drugs):
    print("\n=== S3: Both-unseen ===")
    all_drugs = set(df['smiles1']) | set(df['smiles2'])
    all_drugs_list = list(all_drugs)
    pos_pairs = set(zip(df[df['label'] > 0]['smiles1'], df[df['label'] > 0]['smiles2']))

    # Type-A: both seen → train/val source
    mask_a = ~df['smiles1'].isin(unseen_drugs) & ~df['smiles2'].isin(unseen_drugs)
    df_a = df[mask_a].copy()
    print(f"    Type-A (both seen): {len(df_a)}")

    # Type-C: both unseen → S3 test
    mask_c = df['smiles1'].isin(unseen_drugs) & df['smiles2'].isin(unseen_drugs)
    df_c = df[mask_c].copy()
    print(f"    Type-C (both unseen): {len(df_c)}")

    train_a, val_a, _ = _stratified_split(df_a, 0.8, 0.1)
    train, val, test = train_a, val_a, df_c

    # Verify S3
    seen_tv = (set(train['smiles1']) | set(train['smiles2']) |
               set(val['smiles1'])   | set(val['smiles2']))
    both_u = sum(1 for _, r in test.iterrows()
                 if r['smiles1'] not in seen_tv and r['smiles2'] not in seen_tv)
    print(f"    S3 verify — both_unseen={both_u}/{len(test)}")

    train_bin = make_binary(train, all_drugs_list, pos_pairs)
    val_bin   = make_binary(val,   all_drugs_list, pos_pairs)
    test_bin  = make_binary(test,  all_drugs_list, pos_pairs)

    print(f"    MC  — Train:{len(train)}  Val:{len(val)}  Test:{len(test)}")
    print(f"    Bin — Train:{len(train_bin)}  Val:{len(val_bin)}  Test:{len(test_bin)}")
    return train, val, test, train_bin, val_bin, test_bin


def select_unseen_drugs(df, ratio=0.2):
    all_drugs = list(set(df['smiles1']) | set(df['smiles2']))
    np.random.shuffle(all_drugs)
    n = max(1, int(len(all_drugs) * ratio))
    unseen = set(all_drugs[:n])
    print(f"\nUnseen drugs: {len(unseen)} / {len(all_drugs)}")
    return unseen


def save_splits(splits, out_dir):
    """
    Save splits as .csv and .npy (2D array format + _cols.npy, matching experiment_ddi.py).
    Naming convention: S1_random, S2_one_unseen, S3_both_unseen.
    """
    COLS = ['smiles1', 'smiles2', 'label']

    for sc_key, (mc_tr, mc_va, mc_te, b_tr, b_va, b_te) in splits.items():
        for ltype, df_dict in [('multiclass', {'train': mc_tr, 'val': mc_va, 'test': mc_te}),
                                ('binary',     {'train': b_tr,  'val': b_va,  'test': b_te})]:
            for name, df in df_dict.items():
                base = f"{sc_key}_{ltype}_{name}"
                # CSV
                df[COLS].to_csv(os.path.join(out_dir, f"{base}.csv"), index=False)
                # NPY: 2D array (N, 3) matching experiment_ddi.py format
                arr = df[COLS].values  # shape (N, 3)
                np.save(os.path.join(out_dir, f"{base}.npy"), arr)
                # Companion _cols.npy
                np.save(os.path.join(out_dir, f"{base}_cols.npy"), np.array(COLS))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ratio: 8:1:1 | Binary: all pairs=pos, negatives=sampled non-existing pairs (1:1)")

    df = load_data()

    s1 = split_s1(df)
    unseen = select_unseen_drugs(df, ratio=0.2)
    s2 = split_s2(df, unseen)
    s3 = split_s3(df, unseen)

    splits = {'S1_random': s1, 'S2_one_unseen': s2, 'S3_both_unseen': s3}
    save_splits(splits, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'Scenario':<8} {'Type':<12} {'Train':>10} {'Val':>10} {'Test':>10}")
    print("=" * 68)
    for sc, (mc_t, mc_v, mc_e, b_t, b_v, b_e) in splits.items():
        print(f"{sc:<8} {'multiclass':<12} {len(mc_t):>10} {len(mc_v):>10} {len(mc_e):>10}")
        print(f"{'':8} {'binary':<12} {len(b_t):>10} {len(b_v):>10} {len(b_e):>10}")

    print("\nBinary label distribution (should be 1:1 pos:neg):")
    for sc, (_, _, _, b_t, b_v, b_e) in splits.items():
        for n, d in [('Train', b_t), ('Val', b_v), ('Test', b_e)]:
            p = (d['label'] == 1).sum()
            nz = (d['label'] == 0).sum()
            print(f"  {sc} binary {n}: pos={p}, neg={nz}, ratio={p/max(nz,1):.3f}")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
