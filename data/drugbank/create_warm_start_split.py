"""
DrugBank 86-class warm-start split generator.

Goals:
1. Stratified random split by interaction type.
2. train / valid / test = 8 / 1 / 1.
3. Drug overlap across splits is allowed.
4. The same unordered drug pair must not leak across splits.

Output filenames follow the training loader convention:
    warm_start_multiclass_train.npy / .csv
    warm_start_multiclass_val.npy   / .csv
    warm_start_multiclass_test.npy  / .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def canonical_pair(smiles1: str, smiles2: str):
    return tuple(sorted((smiles1, smiles2)))


def load_rows(data_path: str):
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "smiles1": row["smiles1"],
                "smiles2": row["smiles2"],
                "label": int(row["label"])
            })
    return rows


def build_pair_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[canonical_pair(row["smiles1"], row["smiles2"])].append(row)

    groups = []
    for pair_key, pair_rows in grouped.items():
        label_counter = Counter(row["label"] for row in pair_rows)
        groups.append({
            "pair": pair_key,
            "rows": pair_rows,
            "size": len(pair_rows),
            "label_counter": label_counter,
            "labels": tuple(sorted(label_counter))
        })
    return groups


def proportional_targets(total_count: int, ratios):
    raw = [total_count * r for r in ratios]
    base = [math.floor(x) for x in raw]
    remainder = total_count - sum(base)
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def split_single_label_groups(single_groups, ratios, seed):
    labels = np.array([group["labels"][0] for group in single_groups])
    indices = np.arange(len(single_groups))

    train_size = ratios[0]
    remaining_ratio = ratios[1] + ratios[2]
    val_ratio_within_remaining = ratios[1] / remaining_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, remaining_idx = next(sss1.split(indices, labels))

    remaining_labels = labels[remaining_idx]
    remaining_indices = indices[remaining_idx]

    remaining_counter = Counter(remaining_labels.tolist())
    rare_labels = {label for label, count in remaining_counter.items() if count < 2}
    if rare_labels:
        rare_mask = np.array([label in rare_labels for label in remaining_labels], dtype=bool)
        rare_indices = remaining_indices[rare_mask]
        train_idx = np.concatenate([train_idx, rare_indices])
        remaining_indices = remaining_indices[~rare_mask]
        remaining_labels = labels[remaining_indices]

    if len(remaining_indices) == 0:
        return {
            "train": [single_groups[i] for i in train_idx],
            "val": [],
            "test": []
        }

    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio_within_remaining, random_state=seed + 1)
    val_idx_local, test_idx_local = next(sss2.split(remaining_indices, remaining_labels))

    train_groups = [single_groups[i] for i in train_idx]
    val_groups = [single_groups[remaining_indices[i]] for i in val_idx_local]
    test_groups = [single_groups[remaining_indices[i]] for i in test_idx_local]
    return {"train": train_groups, "val": val_groups, "test": test_groups}


def current_label_counts(assigned):
    counts = {split: Counter() for split in assigned}
    for split_name, groups in assigned.items():
        for group in groups:
            counts[split_name].update(group["label_counter"])
    return counts


def current_size_counts(assigned):
    return {
        split_name: sum(group["size"] for group in groups)
        for split_name, groups in assigned.items()
    }


def assign_groups(groups, ratios, seed):
    split_names = ["train", "val", "test"]
    rng = random.Random(seed)
    label_totals = Counter()
    for group in groups:
        label_totals.update(group["label_counter"])

    target_sizes = proportional_targets(sum(group["size"] for group in groups), ratios)
    target_labels = {
        label: proportional_targets(count, ratios)
        for label, count in sorted(label_totals.items())
    }

    single_label_groups = [group for group in groups if len(group["labels"]) == 1]
    multi_label_groups = [group for group in groups if len(group["labels"]) > 1]

    assigned = split_single_label_groups(single_label_groups, ratios, seed)
    label_counts = current_label_counts(assigned)
    size_counts = {
        split_name: sum(group["size"] for group in assigned[split_name])
        for split_name in split_names
    }

    rng.shuffle(multi_label_groups)
    multi_label_groups.sort(
        key=lambda g: (
            -len(g["labels"]),
            min(label_totals[label] for label in g["labels"]),
            -g["size"]
        )
    )

    for group in multi_label_groups:
        best_split = None
        best_score = None

        for split_idx, split_name in enumerate(split_names):
            new_size = size_counts[split_name] + group["size"]
            size_overflow = max(0, new_size - target_sizes[split_idx])
            size_gap = abs(new_size - target_sizes[split_idx])

            label_gap = 0.0
            label_overflow = 0.0
            for label, count in group["label_counter"].items():
                proposed = label_counts[split_name][label] + count
                target = target_labels[label][split_idx]
                label_gap += abs(proposed - target)
                label_overflow += max(0, proposed - target)

            score = (
                size_overflow * 1000.0 +
                label_overflow * 100.0 +
                label_gap * 10.0 +
                size_gap
            )

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_idx

        split_name = split_names[best_split]
        assigned[split_name].append(group)
        size_counts[split_name] += group["size"]
        label_counts[split_name].update(group["label_counter"])

    return assigned, target_sizes, target_labels


def repair_split_coverage(assigned, target_sizes):
    """
    Ensure train/val/test each cover all labels that appear in the dataset
    when possible, while keeping pair leakage at zero.
    """
    split_names = ["train", "val", "test"]
    label_counts = current_label_counts(assigned)
    size_counts = current_size_counts(assigned)
    all_labels = sorted(set().union(*[set(label_counts[split]) for split in split_names]))

    for target_split in split_names:
        missing_labels = [label for label in all_labels if label_counts[target_split][label] == 0]
        for label in missing_labels:
            candidates = []
            for donor_split in split_names:
                if donor_split == target_split:
                    continue
                for idx, group in enumerate(assigned[donor_split]):
                    if group["label_counter"][label] == 0:
                        continue
                    if label_counts[donor_split][label] - group["label_counter"][label] <= 0:
                        continue
                    new_donor_size = size_counts[donor_split] - group["size"]
                    new_target_size = size_counts[target_split] + group["size"]
                    size_penalty = (
                        abs(new_donor_size - target_sizes[split_names.index(donor_split)]) +
                        abs(new_target_size - target_sizes[split_names.index(target_split)])
                    )
                    label_damage = sum(
                        1 for group_label, count in group["label_counter"].items()
                        if label_counts[donor_split][group_label] - count <= 0
                    )
                    candidates.append((label_damage, size_penalty, group["size"], donor_split, idx, group))

            if not candidates:
                continue

            _, _, _, donor_split, idx, group = min(candidates)
            moved = assigned[donor_split].pop(idx)
            assigned[target_split].append(moved)
            label_counts[donor_split].subtract(moved["label_counter"])
            label_counts[target_split].update(moved["label_counter"])
            size_counts[donor_split] -= moved["size"]
            size_counts[target_split] += moved["size"]

    return assigned


def rebalance_train_tail(assigned, target_sizes, min_train_count=4):
    """
    Pull scarce labels back into train when val/test have spare examples.
    """
    split_names = ["train", "val", "test"]
    label_counts = current_label_counts(assigned)
    size_counts = current_size_counts(assigned)

    low_labels = [label for label, count in label_counts["train"].items() if count < min_train_count]
    for label in sorted(low_labels, key=lambda x: label_counts["train"][x]):
        while label_counts["train"][label] < min_train_count:
            candidates = []
            for donor_split in ["val", "test"]:
                for idx, group in enumerate(assigned[donor_split]):
                    carry = group["label_counter"][label]
                    if carry == 0:
                        continue
                    if label_counts[donor_split][label] - carry <= 0:
                        continue
                    new_train_size = size_counts["train"] + group["size"]
                    new_donor_size = size_counts[donor_split] - group["size"]
                    size_penalty = (
                        abs(new_train_size - target_sizes[0]) +
                        abs(new_donor_size - target_sizes[split_names.index(donor_split)])
                    )
                    label_damage = sum(
                        1 for group_label, count in group["label_counter"].items()
                        if label_counts[donor_split][group_label] - count <= 0
                    )
                    candidates.append((label_damage, size_penalty, -carry, group["size"], donor_split, idx, group))

            if not candidates:
                break

            _, _, _, _, donor_split, idx, group = min(candidates)
            moved = assigned[donor_split].pop(idx)
            assigned["train"].append(moved)
            label_counts[donor_split].subtract(moved["label_counter"])
            label_counts["train"].update(moved["label_counter"])
            size_counts[donor_split] -= moved["size"]
            size_counts["train"] += moved["size"]

    return assigned


def flatten_assigned_groups(assigned):
    flat = {}
    for split_name, groups in assigned.items():
        rows = []
        for group in groups:
            rows.extend(group["rows"])
        flat[split_name] = rows
    return flat


def summarize_splits(flat_splits):
    summary = {}
    pair_sets = {}
    drug_sets = {}

    for split_name, rows in flat_splits.items():
        labels = Counter(row["label"] for row in rows)
        pairs = {canonical_pair(row["smiles1"], row["smiles2"]) for row in rows}
        drugs = {row["smiles1"] for row in rows} | {row["smiles2"] for row in rows}
        pair_sets[split_name] = pairs
        drug_sets[split_name] = drugs

        summary[split_name] = {
            "num_rows": len(rows),
            "num_pairs": len(pairs),
            "num_drugs": len(drugs),
            "label_distribution": dict(sorted(labels.items()))
        }

    summary["overlap"] = {
        "pair_train_val": len(pair_sets["train"] & pair_sets["val"]),
        "pair_train_test": len(pair_sets["train"] & pair_sets["test"]),
        "pair_val_test": len(pair_sets["val"] & pair_sets["test"]),
        "drug_train_val": len(drug_sets["train"] & drug_sets["val"]),
        "drug_train_test": len(drug_sets["train"] & drug_sets["test"]),
        "drug_val_test": len(drug_sets["val"] & drug_sets["test"])
    }
    return summary


def save_split(rows, npy_path, csv_path):
    array = np.array([[row["smiles1"], row["smiles2"], row["label"]] for row in rows], dtype=object)
    np.save(npy_path, array)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles1", "smiles2", "label"])
        writer.writerows(array.tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate DrugBank warm-start 86-class splits without pair leakage.")
    parser.add_argument("--data_path", type=str, default="./data/drugbank.csv")
    parser.add_argument("--output_dir", type=str, default="./data/splits")
    parser.add_argument("--split_name", type=str, default="warm_start")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.data_path)
    groups = build_pair_groups(rows)
    assigned, target_sizes, target_labels = assign_groups(groups, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    assigned = repair_split_coverage(assigned, target_sizes)
    assigned = rebalance_train_tail(assigned, target_sizes, min_train_count=4)
    flat_splits = flatten_assigned_groups(assigned)

    for split_name in ["train", "val", "test"]:
        base = f"{args.split_name}_multiclass_{split_name}"
        save_split(
            flat_splits[split_name],
            npy_path=os.path.join(args.output_dir, f"{base}.npy"),
            csv_path=os.path.join(args.output_dir, f"{base}.csv")
        )

    summary = summarize_splits(flat_splits)
    summary["meta"] = {
        "data_path": os.path.abspath(args.data_path),
        "seed": args.seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "target_sizes": {"train": target_sizes[0], "val": target_sizes[1], "test": target_sizes[2]},
        "target_label_counts": {str(k): {"train": v[0], "val": v[1], "test": v[2]} for k, v in target_labels.items()}
    }

    summary_path = os.path.join(args.output_dir, f"{args.split_name}_multiclass_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved warm-start splits to {os.path.abspath(args.output_dir)}")
    print(json.dumps(summary["overlap"], indent=2))


if __name__ == "__main__":
    main()
