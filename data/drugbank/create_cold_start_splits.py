"""
DrugBank multiclass cold-start split generator.

Outputs:
    cold_semi_multiclass_{train,val,test}.npy/.csv
    cold_strict_multiclass_{train,val,test}.npy/.csv

Definitions:
    cold_semi:
        val/test each contain at least one drug unseen in train,
        and val/test use disjoint holdout drug pools.
    cold_strict:
        val/test each contain two drugs unseen in train,
        and val/test use disjoint holdout drug pools.
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


SPLIT_NAMES = ["train", "val", "test"]
TARGET_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def canonical_pair(smiles1: str, smiles2: str):
    return tuple(sorted((smiles1, smiles2)))


def load_rows(data_path: str):
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {"smiles1": row["smiles1"], "smiles2": row["smiles2"], "label": int(row["label"])}
            for row in reader
        ]


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
            "labels": tuple(sorted(label_counter)),
            "drug_set": set(pair_key),
        })
    return groups


def flatten_groups(groups):
    rows = []
    for group in groups:
        rows.extend(group["rows"])
    return rows


def summarize_rows(rows):
    labels = Counter(row["label"] for row in rows)
    pairs = {canonical_pair(row["smiles1"], row["smiles2"]) for row in rows}
    drugs = {row["smiles1"] for row in rows} | {row["smiles2"] for row in rows}
    return {
        "num_rows": len(rows),
        "num_pairs": len(pairs),
        "num_drugs": len(drugs),
        "num_classes": len(labels),
        "min_class_count": min(labels.values()) if labels else 0,
        "label_distribution": dict(sorted(labels.items())),
    }


def save_split(rows, npy_path, csv_path):
    array = np.array([[row["smiles1"], row["smiles2"], row["label"]] for row in rows], dtype=object)
    np.save(npy_path, array)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles1", "smiles2", "label"])
        writer.writerows(array.tolist())


def partition_groups(groups, val_holdout, test_holdout, mode):
    assigned = {"train": [], "val": [], "test": [], "dropped": []}

    for group in groups:
        pair_drugs = group["drug_set"]
        val_hits = len(pair_drugs & val_holdout)
        test_hits = len(pair_drugs & test_holdout)

        if val_hits == 0 and test_hits == 0:
            assigned["train"].append(group)
            continue

        if val_hits > 0 and test_hits > 0:
            assigned["dropped"].append(group)
            continue

        if mode == "cold_semi":
            if val_hits > 0:
                assigned["val"].append(group)
            elif test_hits > 0:
                assigned["test"].append(group)
            continue

        if mode == "cold_strict":
            if val_hits == 2:
                assigned["val"].append(group)
            elif test_hits == 2:
                assigned["test"].append(group)
            else:
                assigned["dropped"].append(group)
            continue

        raise ValueError(mode)

    return assigned


def score_candidate(assigned):
    split_rows = {split: flatten_groups(assigned[split]) for split in SPLIT_NAMES}
    summaries = {split: summarize_rows(split_rows[split]) for split in SPLIT_NAMES}

    kept_rows = sum(summaries[split]["num_rows"] for split in SPLIT_NAMES)
    if min(summaries[split]["num_rows"] for split in SPLIT_NAMES) == 0:
        return float("inf"), summaries

    ratio_penalty = sum(
        abs((summaries[split]["num_rows"] / kept_rows) - TARGET_RATIOS[split])
        for split in SPLIT_NAMES
    )

    class_penalty = 0
    for split in SPLIT_NAMES:
        class_penalty += (86 - summaries[split]["num_classes"]) * 2000

    train_tail_penalty = max(0, 4 - summaries["train"]["min_class_count"]) * 200
    val_tail_penalty = max(0, 1 - summaries["val"]["min_class_count"]) * 100
    test_tail_penalty = max(0, 1 - summaries["test"]["min_class_count"]) * 100

    score = class_penalty + train_tail_penalty + val_tail_penalty + test_tail_penalty + ratio_penalty * 1000
    return score, summaries


def search_best_split(groups, all_drugs, mode, seed):
    best = None
    drug_list = sorted(all_drugs)

    ratios = [0.08, 0.10, 0.12, 0.14]
    for attempt in range(8):
        rng = random.Random(seed + attempt)
        shuffled = drug_list[:]
        rng.shuffle(shuffled)

        for val_ratio in ratios:
            for test_ratio in ratios:
                val_size = max(1, int(len(shuffled) * val_ratio))
                test_size = max(1, int(len(shuffled) * test_ratio))
                if val_size + test_size >= len(shuffled):
                    continue

                val_holdout = set(shuffled[:val_size])
                test_holdout = set(shuffled[val_size:val_size + test_size])
                assigned = partition_groups(groups, val_holdout, test_holdout, mode=mode)
                score, summaries = score_candidate(assigned)

                candidate = {
                    "score": score,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio,
                    "val_holdout_size": val_size,
                    "test_holdout_size": test_size,
                    "summaries": summaries,
                    "assigned": assigned,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        raise RuntimeError(f"Failed to build {mode} split.")
    return best


def drug_overlap(split_rows):
    drug_sets = {
        split: ({row["smiles1"] for row in split_rows[split]} | {row["smiles2"] for row in split_rows[split]})
        for split in SPLIT_NAMES
    }
    return {
        "train_val": len(drug_sets["train"] & drug_sets["val"]),
        "train_test": len(drug_sets["train"] & drug_sets["test"]),
        "val_test": len(drug_sets["val"] & drug_sets["test"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate DrugBank cold-start multiclass splits.")
    parser.add_argument("--data_path", type=str, default="./data/drugbank.csv")
    parser.add_argument("--output_dir", type=str, default="./data/ddi_cold_start")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.data_path)
    groups = build_pair_groups(rows)
    all_drugs = {row["smiles1"] for row in rows} | {row["smiles2"] for row in rows}

    summary = {}
    for mode in ["cold_semi", "cold_strict"]:
        best = search_best_split(groups, all_drugs, mode=mode, seed=args.seed)
        split_rows = {split: flatten_groups(best["assigned"][split]) for split in SPLIT_NAMES}

        for split in SPLIT_NAMES:
            base = f"{mode}_multiclass_{split}"
            save_split(
                split_rows[split],
                npy_path=os.path.join(args.output_dir, f"{base}.npy"),
                csv_path=os.path.join(args.output_dir, f"{base}.csv"),
            )

        summary[mode] = {
            "val_ratio": best["val_ratio"],
            "test_ratio": best["test_ratio"],
            "val_holdout_size": best["val_holdout_size"],
            "test_holdout_size": best["test_holdout_size"],
            "dropped_rows": summarize_rows(flatten_groups(best["assigned"]["dropped"]))["num_rows"],
            "drug_overlap": drug_overlap(split_rows),
            "train": best["summaries"]["train"],
            "val": best["summaries"]["val"],
            "test": best["summaries"]["test"],
        }

    with open(os.path.join(args.output_dir, "cold_multiclass_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
