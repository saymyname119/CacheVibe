#!/usr/bin/env python3

import os
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CLUSTERING_MODEL_DIR
from app.data_loader import load_and_clean_dataset


def main():
    documents, categories = load_and_clean_dataset()
    cat_labels = [doc.category for doc in documents]

    probs = np.load(os.path.join(CLUSTERING_MODEL_DIR, "membership_probs.npy"))
    hard_labels = np.load(os.path.join(CLUSTERING_MODEL_DIR, "hard_labels.npy"))
    n_clusters = probs.shape[1]

    print("=" * 70)
    print(f"CLUSTERING ANALYSIS  —  {n_clusters} clusters, {len(documents)} documents")
    print("=" * 70)

    print("\n" + "─" * 70)
    print("1. CLUSTER COMPOSITION")
    print("─" * 70)
    for cid in range(n_clusters):
        mask = hard_labels == cid
        cluster_cats = [cat_labels[i] for i in range(len(cat_labels)) if mask[i]]
        size = len(cluster_cats)
        top_cats = Counter(cluster_cats).most_common(5)
        purity = top_cats[0][1] / size if size > 0 else 0

        print(f"\n  Cluster {cid} — {size} docs (purity: {purity:.1%})")
        for cat, count in top_cats:
            bar = "█" * int(count / size * 30)
            print(f"    {cat:<35} {count:>4} ({count/size:>5.1%}) {bar}")

    print("\n" + "─" * 70)
    print("2. CATEGORY → CLUSTER CROSS-TABULATION (% of category in each cluster)")
    print("─" * 70)

    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    cross = np.zeros((len(categories), n_clusters), dtype=int)
    for i, (cat, cl) in enumerate(zip(cat_labels, hard_labels)):
        cross[cat_to_id[cat], int(cl)] += 1

    header = f"{'Category':<35}" + "".join(f"{c:>5}" for c in range(n_clusters))
    print(f"  {header}")
    print("  " + "─" * len(header))
    for cat_idx, cat in enumerate(categories):
        row_total = cross[cat_idx].sum()
        if row_total == 0:
            continue
        pcts = cross[cat_idx] / row_total * 100
        row_str = "".join(
            f"{p:>5.0f}" if p >= 1 else "    ·" for p in pcts
        )
        print(f"  {cat:<35}{row_str}")

    print("\n" + "─" * 70)
    print("3. FUZZY MEMBERSHIP ANALYSIS")
    print("─" * 70)

    THRESHOLD = 0.10
    memberships_per_doc = (probs > THRESHOLD).sum(axis=1)
    print(f"\n  Membership counts (prob > {THRESHOLD}):")
    for n_mem in range(1, min(6, n_clusters + 1)):
        count = (memberships_per_doc == n_mem).sum()
        print(f"    In {n_mem} cluster(s): {count} docs ({count/len(documents):.1%})")
    count_many = (memberships_per_doc >= 6).sum()
    if count_many > 0:
        print(f"    In 6+ clusters:  {count_many} docs ({count_many/len(documents):.1%})")

    print("\n" + "─" * 70)
    print("4. BOUNDARY DOCUMENTS (highest entropy — uncertain cluster membership)")
    print("─" * 70)

    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)

    top_idx = np.argsort(entropy)[-15:][::-1]
    for rank, idx in enumerate(top_idx, 1):
        doc = documents[idx]
        top3 = np.argsort(probs[idx])[-3:][::-1]
        top3_probs = probs[idx][top3]
        clusters_str = ", ".join(
            f"C{c}={p:.2f}" for c, p in zip(top3, top3_probs)
        )
        print(f"\n  #{rank}  Doc {doc.doc_id} [{doc.category}]  entropy={entropy[idx]:.3f}")
        print(f"       Clusters: {clusters_str}")
        print(f"       Text: {doc.text[:150]}…")

    print("\n" + "─" * 70)
    print("5. CLUSTER OVERLAP MATRIX (avg shared membership probability b/w cluster pairs)")
    print("─" * 70)
    print("  (showing pairs where avg shared prob > 0.05)\n")

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mask_i = hard_labels == i
            mask_j = hard_labels == j
            avg_j_in_i = probs[mask_i, j].mean() if mask_i.any() else 0
            avg_i_in_j = probs[mask_j, i].mean() if mask_j.any() else 0
            avg_overlap = (avg_j_in_i + avg_i_in_j) / 2
            if avg_overlap > 0.05:
                print(f"  C{i} ↔ C{j}: avg overlap = {avg_overlap:.3f}  "
                      f"(C{i}→C{j}: {avg_j_in_i:.3f}, C{j}→C{i}: {avg_i_in_j:.3f})")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
