#!/usr/bin/env python3
"""Aggregate student best_acc1 from registry.csv into summary.csv.

Levels:
  overall: across all done students
  expert : grouped by expert_id (one row per expert)
  initial: grouped by initial_id (one row per initial, with parent expert_id)
  recover: grouped by recover_id (one row per recover, with parent expert_id)

Usage:
  python tools/summarize.py experiments/base
"""
import csv
import statistics
import sys
from pathlib import Path


def stats(vals):
    n = len(vals)
    if n == 0:
        return (0, "", "", "", "")
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    return (n, f"{mean:.4f}", f"{std:.4f}", f"{min(vals):.4f}", f"{max(vals):.4f}")


def main(exp_dir):
    exp_dir = Path(exp_dir)
    registry = exp_dir / "registry.csv"
    if not registry.exists():
        print(f"[summarize] no registry at {registry}", file=sys.stderr)
        sys.exit(1)

    students = []
    with registry.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stage"] != "student" or row["status"] != "done":
                continue
            try:
                acc = float(row["best_acc1"])
            except (ValueError, KeyError):
                continue
            students.append({
                "expert_id": row["expert_id"],
                "initial_id": row["initial_id"],
                "recover_id": row["recover_id"],
                "acc": acc,
            })

    if not students:
        print("[summarize] no done students with best_acc1", file=sys.stderr)
        sys.exit(0)

    by_expert = {}
    by_initial = {}
    by_recover = {}
    initial_parent = {}
    recover_parent = {}
    for s in students:
        by_expert.setdefault(s["expert_id"], []).append(s["acc"])
        by_initial.setdefault(s["initial_id"], []).append(s["acc"])
        by_recover.setdefault(s["recover_id"], []).append(s["acc"])
        initial_parent[s["initial_id"]] = s["expert_id"]
        recover_parent[s["recover_id"]] = s["expert_id"]

    out = exp_dir / "summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "group_id", "parent_expert_id",
                    "n", "mean_acc", "std_acc", "min_acc", "max_acc"])
        all_acc = [s["acc"] for s in students]
        w.writerow(["overall", "", "", *stats(all_acc)])
        for eid in sorted(by_expert):
            w.writerow(["expert", eid, "", *stats(by_expert[eid])])
        for iid in sorted(by_initial):
            w.writerow(["initial", iid, initial_parent[iid],
                        *stats(by_initial[iid])])
        for rid in sorted(by_recover):
            w.writerow(["recover", rid, recover_parent[rid],
                        *stats(by_recover[rid])])

    print(f"[summarize] wrote {out} "
          f"(overall n={len(all_acc)}, "
          f"experts={len(by_expert)}, initials={len(by_initial)}, "
          f"recovers={len(by_recover)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
