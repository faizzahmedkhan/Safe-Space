"""Compare two training runs clip by clip, with confidence intervals.

Each notebook run saves its per-clip validation scores as a CSV
(clip_eval.write_rows_csv).  Given two such files this prints, for each run,
its metrics with 95% bootstrap intervals, then a PAIRED comparison: the
difference B - A on the same clips, its interval, and how often B is not ahead.

    python compare_runs.py val_scores_baseline.csv val_scores_seed43.csv \
        --manifest clean_split_manifest.csv --labels baseline seed43

--manifest (optional, recommended) supplies each clip's cluster_id so whole
clusters of near-duplicate clips are resampled together; without it the
intervals come out too narrow.  Paths must match the manifest's `path` column.

Reading the result: an interval that excludes 0 (and P(B not better) below
0.05) means B's lead is unlikely to be luck of the validation sample.  An
interval spanning 0 means the runs are not distinguishable on this data.
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, Optional

import clip_eval


def load_groups(manifest_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not manifest_path:
        return None
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if "cluster_id" not in (reader.fieldnames or []):
            raise SystemExit(f"{manifest_path} has no cluster_id column")
        return {row["path"]: row["cluster_id"] for row in reader}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Paired comparison of two runs' per-clip scores.")
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    p.add_argument("--manifest", help="clean_split_manifest.csv, for cluster-level resampling")
    p.add_argument("--labels", nargs=2, default=("A", "B"), metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows_a = clip_eval.read_rows_csv(args.csv_a)
    rows_b = clip_eval.read_rows_csv(args.csv_b)
    groups = load_groups(args.manifest)
    if groups is None:
        print("Note: no --manifest given; resampling single clips (intervals will be too narrow).\n")
    else:
        missing = sum(1 for r in rows_a if r["path"] not in groups)
        if missing:
            print(f"Note: {missing} clips are not in the manifest; each is treated as its own cluster.\n")

    for label, rows in ((args.labels[0], rows_a), (args.labels[1], rows_b)):
        ci = clip_eval.bootstrap_metrics(rows, groups=groups, n_boot=args.n_boot, seed=args.seed)
        print(f"== {label} ({len(rows)} clips)")
        print(clip_eval.format_intervals(ci))
        print()
    cmp = clip_eval.compare_runs(rows_a, rows_b, groups=groups, n_boot=args.n_boot, seed=args.seed)
    print(f"== {args.labels[1]} vs {args.labels[0]}")
    print(clip_eval.format_comparison(cmp, args.labels[0], args.labels[1]))


if __name__ == "__main__":
    main()
