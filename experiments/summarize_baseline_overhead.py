"""Consolidate per-dataset Task 2 baseline-overhead CSV files."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


EXCLUSIONS = {
    "patents": "Excluded from Task 2 until the remaining tasks are satisfied.",
    "reddit2": "Excluded from Task 2 until the remaining tasks are satisfied.",
}


def mean_std(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the consolidated Task 2 baseline-overhead table.")
    parser.add_argument("--results-dir", type=Path, default=Path("plots") / "random_set_results")
    parser.add_argument("--output", type=Path, default=Path("plots") / "random_set_results" / "task_2_baseline_overhead_summary")
    args = parser.parse_args()

    groups = defaultdict(list)
    for csv_path in sorted(args.results_dir.glob("baseline_overhead_*.csv")):
        dataset = csv_path.stem.removeprefix("baseline_overhead_")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["status"] == "ok":
                    groups[(dataset, row["method"])].append(row)

    if not groups:
        raise SystemExit("No baseline-overhead CSV files were found.")

    rows = []
    for (dataset, method), method_rows in sorted(groups.items()):
        setup_mean, setup_std = mean_std([float(row["setup_mean_seconds"]) for row in method_rows])
        overhead_mean, overhead_std = mean_std([float(row["additional_forward_mean_seconds"]) for row in method_rows])
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "status": "complete",
                "checkpoints": len(method_rows),
                "setup_mean_seconds": round(setup_mean, 6),
                "setup_std_seconds": round(setup_std, 6),
                "additional_forward_mean_seconds": round(overhead_mean, 6),
                "additional_forward_std_seconds": round(overhead_std, 6),
                "note": "",
            }
        )
    for dataset, reason in EXCLUSIONS.items():
        rows.append(
            {
                "dataset": dataset,
                "method": "all",
                "status": "excluded",
                "checkpoints": 0,
                "setup_mean_seconds": "",
                "setup_std_seconds": "",
                "additional_forward_mean_seconds": "",
                "additional_forward_std_seconds": "",
                "note": reason,
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["method"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Task 2: post-hoc baseline overhead",
        "",
        "Setup is one-time post-hoc work; additional forward time is detector time minus vanilla-GNN forward time.",
        "Values are mean ± sample standard deviation across the three fixed-seed vanilla checkpoints.",
        "",
        "| Dataset | Method | Status | Checkpoints | Setup mean ± std (s) | Additional forward mean ± std (s) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "excluded":
            lines.append(f"| {row['dataset']} | all | excluded | 0 | — | — |")
        else:
            lines.append(
                f"| {row['dataset']} | {row['method']} | complete | {row['checkpoints']} | "
                f"{row['setup_mean_seconds']:.3f} ± {row['setup_std_seconds']:.3f} | "
                f"{row['additional_forward_mean_seconds']:.3f} ± {row['additional_forward_std_seconds']:.3f} |"
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Task 2 summary to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
