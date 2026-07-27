"""Create a compact before/after table for RS-GNN isotonic-calibration runs."""

import argparse
import csv
from pathlib import Path


METRICS = (
    ("test ID ECE", "isotonic_test_id_ece_before", "isotonic_test_id_ece_after"),
    ("test entropy AUROC", "isotonic_test_auroc_entropy_before", "isotonic_test_auroc_entropy_after"),
)


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else None


def display(mean, std, count):
    if mean is None:
        return "—"
    if count is None or count <= 1 or std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("plots") / "random_set_results" / "aggregate_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / "random_set_results" / "isotonic_calibration_summary",
    )
    parser.add_argument("--dataset", help="Optional dataset name to include in the table.")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("isotonic_calibration") == "enabled"]
    if args.dataset:
        rows = [row for row in rows if row.get("dataset") == args.dataset]
    if not rows:
        raise SystemExit("No completed isotonic-calibration runs were found.")

    output_rows = []
    for row in sorted(rows, key=lambda item: item["dataset"]):
        output = {"dataset": row["dataset"], "runs": row["runs"]}
        for _, before_key, after_key in METRICS:
            before_mean = number(row, f"{before_key}_mean")
            after_mean = number(row, f"{after_key}_mean")
            output[f"{before_key}_mean"] = before_mean
            output[f"{before_key}_std"] = number(row, f"{before_key}_std")
            output[f"{after_key}_mean"] = after_mean
            output[f"{after_key}_std"] = number(row, f"{after_key}_std")
            output[f"{after_key}_delta"] = (
                after_mean - before_mean if before_mean is not None and after_mean is not None else None
            )
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# RS-GNN with post-hoc isotonic calibration",
        "",
        "Each value is mean ± sample standard deviation across runs. The calibrator is fitted on labelled ID validation nodes only.",
        "",
        "| Dataset | Runs | Test ID ECE: before → after | Δ ECE | Test entropy AUROC: before → after | Δ AUROC |",
        "|---|---:|---|---:|---|---:|",
    ]
    for row in output_rows:
        ece_before = row["isotonic_test_id_ece_before_mean"]
        ece_after = row["isotonic_test_id_ece_after_mean"]
        auroc_before = row["isotonic_test_auroc_entropy_before_mean"]
        auroc_after = row["isotonic_test_auroc_entropy_after_mean"]
        lines.append(
            "| {dataset} | {runs} | {ece_before} → {ece_after} | {ece_delta:+.4f} | "
            "{auroc_before} → {auroc_after} | {auroc_delta:+.4f} |".format(
                dataset=row["dataset"],
                runs=row["runs"],
                ece_before=display(ece_before, row["isotonic_test_id_ece_before_std"], int(row["runs"])),
                ece_after=display(ece_after, row["isotonic_test_id_ece_after_std"], int(row["runs"])),
                ece_delta=row["isotonic_test_id_ece_after_delta"],
                auroc_before=display(auroc_before, row["isotonic_test_auroc_entropy_before_std"], int(row["runs"])),
                auroc_after=display(auroc_after, row["isotonic_test_auroc_entropy_after_std"], int(row["runs"])),
                auroc_delta=row["isotonic_test_auroc_entropy_after_delta"],
            )
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved isotonic-calibration summary to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
