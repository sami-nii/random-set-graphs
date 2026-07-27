"""Aggregate full focal-set budget construction timings from local W&B runs."""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


TIMING_PATTERN = re.compile(r"Focal set budget construction time:\s*([0-9.]+)\s*s")
DATASET_PATTERN = re.compile(r"args:.*?\n\s+- -d\s*\n\s+- ([^\r\n]+)", re.DOTALL)
MODEL_PATTERN = re.compile(r"args:.*?\n\s+- -m\s*\n\s+- ([^\r\n]+)", re.DOTALL)
FOCAL_SET_PATTERN = re.compile(r"Budgeted focal sets size:\s*(\d+)")
ID_CLASS_PATTERN = re.compile(r"Number of ID Classes:\s*(\d+)")
CONFIG_VALUE_PATTERN = r"^{key}:\s*$\n^\s+value:\s*(.+?)\s*$"
DEFAULT_EXCLUSIONS = {}


def config_value(config_text: str, key: str, default):
    match = re.search(CONFIG_VALUE_PATTERN.format(key=re.escape(key)), config_text, re.MULTILINE)
    if not match:
        return default
    value = match.group(1).strip().strip('"')
    try:
        return int(value)
    except ValueError:
        return value


def write_outputs(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Task 1: full focal-set budget construction time",
        "",
        "Times include sampling, auxiliary embeddings, GMM fitting, ellipsoid construction, and overlap selection.",
        "",
        "| Dataset | Strategy | Status | K | Repetitions | Focal sets | Mean (s) | Sample std (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mean = "—" if row["mean_seconds"] == "" else f"{row['mean_seconds']:.3f}"
        std = "—" if row["std_seconds"] == "" else f"{row['std_seconds']:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['focal_strategy']} | {row['status']} | {row['budget_k']} | {row['repetitions']} | {row['num_focal_sets']} | "
            f"{mean} | {std} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize focal-set budget timings from local W&B logs.")
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--output", type=Path, default=Path("plots") / "random_set_results" / "task_1_focal_set_budget_summary")
    args = parser.parse_args()

    groups = defaultdict(list)
    observed_runs = {}
    for run_dir in sorted(args.wandb_dir.glob("run-*")):
        config_path = run_dir / "files" / "config.yaml"
        output_path = run_dir / "files" / "output.log"
        if not config_path.exists() or not output_path.exists():
            continue
        config_text = config_path.read_text(encoding="utf-8", errors="ignore")
        output_text = output_path.read_text(encoding="utf-8", errors="ignore")
        dataset_match = DATASET_PATTERN.search(config_text)
        model_match = MODEL_PATTERN.search(config_text)
        if not dataset_match or not model_match or model_match.group(1).strip() != "random_set":
            continue
        dataset = dataset_match.group(1).strip()
        if "Budgeted focal sets size:" in output_text:
            focal_strategy = "budgeted"
        elif "Focal Sets Strategy: Full Power Set" in output_text:
            focal_strategy = "full_power_set"
        elif "Focal Sets Strategy: Singletons Only" in output_text:
            focal_strategy = "singletons_only"
        else:
            focal_strategy = "incomplete_or_unknown"
        metadata = observed_runs.setdefault(
            dataset,
            {"runs": 0, "focal_strategy": focal_strategy, "num_id_classes": ""},
        )
        metadata["runs"] += 1
        metadata["focal_strategy"] = focal_strategy
        id_class_match = ID_CLASS_PATTERN.search(output_text)
        if id_class_match:
            metadata["num_id_classes"] = int(id_class_match.group(1))
        timings = [float(match.group(1)) for match in TIMING_PATTERN.finditer(output_text)]
        if not timings:
            continue
        focal_matches = FOCAL_SET_PATTERN.findall(output_text)
        focal_sets = int(focal_matches[-1]) if focal_matches else None
        key = (dataset, config_value(config_text, "budget_k", 32), focal_sets)
        groups[key].extend(timings)

    if not observed_runs:
        raise SystemExit("No local random-set runs were found.")

    rows = []
    for (dataset, budget_k, focal_sets), timings in sorted(groups.items()):
        exclusion_reason = DEFAULT_EXCLUSIONS.get(dataset, "")
        rows.append(
            {
                "dataset": dataset,
                "focal_strategy": "budgeted",
                "status": "excluded" if exclusion_reason else "complete",
                "exclusion_reason": exclusion_reason,
                "budget_k": budget_k,
                "repetitions": len(timings),
                "num_focal_sets": focal_sets if focal_sets is not None else "",
                "mean_seconds": round(statistics.fmean(timings), 6),
                "std_seconds": round(statistics.stdev(timings) if len(timings) > 1 else 0.0, 6),
            }
        )
    timed_datasets = {dataset for dataset, _, _ in groups}
    for dataset, metadata in sorted(observed_runs.items()):
        if dataset in timed_datasets:
            continue
        if metadata["focal_strategy"] == "full_power_set":
            reason = f"Not applicable: {metadata['num_id_classes']} ID classes use the full power set."
            status = "not_applicable"
        else:
            reason = "No completed budget-construction timing was found."
            status = "incomplete"
        rows.append(
            {
                "dataset": dataset,
                "focal_strategy": metadata["focal_strategy"],
                "status": status,
                "exclusion_reason": reason,
                "budget_k": "",
                "repetitions": metadata["runs"],
                "num_focal_sets": "",
                "mean_seconds": "",
                "std_seconds": "",
            }
        )
    rows.sort(key=lambda row: row["dataset"])
    write_outputs(rows, args.output)
    print(f"Saved Task 1 timing tables to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
