"""Measure RandomSetLayer inference overhead against a standard linear class head."""

import argparse
import csv
import itertools
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_loader.dataset_loader import dataset_loader
from models.VanillaGNN import VanillaGNN
from models.random_set_gnn import RandomSetLayer
from utils.model_manager import find_best_checkpoints


DATASET_PATTERN = re.compile(r"args:.*?\n\s+- -d\s*\n\s+- ([^\r\n]+)", re.DOTALL)
FOCAL_SET_PATTERN = re.compile(r"Budgeted focal sets size:\s*(\d+)")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_base_data(loader):
    if hasattr(loader, "data") and hasattr(loader.data, "x"):
        return loader.data
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and len(dataset) == 1:
        return dataset[0]
    return next(iter(loader))


def num_focal_sets(dataset: str, num_classes: int, wandb_dir: Path) -> int:
    """Use the observed budget size for large-class datasets; enumerate small ones."""
    if num_classes < 10:
        return sum(1 for size in range(1, num_classes + 1) for _ in itertools.combinations(range(num_classes), size))
    for run_dir in sorted(wandb_dir.glob("run-*"), reverse=True):
        config_path = run_dir / "files" / "config.yaml"
        output_path = run_dir / "files" / "output.log"
        if not config_path.exists() or not output_path.exists():
            continue
        dataset_match = DATASET_PATTERN.search(config_path.read_text(encoding="utf-8", errors="ignore"))
        if not dataset_match or dataset_match.group(1).strip() != dataset:
            continue
        matches = FOCAL_SET_PATTERN.findall(output_path.read_text(encoding="utf-8", errors="ignore"))
        if matches:
            return int(matches[-1])
    raise FileNotFoundError(f"No recorded budgeted focal-set size found for {dataset}.")


def average_call_time(layer, embeddings, device: torch.device, inner_iterations: int) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(inner_iterations):
            layer(embeddings)
    synchronize(device)
    return (time.perf_counter() - start) / inner_iterations


def summarize(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark layer-only random-set inference overhead.")
    parser.add_argument("--results-dir", type=Path, default=Path("plots") / "random_set_results", help="Uses Task 2 baseline CSVs to select datasets.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional explicit dataset list.")
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--inner-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("plots") / "random_set_results" / "task_2_random_set_layer_overhead")
    args = parser.parse_args()
    if args.repetitions < 2 or args.inner_iterations < 1:
        raise ValueError("Use at least two repetitions and one inner iteration.")

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(path.stem.removeprefix("baseline_overhead_") for path in args.results_dir.glob("baseline_overhead_*.csv"))
    if not datasets:
        raise SystemExit("No Task 2 baseline CSVs were found to select datasets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_rows = []
    aggregate_rows = []
    for dataset in datasets:
        config = {"batch_size": -1, "num_layers": 2, "num_neighbors": 10, "num_workers": 0}
        train_loader, _, _ = dataset_loader(dataset, config)
        data = get_base_data(train_loader)
        if data.y.dim() > 1 and data.y.size(1) > 1:
            classes = data.y.size(1)
        else:
            classes = int(data.y[data.train_mask].max().item()) + 1
        focal_sets = num_focal_sets(dataset, classes, args.wandb_dir)
        checkpoints = find_best_checkpoints(dataset, 3)

        dataset_rows = []
        for checkpoint in checkpoints:
            vanilla = VanillaGNN.load_from_checkpoint(checkpoint)
            hidden_channels = int(vanilla.hparams.hidden_channels)
            embeddings = torch.randn((data.num_nodes, hidden_channels), device=device)
            standard_head = nn.Linear(hidden_channels, classes).to(device).eval()
            random_set_layer = RandomSetLayer(hidden_channels, focal_sets).to(device).eval()
            with torch.no_grad():
                standard_head(embeddings)
                random_set_layer(embeddings)
            standard_times = [average_call_time(standard_head, embeddings, device, args.inner_iterations) for _ in range(args.repetitions)]
            random_set_times = [average_call_time(random_set_layer, embeddings, device, args.inner_iterations) for _ in range(args.repetitions)]
            standard_mean, standard_std = summarize(standard_times)
            random_set_mean, random_set_std = summarize(random_set_times)
            row = {
                "dataset": dataset,
                "checkpoint": Path(checkpoint).name,
                "num_nodes": data.num_nodes,
                "hidden_channels": hidden_channels,
                "num_classes": classes,
                "num_focal_sets": focal_sets,
                "standard_head_mean_ms": standard_mean * 1000,
                "standard_head_std_ms": standard_std * 1000,
                "random_set_layer_mean_ms": random_set_mean * 1000,
                "random_set_layer_std_ms": random_set_std * 1000,
                "additional_random_set_layer_mean_ms": (random_set_mean - standard_mean) * 1000,
            }
            raw_rows.append(row)
            dataset_rows.append(row)

        def aggregate(key):
            return summarize([row[key] for row in dataset_rows])

        standard_mean, standard_std = aggregate("standard_head_mean_ms")
        random_set_mean, random_set_std = aggregate("random_set_layer_mean_ms")
        overhead_mean, overhead_std = aggregate("additional_random_set_layer_mean_ms")
        aggregate_rows.append(
            {
                "dataset": dataset,
                "checkpoints": len(dataset_rows),
                "num_nodes": data.num_nodes,
                "num_classes": classes,
                "num_focal_sets": focal_sets,
                "standard_head_mean_ms": standard_mean,
                "standard_head_std_ms": standard_std,
                "random_set_layer_mean_ms": random_set_mean,
                "random_set_layer_std_ms": random_set_std,
                "additional_random_set_layer_mean_ms": overhead_mean,
                "additional_random_set_layer_std_ms": overhead_std,
            }
        )
        print(f"Measured RandomSetLayer overhead for {dataset}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with args.output.with_name(args.output.name + "_per_checkpoint").with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    lines = [
        "# Task 2: RS-GNN random-set layer inference overhead",
        "",
        "The shared GNN backbone is excluded. The table compares a standard linear class head with RandomSetLayer on the same node embeddings.",
        "Values are mean ± sample standard deviation across three vanilla-checkpoint hidden dimensions; each checkpoint uses repeated layer calls.",
        "",
        "| Dataset | Checkpoints | Nodes | Classes | Focal sets | Standard head (ms) | Random-set layer (ms) | Additional RS layer (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {row['checkpoints']} | {row['num_nodes']} | {row['num_classes']} | {row['num_focal_sets']} | "
            f"{row['standard_head_mean_ms']:.3f} ± {row['standard_head_std_ms']:.3f} | "
            f"{row['random_set_layer_mean_ms']:.3f} ± {row['random_set_layer_std_ms']:.3f} | "
            f"{row['additional_random_set_layer_mean_ms']:.3f} ± {row['additional_random_set_layer_std_ms']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved RS layer overhead tables to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
