"""Benchmark post-hoc OOD baseline overhead relative to a frozen vanilla GNN."""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_loader.dataset_loader import dataset_loader
from models.VanillaGNN import VanillaGNN
from utils.model_manager import find_best_checkpoints


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


def timed_call(callback, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    callback()
    synchronize(device)
    return time.perf_counter() - start


def mean_std(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def detector_for(method: str, checkpoint: str, args):
    if method == "energy":
        from models.energy_detector import EnergyDetector

        return EnergyDetector(checkpoint)
    if method == "knn":
        from models.knn_detector import KNNDetector

        return KNNDetector(checkpoint, k=args.knn_k)
    if method == "mahalanobis":
        from models.mahalanobis_Detector import MahalanobisDetector

        return MahalanobisDetector(checkpoint, noise_magnitude=args.mahalanobis_noise)
    if method == "odin":
        from models.ODIN_Detector import ODINDetector

        return ODINDetector(checkpoint, temperature=args.odin_temperature, noise_magnitude=args.odin_noise)
    raise ValueError(f"Unsupported method: {method}")


def prepare_detector(method: str, detector, train_data, device: torch.device) -> None:
    if method in {"knn", "mahalanobis"}:
        detector.precompute_statistics(train_data)


def benchmark_method(method: str, checkpoint: str, baseline, train_data, eval_data, device: torch.device, args):
    detector = detector_for(method, checkpoint, args).to(device)
    detector.eval()

    setup_times = []
    if method in {"knn", "mahalanobis"}:
        for _ in range(args.repetitions):
            setup_times.append(timed_call(lambda: prepare_detector(method, detector, train_data, device), device))
    else:
        setup_times = [0.0] * args.repetitions

    baseline_times = []
    detector_times = []
    for _ in range(args.repetitions):
        baseline_times.append(timed_call(lambda: baseline(eval_data), device))
        if method == "odin":
            # ODIN deliberately performs a backward pass on the input.
            detector_times.append(timed_call(lambda: detector(eval_data), device))
        else:
            with torch.no_grad():
                detector_times.append(timed_call(lambda: detector(eval_data), device))

    setup_mean, setup_std = mean_std(setup_times)
    baseline_mean, baseline_std = mean_std(baseline_times)
    detector_mean, detector_std = mean_std(detector_times)
    return {
        "method": method,
        "status": "ok",
        "setup_mean_seconds": setup_mean,
        "setup_std_seconds": setup_std,
        "vanilla_forward_mean_seconds": baseline_mean,
        "vanilla_forward_std_seconds": baseline_std,
        "detector_forward_mean_seconds": detector_mean,
        "detector_forward_std_seconds": detector_std,
        "additional_forward_mean_seconds": detector_mean - baseline_mean,
        "additional_forward_std_seconds": (detector_std**2 + baseline_std**2) ** 0.5,
        "error": "",
    }


def write_results(rows, output: Path, dataset: str, checkpoint: str, repetitions: int, device: torch.device) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Post-hoc baseline overhead",
        "",
        f"Dataset: `{dataset}`  ",
        f"Checkpoints: {checkpoint}  ",
        f"Device: `{device}`; repetitions: {repetitions}.",
        "",
        "Setup is one-time post-hoc work. Additional forward time is detector forward time minus a vanilla GNN forward time on the same graph.",
        "",
        "| Method | Checkpoints | Setup mean ± sample std (s) | Vanilla forward mean ± sample std (s) | Detector forward mean ± sample std (s) | Additional forward time (s) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    for method, method_rows in sorted(grouped.items()):
        successful = [row for row in method_rows if row["status"] == "ok"]
        if not successful:
            errors = "; ".join(row["error"] for row in method_rows)
            lines.append(f"| {method} | 0 | — | — | — | — | {errors} |")
            continue
        def aggregate(key):
            values = [row[key] for row in successful]
            return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0
        setup_mean, setup_std = aggregate("setup_mean_seconds")
        vanilla_mean, vanilla_std = aggregate("vanilla_forward_mean_seconds")
        detector_mean, detector_std = aggregate("detector_forward_mean_seconds")
        overhead_mean, overhead_std = aggregate("additional_forward_mean_seconds")
        lines.append(
            f"| {method} | {len(successful)} | {setup_mean:.3f} ± {setup_std:.3f} | "
            f"{vanilla_mean:.3f} ± {vanilla_std:.3f} | {detector_mean:.3f} ± {detector_std:.3f} | "
            f"{overhead_mean:.3f} ± {overhead_std:.3f} | ok |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure post-hoc OOD baseline overhead against a vanilla GNN.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=None, help="VanillaGNN checkpoint. Repeat to provide multiple checkpoints.")
    parser.add_argument("--num-checkpoints", type=int, default=1, help="Number of top local checkpoints to use when --checkpoint is omitted.")
    parser.add_argument("--methods", nargs="+", choices=["energy", "knn", "mahalanobis", "odin"], default=["energy", "knn", "mahalanobis", "odin"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--mahalanobis-noise", type=float, default=0.0)
    parser.add_argument("--odin-temperature", type=float, default=1.0)
    parser.add_argument("--odin-noise", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("plots") / "random_set_results" / "baseline_overhead")
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("Use at least two repetitions to report sample standard deviation.")

    if args.num_checkpoints < 1:
        raise ValueError("--num-checkpoints must be positive.")
    checkpoints = [str(path) for path in args.checkpoint] if args.checkpoint else find_best_checkpoints(args.dataset, args.num_checkpoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "num_neighbors": args.num_neighbors,
        "num_workers": args.num_workers,
    }
    train_loader, _, test_loader = dataset_loader(args.dataset, config)
    train_data = get_base_data(train_loader).to(device)
    eval_data = get_base_data(test_loader).to(device)

    rows = []
    for checkpoint in checkpoints:
        baseline = VanillaGNN.load_from_checkpoint(checkpoint).to(device)
        baseline.eval()
        with torch.no_grad():
            baseline(eval_data)  # warm-up outside the timed measurements
        for method in args.methods:
            try:
                row = benchmark_method(method, checkpoint, baseline, train_data, eval_data, device, args)
            except Exception as exc:  # Preserve the other baseline results if one dependency is unavailable.
                row = {
                    "method": method,
                    "status": "unavailable",
                    "setup_mean_seconds": "",
                    "setup_std_seconds": "",
                    "vanilla_forward_mean_seconds": "",
                    "vanilla_forward_std_seconds": "",
                    "detector_forward_mean_seconds": "",
                    "detector_forward_std_seconds": "",
                    "additional_forward_mean_seconds": "",
                    "additional_forward_std_seconds": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            row["checkpoint"] = Path(checkpoint).name
            rows.append(row)

    write_results(rows, args.output, args.dataset, ", ".join(Path(path).name for path in checkpoints), args.repetitions, device)
    print(f"Saved baseline-overhead tables to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
