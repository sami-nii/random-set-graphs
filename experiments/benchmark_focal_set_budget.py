"""Tab. focal_budget: focal-set construction time as a function of the budget K.

The K-independent stages (node sampling, warm-up GNN, embeddings, t-SNE, Gaussian fits and
ellipsoid regions) run once; the overlap-scoring + top-K selection stage is then timed
``--repetitions`` times for every K. "Estimated total" = setup + mean selection time.
Writes ``<output>.csv`` and ``<output>.md``. Datasets need >= 10 ID classes (budgeted path).

    python experiments/benchmark_focal_set_budget.py --dataset roman_empire --k-values 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import lightning as L
import torch

from cgnn.config import parse_overrides, resolve_config
from cgnn.data import get_dataset, load_dataset
from cgnn.hardware import pin_gpus
from cgnn.methods.random_set import budget_device, budget_embeddings
from cgnn.models.focal_sets import class_regions, overlap_scores, top_k_focal_sets, tsne_3d
from cgnn.paths import Paths


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="RS-GNN config overrides")
    parser.add_argument(
        "--output", type=Path, default=None, help="default: outputs/tables/focal_budget_<dataset>"
    )
    args = parser.parse_args()

    if not args.cpu:
        pin_gpus(1)
    spec = get_dataset(args.dataset)
    overrides = parse_overrides(args.set)
    if args.cpu:
        overrides["accelerator"] = "cpu"
    cfg = resolve_config("random_set", spec.default_config(), overrides)
    if spec.num_id_classes < int(cfg["power_set_max_classes"]):
        raise SystemExit(f"{args.dataset} has {spec.num_id_classes} ID classes: it uses the full power set.")
    L.seed_everything(int(cfg["seed"]), workers=True)
    device = budget_device(cfg)
    paths = Paths.from_env()
    bundle = load_dataset(args.dataset, cfg, paths=paths)
    C = spec.num_id_classes

    synchronize(device)
    start = time.perf_counter()
    embeddings, labels = budget_embeddings(cfg, bundle, paths.output_dir)
    regions = class_regions(tsne_3d(embeddings, int(cfg["seed"])), labels, C, device=device)
    synchronize(device)
    setup = time.perf_counter() - start

    rows = []
    for k in args.k_values:
        times, focal_sets = [], None
        for _ in range(args.repetitions):
            synchronize(device)
            start = time.perf_counter()
            scores = overlap_scores(regions, C, int(cfg["budget_max_cardinality"]))
            focal_sets = top_k_focal_sets(scores, C, k)
            synchronize(device)
            times.append(time.perf_counter() - start)
        mean, std = statistics.fmean(times), statistics.stdev(times) if len(times) > 1 else 0.0
        rows.append(
            {
                "dataset": args.dataset,
                "K": k,
                "focal_sets": len(focal_sets),
                "setup_s": setup,
                "selection_mean_s": mean,
                "selection_std_s": std,
                "total_s": setup + mean,
            }
        )
        print(rows[-1])

    output = args.output or Path("outputs") / "tables" / f"focal_budget_{args.dataset}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Focal-set construction time vs K: {spec.paper_name}",
        "",
        "| Dataset | K | # Focal Sets | Setup (s) | Selection (s) | Estimated Total Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {spec.paper_name} | {r['K']} | {r['focal_sets']} | {r['setup_s']:.3f} | "
            f"{r['selection_mean_s']:.3f} ± {r['selection_std_s']:.3f} | {r['total_s']:.3f} |"
            for r in rows
        ),
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
