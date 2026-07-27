"""Benchmark the one-time, budgeted focal-set construction for several K values.

The expensive embedding, GMM, and ellipsoid stages are prepared once.  Each K
then re-runs the overlap-selection stage on those fixed inputs, isolating the
part of the focal-set budget computation affected by K.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
from torch_geometric.utils import subgraph

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_loader.dataset_loader import dataset_loader
from models.random_set_gnn import RandomSetGNN
from trainers.budgeting import ellipse, fit_gmm, overlaps, train_embeddings


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def get_base_data(loader):
    if hasattr(loader, "data") and hasattr(loader.data, "x") and hasattr(loader.data, "y"):
        return loader.data
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and len(dataset) == 1:
        return dataset[0]
    return next(iter(loader))


def prepare_budget_inputs(args, device: str):
    """Compute the K-independent part of the focal-set budget exactly once."""
    config = {
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "num_neighbors": args.num_neighbors,
        "num_workers": args.num_workers,
    }
    train_loader, _, _ = dataset_loader(args.dataset, config)
    data = get_base_data(train_loader)

    if data.y.dim() > 1 and data.y.size(1) > 1:
        num_classes = data.y.shape[1]
        is_id = data.y.sum(dim=1) == 1
        y_indices = data.y.argmax(dim=-1)
    else:
        train_mask = data.train_mask.bool()
        num_classes = int(data.y[train_mask].max().item()) + 1
        is_id = torch.ones_like(data.y, dtype=torch.bool)
        y_indices = data.y

    if num_classes < 10:
        raise ValueError(
            f"{args.dataset} has {num_classes} ID classes and uses the full power set, "
            "not the budgeted focal-set path."
        )

    aux_model = RandomSetGNN(
        gnn_type=args.gnn_type,
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        focal_sets=[{class_idx} for class_idx in range(num_classes)],
        num_classes=num_classes,
    ).gnn_model

    budget_mask = data.train_mask.bool() & is_id
    sampled_indices = []
    for class_idx in range(num_classes):
        class_nodes = torch.nonzero(budget_mask & (y_indices == class_idx), as_tuple=False).view(-1)
        if class_nodes.numel() == 0:
            raise ValueError(f"No training samples found for class {class_idx}.")
        if class_nodes.numel() > args.samples_per_class:
            class_nodes = class_nodes[: args.samples_per_class]
        sampled_indices.append(class_nodes)

    sampled_indices = torch.cat(sampled_indices).cpu()
    sampled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    sampled_mask[sampled_indices] = True
    sampled_edge_index, _ = subgraph(sampled_mask, data.edge_index, relabel_nodes=True)

    sampled_x = data.x[sampled_mask].to(device)
    sampled_y = y_indices[sampled_mask]
    embedded = train_embeddings(
        aux_model,
        sampled_x,
        batch_size=args.batch_size,
        device=device,
        edge_index=sampled_edge_index.to(device),
    )
    classes = list(range(num_classes))
    gms = fit_gmm(classes, embedded, sampled_y)
    regions, means, max_len = ellipse(gms, num_classes, device=device)
    return classes, regions, means, max_len


def write_results(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Focal-set budget K ablation",
        "",
        "Setup time is the shared, one-time embedding/GMM/ellipsoid preparation. "
        "Overlap time is measured independently for each K; total is their sum.",
        "",
        "| Dataset | K | Repetitions | Focal sets | Setup (s) | Overlap mean ± sample std (s) | Estimated total (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        overlap = f"{row['overlap_mean_seconds']:.3f} ± {row['overlap_std_seconds']:.3f}"
        markdown.append(
            f"| {row['dataset']} | {row['k']} | {row['repetitions']} | {row['num_focal_sets']} | "
            f"{row['setup_seconds']:.3f} | {overlap} | {row['estimated_total_seconds']:.3f} |"
        )
    output_path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark budgeted focal-set construction across K values.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--max-cardinality", type=int, default=3)
    parser.add_argument("--gnn-type", default="GAT")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("plots") / "random_set_results" / "focal_set_budget_k_ablation")
    args = parser.parse_args()

    if args.repetitions < 2:
        raise ValueError("Use at least two repetitions to report a sample standard deviation.")
    if any(k <= 0 for k in args.k_values):
        raise ValueError("All K values must be positive.")

    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synchronize(device)
    setup_start = time.perf_counter()
    classes, regions, means, max_len = prepare_budget_inputs(args, device)
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    print(f"Prepared shared focal-set budget inputs in {setup_seconds:.3f} s on {device}.")

    rows = []
    for k in sorted(set(args.k_values)):
        elapsed = []
        num_focal_sets = None
        for repetition in range(args.repetitions):
            synchronize(device)
            start = time.perf_counter()
            focal_sets = overlaps(
                k=k,
                classes=[str(class_idx) for class_idx in classes],
                num_clusters=len(classes),
                classes_dict={str(class_idx): class_idx for class_idx in classes},
                regions=regions,
                means=means,
                max_len=max_len,
                max_cardinality=args.max_cardinality,
            )
            synchronize(device)
            elapsed.append(time.perf_counter() - start)
            num_focal_sets = len(focal_sets)
            print(f"K={k}, repetition {repetition + 1}/{args.repetitions}: {elapsed[-1]:.3f} s")

        overlap_mean = statistics.fmean(elapsed)
        rows.append(
            {
                "dataset": args.dataset,
                "k": k,
                "repetitions": args.repetitions,
                "num_focal_sets": num_focal_sets,
                "setup_seconds": round(setup_seconds, 6),
                "overlap_mean_seconds": round(overlap_mean, 6),
                "overlap_std_seconds": round(statistics.stdev(elapsed), 6),
                "estimated_total_seconds": round(setup_seconds + overlap_mean, 6),
            }
        )

    write_results(rows, args.output)
    print(f"Saved K-ablation tables to {args.output.with_suffix('.md')} and .csv")


if __name__ == "__main__":
    main()
