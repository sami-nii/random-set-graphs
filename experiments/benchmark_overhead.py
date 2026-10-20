"""Tab. runtime: additional per-forward time of RS-GNN and of post-hoc OOD detectors over the backbone.

For each of the top ``--num-checkpoints`` vanilla backbones of a dataset (train them first, e.g. with
``cgnn run -m vanilla -d <ds> --seeds 0 1 2``):

- post-hoc detectors (Energy, ODIN, KNN, Mahalanobis): ``ood_scores`` time on the full graph minus the
  backbone forward time; their one-off ``prepare`` time is reported as setup;
- RS-GNN: belief head (linear + sigmoid over ``|F|`` focal sets) minus a linear class head on the same
  ``N x hidden`` node embeddings (the backbone is shared and excluded), averaged over inner iterations.

Values are mean +- sample std across checkpoints (ms). Writes ``<output>.csv`` and ``<output>.md``.

    python experiments/benchmark_overhead.py --dataset cora --num-checkpoints 3
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from cgnn.checkpoints import select_backbones
from cgnn.config import resolve_config
from cgnn.data import get_dataset, load_dataset
from cgnn.hardware import pin_gpus

DETECTORS = {
    "energy": ("EnergyDetector", {}),
    "odin": ("ODINDetector", {"temperature": 1.0, "noise_magnitude": 0.0}),
    "knn": ("KNNDetector", {"k": 50}),
    "mahalanobis": ("MahalanobisDetector", {"noise_magnitude": 0.0}),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - start


def mean_std(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def num_focal_sets(cfg, num_classes: int) -> int:
    """Size of the RS-GNN focal family: power set below ``power_set_max_classes``, else C + K."""
    if num_classes < int(cfg.get("power_set_max_classes", 10)):
        return 2**num_classes - 1
    return num_classes + int(cfg.get("budget_k", 64))


def detector_overhead(method, checkpoint, data, backbone, device, repetitions):
    from cgnn.models import detectors

    cls_name, kwargs = DETECTORS[method]
    detector = getattr(detectors, cls_name)(backbone_ckpt_path=str(checkpoint), **kwargs).to(device).eval()
    setup = [timed(lambda: detector.prepare(data), device) for _ in range(repetitions)]
    base, extra = [], []
    for _ in range(repetitions):
        with torch.no_grad():
            base.append(timed(lambda: backbone(data), device))
        if getattr(detector, "needs_input_grad", False):
            extra.append(timed(lambda: detector.ood_scores(data), device))
        else:
            with torch.no_grad():
                extra.append(timed(lambda: detector.ood_scores(data), device))
    return mean_std(setup)[0], mean_std(extra)[0] - mean_std(base)[0]


def head_overhead(num_nodes, hidden, num_classes, focal_sets, device, repetitions, inner):
    z = torch.randn(num_nodes, hidden, device=device)
    linear = nn.Linear(hidden, num_classes).to(device).eval()
    belief = nn.Linear(hidden, focal_sets).to(device).eval()

    def call(head, activation):
        with torch.no_grad():
            for _ in range(inner):
                activation(head(z))

    call(linear, lambda t: t), call(belief, torch.sigmoid)  # warm-up
    std = [timed(lambda: call(linear, lambda t: t), device) / inner for _ in range(repetitions)]
    rs = [timed(lambda: call(belief, torch.sigmoid), device) / inner for _ in range(repetitions)]
    return mean_std(rs)[0] - mean_std(std)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-checkpoints", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=sorted(DETECTORS), default=sorted(DETECTORS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--inner-iterations", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=None, help="default: outputs/tables/overhead_<dataset>"
    )
    args = parser.parse_args()

    if not args.cpu:
        pin_gpus(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    spec = get_dataset(args.dataset)
    cfg = resolve_config("random_set", spec.default_config())
    bundle = load_dataset(args.dataset, cfg)
    data = bundle.data.to(device)
    records = select_backbones(args.dataset, count=args.num_checkpoints)
    C, F = spec.num_id_classes, num_focal_sets(cfg, spec.num_id_classes)

    from cgnn.models.detectors.base import load_frozen_backbone

    per_ckpt = []
    for record in records:
        backbone = load_frozen_backbone(str(record.path)).to(device)
        row = {"checkpoint": record.path.name}
        for method in args.methods:
            row[f"{method}_setup_s"], row[f"{method}_ms"] = detector_overhead(
                method, record.path, data, backbone, device, args.repetitions
            )
            row[f"{method}_ms"] *= 1000
        hidden = int(backbone.hparams["hidden_channels"])
        row["random_set_ms"] = 1000 * head_overhead(
            data.num_nodes, hidden, C, F, device, args.repetitions, args.inner_iterations
        )
        per_ckpt.append(row)
        print(row)

    output = args.output or Path("outputs") / "tables" / f"overhead_{args.dataset}"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["random_set", *args.methods]
    summary = {c: mean_std([r[f"{c}_ms"] for r in per_ckpt]) for c in columns}
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_ckpt[0]))
        writer.writeheader()
        writer.writerows(per_ckpt)
    lines = [
        f"# Additional forward time over the backbone: {spec.paper_name}",
        "",
        f"{len(per_ckpt)} checkpoints, {data.num_nodes} nodes, {C} ID classes, {F} focal sets, device {device}.",
        "",
        "| Dataset | " + " | ".join(f"{c} (ms)" for c in columns) + " |",
        "|---|" + "---:|" * len(columns),
        f"| {spec.paper_name} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in summary.values()) + " |",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
