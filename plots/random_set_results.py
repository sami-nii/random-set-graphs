import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_FILENAME = "wandb-summary.json"
CONFIG_FILENAME = "config.yaml"
OUTPUT_LOG_FILENAME = "output.log"


def _coerce_scalar(raw: str):
    value = raw.strip().strip('"').strip("'")
    lowered = value.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _extract_arg_from_config(text: str, arg_name: str) -> Optional[str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == f"- {arg_name}" and idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if next_line.startswith("- "):
                return next_line[2:].strip().strip('"').strip("'")
    return None


def _extract_value_from_config(text: str, key: str):
    pattern = re.compile(
        rf"^{re.escape(key)}:\s*$\n^\s+value:\s*(.+?)\s*$",
        flags=re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return None
    return _coerce_scalar(match.group(1))


def _extract_focal_metadata(text: str) -> Dict[str, Optional[object]]:
    metadata = {
        "num_id_classes": None,
        "focal_strategy": None,
        "num_output_heads": None,
    }

    class_match = re.search(r"Number of ID Classes:\s*(\d+)", text)
    if class_match:
        metadata["num_id_classes"] = int(class_match.group(1))

    if "Focal Sets Strategy: Full Power Set" in text:
        metadata["focal_strategy"] = "full_power_set"
    elif "Constructing budgeted focal sets" in text:
        metadata["focal_strategy"] = "budgeted"

    heads_match = re.search(r"Total Output Heads \(2\^N - 1\):\s*(\d+)", text)
    if heads_match:
        metadata["num_output_heads"] = int(heads_match.group(1))

    budgeted_match = re.search(r"Budgeted focal sets size:\s*(\d+)", text)
    if budgeted_match:
        metadata["num_output_heads"] = int(budgeted_match.group(1))

    return metadata


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _build_ablation_label(singletons_only: bool, loss_ablation: str) -> str:
    focal_label = "singletons_only" if singletons_only else "standard_focal_sets"
    return f"{focal_label}__{loss_ablation}"


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def load_random_set_runs(wandb_dir: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []

    for run_dir in sorted(wandb_dir.glob("run-*")):
        files_dir = run_dir / "files"
        config_path = files_dir / CONFIG_FILENAME
        summary_path = files_dir / SUMMARY_FILENAME
        output_log_path = files_dir / OUTPUT_LOG_FILENAME

        if not config_path.exists() or not summary_path.exists():
            continue

        config_text = config_path.read_text(encoding="utf-8", errors="ignore")
        model_name = _extract_arg_from_config(config_text, "--model")
        if model_name != "random_set":
            continue

        dataset_name = _extract_arg_from_config(config_text, "--dataset") or "unknown"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        focal_metadata = {}
        if output_log_path.exists():
            output_text = output_log_path.read_text(encoding="utf-8", errors="ignore")
            focal_metadata = _extract_focal_metadata(output_text)

        run = {
            "run_id": run_dir.name.split("-")[-1],
            "run_name": run_dir.name,
            "dataset": dataset_name,
            "model": model_name,
            "gnn_type": _extract_value_from_config(config_text, "gnn_type"),
            "hidden_channels": _extract_value_from_config(config_text, "hidden_channels"),
            "num_layers": _extract_value_from_config(config_text, "num_layers"),
            "lr": _extract_value_from_config(config_text, "lr"),
            "weight_decay": _extract_value_from_config(config_text, "weight_decay"),
            "batch_size": _extract_value_from_config(config_text, "batch_size"),
            "singletons_only": _normalize_bool(_extract_value_from_config(config_text, "singletons_only")),
            "loss_ablation": _extract_value_from_config(config_text, "loss_ablation") or "full",
            "use_bce_loss": _normalize_bool(_extract_value_from_config(config_text, "use_bce_loss")),
            "use_mr_loss": _normalize_bool(_extract_value_from_config(config_text, "use_mr_loss")),
            "use_ms_loss": _normalize_bool(_extract_value_from_config(config_text, "use_ms_loss")),
            "train_loss": _safe_float(summary.get("train_loss")),
            "train_bce": _safe_float(summary.get("train_bce")),
            "train_mr": _safe_float(summary.get("train_mr")),
            "train_ms": _safe_float(summary.get("train_ms")),
            "train_acc": _safe_float(summary.get("train_acc")),
            "val_loss": _safe_float(summary.get("val_loss")),
            "val_f1": _safe_float(summary.get("val_f1")),
            "val_auroc_entropy": _safe_float(summary.get("val_auroc_entropy")),
            "test_acc_id": _safe_float(summary.get("test_acc_id")),
            "test_auroc_entropy": _safe_float(summary.get("test_auroc_entropy")),
            "epoch": _safe_float(summary.get("epoch")),
        }
        run.update(focal_metadata)
        run["ablation_group"] = _build_ablation_label(
            bool(run["singletons_only"]),
            str(run["loss_ablation"]),
        )
        runs.append(run)

    return runs


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            text = "" if value is None else str(value)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_by_dataset(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    return grouped


def _group_by_ablation(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation_group"]), []).append(row)
    return grouped


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def plot_tradeoff_scatter(rows: List[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    grouped = _group_by_dataset(rows)
    cmap = plt.get_cmap("tab10")

    for idx, (dataset, dataset_rows) in enumerate(sorted(grouped.items())):
        xs = [row["test_acc_id"] for row in dataset_rows if row["test_acc_id"] is not None and row["test_auroc_entropy"] is not None]
        ys = [row["test_auroc_entropy"] for row in dataset_rows if row["test_acc_id"] is not None and row["test_auroc_entropy"] is not None]
        sizes = []
        valid_rows = [row for row in dataset_rows if row["test_acc_id"] is not None and row["test_auroc_entropy"] is not None]
        for row in valid_rows:
            heads = row.get("num_output_heads")
            sizes.append(50 if heads is None else max(40, min(240, float(heads) * 3)))

        if xs:
            ax.scatter(xs, ys, s=sizes, alpha=0.75, color=cmap(idx % 10), label=dataset, edgecolors="black", linewidths=0.4)

            best_row = max(valid_rows, key=lambda r: (r["test_auroc_entropy"], r["test_acc_id"]))
            ax.annotate(
                f"{dataset}:{best_row.get('gnn_type', 'NA')}",
                (best_row["test_acc_id"], best_row["test_auroc_entropy"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.set_title("Random-Set Tradeoff: ID Accuracy vs OOD AUROC")
    ax.set_xlabel("Test Accuracy on ID Nodes")
    ax.set_ylabel("Test AUROC from Entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ablation_tradeoff(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [row for row in rows if row.get("test_acc_id") is not None and row.get("test_auroc_entropy") is not None]
    if not valid_rows:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    grouped = _group_by_ablation(valid_rows)

    for idx, (ablation, ablation_rows) in enumerate(sorted(grouped.items())):
        xs = [row["test_acc_id"] for row in ablation_rows]
        ys = [row["test_auroc_entropy"] for row in ablation_rows]
        ax.scatter(
            xs,
            ys,
            alpha=0.7,
            color=cmap(idx % 10),
            label=ablation,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_title("Random-Set Ablation Tradeoff: ID Accuracy vs OOD AUROC")
    ax.set_xlabel("Test Accuracy on ID Nodes")
    ax.set_ylabel("Test AUROC from Entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ablation_distribution(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_ablation(rows)
    labels = []
    values = []

    for ablation, ablation_rows in sorted(grouped.items()):
        scores = [row["test_auroc_entropy"] for row in ablation_rows if row["test_auroc_entropy"] is not None]
        if scores:
            labels.append(ablation)
            values.append(scores)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels)
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.85)

    ax.set_ylabel("Test AUROC from Entropy")
    ax.set_title("Random-Set AUROC Distribution by Ablation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_best_per_ablation(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_ablation(rows)
    best_rows = []
    for ablation, ablation_rows in sorted(grouped.items()):
        valid_rows = [row for row in ablation_rows if row.get("test_auroc_entropy") is not None]
        if not valid_rows:
            continue
        best_rows.append(max(valid_rows, key=lambda r: (r["test_auroc_entropy"], r.get("test_acc_id") or -1)))

    if not best_rows:
        return

    labels = [row["ablation_group"] for row in best_rows]
    aurocs = [row["test_auroc_entropy"] for row in best_rows]
    accs = [row["test_acc_id"] or 0.0 for row in best_rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(best_rows)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], aurocs, width=width, label="Best Test AUROC", color="#355070")
    ax.bar([i + width / 2 for i in x], accs, width=width, label="Matching Test ID Accuracy", color="#e56b6f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Best Random-Set Result Per Ablation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_best_per_dataset(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_dataset(rows)
    best_rows = []

    for dataset, dataset_rows in sorted(grouped.items()):
        valid_rows = [row for row in dataset_rows if row["test_auroc_entropy"] is not None]
        if not valid_rows:
            continue
        best_rows.append(max(valid_rows, key=lambda r: (r["test_auroc_entropy"], r.get("test_acc_id") or -1)))

    if not best_rows:
        return

    labels = [f"{row['dataset']} ({row.get('gnn_type', 'NA')})" for row in best_rows]
    aurocs = [row["test_auroc_entropy"] for row in best_rows]
    accs = [row["test_acc_id"] or 0.0 for row in best_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(best_rows)))
    width = 0.38

    ax.bar([i - width / 2 for i in x], aurocs, width=width, label="Best Test AUROC", color="#2a6f97")
    ax.bar([i + width / 2 for i in x], accs, width=width, label="Matching Test ID Accuracy", color="#c97b63")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Best Random-Set Result Per Dataset")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dataset_distributions(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_dataset(rows)
    labels = []
    values = []

    for dataset, dataset_rows in sorted(grouped.items()):
        scores = [row["test_auroc_entropy"] for row in dataset_rows if row["test_auroc_entropy"] is not None]
        if scores:
            labels.append(dataset)
            values.append(scores)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels)
    palette = ["#6baed6", "#9ecae1", "#c6dbef", "#fd8d3c", "#fdae6b", "#fdd0a2"]
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.85)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test AUROC from Entropy")
    ax.set_title("Random-Set AUROC Distribution by Dataset")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_focal_set_size(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("num_output_heads") is not None and row.get("test_auroc_entropy") is not None
    ]
    if not valid_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _group_by_dataset(valid_rows)
    cmap = plt.get_cmap("Dark2")

    for idx, (dataset, dataset_rows) in enumerate(sorted(grouped.items())):
        xs = [row["num_output_heads"] for row in dataset_rows]
        ys = [row["test_auroc_entropy"] for row in dataset_rows]
        ax.scatter(xs, ys, color=cmap(idx % 8), alpha=0.8, label=dataset, edgecolors="black", linewidths=0.4)

    ax.set_xlabel("Number of Focal Sets / Output Heads")
    ax.set_ylabel("Test AUROC from Entropy")
    ax.set_title("Random-Set OOD Performance vs Focal-Set Budget")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_bars(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("test_auroc_entropy") is not None and row.get("test_acc_id") is not None
    ]
    if not valid_rows:
        return

    sorted_rows = sorted(
        valid_rows,
        key=lambda row: (row["test_auroc_entropy"], row["test_acc_id"]),
        reverse=True,
    )[:12]

    labels = [f"{row.get('gnn_type', 'NA')}-{row['run_id']}" for row in sorted_rows]
    aurocs = [row["test_auroc_entropy"] for row in sorted_rows]
    accs = [row["test_acc_id"] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(9, len(sorted_rows) * 0.8), 5))
    x = list(range(len(sorted_rows)))
    width = 0.38

    ax.bar([i - width / 2 for i in x], aurocs, width=width, label="Test AUROC", color="#33658a")
    ax.bar([i + width / 2 for i in x], accs, width=width, label="Test ID Accuracy", color="#f6ae2d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Top Random-Set Runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_learning_diagnostics(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("train_loss") is not None and row.get("val_loss") is not None and row.get("train_mr") is not None
    ]
    if not valid_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(
        [row["train_loss"] for row in valid_rows],
        [row["val_loss"] for row in valid_rows],
        color="#758e4f",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
    )
    axes[0].set_title("Train vs Validation Loss")
    axes[0].set_xlabel("Train Loss")
    axes[0].set_ylabel("Validation Loss")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        [row["train_mr"] for row in valid_rows],
        [row["test_auroc_entropy"] for row in valid_rows],
        color="#bc4b51",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
    )
    axes[1].set_title("Mass-Regularization vs OOD AUROC")
    axes[1].set_xlabel("Train MR")
    axes[1].set_ylabel("Test AUROC")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_top_runs_report(rows: List[Dict[str, object]], output_path: Path, top_k: int) -> None:
    sortable = [
        row
        for row in rows
        if row.get("test_auroc_entropy") is not None
    ]
    sortable.sort(key=lambda row: (row["test_auroc_entropy"], row.get("test_acc_id") or -1), reverse=True)

    lines = ["Top random-set runs by test_auroc_entropy", ""]
    for idx, row in enumerate(sortable[:top_k], start=1):
        lines.append(
            (
                f"{idx}. dataset={row['dataset']} run={row['run_id']} "
                f"gnn={row.get('gnn_type')} "
                f"test_auroc_entropy={row.get('test_auroc_entropy'):.4f} "
                f"test_acc_id={(row.get('test_acc_id') or 0.0):.4f} "
                f"focal_strategy={row.get('focal_strategy')} "
                f"num_output_heads={row.get('num_output_heads')}"
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_report(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_ablation(rows)
    lines = ["Random-set ablation summary", ""]
    for ablation, ablation_rows in sorted(grouped.items()):
        valid_rows = [row for row in ablation_rows if row.get("test_auroc_entropy") is not None]
        if not valid_rows:
            continue
        best_row = max(valid_rows, key=lambda r: (r["test_auroc_entropy"], r.get("test_acc_id") or -1))
        mean_auroc = sum(row["test_auroc_entropy"] for row in valid_rows) / len(valid_rows)
        mean_acc = sum((row.get("test_acc_id") or 0.0) for row in valid_rows) / len(valid_rows)
        lines.append(
            f"{ablation}: runs={len(ablation_rows)} valid={len(valid_rows)} "
            f"best_auroc={best_row['test_auroc_entropy']:.4f} "
            f"mean_auroc={mean_auroc:.4f} mean_acc={mean_acc:.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset_bundle(dataset: str, rows: List[Dict[str, object]], output_dir: Path, top_k: int) -> None:
    dataset_dir = output_dir / _slugify(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, dataset_dir / "runs.csv")
    write_top_runs_report(rows, dataset_dir / "top_runs.txt", top_k)
    plot_tradeoff_scatter(rows, dataset_dir / "tradeoff_scatter.png")
    plot_focal_set_size(rows, dataset_dir / "focal_set_size_vs_auroc.png")
    plot_metric_bars(rows, dataset_dir / "top_run_bars.png")
    plot_learning_diagnostics(rows, dataset_dir / "learning_diagnostics.png")


def generate_ablation_bundle(ablation: str, rows: List[Dict[str, object]], output_dir: Path, top_k: int) -> None:
    ablation_dir = output_dir / _slugify(ablation)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, ablation_dir / "runs.csv")
    write_top_runs_report(rows, ablation_dir / "top_runs.txt", top_k)
    plot_tradeoff_scatter(rows, ablation_dir / "tradeoff_scatter.png")
    plot_focal_set_size(rows, ablation_dir / "focal_set_size_vs_auroc.png")
    plot_metric_bars(rows, ablation_dir / "top_run_bars.png")
    plot_learning_diagnostics(rows, ablation_dir / "learning_diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline random-set result plots from local W&B runs.")
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"), help="Path to the local wandb directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "random_set_results", help="Directory for generated plots and tables.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top runs to include in the text summary.")
    args = parser.parse_args()

    rows = load_random_set_runs(args.wandb_dir)
    if args.dataset:
        rows = [row for row in rows if row["dataset"] == args.dataset]

    if not rows:
        raise SystemExit("No local random_set runs were found for the requested filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, args.output_dir / "random_set_runs.csv")
    write_top_runs_report(rows, args.output_dir / "top_runs.txt", args.top_k)
    write_ablation_report(rows, args.output_dir / "ablation_summary.txt")
    plot_tradeoff_scatter(rows, args.output_dir / "tradeoff_scatter.png")
    plot_best_per_dataset(rows, args.output_dir / "best_per_dataset.png")
    plot_dataset_distributions(rows, args.output_dir / "dataset_auroc_distribution.png")
    plot_focal_set_size(rows, args.output_dir / "focal_set_size_vs_auroc.png")
    plot_metric_bars(rows, args.output_dir / "top_run_bars.png")
    plot_learning_diagnostics(rows, args.output_dir / "learning_diagnostics.png")
    plot_ablation_tradeoff(rows, args.output_dir / "ablation_tradeoff_scatter.png")
    plot_ablation_distribution(rows, args.output_dir / "ablation_auroc_distribution.png")
    plot_best_per_ablation(rows, args.output_dir / "best_per_ablation.png")

    per_dataset_dir = args.output_dir / "by_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_rows in sorted(_group_by_dataset(rows).items()):
        generate_dataset_bundle(dataset, dataset_rows, per_dataset_dir, args.top_k)

    per_ablation_dir = args.output_dir / "by_ablation"
    per_ablation_dir.mkdir(parents=True, exist_ok=True)
    for ablation, ablation_rows in sorted(_group_by_ablation(rows).items()):
        generate_ablation_bundle(ablation, ablation_rows, per_ablation_dir, args.top_k)

    print(f"Saved random-set plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
