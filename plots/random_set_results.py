import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional


SUMMARY_FILENAME = "wandb-summary.json"
CONFIG_FILENAME = "config.yaml"
OUTPUT_LOG_FILENAME = "output.log"
AUROC_KEY = "test_auroc_entropy"
VAL_AUROC_KEY = "val_auroc_entropy"
AUROC_LABEL = "Test AUROC"
VAL_AUROC_LABEL = "Validation AUROC"
SUMMARY_METRICS = (
    ("test_acc_id", "Test ID accuracy"),
    ("test_ece_id", "Test ID ECE"),
    ("test_auroc_entropy", "Test AUROC"),
    ("val_f1", "Validation F1"),
    ("val_auroc_entropy", "Validation AUROC"),
    ("focal_set_budget_seconds", "Budget time (s)"),
    ("isotonic_test_id_ece_before", "Test ID ECE (before)"),
    ("isotonic_test_id_ece_after", "Test ID ECE (isotonic)"),
    ("isotonic_test_auroc_entropy_before", "Test AUROC (before)"),
    ("isotonic_test_auroc_entropy_after", "Test AUROC (isotonic)"),
)
PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#2f3437",
    "axes.labelcolor": "#1f2933",
    "axes.titlecolor": "#111827",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#d8dee4",
    "grid.linewidth": 0.8,
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "font.size": 10,
}
SERIES_PALETTE = ["#2f6f9f", "#c05640", "#4f8f62", "#8f5c9f", "#c08a2d", "#4f7f82", "#7a6f5f", "#b64d6a"]
STANDARD_ABLATION_GROUP = "standard_focal_sets__full"

plt = None


def _initialize_plotting() -> None:
    """Load Matplotlib only when plots, rather than summary tables, are requested."""
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting

    plotting.rcParams.update(PLOT_STYLE)
    plt = plotting


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
        model_name = (
            _extract_arg_from_config(config_text, "--model")
            or _extract_arg_from_config(config_text, "-m")
        )
        if model_name != "random_set":
            continue

        dataset_name = (
            _extract_arg_from_config(config_text, "--dataset")
            or _extract_arg_from_config(config_text, "-d")
            or "unknown"
        )
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
            "time_focal_set_budget": _normalize_bool(_extract_value_from_config(config_text, "time_focal_set_budget")),
            "isotonic_calibration": _normalize_bool(_extract_value_from_config(config_text, "isotonic_calibration")),
            "loss_ablation": _extract_value_from_config(config_text, "loss_ablation") or "full",
            "use_bce_loss": _normalize_bool(_extract_value_from_config(config_text, "use_bce_loss")),
            "use_mr_loss": _normalize_bool(_extract_value_from_config(config_text, "use_mr_loss")),
            "use_ms_loss": _normalize_bool(_extract_value_from_config(config_text, "use_ms_loss")),
            "train_loss": _safe_float(summary.get("train_loss")),
            "train_bce": _safe_float(summary.get("train_bce")),
            "train_mr": _safe_float(summary.get("train_mr")),
            "train_ms": _safe_float(summary.get("train_ms")),
            "val_loss": _safe_float(summary.get("val_loss")),
            "val_f1": _safe_float(summary.get("val_f1")),
            "val_auroc_entropy": _safe_float(summary.get("val_auroc_entropy")),
            "test_acc_id": _safe_float(summary.get("test_acc_id")),
            "test_ece_id": _safe_float(summary.get("test_ece_id")),
            "test_auroc_entropy": _safe_float(summary.get("test_auroc_entropy")),
            "focal_set_budget_seconds": _safe_float(summary.get("focal_set_budget_seconds")),
            "isotonic_test_id_ece_before": _safe_float(summary.get("isotonic_test_id_ece_before")),
            "isotonic_test_id_ece_after": _safe_float(summary.get("isotonic_test_id_ece_after")),
            "isotonic_test_auroc_entropy_before": _safe_float(summary.get("isotonic_test_auroc_entropy_before")),
            "isotonic_test_auroc_entropy_after": _safe_float(summary.get("isotonic_test_auroc_entropy_after")),
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


def _summary_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _format_mean_std(statistics_row: Dict[str, Optional[float]]) -> str:
    if not statistics_row["count"]:
        return "—"
    mean = statistics_row["mean"]
    std = statistics_row["std"]
    if std is None:
        return f"{mean:.6f} (n=1)"
    return f"{mean:.6f} ± {std:.6f} (n={statistics_row['count']})"


def write_aggregate_results_table(rows: List[Dict[str, object]], output_dir: Path) -> None:
    """Write per-dataset, per-ablation mean and sample standard-deviation tables."""
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        group_key = (
            str(row["dataset"]),
            str(row["ablation_group"]),
            str(row.get("focal_strategy") or "unknown"),
            bool(row.get("time_focal_set_budget")),
            bool(row.get("isotonic_calibration")),
        )
        grouped.setdefault(group_key, []).append(row)

    table_rows = []
    for (dataset, ablation, focal_strategy, timer_enabled, isotonic_enabled), group_rows in sorted(grouped.items()):
        table_row = {
            "dataset": dataset,
            "ablation": ablation,
            "focal_strategy": focal_strategy,
            "budget_timer": "enabled" if timer_enabled else "disabled",
            "isotonic_calibration": "enabled" if isotonic_enabled else "disabled",
            "runs": len(group_rows),
        }
        for metric_key, _ in SUMMARY_METRICS:
            values = [row[metric_key] for row in group_rows if row.get(metric_key) is not None]
            stats = _summary_statistics(values)
            table_row[f"{metric_key}_n"] = stats["count"]
            table_row[f"{metric_key}_mean"] = stats["mean"]
            table_row[f"{metric_key}_std"] = stats["std"]
        table_rows.append(table_row)

    write_csv(table_rows, output_dir / "aggregate_results.csv")

    headers = ["Dataset", "Ablation", "Focal sets", "Budget timer", "Isotonic", "Runs"] + [label for _, label in SUMMARY_METRICS]
    lines = [
        "# Aggregate random-set results",
        "",
        "Values are mean ± sample standard deviation. `n` is the number of runs with that metric; budget time is blank for non-budgeted runs.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for table_row in table_rows:
        cells = [
            table_row["dataset"],
            table_row["ablation"],
            table_row["focal_strategy"],
            table_row["budget_timer"],
            table_row["isotonic_calibration"],
            str(table_row["runs"]),
        ]
        for metric_key, _ in SUMMARY_METRICS:
            cells.append(
                _format_mean_std(
                    {
                        "count": table_row[f"{metric_key}_n"],
                        "mean": table_row[f"{metric_key}_mean"],
                        "std": table_row[f"{metric_key}_std"],
                    }
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    (output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _format_label(value: object) -> str:
    return str(value).replace("__", ": ").replace("_", " ")


def _valid_auroc_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [row for row in rows if row.get(AUROC_KEY) is not None]


def _best_by_auroc(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return max(rows, key=lambda row: row[AUROC_KEY])


def _set_auroc_axis(ax) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(AUROC_LABEL)
    ax.grid(axis="y", alpha=0.8)


def _ablation_summaries(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    for ablation, ablation_rows in sorted(_group_by_ablation(rows).items()):
        scores = [row[AUROC_KEY] for row in ablation_rows if row[AUROC_KEY] is not None]
        if not scores:
            continue
        summaries.append(
            {
                "group": ablation,
                "label": _format_label(ablation),
                "scores": sorted(scores),
                "mean": sum(scores) / len(scores),
                "best": max(scores),
                "runs": len(scores),
            }
        )
    return summaries


def plot_tradeoff_scatter(rows: List[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    grouped = _group_by_dataset(rows)

    for idx, (dataset, dataset_rows) in enumerate(sorted(grouped.items())):
        valid_rows = [row for row in dataset_rows if row.get(VAL_AUROC_KEY) is not None and row.get(AUROC_KEY) is not None]
        xs = [row[VAL_AUROC_KEY] for row in valid_rows]
        ys = [row[AUROC_KEY] for row in valid_rows]
        sizes = []
        for row in valid_rows:
            heads = row.get("num_output_heads")
            sizes.append(50 if heads is None else max(40, min(240, float(heads) * 3)))

        if xs:
            color = SERIES_PALETTE[idx % len(SERIES_PALETTE)]
            ax.scatter(xs, ys, s=sizes, alpha=0.78, color=color, label=dataset, edgecolors="white", linewidths=0.5)

            best_row = _best_by_auroc(valid_rows)
            ax.annotate(
                f"{dataset}:{best_row.get('gnn_type', 'NA')}",
                (best_row[VAL_AUROC_KEY], best_row[AUROC_KEY]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.plot([0, 1], [0, 1], color="#9aa4b2", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Random-Set AUROC Generalization")
    ax.set_xlabel(VAL_AUROC_LABEL)
    _set_auroc_axis(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ablation_tradeoff(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [row for row in rows if row.get(VAL_AUROC_KEY) is not None and row.get(AUROC_KEY) is not None]
    if not valid_rows:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = _group_by_ablation(valid_rows)

    for idx, (ablation, ablation_rows) in enumerate(sorted(grouped.items())):
        xs = [row[VAL_AUROC_KEY] for row in ablation_rows]
        ys = [row[AUROC_KEY] for row in ablation_rows]
        ax.scatter(
            xs,
            ys,
            alpha=0.7,
            color=SERIES_PALETTE[idx % len(SERIES_PALETTE)],
            label=_format_label(ablation),
            edgecolors="white",
            linewidths=0.5,
        )

    ax.plot([0, 1], [0, 1], color="#9aa4b2", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Random-Set Ablation AUROC Generalization")
    ax.set_xlabel(VAL_AUROC_LABEL)
    _set_auroc_axis(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_ablation_distribution(rows: List[Dict[str, object]], output_path: Path) -> None:
    summaries = _ablation_summaries(rows)
    if not summaries:
        return

    standard = next((item for item in summaries if item["group"] == STANDARD_ABLATION_GROUP), None)
    standard_best = standard["best"] if standard else max(item["best"] for item in summaries)
    summaries.sort(key=lambda item: (item["group"] != STANDARD_ABLATION_GROUP, -item["best"]))

    labels = [item["label"] for item in summaries]
    bests = [item["best"] for item in summaries]
    means = [item["mean"] for item in summaries]
    x = list(range(len(summaries)))
    colors = [SERIES_PALETTE[0] if item["group"] == STANDARD_ABLATION_GROUP else SERIES_PALETTE[1] for item in summaries]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(x, bests, color=colors, alpha=0.9, label="Best AUROC")
    ax.scatter(x, means, marker="D", s=58, color="#111827", zorder=4, label="Mean AUROC")
    ax.axhline(
        standard_best,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"Standard best ({standard_best:.3f})",
    )

    for idx, item in enumerate(summaries):
        delta = item["best"] - standard_best
        ax.text(idx, item["best"] + 0.018, f"{item['best']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx, item["mean"] - 0.026, f"mean {item['mean']:.3f}", ha="center", va="top", fontsize=8, color="#374151")
        if item["group"] != STANDARD_ABLATION_GROUP:
            ax.text(idx, 0.055, f"Delta {delta:+.3f}", ha="center", va="bottom", fontsize=8, color="#374151")
        else:
            ax.text(idx, 0.055, "standard", ha="center", va="bottom", fontsize=8, color="#374151")

    ax.bar_label(bars, labels=[f"n={item['runs']}" for item in summaries], label_type="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _set_auroc_axis(ax)
    ax.set_title("Ablation AUROC Compared with Standard Best Run")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_mean_best(rows: List[Dict[str, object]], output_path: Path) -> None:
    summaries = _ablation_summaries(rows)
    if not summaries:
        return

    standard = next((item for item in summaries if item["group"] == STANDARD_ABLATION_GROUP), None)
    standard_best = standard["best"] if standard else max(item["best"] for item in summaries)
    summaries.sort(key=lambda item: (item["group"] != STANDARD_ABLATION_GROUP, -item["best"]))

    labels = [item["label"] for item in summaries]
    means = [item["mean"] for item in summaries]
    bests = [item["best"] for item in summaries]
    x = list(range(len(summaries)))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.vlines(x, means, bests, color="#9aa4b2", linewidth=2.5, alpha=0.9)
    ax.scatter(x, means, marker="D", s=74, color="#111827", zorder=4, label="Mean AUROC")
    ax.scatter(x, bests, marker="*", s=160, color="#f2a541", edgecolors="#111827", linewidths=0.6, zorder=5, label="Best AUROC")
    ax.axhline(
        standard_best,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"Standard best ({standard_best:.3f})",
    )
    for idx, item in enumerate(summaries):
        delta = item["best"] - standard_best
        ax.text(idx, item["best"] + 0.015, f"{item['best']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx, item["mean"] - 0.025, f"{item['mean']:.3f}", ha="center", va="top", fontsize=8, color="#374151")
        if item["group"] != STANDARD_ABLATION_GROUP:
            ax.text(idx, 0.055, f"Delta {delta:+.3f}", ha="center", va="bottom", fontsize=8, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _set_auroc_axis(ax)
    ax.set_title("Mean and Best AUROC by Ablation")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_best_per_dataset(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_dataset(rows)
    best_rows = []

    for dataset, dataset_rows in sorted(grouped.items()):
        valid_rows = [row for row in dataset_rows if row[AUROC_KEY] is not None]
        if not valid_rows:
            continue
        best_rows.append(_best_by_auroc(valid_rows))

    if not best_rows:
        return

    labels = [f"{row['dataset']} ({row.get('gnn_type', 'NA')})" for row in best_rows]
    aurocs = [row[AUROC_KEY] for row in best_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(best_rows)))

    bars = ax.bar(x, aurocs, color=SERIES_PALETTE[0])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in aurocs], padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _set_auroc_axis(ax)
    ax.set_title("Best Random-Set AUROC Per Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dataset_distributions(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_dataset(rows)
    labels = []
    values = []

    for dataset, dataset_rows in sorted(grouped.items()):
        scores = [row[AUROC_KEY] for row in dataset_rows if row[AUROC_KEY] is not None]
        if scores:
            labels.append(dataset)
            values.append(scores)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels)
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(SERIES_PALETTE[idx % len(SERIES_PALETTE)])
        patch.set_alpha(0.85)

    _set_auroc_axis(ax)
    ax.set_title("Random-Set AUROC Distribution by Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_focal_set_size(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("num_output_heads") is not None and row.get(AUROC_KEY) is not None
    ]
    if not valid_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = _group_by_dataset(valid_rows)

    for idx, (dataset, dataset_rows) in enumerate(sorted(grouped.items())):
        xs = [row["num_output_heads"] for row in dataset_rows]
        ys = [row[AUROC_KEY] for row in dataset_rows]
        ax.scatter(xs, ys, color=SERIES_PALETTE[idx % len(SERIES_PALETTE)], alpha=0.8, label=dataset, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Number of Focal Sets / Output Heads")
    _set_auroc_axis(ax)
    ax.set_title("Random-Set OOD Performance vs Focal-Set Budget")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_bars(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get(AUROC_KEY) is not None
    ]
    if not valid_rows:
        return

    sorted_rows = sorted(
        valid_rows,
        key=lambda row: row[AUROC_KEY],
        reverse=True,
    )[:12]

    labels = [f"{row.get('gnn_type', 'NA')}-{row['run_id']}" for row in sorted_rows]
    aurocs = [row[AUROC_KEY] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(9, len(sorted_rows) * 0.8), 5))
    x = list(range(len(sorted_rows)))

    bars = ax.bar(x, aurocs, color=SERIES_PALETTE[0])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in aurocs], padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _set_auroc_axis(ax)
    ax.set_title("Top Random-Set Runs by AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_learning_diagnostics(rows: List[Dict[str, object]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get(AUROC_KEY) is not None and row.get(VAL_AUROC_KEY) is not None
    ]
    if not valid_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(
        [row[VAL_AUROC_KEY] for row in valid_rows],
        [row[AUROC_KEY] for row in valid_rows],
        color=SERIES_PALETTE[0],
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[0].plot([0, 1], [0, 1], color="#9aa4b2", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Validation vs Test AUROC")
    axes[0].set_xlabel(VAL_AUROC_LABEL)
    _set_auroc_axis(axes[0])
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(
        [row["epoch"] for row in valid_rows if row.get("epoch") is not None],
        [row[AUROC_KEY] for row in valid_rows if row.get("epoch") is not None],
        color=SERIES_PALETTE[1],
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    axes[1].set_title("Training Epoch vs Test AUROC")
    axes[1].set_xlabel("Epoch")
    _set_auroc_axis(axes[1])
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_top_runs_report(rows: List[Dict[str, object]], output_path: Path, top_k: int) -> None:
    sortable = [
        row
        for row in rows
        if row.get(AUROC_KEY) is not None
    ]
    sortable.sort(key=lambda row: row[AUROC_KEY], reverse=True)

    lines = ["Top random-set runs by test AUROC", ""]
    for idx, row in enumerate(sortable[:top_k], start=1):
        lines.append(
            (
                f"{idx}. dataset={row['dataset']} run={row['run_id']} "
                f"gnn={row.get('gnn_type')} "
                f"test_auroc={row.get(AUROC_KEY):.4f} "
                f"focal_strategy={row.get('focal_strategy')} "
                f"num_output_heads={row.get('num_output_heads')}"
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_report(rows: List[Dict[str, object]], output_path: Path) -> None:
    grouped = _group_by_ablation(rows)
    lines = ["Random-set ablation summary", ""]
    for ablation, ablation_rows in sorted(grouped.items()):
        valid_rows = _valid_auroc_rows(ablation_rows)
        if not valid_rows:
            continue
        best_row = _best_by_auroc(valid_rows)
        mean_auroc = sum(row[AUROC_KEY] for row in valid_rows) / len(valid_rows)
        lines.append(
            f"{ablation}: runs={len(ablation_rows)} valid={len(valid_rows)} "
            f"best_auroc={best_row[AUROC_KEY]:.4f} "
            f"mean_auroc={mean_auroc:.4f}"
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
    parser.add_argument("--summary-only", action="store_true", help="Only refresh the aggregate mean/std tables.")
    args = parser.parse_args()

    rows = load_random_set_runs(args.wandb_dir)
    if args.dataset:
        rows = [row for row in rows if row["dataset"] == args.dataset]

    if not rows:
        raise SystemExit("No local random_set runs were found for the requested filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_aggregate_results_table(rows, args.output_dir)
    if args.summary_only:
        print(f"Saved aggregate result tables to: {args.output_dir}")
        return

    _initialize_plotting()

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
    plot_ablation_mean_best(rows, args.output_dir / "best_per_ablation.png")

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
