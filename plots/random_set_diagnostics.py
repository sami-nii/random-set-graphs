import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = ["#2f6f9f", "#c05640", "#4f8f62", "#8f5c9f", "#c08a2d", "#4f7f82"]
AUROC_KEY = "test_auroc_entropy"
VAL_KEY = "val_auroc_entropy"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2f3437",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#d8dee4",
        "grid.linewidth": 0.8,
        "font.size": 10,
    }
)


def maybe_float(value):
    if value in (None, ""):
        return None
    try:
        converted = float(value)
    except ValueError:
        return None
    if math.isnan(converted):
        return None
    return converted


def label(value):
    return str(value).replace("__", ": ").replace("_", " ")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            AUROC_KEY,
            VAL_KEY,
            "lr",
            "weight_decay",
            "hidden_channels",
            "num_layers",
            "epoch",
            "num_output_heads",
        ]:
            row[key] = maybe_float(row.get(key))
    return rows


def valid(rows, *keys):
    return [row for row in rows if all(row.get(key) is not None for key in keys)]


def ordered_unique(rows, key):
    return sorted({str(row[key]) for row in rows if row.get(key) not in (None, "")})


def grouped_best(rows, row_key, col_key):
    table = defaultdict(dict)
    for row in valid(rows, AUROC_KEY):
        r = str(row[row_key])
        c = str(row[col_key])
        score = row[AUROC_KEY]
        current = table[r].get(c)
        if current is None or score > current:
            table[r][c] = score
    return table


def draw_heatmap(matrix, row_labels, col_labels, output_path, title, value_format="{:.3f}", cmap="viridis"):
    if not row_labels or not col_labels:
        return

    fig_w = max(7.5, 1.15 * len(col_labels) + 2.2)
    fig_h = max(4.5, 0.52 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image_values = [[math.nan if value is None else value for value in row] for row in matrix]
    image = ax.imshow(image_values, aspect="auto", vmin=0, vmax=1, cmap=cmap)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([label(item) for item in col_labels], rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([label(item) for item in row_labels])
    ax.set_title(title)

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value is None:
                ax.text(x, y, "-", ha="center", va="center", color="#6b7280", fontsize=9)
                continue
            text_color = "white" if value < 0.62 else "#111827"
            ax.text(x, y, value_format.format(value), ha="center", va="center", color=text_color, fontsize=8.5)

    fig.colorbar(image, ax=ax, label="Best Test AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_ablation_dataset_heatmap(rows, output_dir):
    datasets = ordered_unique(rows, "dataset")
    ablations = ordered_unique(rows, "ablation_group")
    bests = grouped_best(rows, "dataset", "ablation_group")
    matrix = [[bests.get(dataset, {}).get(ablation) for ablation in ablations] for dataset in datasets]
    draw_heatmap(
        matrix,
        datasets,
        ablations,
        output_dir / "best_auroc_by_dataset_ablation.png",
        "Best AUROC by Dataset and Ablation",
    )


def plot_backbone_dataset_heatmap(rows, output_dir):
    datasets = ordered_unique(rows, "dataset")
    backbones = ordered_unique(rows, "gnn_type")
    bests = grouped_best(rows, "dataset", "gnn_type")
    matrix = [[bests.get(dataset, {}).get(backbone) for backbone in backbones] for dataset in datasets]
    draw_heatmap(
        matrix,
        datasets,
        backbones,
        output_dir / "best_auroc_by_dataset_backbone.png",
        "Best AUROC by Dataset and Backbone",
    )


def plot_ablation_run_coverage(rows, output_dir):
    datasets = ordered_unique(rows, "dataset")
    ablations = ordered_unique(rows, "ablation_group")
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[str(row["dataset"])][str(row["ablation_group"])] += 1

    matrix = [[counts[dataset].get(ablation, 0) for ablation in ablations] for dataset in datasets]
    max_count = max(max(row) for row in matrix) if matrix else 1
    scaled = [[value / max_count if value else None for value in row] for row in matrix]

    fig_w = max(7.5, 1.15 * len(ablations) + 2.2)
    fig_h = max(4.5, 0.52 * len(datasets) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow([[value or 0 for value in row] for row in scaled], aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels([label(item) for item in ablations], rotation=25, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Run Coverage by Dataset and Ablation")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, str(value) if value else "-", ha="center", va="center", color="#111827", fontsize=8.5)
    fig.colorbar(image, ax=ax, label="Relative run count")
    fig.tight_layout()
    fig.savefig(output_dir / "run_coverage_by_dataset_ablation.png", dpi=240)
    plt.close(fig)


def plot_validation_test_gap(rows, output_dir):
    by_dataset = defaultdict(list)
    for row in valid(rows, AUROC_KEY, VAL_KEY):
        by_dataset[str(row["dataset"])].append(row[AUROC_KEY] - row[VAL_KEY])

    labels = sorted(by_dataset)
    if not labels:
        return

    values = [by_dataset[item] for item in labels]
    means = [sum(item) / len(item) for item in values]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=False)
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[idx % len(PALETTE)])
        patch.set_alpha(0.8)
    ax.scatter(range(1, len(labels) + 1), means, marker="D", s=52, color="#111827", zorder=4, label="Mean gap")
    ax.axhline(0, color="#111827", lw=1.1, linestyle="--", alpha=0.75)
    ax.set_ylabel("Test AUROC - Validation AUROC")
    ax.set_title("Validation/Test AUROC Gap by Dataset")
    ax.grid(axis="y", alpha=0.6)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_test_gap_by_dataset.png", dpi=240)
    plt.close(fig)


def plot_hyperparameter_map(rows, output_dir):
    points = [row for row in valid(rows, AUROC_KEY, "lr", "weight_decay") if row["lr"] > 0 and row["weight_decay"] > 0]
    if not points:
        return

    xs = [math.log10(row["lr"]) for row in points]
    ys = [math.log10(row["weight_decay"]) for row in points]
    colors = [row[AUROC_KEY] for row in points]
    best = max(points, key=lambda row: row[AUROC_KEY])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", vmin=0, vmax=1, alpha=0.75, s=48, edgecolors="white", linewidths=0.35)
    ax.scatter([math.log10(best["lr"])], [math.log10(best["weight_decay"])], marker="*", s=260, color="#f59e0b", edgecolors="#111827", linewidths=0.8)
    ax.annotate(
        f"best {best[AUROC_KEY]:.3f}\n{best['dataset']}",
        (math.log10(best["lr"]), math.log10(best["weight_decay"])),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=8,
    )
    ax.set_xlabel("log10(learning rate)")
    ax.set_ylabel("log10(weight decay)")
    ax.set_title("Hyperparameter Map Colored by Test AUROC")
    ax.grid(alpha=0.55)
    fig.colorbar(scatter, ax=ax, label="Test AUROC")
    fig.tight_layout()
    fig.savefig(output_dir / "lr_weight_decay_auroc_map.png", dpi=240)
    plt.close(fig)


def plot_focal_head_ablation(rows, output_dir):
    points = valid(rows, AUROC_KEY, "num_output_heads")
    if not points:
        return

    by_ablation = defaultdict(list)
    for row in points:
        by_ablation[str(row["ablation_group"])].append(row)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for idx, (ablation, ablation_rows) in enumerate(sorted(by_ablation.items())):
        xs = [row["num_output_heads"] for row in ablation_rows]
        ys = [row[AUROC_KEY] for row in ablation_rows]
        ax.scatter(xs, ys, s=44, alpha=0.72, color=PALETTE[idx % len(PALETTE)], label=label(ablation), edgecolors="white", linewidths=0.35)

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of focal sets / output heads (log scale)")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Focal-Set Budget vs AUROC by Ablation")
    ax.grid(axis="both", alpha=0.45)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "focal_heads_vs_auroc_by_ablation.png", dpi=240)
    plt.close(fig)


def write_notes(rows, output_dir):
    lines = ["Random-set diagnostic chart notes", ""]
    valid_rows = valid(rows, AUROC_KEY)
    if valid_rows:
        best = max(valid_rows, key=lambda row: row[AUROC_KEY])
        lines.append(
            f"Best overall: dataset={best['dataset']} run={best['run_id']} "
            f"ablation={best['ablation_group']} AUROC={best[AUROC_KEY]:.4f}"
        )

    for dataset in ordered_unique(rows, "dataset"):
        dataset_rows = [row for row in valid_rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        best = max(dataset_rows, key=lambda row: row[AUROC_KEY])
        lines.append(
            f"{dataset}: best={best[AUROC_KEY]:.4f}, ablation={best['ablation_group']}, "
            f"heads={best.get('num_output_heads')}, lr={best.get('lr')}, wd={best.get('weight_decay')}"
        )

    (output_dir / "diagnostic_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate additional diagnostic charts from random-set CSV results.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("plots") / "random_set_results" / "random_set_runs.csv",
        help="CSV generated by plots/random_set_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "random_set_results" / "diagnostics",
        help="Directory for additional diagnostic charts.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_dataset_heatmap(rows, args.output_dir)
    plot_backbone_dataset_heatmap(rows, args.output_dir)
    plot_ablation_run_coverage(rows, args.output_dir)
    plot_validation_test_gap(rows, args.output_dir)
    plot_hyperparameter_map(rows, args.output_dir)
    plot_focal_head_ablation(rows, args.output_dir)
    write_notes(rows, args.output_dir)
    print(f"Saved diagnostic charts to: {args.output_dir}")


if __name__ == "__main__":
    main()
