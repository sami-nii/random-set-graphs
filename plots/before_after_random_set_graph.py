from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


OUT_DIR = Path(__file__).resolve().parent


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            color="black",
        )
    )
    if text:
        ax.text((x1 + x2) / 2, y1 + 0.035, text, ha="center", va="center", fontsize=8.5)


def graph(ax, ox, oy, s=1.0):
    pts = {
        "a": (ox + 0.00 * s, oy + 0.24 * s),
        "b": (ox + 0.12 * s, oy + 0.34 * s),
        "c": (ox + 0.06 * s, oy + 0.07 * s),
        "d": (ox + 0.25 * s, oy + 0.16 * s),
        "e": (ox + 0.16 * s, oy - 0.10 * s),
    }
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")]:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color="#b9c0c9", lw=1.1, zorder=1)
    colors = {"a": "#224a9b", "b": "#224a9b", "d": "#224a9b", "c": "#07b47b", "e": "#d63d32"}
    for k, (x, y) in pts.items():
        ax.add_patch(Circle((x, y), 0.026 * s, facecolor=colors[k], edgecolor="black", lw=0.7, zorder=3))


def bars(ax, x, y, vals, labels, colors, width=0.028, scale=0.17, title=None):
    if title:
        ax.text(x + len(vals) * width * 0.55, y + scale + 0.04, title, ha="center", va="bottom", fontsize=8.5, color="#344054")
    for i, (v, label, color) in enumerate(zip(vals, labels, colors)):
        bx = x + i * width * 1.35
        ax.add_patch(Polygon(
            [(bx, y), (bx + width, y), (bx + width, y + v * scale), (bx, y + v * scale)],
            closed=True,
            facecolor=color,
            edgecolor="black",
            lw=0.6,
        ))
        ax.text(bx + width / 2, y + v * scale + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(bx + width / 2, y - 0.018, label, ha="center", va="top", fontsize=7)


def simplex(ax, x, y, w=0.17, h=0.23, labels=True):
    c1 = (x + w / 2, y + h)
    c2 = (x, y)
    c3 = (x + w, y)
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color="black", lw=1.2)
    ax.plot([c2[0], c3[0]], [c2[1], c3[1]], color="black", lw=1.2)
    ax.plot([c3[0], c1[0]], [c3[1], c1[1]], color="black", lw=1.2)
    if labels:
        ax.text(c1[0], c1[1] + 0.025, "$c_1$", ha="center", va="center", fontsize=9)
        ax.text(c2[0] - 0.02, c2[1] - 0.014, "$c_2$", ha="center", va="center", fontsize=9)
        ax.text(c3[0] + 0.02, c3[1] - 0.014, "$c_3$", ha="center", va="center", fontsize=9)
    return c1, c2, c3


def left_panel(ax):
    ax.text(0.015, 0.965, "(a) Standard GNN: point estimate on the simplex", fontsize=8.8, weight="bold")
    ax.text(0.08, 0.885, "Input graph\n(OOD node in red)", ha="center", fontsize=8.5, color="#17345c")
    graph(ax, 0.045, 0.61, s=0.55)

    arrow(ax, 0.165, 0.70, 0.225, 0.70, "GNN")
    bars(
        ax,
        0.255,
        0.63,
        [0.04, 0.91, 0.05],
        ["$c_1$", "$c_2$", "$c_3$"],
        ["#2aa7df", "#09b26f", "#d63d32"],
        title="softmax output",
    )
    arrow(ax, 0.385, 0.70, 0.43, 0.70)

    simplex(ax, 0.455, 0.55, w=0.145, h=0.22)
    ax.scatter([0.49], [0.565], marker="x", s=80, color="#d92323", linewidths=2.2, zorder=5)
    ax.plot([0.49, 0.535], [0.565, 0.49], color="#d92323", lw=1.0)
    ax.text(0.535, 0.47, "confident\nerror", ha="center", va="top", fontsize=8, color="#d92323")
    ax.text(
        0.29,
        0.34,
        "Before: local uncertainty collapses each node to one class,\nso structural inconsistency can be hidden by high confidence.",
        ha="center",
        va="center",
        fontsize=9,
        color="#667085",
        style="italic",
    )


def right_panel(ax):
    ax.text(0.66, 0.965, "(b) RS-GNN: candidate focal sets induce a credal region", fontsize=8.8, weight="bold")
    ax.text(0.71, 0.885, "Input graph\n(OOD node in red)", ha="center", fontsize=8.5, color="#17345c")
    graph(ax, 0.675, 0.61, s=0.55)

    arrow(ax, 0.795, 0.70, 0.85, 0.70, "GNN")
    bars(
        ax,
        0.88,
        0.63,
        [0.10, 0.30, 0.10, 0.20, 0.30],
        ["$c_1$", "$\\{c_1,c_2\\}$", "$c_2$", "$\\{c_2,c_3\\}$", "$\\Omega$"],
        ["#2aa7df", "#96d8f0", "#09b26f", "#f3a6a0", "#344054"],
        width=0.021,
        title="mass over focal sets $m(\\cdot)$",
    )
    arrow(ax, 1.015, 0.70, 1.06, 0.70)

    simplex(ax, 1.09, 0.55, w=0.145, h=0.22)
    region = Polygon(
        [(1.155, 0.61), (1.19, 0.64), (1.22, 0.62), (1.228, 0.575), (1.19, 0.54), (1.155, 0.56)],
        closed=True,
        facecolor="#dfe9ff",
        edgecolor="#3155b7",
        lw=1.3,
    )
    ax.add_patch(region)
    ax.scatter([1.185], [0.585], marker="x", s=70, color="#3155b7", linewidths=1.7, zorder=5)
    ax.plot([1.196, 1.22], [0.575, 0.505], color="#3155b7", lw=1.0)
    ax.text(1.22, 0.485, "verified credal set\nwidth = uncertainty", ha="center", va="top", fontsize=8, color="#183c9a")
    ax.text(
        0.93,
        0.34,
        "After: mass spreads to non-singleton focal sets,\nso novel inputs enlarge the set instead of forcing a point prediction.",
        ha="center",
        va="center",
        fontsize=9,
        color="#183c9a",
        style="italic",
        weight="bold",
    )


def main():
    fig, ax = plt.subplots(figsize=(12.0, 4.25), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1.28)
    ax.set_ylim(0.25, 1.0)
    ax.axis("off")

    left_panel(ax)
    right_panel(ax)

    ax.plot([0.635, 0.635], [0.28, 0.96], color="#e5e7eb", lw=1.0)
    fig.savefig(OUT_DIR / "before_after_random_set_graph.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "before_after_random_set_graph.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
