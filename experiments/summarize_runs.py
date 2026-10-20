"""Mean +- sample std of metrics over the local run summaries (``outputs/runs/*.json``) per dataset/method.

Presets:

- ``isotonic``: Tab. isotonic_calibration (ID ECE and entropy AUROC before/after isotonic calibration;
  runs of ``random_set_isotonic``);
- ``road``: Tabs. id_core_metrics and ood_uncertainty_auroc (ID accuracy/ECE/NLL/Brier, mean
  uncertainties, AUROC/AUPRC of entropy and credal width; ``vanilla`` and ``random_set``);
- ``budget``: full focal-set construction time (``focal_set_budget_seconds``) and family size.

    python experiments/summarize_runs.py isotonic
    python experiments/summarize_runs.py road -d road nuscenes road_to_nuscenes
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cgnn.paths import Paths

PRESETS = {
    "isotonic": (
        ("random_set_isotonic",),
        [
            "isotonic_test_id_ece_before",
            "isotonic_test_id_ece_after",
            "isotonic_test_auroc_entropy_before",
            "isotonic_test_auroc_entropy_after",
        ],
    ),
    "road": (
        ("vanilla", "random_set"),
        [
            "test_id_acc",
            "test_id_ece",
            "test_id_nll",
            "test_id_brier",
            "test_mean_id_entropy",
            "test_mean_id_credal",
            "test_mean_ood_entropy",
            "test_mean_ood_credal",
            "test_auroc_entropy",
            "test_auprc_entropy",
            "test_auroc_credal",
            "test_auprc_credal",
        ],
    ),
    "budget": (("random_set",), ["num_focal_sets", "focal_set_budget_seconds"]),
}


def fmt(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.fmean(values):.4f} ± {statistics.stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("-d", "--datasets", nargs="*")
    parser.add_argument("-m", "--methods", nargs="*")
    parser.add_argument("--runs-dir", type=Path, default=None, help="default: $CGNN_OUTPUT_DIR/runs")
    args = parser.parse_args()

    methods, keys = PRESETS[args.preset]
    methods = tuple(args.methods or methods)
    runs_dir = args.runs_dir or Paths.from_env().output_dir / "runs"
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(runs_dir.glob("*.json")):
        run = json.loads(path.read_text())
        if run["method"] not in methods or (args.datasets and run["dataset"] not in args.datasets):
            continue
        for key in keys:
            if key in run["metrics"]:
                groups[(run["dataset"], run["method"])][key].append(run["metrics"][key])

    print("| Dataset | Method | Runs | " + " | ".join(keys) + " |")
    print("|---|---|---:|" + "---|" * len(keys))
    for (dataset, method), metrics in sorted(groups.items()):
        runs = max((len(v) for v in metrics.values()), default=0)
        print(
            f"| {dataset} | {method} | {runs} | " + " | ".join(fmt(metrics.get(k, [])) for k in keys) + " |"
        )


if __name__ == "__main__":
    main()
