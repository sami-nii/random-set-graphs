#!/usr/bin/env bash
# Experiments of "Random-Set Graph Neural Networks" (NeurIPS 2026), in dependency order.
# Prints the commands; set EXECUTE=1 to run them. Every command pins ONE free GPU (`--gpus 1`);
# never run more than two at the same time on the shared server.
#
#   EXECUTE=1 bash scripts/reproduce_paper.sh                 # everything
#   STEPS="ood ablation" DATASETS="cora chameleon" EXECUTE=1 bash scripts/reproduce_paper.sh
#
# Steps (paper artefact in brackets):
#   ood        OOD detection on the nine standard benchmarks            [Tab. ood, Figs. 5-7]
#   road       ROAD / nuScenes / ROAD->nuScenes, vanilla vs RS-GNN     [Tabs. id_core_metrics, ood_uncertainty_auroc]
#   ablation   focal-set and loss ablations on Cora and Roman Empire   [Tab. ablation_main, Fig. ablation_summary]
#   isotonic   RS-GNN + isotonic calibration, 5 seeds                  [Tab. isotonic_calibration]
#   cost       focal-set budget vs K and per-forward overhead          [Tabs. focal_budget, runtime]
#
# Post-hoc baselines, ensembles, CaGCN and CGNN post train need the vanilla backbones of the same
# dataset (first command of `ood`). Every trainable model uses the paper's GATv2 backbone
# (configs/methods/*.yaml). Road-scene data: dataset/road/road_trainval_v1.0.json and
# dataset/nuscenes/v1.0-mini/ (see README).
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-"ood road ablation isotonic cost"}
DATASETS=${DATASETS:-"chameleon squirrel arxiv patents amazon_ratings roman_empire coauthor reddit2 cora"}
PROJECT=${PROJECT:-rs-gnn}
VANILLA_RUNS=${VANILLA_RUNS:-100}
TRAINABLE_RUNS=${TRAINABLE_RUNS:-30}
SEEDS=${SEEDS:-"0 1 2 3 4"}

run() {
    echo "+ $*"
    if [[ "${EXECUTE:-0}" == "1" ]]; then "$@"; fi
}
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

if has_step ood; then
    for ds in $DATASETS; do
        # 1) backbones: checkpoints ranked by val_auroc feed every post-hoc method
        run cgnn sweep -m vanilla -d "$ds" -c "$VANILLA_RUNS" -p "$PROJECT" --gpus 1
        # 2) post-hoc baselines (grid sweeps), CaGCN and CGNN post train
        for m in energy odin mahalanobis knn knn_LJ gnnsafe gebm ensemble; do
            run cgnn sweep -m "$m" -d "$ds" -p "$PROJECT" --gpus 1
        done
        for m in cagcn frozen; do
            run cgnn sweep -m "$m" -d "$ds" -c "$TRAINABLE_RUNS" -p "$PROJECT" --gpus 1
        done
        # 3) credal baselines (CGNN family) and RS-GNN
        for m in credal credal_LJ credal_LJ_dual_head_detached random_set; do
            run cgnn sweep -m "$m" -d "$ds" -c "$TRAINABLE_RUNS" -p "$PROJECT" --gpus 1
        done
    done
    # Tab. 1 layout: per score, mean over the top-5 runs of each sweep by validation AUROC
    run cgnn results -p "$PROJECT" -d $DATASETS --format latex --out outputs/tables/ood.tex --details
fi

if has_step road; then
    # Fixed protocol of Tab. hyperparams (dataset specs in cgnn/data/sources/road_scene.py)
    for ds in road nuscenes road_to_nuscenes; do
        for m in vanilla random_set; do
            run cgnn run -m "$m" -d "$ds" -p "$PROJECT" --gpus 1 --seeds 0
        done
    done
    run python experiments/summarize_runs.py road -d road nuscenes road_to_nuscenes
fi

if has_step ablation; then
    for ds in cora roman_empire; do
        for m in random_set random_set_singletons random_set_bce_only random_set_bce_mr; do
            run cgnn sweep -m "$m" -d "$ds" -c "$TRAINABLE_RUNS" -p "$PROJECT" --gpus 1
        done
    done
    # Tab. 4 reports the best OOD AUROC over each sweep (-k 1)
    run cgnn results -p "$PROJECT" -d cora roman_empire -k 1 \
        -m random_set random_set_singletons random_set_bce_only random_set_bce_mr --details
fi

if has_step isotonic; then
    # Best RS-GNN configuration of each `ood` sweep, re-run with isotonic calibration over 5 seeds.
    # Set SWEEP_<dataset>=<W&B sweep id of `random_set` on that dataset>.
    for ds in amazon_ratings arxiv chameleon coauthor cora roman_empire squirrel road nuscenes; do
        var="SWEEP_${ds}"
        from_sweep=()
        if [[ -n "${!var:-}" ]]; then from_sweep=(--from-sweep "${!var}"); fi
        run cgnn run -m random_set_isotonic -d "$ds" -p "$PROJECT" --gpus 1 --seeds $SEEDS "${from_sweep[@]}"
    done
    run python experiments/summarize_runs.py isotonic
fi

if has_step cost; then
    for ds in coauthor reddit2 roman_empire; do
        run python experiments/benchmark_focal_set_budget.py --dataset "$ds" --k-values 4 8 16 32 64 --repetitions 3
    done
    for ds in amazon_ratings arxiv chameleon coauthor cora roman_empire squirrel; do
        # three backbones with the default configuration (seeds 0, 1, 2), then the timings
        run cgnn run -m vanilla -d "$ds" -p "$PROJECT" --gpus 1 --seeds 0 1 2
        run python experiments/benchmark_overhead.py --dataset "$ds" --num-checkpoints 3 --repetitions 3
    done
fi
