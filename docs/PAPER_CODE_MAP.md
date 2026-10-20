# Paper ↔ code map

Paper: *Random-Set Graph Neural Networks*, NeurIPS 2026 (camera-ready source: `neurips_2026.tex`).
Numbers refer to the compiled camera-ready (LaTeX labels in brackets).

## Method (Sec. 3, App. D)

| Paper | Code |
|---|---|
| Belief head `Bel_v(A) = σ(g_θ(z_v))` | `cgnn/models/random_set.py::RandomSetGNN.forward` (`belief_head`, sigmoid) |
| Möbius inversion, Eq. 1 [`eq:mobius`] | `cgnn/models/focal_sets.py::focal_set_matrices` (`moebius`), `RandomSetGNN.valid_masses` |
| Belief head for node v, Fig. 11 [`fig:rsgnn_belief_head`]: clip, residual → Θ, renormalise, uniform fallback | `RandomSetGNN.valid_masses`, `RandomSetGNN.predict` |
| Pignistic probability `BetP` | `focal_set_matrices` (`pignistic`), `RandomSetGNN.predict` |
| Pignistic entropy, Eq. 2 [`eq:pignistic_entropy`] | `RandomSetGNN.predict` → score key `entropy` |
| Credal width `W = Pl({ŷ}) − Bel({ŷ})`, Eq. 3 [`eq:credal_width`] | `RandomSetGNN.predict` (`membership`, `singleton` matrices) → score key `credal` |
| Proposition 1 [`prop:validity`] | checked in `tests/test_random_set.py` (BetP ∈ credal set, W formula, singletons ⇒ W = 0) |
| Worked example (Sec. 3, "Example") | `tests/test_random_set.py::test_worked_example_of_the_paper` |
| Loss, Eq. 4 [`eq:final_loss`], α = β = 1e-3 | `cgnn/models/random_set.py::RandomSetLoss`; `alpha`, `beta` in `configs/methods/random_set.yaml` |
| Power set when \|Y\| < 10 | `cgnn/methods/random_set.py::build_focal_sets` (`power_set_max_classes: 10`) |
| Alg. 1 [`alg:focal_sets`]: 200 nodes/class, induced subgraph, warm-up GNN on labelled ID nodes, 3-D t-SNE, one Gaussian per class, 95% ellipsoid (χ²₃ = 7.815), Jaccard of voxelised regions, pairs → triples, top-K | `cgnn/methods/random_set.py::budget_embeddings`, `train_warmup_backbone`; `cgnn/models/focal_sets.py::tsne_3d`, `class_regions`, `overlap_scores`, `top_k_focal_sets` |
| K = 64 non-singletons, max cardinality 3 | `budget_k`, `budget_max_cardinality` |
| Alg. 2 [`alg:training`] | `RandomSetGNN.training_step` / `test_step`, `cgnn/runner.py` |
| Hard-validity alternative (App. D.4) | not implemented (the paper describes it as an alternative) |
| Isotonic calibration on ID validation nodes (App. D.11, Tab. 12) | `cgnn/uncertainty/isotonic.py`; method `random_set_isotonic` |

## Experimental setup (Sec. 4.1, App. A, D)

| Paper | Code |
|---|---|
| Leave-out-class transductive protocol (App. D.9) | `cgnn/data/ood.py::apply_leave_out_classes` |
| ID/OOD classes and split sizes, Tab. 10 [`tab:dataset_stats`] | `cgnn/data/sources/*.py` (`id_classes`, `ood_classes`); print with `cgnn data <dataset>` |
| Feature sparsity, Tab. 11 [`tab:feature_sparsity`] | `experiments/feature_sparsity.py` |
| GATv2 backbone for every model (App. D.2): 2 layers, hidden 128, heads 4 → 1, ReLU, dropout 0.2 | `cgnn/models/backbones.py::GATv2`; `gnn_type/gat_heads/act/dropout` in `configs/methods/{vanilla,credal*,random_set*}.yaml` |
| Vanilla head `softmax(W z_v)` (App. D.1, D.3) | `cgnn/models/vanilla.py` (`head: linear`) |
| Baselines (Sec. 4.1) | `cgnn/models/detectors/`, `cgnn/models/{credal,frozen,cagcn}.py`; keys in `README.md` |
| Temporal road-scene graphs (App. A.2): scene + agent nodes, scene-centred and temporal edges, self-loops; ROAD window 4 / stride 4 / step 5, 37 features; nuScenes window 3 / stride 1, 10 features | `cgnn/data/sources/road_scene.py` (`road`, `nuscenes`, `road_to_nuscenes`) |
| Focal sets on road scenes, Tab. 6 [`tab:focal_sets`] | dataset protocol `focal_set_strategy: budgeted`; `num_total_classes` adds OOD singletons |
| Road-scene hyperparameters, Tab. 8 [`tab:hyperparams`] | `_ROAD_PROTOCOL`, `_NUSCENES_PROTOCOL` in `road_scene.py` |
| Evaluation metrics (App. D.11): AUROC, AUPRC, FPR95, ECE (15 bins), NLL, Brier | `cgnn/metrics.py` (`ood_detection_metrics`, `calibration_metrics`) |

## Results

| Paper | How to produce |
|---|---|
| Tab. 1 [`tab:ood`], Figs. 5–7 | `scripts/reproduce_paper.sh` step `ood`; `cgnn results` |
| Tabs. 2–3 [`tab:id_core_metrics`, `tab:ood_uncertainty_auroc`], Figs. 8–9 | step `road`; `experiments/summarize_runs.py road` |
| Tab. 4 [`tab:ablation_main`], Fig. 10 | step `ablation` (`random_set_singletons`, `random_set_bce_only`, `random_set_bce_mr`) |
| Tab. 7 [`tab:focal_budget`] | `experiments/benchmark_focal_set_budget.py` |
| Tab. 9 [`tab:runtime`] | `experiments/benchmark_overhead.py` |
| Tab. 12 [`tab:isotonic_calibration`] | step `isotonic`; `experiments/summarize_runs.py isotonic` |

Figures 5–10 are plots of these tables. Their plotting scripts are not in this repository.

## Where the paper and the code cannot fully agree

These are the points where the paper text is ambiguous or inconsistent. The code follows the choice stated here.

1. **nuScenes focal sets.** Tab. 6 gives 72 (= 8 + 64) focal sets. With 6 ID classes there are only
   C(6,2) + C(6,3) = 35 candidate pairs and triples, so the code gives at most 6 + 35 + 2 = 43. Reaching 72
   would need pairs and triples that include OOD classes, which would use OOD labels, and App. D.6 says
   these are never used. ROAD gives 74 as in the table (8 ID classes give 84 candidates).
2. **Road-scene graph counts** (Tab. 5). The window parameters are taken from the text. The ROAD
   train/val/test counts (49,123 / 26,302 / 26,301) depend on the video split and window semantics, which
   the paper does not fully specify: the code uses `train_1` videos for training and halves the `val_1`
   windows into validation and test. For nuScenes, 307/39/38 graphs fit the official v1.0-mini split
   (8 train scenes; scene-0103 and scene-0916), which is the default.
3. **ROAD → nuScenes features.** ROAD graphs have 37 features and nuScenes graphs have 10. The paper does
   not say how the features were aligned, so the code zero-pads nuScenes to the ROAD width with the
   node-type columns aligned (`_pad_features`).
4. **Mini-batching.** App. D.2 mentions NeighborLoader mini-batching. The code mini-batches where the
   dataset metadata sets `batch_size > 0` (ArXiv, Patents, Reddit2) and trains the smaller graphs
   full-batch. Road-scene graphs are batched 16 at a time.
5. **Warm-up GNN.** Alg. 1 does not give the warm-up training length. The code uses
   `budget_warmup_epochs: 20` with the RS-GNN learning rate.
6. **Model selection.** As in the CGNN protocol, sweeps select configurations by validation AUROC (ID + OOD
   validation nodes, App. D.9). RS-GNN early-stops on the ID validation loss (patience 50, at most 200
   epochs), as in the code that produced the paper's RS-GNN results.
7. **Ablation names.** "BCE + margin" (Tab. 4) is BCE plus the negative-mass regulariser R_neg (`bce_mr`).
8. **ODIN on Squirrel** in Tab. 1 is 47.28. The CGNN paper reports 62.28 for the same configuration.
9. **Backbones.** The baseline rows of Tab. 1 on the seven datasets shared with the CGNN paper come from the
   CGNN pipeline (GCN/SAGE backbones with sigmoid). This code uses the GATv2 backbone for every model, as
   App. D.2 states, so baselines re-run with it will differ from those rows.
