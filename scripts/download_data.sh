#!/usr/bin/env bash
# Download / prepare the datasets of the RS-GNN paper under $CGNN_DATA_DIR (default: ./dataset).
#
#   bash scripts/download_data.sh                 # all datasets that can be fetched automatically
#   DATASETS="cora chameleon" bash scripts/download_data.sh
#
# Automatic: cora, chameleon, squirrel (PyG), coauthor (gnn-benchmark), reddit2 (GraphSAINT, Google Drive),
# arxiv (OGB), amazon_ratings and roman_empire (Platonov et al., GitHub). Manual: patents, road, nuscenes
# (instructions printed at the end). `cgnn doctor` reports what is still missing.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR=${CGNN_DATA_DIR:-dataset}
DATASETS=${DATASETS:-"cora chameleon squirrel coauthor arxiv amazon_ratings roman_empire reddit2"}
HETEROPHILOUS=https://github.com/yandex-research/heterophilous-graphs/raw/main/data

for ds in $DATASETS; do
    case "$ds" in
        amazon_ratings | roman_empire)
            mkdir -p "$DATA_DIR/$ds"
            [[ -f "$DATA_DIR/$ds/$ds.npz" || -f "$DATA_DIR/$ds/raw/$ds.npz" ]] ||
                curl -L --fail -o "$DATA_DIR/$ds/$ds.npz" "$HETEROPHILOUS/$ds.npz"
            ;;
    esac
    echo "== $ds"
    cgnn data "$ds" # loads the dataset (PyG/OGB download on first use) and prints the ID/OOD split sizes
done

cat <<'MSG'

Manual downloads (place the files as shown, relative to the data directory):
  patents   snap-patents.mat from the LINKX benchmark
            (https://github.com/CUAI/Non-Homophily-Large-Scale, `snap-patents` in its data loaders)
            -> dataset/snap-patents.mat
  road      road_trainval_v1.0.json of the ROAD dataset (https://github.com/gurkirt/road-dataset)
            -> dataset/road/road_trainval_v1.0.json
  nuscenes  nuScenes v1.0-mini metadata (https://www.nuscenes.org/download, free registration;
            only the JSON tables are used) -> dataset/nuscenes/v1.0-mini/*.json
Then check with: cgnn doctor
MSG
