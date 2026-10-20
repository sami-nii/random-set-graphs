# Ensure ~/.netrc exists before running the container
[ -f "$HOME/.netrc" ] || touch "$HOME/.netrc"
chmod 600 "$HOME/.netrc"  

# Check if an NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "NVIDIA GPU detected. Enabling GPU support."
else
    GPU_FLAG=""
    echo "No NVIDIA GPU detected. Running without GPU support."
fi

# Run the container (installs the mounted repository and the graph-ebm submodule in editable mode)
docker run -it --user $(id -u):$(id -g) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(cd "$(dirname "$0")/.." && pwd)":/workspace \
    -v "$HOME/.config/wandb":/home/$(whoami)/.config/wandb \
    -v "$HOME/.netrc:/home/$(whoami)/.netrc:rw" \
    $GPU_FLAG \
    --rm custom-pyg \
    bash -c 'pip install --user --no-deps -e /workspace && pip install --user --no-deps --no-build-isolation -e /workspace/graph-ebm && exec bash'
