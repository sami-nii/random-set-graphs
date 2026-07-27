$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\tommy\anaconda3\envs\graph_uncertainty\python.exe"

# Add or remove datasets as required. Runs are sequential to avoid contention.
$datasets = @(
    "arxiv",  
    "coauthor", 
    "squirrel", 
    "amazon_ratings", 
    "cora", 
    "roman_empire"
)

& $pythonExe -c "import sys, torch; print(sys.executable); print(torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The graph_uncertainty Python environment could not import torch."
}

foreach ($dataset in $datasets) {
    Write-Host "Training three fixed-config vanilla GNNs for $dataset..."
    & $pythonExe .\main.py -d $dataset -m vanilla --sweep_name sweep_vanilla_overhead_benchmark --count 3
    if ($LASTEXITCODE -ne 0) {
        throw "Vanilla training failed for $dataset with exit code $LASTEXITCODE."
    }

    Write-Host "Benchmarking post-hoc baseline overhead for $dataset..."
    & $pythonExe .\experiments\benchmark_baseline_overhead.py --dataset $dataset --num-checkpoints 3 --repetitions 3 --output ".\plots\random_set_results\baseline_overhead_$dataset"
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline-overhead benchmark failed for $dataset."
    }
}
