$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\tommy\anaconda3\envs\graph_uncertainty\python.exe"

# Task 3 applies only to budgeted datasets; Reddit2 is queued last.
$datasets = @("coauthor", "roman_empire", "reddit2")
$kValues = @(4, 8, 16, 32, 64)

& $pythonExe -c "import sys, torch; print(sys.executable); print(torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The graph_uncertainty Python environment could not import torch."
}

foreach ($dataset in $datasets) {
    Write-Host "Running Task 3 K ablation for $dataset..."
    $output = ".\plots\random_set_results\task_3_focal_set_budget_k_ablation_$dataset"
    & $pythonExe .\experiments\benchmark_focal_set_budget.py --dataset $dataset --k-values $kValues --repetitions 3 --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "Task 3 K ablation failed for $dataset with exit code $LASTEXITCODE."
    }
}
