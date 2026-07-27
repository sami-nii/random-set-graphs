$ErrorActionPreference = "Stop"

$python = "C:\Users\tommy\anaconda3\envs\graph_uncertainty\python.exe"

$datasets = @(
    "reddit2",
    "arxiv"
)

foreach ($dataset in $datasets) {
    Write-Host "Starting $dataset..."

    & $python .\main.py -d $dataset -m random_set --sweep_name sweep_random_set_ablation_budget_timing --count 3

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for $dataset with exit code $LASTEXITCODE."
    }

    & $python .\plots\random_set_results.py --summary-only
}