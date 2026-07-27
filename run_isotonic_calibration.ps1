# Sequential RS-GNN + post-hoc isotonic-calibration runs for the reviewer study.
# Defaults to every current in-scope dataset. Patents and Reddit2 are omitted
# because they are deferred for the time-limited experiments.
# Safe to run alongside other terminals: this script only starts its own child
# Python processes and waits for each one to finish before starting the next.

param(
    [string[]]$Datasets = @(
        "amazon_ratings",
        "arxiv",
        "chameleon",
        "coauthor",
        "cora",
        "roman_empire",
        "squirrel"
    ),
    [int]$Count = 3,
    [string]$ProjectName = "graph-uncertainty"
)

$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\tommy\anaconda3\envs\graph_uncertainty\python.exe"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Could not find the graph_uncertainty Python interpreter at: $pythonExe"
}

Set-Location -LiteralPath $repoRoot

foreach ($dataset in $Datasets) {
    Write-Host "Starting isotonic RS-GNN runs for $dataset ($Count repetitions)..." -ForegroundColor Cyan
    & $pythonExe main.py `
        --dataset $dataset `
        --model random_set `
        --sweep_name sweep_random_set_isotonic_calibration `
        --count $Count `
        --project_name $ProjectName

    if ($LASTEXITCODE -ne 0) {
        throw "The isotonic-calibration run for '$dataset' failed with exit code $LASTEXITCODE."
    }
}

Write-Host "All isotonic-calibration runs finished. Refreshing the aggregate table..." -ForegroundColor Cyan
& $pythonExe plots/random_set_results.py --summary-only

if ($LASTEXITCODE -ne 0) {
    throw "Result-summary refresh failed with exit code $LASTEXITCODE."
}

Write-Host "Done. Results are in plots/random_set_results/aggregate_results.md and WandB." -ForegroundColor Green
