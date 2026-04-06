# Auto-push training logs every hour for remote monitoring.
# Usage: powershell scripts/auto_push.ps1
# Run this in a SEPARATE terminal alongside training.

$INTERVAL = 3600  # seconds (1 hour)

Write-Host "=== Auto-push started (every $($INTERVAL)s) ===" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop"

while ($true) {
    Start-Sleep -Seconds $INTERVAL

    # Stage only log files (not large checkpoint .pt files)
    git add experiments/*_train_log.csv 2>$null
    git add experiments/*_results.json 2>$null
    git add experiments/*_emissions.csv 2>$null

    # Check if there's anything to commit
    $diff = git diff --cached --quiet 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date -Format 'HH:mm')] No new log changes to push" -ForegroundColor DarkGray
        continue
    }

    # Get latest stats from logs for commit message
    $summary = ""
    Get-ChildItem experiments/*_train_log.csv -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace '_train_log$', ''
        $last = Get-Content $_.FullName -Tail 1
        if ($last -and $last -notmatch '^epoch') {
            $fields = ($last -split ',')[0..3] -join ','
            $summary += " ${name}:[$fields]"
        }
    }

    if (-not $summary) { $summary = " update" }
    git commit -m "training progress:$summary"

    # Pull with rebase to handle pushes from other machines
    git pull --rebase 2>$null

    if (git push 2>$null) {
        Write-Host "[$(Get-Date -Format 'HH:mm')] Pushed training logs" -ForegroundColor Green
    } else {
        Write-Host "[$(Get-Date -Format 'HH:mm')] Push failed -- will retry next cycle" -ForegroundColor Yellow
    }
}
