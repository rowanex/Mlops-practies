$ErrorActionPreference = "Stop"

$branchName = "no-git"
$commitSha = (Get-Date -Format "yyyyMMddHHmmss")

try {
    $gitBranch = git branch --show-current 2>$null
    if ($LASTEXITCODE -eq 0 -and $gitBranch) {
        $branchName = $gitBranch.Trim()
    }

    $gitSha = git rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $gitSha) {
        $commitSha = $gitSha.Trim()
    }
} catch {
    Write-Host "Git metadata not found, using timestamp tag."
}

$safeBranch = $branchName.ToLower() -replace "[^a-z0-9_.-]", "-"
$imageName = "iris-api"
$tag = "$safeBranch-$commitSha"

Write-Host "Building image $imageName`:$tag"
docker build -t "${imageName}:${tag}" -t "${imageName}:latest" .

Write-Host "Done. Tags:"
Write-Host " - ${imageName}:${tag}"
Write-Host " - ${imageName}:latest"
