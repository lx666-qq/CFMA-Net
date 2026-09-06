$ErrorActionPreference = "Stop"

$config = "config/fold0_resnet50_5shot.yaml"
$seed = 2025
$modes = @("direct", "cross_attention", "pma")

foreach ($mode in $modes) {
    $out = "runs\cmad_ablation\5shot\${mode}_s${seed}"
    Write-Host "============================================================"
    Write-Host "Training CMAD interaction: $mode (5-shot)"
    Write-Host "Output: $out"
    Write-Host "============================================================"

    python train.py `
      --config $config `
      --cmad_match_mode $mode `
      --pma_mode sinkhorn `
      --seed $seed `
      --output_dir $out `
      --no-viz `
      --no-pma-analysis

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $ckpt = Get-ChildItem -Path $out -Filter "train_epoch_*.pth" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $ckpt) {
        throw "No best checkpoint found under $out"
    }

    python train.py `
      --config $config `
      --cmad_match_mode $mode `
      --pma_mode sinkhorn `
      --seed $seed `
      --output_dir $out `
      --checkpoint_path $ckpt.FullName `
      --eval_only `
      --pma_analysis `
      --no-viz `
      --pma_run_tag "cmad-${mode}-5shot-s${seed}"

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python analyze_cmad_ablation.py --root "runs\cmad_ablation\5shot" --seed $seed
