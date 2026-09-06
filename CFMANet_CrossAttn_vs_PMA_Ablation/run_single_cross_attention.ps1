$ErrorActionPreference = "Stop"
$config = "config/fold0_resnet50.yaml"
$seed = 2025
$out = "runs\cmad_ablation\1shot\cross_attention_s${seed}"

python train.py `
  --config $config `
  --cmad_match_mode cross_attention `
  --pma_mode sinkhorn `
  --seed $seed `
  --output_dir $out `
  --no-viz `
  --no-pma-analysis
