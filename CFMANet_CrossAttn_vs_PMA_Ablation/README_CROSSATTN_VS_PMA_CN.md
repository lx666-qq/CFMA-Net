# CFMANet：Cross-Attention vs PMA 的 CMAD 消融实验包

## 1. 实验目的

这个版本专门验证：**在 CMAD 内部，常规 Patch Cross-Attention 和本文 PMA 哪个更有效。**

为了保证控制变量干净：

- 网络前面的 `pma_ref` 始终保留；
- 正式比较时 `pma_ref` 固定为 `pma_mode=sinkhorn`；
- 只替换 CMAD 内部两处交互：`pma_fusion` 和 `pma_high`；
- PFE、FDPG、SPM prior、CMAD 的上采样、多尺度卷积、detail/semantic branch、loss、训练设置全部不变。

支持三个 CMAD 模式：

- `direct`：直接拼接 + GBM，无显式对应关系；
- `cross_attention`：轻量 Patch Cross-Attention；
- `pma`：原始 Sinkhorn-style PMA。

## 2. Cross-Attention 的公平性设计

Cross-Attention 与 PMA 使用：

- 相同输入特征；
- 相同 `patch_size=7`；
- 相同 Q/K 降维宽度 `256 -> 64`；
- 相同形式的 `Conv1x1 + BN + Sigmoid` confidence gate；
- 不添加 Transformer FFN、LayerNorm、多头堆叠；
- 不额外增加 V projection。

Cross-Attention：

`A = softmax(Q K^T / sqrt(d))`

然后从 support/guidance patch 聚合 Value，并通过相同风格 gate 融合回 query。

当前 smoke test 中，`cross_attention` 与 `pma` 的 **CMAD 参数量完全相同**（均约 0.886888M；实际完整模型参数量以训练日志为准），因此非常适合用于论文控制实验。

## 3. 安装方法

这是一个**覆盖到你现有 MFANet-master/CFMANet 工程根目录的可运行补丁包**。它依赖你项目中原有的 `util/`、`model/resnet.py`、`model/vgg.py`、数据集和 data_list。

将本包内容复制到工程根目录，目录对应关系：

```text
train.py                         -> 工程根目录/train.py
model/mfanet.py                  -> 工程根目录/model/mfanet.py
config/fold0_resnet50.yaml       -> 工程根目录/config/fold0_resnet50.yaml
config/fold0_resnet50_5shot.yaml -> 工程根目录/config/fold0_resnet50_5shot.yaml
```

建议先备份你现有文件。

## 4. 先做 smoke test

在工程根目录：

```powershell
python smoke_test_matchers.py
```

应看到三个模式都输出 `(1, 2, 200, 200)` 且 `finite=True`。

## 5. 单独跑 Cross-Attention（1-shot）

```powershell
python train.py `
  --config config/fold0_resnet50.yaml `
  --cmad_match_mode cross_attention `
  --pma_mode sinkhorn `
  --seed 2025 `
  --output_dir runs\cmad_ablation\1shot\cross_attention_s2025 `
  --no-viz `
  --no-pma-analysis
```

也可以直接：

```powershell
.\run_single_cross_attention.ps1
```

## 6. 一键跑完整 1-shot 三组

```powershell
.\run_cmad_ablation_1shot.ps1
```

依次训练：

1. Direct Fusion
2. Cross-Attention
3. PMA

每组训练完成后脚本会自动找到该目录下最新的 `train_epoch_*.pth`，重新做一次 `eval_only + pma_analysis`，最后运行统计脚本。

## 7. 一键跑完整 5-shot 三组

```powershell
.\run_cmad_ablation_5shot.ps1
```

5-shot 使用 `config/fold0_resnet50_5shot.yaml`，除了 `shot=5` 之外与 1-shot 配置保持一致。

## 8. YAML / CLI 新参数

新增：

```yaml
cmad_match_mode: pma
```

可选：

```text
direct
cross_attention
pma
```

CLI 示例：

```powershell
python train.py --config config/fold0_resnet50.yaml --cmad_match_mode cross_attention --pma_mode sinkhorn
```

注意：`pma_mode` 和 `cmad_match_mode` 是两个不同概念。

- `pma_mode=sinkhorn`：固定原始 PMA 内部匹配规则；
- `cmad_match_mode=cross_attention`：只把 CMAD 内部两处 PMA 替换为 Cross-Attention；
- 前面的 `pma_ref` 不会被 `cmad_match_mode` 替换。

## 9. 论文正式比较建议

主表：

| CMAD interaction | Explicit matching | Structured norm. | 1-shot mIoU | FB-IoU | 5-shot mIoU | FB-IoU | FPS |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| Direct Fusion | × | × | | | | | |
| Cross-Attention | ✓ | × | | | | | |
| PMA | ✓ | ✓ | | | | | |

所有三组必须：

- 同一个 seed；
- 同一个训练/验证划分；
- 同一个 optimizer、LR、epoch、batch size；
- 不加载其他模式训练得到的 checkpoint；
- 正式 PMA 与 Cross-Attention 对比时 `pma_mode` 固定 `sinkhorn`。

趋势验证可以先只跑 seed=2025。论文最终建议再补 2026、2027 并报告 mean±std。

## 10. Spatial-mismatch 分析

训练脚本原有的 `pma_analysis` 已经记录每个 episode 的：

- `disp_mean`
- `disp_max`
- `fg_iou`
- `episode_fb_iou`

本包额外记录 `cmad_match_mode`。

运行：

```powershell
python analyze_cmad_ablation.py --root runs\cmad_ablation\1shot --seed 2025
```

会生成：

```text
analysis_results/
  official_metrics.csv
  spatial_mismatch_shared_thresholds.csv
  paired_bootstrap.csv
  README_RESULTS.txt
```

其中：

- `official_metrics.csv`：主论文表优先使用这里的官方 mIoU / FB-IoU；
- `spatial_mismatch_shared_thresholds.csv`：使用 PMA run 的 1/3、2/3 分位点作为**所有方法共同阈值**，避免三组各自划分 Low/Medium/High；
- `paired_bootstrap.csv`：PMA − Cross-Attention 的 episode foreground-IoU paired bootstrap 95% CI。

注意：现有 `pma_analysis` 的 `fg_conf/bg_conf/alignment_auroc` 来自网络前面的 `pma_ref`。本实验中 `pma_ref`是固定的，所以这些 confidence 指标**不能用于声称 CMAD 的 Cross-Attention/PMA 谁的匹配质量更好**。本次机制分析主要使用 displacement 与 episode IoU。

## 11. 当前配置

随包 YAML 已设为：

```yaml
pma_mode: sinkhorn
cmad_match_mode: pma
pma_patch_size: 7
pma_epsilon: 0.05
pma_sinkhorn_iter: 2
pma_no_adapter: false
pma_analysis: false
resume:
```

这保证默认启动仍然是完整 PMA 模型。
