import os
import random
import time
import csv
import json
import math
import cv2
import numpy as np
import logging
import argparse
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model.mfanet import mfanet
from util.dataset_fss import SemData as dataset
from util import transform, config
from util.util import *
# SVG生成
import base64
import io
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#解决样本不平衡
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

global best_iou
global best_epoch

def feat_to_gray(feat):
    """将中间特征转换为归一化二维能量图。"""
    if feat is None:
        raise ValueError('feature tensor is None')
    if not torch.is_tensor(feat):
        feat = torch.as_tensor(feat)

    feat = feat.detach().float().cpu()
    if feat.dim() == 4:
        feat = feat[0]
    if feat.dim() == 3:
        # RMS energy 比 abs-mean 更适合显示小波高频响应
        feat = torch.sqrt(torch.mean(feat ** 2, dim=0) + 1e-12)
    elif feat.dim() != 2:
        raise ValueError(f'Unsupported feature shape: {tuple(feat.shape)}')

    arr = feat.numpy()
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom > 1e-8:
        arr = arr / denom
    else:
        arr = np.zeros_like(arr)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def denorm_image(img_tensor, mean, std):
    """反归一化 CHW 图像，并安全转换为 uint8 RGB。"""
    img = img_tensor.detach().float().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    img = img * std + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def colorize_mask(mask_np, palette, num_classes):
    mask_np = np.asarray(mask_np)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    color = np.zeros((*mask_np.shape[-2:], 3), dtype=np.uint8)
    for cls_id in range(num_classes):
        color[mask_np == cls_id] = palette[cls_id * 3:cls_id * 3 + 3]
    return color


def slice_vis_dict(vis_dict, sample_index, batch_size):
    """从包含 batch 维的可视化字典中取出单个样本。"""
    result = {}
    for key, value in vis_dict.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
            result[key] = value[sample_index:sample_index + 1]
        else:
            result[key] = value
    return result


def save_fdpg_figure(support_img, support_mask, fdpg_vis, save_path,
                     mean, std, palette, num_classes):
    if fdpg_vis is None:
        raise ValueError('FDPG visualization dictionary is empty')

    support_np = denorm_image(support_img, mean, std)
    supp_mask_np = support_mask.detach().cpu().numpy()
    supp_mask_color = colorize_mask(supp_mask_np.astype(np.uint8), palette, num_classes)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    items = [
        (support_np, None, 'Support'),
        (supp_mask_color, None, 'Support Mask'),
        (feat_to_gray(fdpg_vis['LL']), 'inferno', 'LL (Low-frequency)'),
        (feat_to_gray(fdpg_vis['LH']), 'magma', 'LH (Vertical detail)'),
        (feat_to_gray(fdpg_vis['HL']), 'magma', 'HL (Horizontal detail)'),
        (feat_to_gray(fdpg_vis['HH']), 'magma', 'HH (Diagonal detail)'),
        (feat_to_gray(fdpg_vis['high_freq_feat']), 'viridis', 'High-frequency Fusion'),
        (feat_to_gray(fdpg_vis['low_freq_feat']), 'viridis', 'Low-frequency Enhancement'),
        (feat_to_gray(fdpg_vis['mca_feat_up']), 'viridis', 'Frequency Fusion'),
        (feat_to_gray(fdpg_vis['prototype']), 'viridis', 'Dynamic Prototype'),
    ]

    for ax, (img, cmap, title) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    dir_w = fdpg_vis['dir_weight'][0].detach().float().cpu().numpy()
    freq_w_tensor = fdpg_vis['freq_weight']
    if freq_w_tensor.dim() == 2:
        freq_w_tensor = freq_w_tensor[0]
    freq_w = freq_w_tensor.detach().float().cpu().numpy()

    fig.suptitle(
        f'FDPG: direction=[{dir_w[0]:.3f}, {dir_w[1]:.3f}, {dir_w[2]:.3f}], '
        f'frequency=[high {freq_w[0]:.3f}, low {freq_w[1]:.3f}]',
        fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _normalize_vis_map(x):
    """Convert a tensor/array to a robust [0, 1] 2-D map for visualization."""
    if torch.is_tensor(x):
        x = x.detach().float().cpu()
        while x.dim() > 2 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() == 3:
            # L2 channel energy is less cancellation-prone than a signed mean.
            x = torch.linalg.vector_norm(x, ord=2, dim=0)
        elif x.dim() != 2:
            raise ValueError(f'Unsupported visualization tensor shape: {tuple(x.shape)}')
        x = x.numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = np.linalg.norm(x, axis=0)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [1.0, 99.0]) if x.size else (0.0, 1.0)
    if hi <= lo + 1e-8:
        lo, hi = float(x.min()), float(x.max())
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _resize_vis_map(x, out_hw):
    h, w = out_hw
    return cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_label_map(mask, out_hw):
    """Resize a discrete label/prediction map for visualization only.

    Nearest-neighbor interpolation is required so class IDs are not mixed. This
    helper is intentionally confined to figure generation and does not modify
    model outputs or evaluation metrics.
    """
    arr = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    h, w = int(out_hw[0]), int(out_hw[1])
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return arr


def _overlay_heatmap(rgb, heat, alpha=0.48, cmap_name='magma'):
    """Overlay a scalar heatmap on an RGB uint8 image."""
    heat = _normalize_vis_map(heat)
    heat = _resize_vis_map(heat, rgb.shape[:2])
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(heat)[..., :3] * 255.0).astype(np.uint8)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8), heat


def _draw_mask_contour(ax, mask, color='white', linewidth=1.4):
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.max() > 0:
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=linewidth)


def _draw_patch_grid(ax, h_patch, w_patch, prefix, image_hw, label=True):
    """Draw an interpretable patch grid. The grid is for visualization only."""
    h, w = image_hw
    for r in range(1, h_patch):
        ax.axhline(r * h / h_patch - 0.5, color='white', lw=0.55, alpha=0.75)
    for c in range(1, w_patch):
        ax.axvline(c * w / w_patch - 0.5, color='white', lw=0.55, alpha=0.75)
    if label and h_patch * w_patch <= 16:
        for r in range(h_patch):
            for c in range(w_patch):
                idx = r * w_patch + c + 1
                x = (c + 0.08) * w / w_patch
                y = (r + 0.20) * h / h_patch
                ax.text(
                    x, y, f'{prefix}{idx}', color='white', fontsize=7,
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.12', fc='black', ec='none', alpha=0.55)
                )


def _draw_correspondence_schematic(ax, transport_plan, h_patch, w_patch, topk=6):
    """Reviewer-friendly top-k patch correspondence diagram."""
    from matplotlib.patches import Rectangle

    T = np.asarray(transport_plan, dtype=np.float32)
    n = h_patch * w_patch
    if T.shape[0] != n or T.shape[1] != n:
        ax.text(0.5, 0.5, f'Unexpected transport shape {T.shape}', ha='center', va='center')
        ax.axis('off')
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Top Transport Correspondences', fontsize=10)

    left_x0, left_x1 = 0.03, 0.34
    right_x0, right_x1 = 0.66, 0.97
    y0, y1 = 0.10, 0.90
    cell_w_l = (left_x1 - left_x0) / w_patch
    cell_w_r = (right_x1 - right_x0) / w_patch
    cell_h = (y1 - y0) / h_patch

    def center(side, idx):
        r, c = divmod(idx, w_patch)
        x0 = left_x0 if side == 'p' else right_x0
        cw = cell_w_l if side == 'p' else cell_w_r
        return x0 + (c + 0.5) * cw, y1 - (r + 0.5) * cell_h

    for side, x0, cw, prefix in [('p', left_x0, cell_w_l, 'P'), ('q', right_x0, cell_w_r, 'Q')]:
        for r in range(h_patch):
            for c in range(w_patch):
                idx = r * w_patch + c
                rect = Rectangle(
                    (x0 + c * cw, y1 - (r + 1) * cell_h), cw, cell_h,
                    fill=False, edgecolor='0.45', linewidth=0.7
                )
                ax.add_patch(rect)
                cx, cy = center(side, idx)
                ax.text(cx, cy, f'{prefix}{idx + 1}', ha='center', va='center', fontsize=7)

    ax.text((left_x0 + left_x1) / 2, 0.96, 'Guidance / prototype patches', ha='center', fontsize=8)
    ax.text((right_x0 + right_x1) / 2, 0.96, 'Query patches', ha='center', fontsize=8)

    # Select global strongest entries to avoid a spaghetti plot.
    topk = int(min(max(topk, 1), T.size))
    flat_idx = np.argpartition(T.reshape(-1), -topk)[-topk:]
    flat_idx = flat_idx[np.argsort(T.reshape(-1)[flat_idx])[::-1]]
    values = T.reshape(-1)[flat_idx]
    vmin, vmax = float(values.min()), float(values.max())

    for rank, flat in enumerate(flat_idx):
        pi, qi = np.unravel_index(int(flat), T.shape)
        x1c, y1c = center('p', pi)
        x2c, y2c = center('q', qi)
        score = float(T[pi, qi])
        strength = (score - vmin) / (vmax - vmin + 1e-8)
        lw = 0.8 + 2.2 * strength
        alpha = 0.38 + 0.55 * strength
        ax.plot([x1c, x2c], [y1c, y2c], '-', lw=lw, alpha=alpha, color=plt.get_cmap('viridis')(0.15 + 0.75 * strength))
        if rank < 3:
            ax.text((x1c + x2c) / 2, (y1c + y2c) / 2, f'{score:.2g}', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.70))


def _make_error_map(pred_np, gt_np):
    pred = (np.asarray(pred_np) > 0).astype(np.uint8)
    gt = (np.asarray(gt_np) > 0).astype(np.uint8)
    while pred.ndim > 2:
        pred = pred[0]
    while gt.ndim > 2:
        gt = gt[0]
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[:] = 28
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = (46, 160, 67)    # TP: green
    out[fp] = (220, 53, 69)    # FP: red
    out[fn] = (48, 116, 214)   # FN: blue
    return out


def save_pma_figure(query_img, pred_mask, gt_mask, pma_vis, save_path,
                    mean, std, palette, num_classes,
                    support_img=None, support_mask=None, topk=6):
    """Save a reviewer-oriented PMA mechanism visualization.

    Main design:
      (a) support+mask, (b) query+GT boundary, (c) prototype response + patch IDs,
      (d) query-side transport confidence overlay, (e) top-k patch correspondences,
      (f) labeled transport plan, (g) prediction, (h) TP/FP/FN error map.

    Important: `query_align_patch_up` is a visualization-only diagnostic obtained by
    reducing the transport plan over the guidance-patch axis. It is NOT used by the
    model forward pass, so this figure does not alter or re-evaluate the model.
    """
    if pma_vis is None:
        raise ValueError('PMA visualization dictionary is empty')

    from matplotlib.patches import Patch

    query_np = denorm_image(query_img, mean, std)

    # Evaluation can optionally resize logits/labels back to the original image
    # size (`ori_resize=True`) while `query_img` remains at the network input
    # resolution. For a spatially meaningful overlay, map the discrete GT and
    # prediction back to the displayed query resolution using nearest-neighbor
    # interpolation. This is visualization-only and does NOT affect metrics.
    query_hw = query_np.shape[:2]
    gt_np = _resize_label_map(gt_mask, query_hw)
    pred_np = _resize_label_map(pred_mask, query_hw)

    # Determine patch grid from actual pooled tensors.
    q_patch = pma_vis['q_patch']
    if torch.is_tensor(q_patch):
        q_patch_t = q_patch.detach().cpu()
        while q_patch_t.dim() > 4 and q_patch_t.size(0) == 1:
            q_patch_t = q_patch_t.squeeze(0)
        if q_patch_t.dim() == 4:
            h_patch, w_patch = int(q_patch_t.shape[-2]), int(q_patch_t.shape[-1])
        else:
            h_patch, w_patch = pma_vis.get('patch_grid_hw', (1, 1))
    else:
        h_patch, w_patch = pma_vis.get('patch_grid_hw', (1, 1))
    if torch.is_tensor(h_patch):
        h_patch = int(h_patch.item())
    if torch.is_tensor(w_patch):
        w_patch = int(w_patch.item())

    prototype_map = _normalize_vis_map(pma_vis['guidance_feat'])
    transport = pma_vis['transport_plan']
    if torch.is_tensor(transport):
        transport = transport.detach().float().cpu().numpy()
    transport = np.asarray(transport)
    while transport.ndim > 2 and transport.shape[0] == 1:
        transport = transport[0]

    if 'query_align_patch_up' in pma_vis:
        query_conf = _normalize_vis_map(pma_vis['query_align_patch_up'])
    else:
        # Backward-compatible fallback: T=[prototype, query], reduce prototype axis.
        q_conf = transport.max(axis=0).reshape(h_patch, w_patch)
        query_conf = _resize_vis_map(q_conf, pma_vis['base_feat'].shape[-2:])

    query_overlay, query_conf_resized = _overlay_heatmap(query_np, query_conf, alpha=0.48, cmap_name='magma')

    fig = plt.figure(figsize=(16.0, 8.4))
    gs = fig.add_gridspec(2, 4, wspace=0.16, hspace=0.24)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    # (a) Support + mask (or prototype-only fallback).
    ax = axes[0]
    if support_img is not None:
        supp_np = denorm_image(support_img, mean, std)
        ax.imshow(supp_np)
        if support_mask is not None:
            sm = _resize_label_map(support_mask, supp_np.shape[:2])
            ax.imshow(sm, cmap='Reds', alpha=0.20, vmin=0, vmax=1)
            _draw_mask_contour(ax, sm, color='white', linewidth=1.2)
        ax.set_title('Support + Mask', fontsize=10)
    else:
        ax.imshow(prototype_map, cmap='viridis')
        ax.set_title('Guidance Prototype', fontsize=10)
    ax.axis('off')

    # (b) Query with GT contour.
    ax = axes[1]
    ax.imshow(query_np)
    _draw_mask_contour(ax, gt_np, color='white', linewidth=1.2)
    ax.set_title('Query (GT boundary)', fontsize=10)
    ax.axis('off')

    # (c) Prototype response + patch IDs.
    ax = axes[2]
    ax.imshow(prototype_map, cmap='viridis')
    _draw_patch_grid(ax, h_patch, w_patch, 'P', prototype_map.shape, label=True)
    ax.set_title('Guidance Prototype Patches', fontsize=10)
    ax.axis('off')

    # (d) Diagnostic query-side confidence overlay.
    ax = axes[3]
    ax.imshow(query_overlay)
    _draw_mask_contour(ax, gt_np, color='white', linewidth=1.1)
    _draw_patch_grid(ax, h_patch, w_patch, 'Q', query_np.shape[:2], label=True)
    ax.set_title('Query-side Transport Confidence*', fontsize=10)
    ax.axis('off')

    # (e) Top-k correspondence schematic.
    _draw_correspondence_schematic(axes[4], transport, h_patch, w_patch, topk=topk)

    # (f) Transport matrix with semantic patch labels.
    ax = axes[5]
    im = ax.imshow(transport, cmap='Blues', aspect='equal')
    ax.set_title('Transport Plan $T$', fontsize=10)
    ax.set_xlabel('Query patch')
    ax.set_ylabel('Prototype patch')
    n = h_patch * w_patch
    if n <= 16:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f'Q{i+1}' for i in ticks], fontsize=6, rotation=45, ha='right')
        ax.set_yticklabels([f'P{i+1}' for i in ticks], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # (g) Prediction.
    ax = axes[6]
    ax.imshow(query_np)
    pred_color = np.zeros((query_hw[0], query_hw[1], 3), dtype=np.uint8)
    pred_color[..., 0] = (pred_np > 0).astype(np.uint8) * 220
    ax.imshow(pred_color, alpha=0.38)
    _draw_mask_contour(ax, gt_np, color='white', linewidth=1.0)
    ax.set_title('Prediction (GT boundary)', fontsize=10)
    ax.axis('off')

    # (h) Error map.
    ax = axes[7]
    ax.imshow(_make_error_map(pred_np, gt_np))
    ax.set_title('Error Map', fontsize=10)
    ax.axis('off')
    ax.legend(
        handles=[
            Patch(facecolor=(46/255,160/255,67/255), label='TP'),
            Patch(facecolor=(220/255,53/255,69/255), label='FP'),
            Patch(facecolor=(48/255,116/255,214/255), label='FN'),
        ],
        loc='lower right', fontsize=7, framealpha=0.72, borderpad=0.3,
        handlelength=1.1, handletextpad=0.35
    )

    fig.suptitle(
        'PMA: Patchwise Transport-Guided Matching and Alignment',
        fontsize=13, y=0.995
    )
    fig.text(
        0.5, 0.008,
        '* Query-side confidence is a visualization-only diagnostic derived from the same transport plan; '
        'it is not an additional model input or inference operation.',
        ha='center', va='bottom', fontsize=8
    )
    # Do not call tight_layout here. The colorbar created for the transport
    # matrix owns an auxiliary Axes without a normal SubplotSpec, which causes
    # Matplotlib's `Axes that are not compatible with tight_layout` warning.
    # Explicit margins are deterministic and leave room for the title/note.
    fig.subplots_adjust(
        left=0.035, right=0.985, bottom=0.085, top=0.925,
        wspace=0.16, hspace=0.24
    )

    # Keep the requested raster output and additionally save a vector PDF for LaTeX.
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    base, ext = os.path.splitext(save_path)
    if ext.lower() != '.pdf':
        fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)

def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Few-Shot Semantic Segmentation / PMA Ablation'
    )

    parser.add_argument('--arch', type=str, default='MFANet')

    # Visualization controls.
    parser.add_argument('--viz', dest='viz', action='store_true', default=None,
                        help='enable qualitative visualization')
    parser.add_argument('--no-viz', dest='viz', action='store_false',
                        help='disable qualitative visualization')
    parser.add_argument('--vis_num', type=int, default=None,
                        help='maximum number of validation samples to visualize')
    parser.add_argument('--eval_only', dest='eval_only', action='store_true',
                        default=None,
                        help='load checkpoint from YAML resume and only validate')

    # PMA matching-strategy ablations.
    parser.add_argument(
        '--pma_mode',
        type=str,
        choices=['concat', 'cosine', 'softmax', 'sinkhorn', 'balanced'],
        default=None,
        help='PMA strategy: concat/cosine/softmax/sinkhorn/balanced'
    )
    parser.add_argument('--pma_patch_size', type=int, default=None,
                        help='patch pooling window; main model uses 7')
    parser.add_argument(
        '--cmad_match_mode',
        type=str,
        choices=['direct', 'cross_attention', 'pma'],
        default=None,
        help=(
            'replace ONLY the two CMAD interaction stages: direct / '
            'cross_attention / pma. The pre-alignment pma_ref is unchanged.'
        )
    )
    parser.add_argument('--pma_epsilon', type=float, default=None,
                        help='matching temperature epsilon; main model uses 0.05')
    parser.add_argument('--pma_sinkhorn_iter', type=int, default=None,
                        help='iterations for lightweight Sinkhorn-style PMA')
    parser.add_argument('--pma_balanced_iter', type=int, default=None,
                        help='iterations for log-domain balanced Sinkhorn reference')
    parser.add_argument(
        '--pma_no_adapter',
        dest='pma_no_adapter',
        action='store_true',
        default=None,
        help='bypass learned confidence adapter and use normalized raw confidence'
    )
    parser.add_argument(
        '--pma_use_adapter',
        dest='pma_no_adapter',
        action='store_false',
        help='use the learned confidence adapter'
    )
    parser.add_argument(
        '--pma_analysis',
        dest='pma_analysis',
        action='store_true',
        default=None,
        help='save per-episode PMA analysis CSV/JSON during validation'
    )
    parser.add_argument(
        '--no-pma-analysis',
        dest='pma_analysis',
        action='store_false',
        help='disable PMA analysis logging'
    )
    parser.add_argument('--pma_run_tag', type=str, default=None,
                        help='tag used in PMA analysis filenames')
    parser.add_argument('--seed', type=int, default=None,
                        help='override YAML manual_seed for repeated runs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='override YAML save_path; useful to isolate ablation runs')
    parser.add_argument('--resume', type=str, default=None,
                        help='override YAML resume checkpoint filename')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='absolute/relative checkpoint path; bypasses snapshot_path lookup')

    parser.add_argument(
        '--config',
        type=str,
        default='config/fold0_resnet50.yaml',
        help='config file'
    )

    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)

    if cli_args.viz is not None:
        cfg.viz = cli_args.viz
    elif not hasattr(cfg, 'viz'):
        cfg.viz = False

    if cli_args.vis_num is not None:
        cfg.vis_num = max(0, cli_args.vis_num)
    elif not hasattr(cfg, 'vis_num'):
        cfg.vis_num = 20

    if cli_args.eval_only is not None:
        cfg.eval_only = cli_args.eval_only
    elif not hasattr(cfg, 'eval_only'):
        cfg.eval_only = False

    # PMA defaults exactly reproduce the current CFMANet implementation.
    defaults = {
        'pma_mode': 'sinkhorn',
        'pma_patch_size': 7,
        'pma_epsilon': 0.05,
        'pma_sinkhorn_iter': 2,
        'pma_balanced_iter': 20,
        'pma_no_adapter': False,
        'pma_analysis': False,
        'cmad_match_mode': 'pma',
    }
    for key, value in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)

    for key in (
        'pma_mode', 'pma_patch_size', 'pma_epsilon',
        'pma_sinkhorn_iter', 'pma_balanced_iter',
        'pma_no_adapter', 'pma_analysis', 'cmad_match_mode'
    ):
        value = getattr(cli_args, key)
        if value is not None:
            setattr(cfg, key, value)

    if cli_args.seed is not None:
        cfg.manual_seed = int(cli_args.seed)

    if cli_args.output_dir is not None:
        cfg.save_path = cli_args.output_dir

    if cli_args.resume is not None:
        cfg.resume = cli_args.resume
    cfg.checkpoint_path = cli_args.checkpoint_path

    adapter_tag = 'raw' if bool(cfg.pma_no_adapter) else 'adapter'
    default_tag = (
        f"cmad-{cfg.cmad_match_mode}_pre-{cfg.pma_mode}_"
        f"p{int(cfg.pma_patch_size)}_{adapter_tag}_"
        f"s{int(getattr(cfg, 'manual_seed', 0))}"
    )
    cfg.pma_run_tag = cli_args.pma_run_tag or getattr(
        cfg, 'pma_run_tag', default_tag
    )
    if not cfg.pma_run_tag:
        cfg.pma_run_tag = default_tag

    return cfg

def get_model(args):
    model = mfanet(args)
    optimizer = model.get_optim(args, args.base_lr)
    freeze_modules(model)
    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    checkpoint_path = getattr(args, 'checkpoint_path', None)
    if checkpoint_path or args.resume:
        resume_path = checkpoint_path if checkpoint_path else osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:
                # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))
    return model, optimizer


def main_process():
    return True


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    logger.info(" -------------------- creating model-------------------")
    model, optimizer = get_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model created: %s | total params %.2fM | trainable params %.2fM',
                model.__class__.__name__, total_params / 1e6, trainable_params / 1e6)
    logger.info(
        'CMAD interaction ablation: cmad_match_mode=%s | pre-PMA pma_mode=%s | patch=%s',
        getattr(args, 'cmad_match_mode', 'pma'),
        getattr(args, 'pma_mode', 'sinkhorn'),
        getattr(args, 'pma_patch_size', 7),
    )

    # ---------------------- DATASET ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # Train
    train_transform = [
        transform.Resize(args.train_h, args.train_w),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, \
                         data_list=args.train_list, transform=train_transform, mode='train', data_set=args.data_set)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    # #解决样本不平衡
    # #读取train_list文件解析每个样本的类别标签
    # labels = []
    # if os.path.exists(args.train_list):
    #     with open(args.train_list, 'r') as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             if len(parts) >= 2:
    #                 labels.append(int(parts[1]))  # 获取类别ID
    # else:
    #     raise FileNotFoundError(f"训练列表文件未找到: {args.train_list}")
    # #计算每个类别的样本数量
    # class_counts = Counter(labels)
    # num_samples = len(labels)
    # num_classes = len(class_counts)
    #
    # # 打印确认分布
    # if main_process():
    #     logger.info(f"Dataset Class Distribution: {dict(class_counts)}")
    # #计算类别权重
    # #样本越少的类，权重越大
    # class_weights = {cls: num_samples / (num_classes * count) for cls, count in class_counts.items()}
    # #为每个样本生成对应的权重列表
    # sample_weights = [class_weights[label] for label in labels]
    # #创建加权采样器
    # #replacement=True 表示允许重复采样少数类，这是解决不平衡的关键
    # train_sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(train_data),
    #     replacement=True
    # )
    # #使用了sampler后，shuffle必须为False
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
    #                                            shuffle=False,  # 这里必须改为 False
    #                                            num_workers=args.workers, pin_memory=True,
    #                                            sampler=train_sampler, drop_last=True)

    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(h=args.val_size, w=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        val_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list,
                           transform=val_transform, mode='val', data_set=args.data_set)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # 仅评估/可视化模式：需要 YAML 中 evaluate=True，并通过 resume 指定权重。
    if getattr(args, 'eval_only', False):
        if not args.evaluate:
            raise ValueError('--eval_only requires evaluate: True in the YAML configuration')
        if not (getattr(args, 'checkpoint_path', None) or args.resume):
            raise ValueError(
                '--eval_only requires --checkpoint_path or a checkpoint filename in YAML field TRAIN.resume'
            )
        validate(val_loader, model)
        if main_process():
            writer.flush()
            writer.close()
        return

    # ---------------------- TRAINVAL ----------------------
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m
    best_miou = 0.
    best_FBiou = 0.
    best_piou = 0.
    best_epoch = 0
    val_num = 0
    max_iou = 0.
    max_fbiou = 0
    best_epoch = 0
    filename = 'SD-AANet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        # loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train, mani_loss_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            writer.add_scalar('mani_loss_train', mani_loss_train, epoch_log)

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, class_iou_class = validate(val_loader, model)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                if class_miou > max_iou:
                    max_iou = class_miou
                    best_epoch = epoch
                    if os.path.exists(filename):
                        os.remove(filename)
                    filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, filename)
                if mIoU_val > max_fbiou:
                    max_fbiou = mIoU_val
                logger.info(
                    'Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_iou, max_fbiou))

                #写入带轮数标记的log文件
                log_file_path = os.path.join(args.save_path, 'val_results_log.txt')
                prefix = "[{}/{}] ".format(epoch, args.epochs - 1)
                log_line = "meanIoU---Val result: mIoU {:.4f}. ".format(class_miou)
                for i in range(len(class_iou_class)):
                    log_line += "Class_{} Result: iou {:.4f}. ".format(i + 1, class_iou_class[i])
                log_line += "FBIoU---Val result: mIoU {:.4f}".format(mIoU_val)
                with open(log_file_path, 'a') as f:
                    f.write(prefix + log_line + '\n')

        if main_process():
            writer.flush()

    if main_process():
        writer.flush()
        writer.close()


def train(train_loader, model, optimizer, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<')
    batch_time = AverageMeter()
    data_time = AverageMeter()

    manifold_loss_meter = AverageMeter()

    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    print('Warmup: {}'.format(args.warmup))

    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, total_loss, seg_loss, m_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        # output, main_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)

        loss = total_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        #更新Manifold Loss 监控器
        manifold_loss_meter.update(m_loss.item(), n)
        main_loss_meter.update(seg_loss.item(), n)  # 分割损失
        loss_meter.update(loss.item(), n)  # 总损失
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if (i + 1) % args.print_freq == 0 and main_process():
        #     logger.info('Epoch: [{}/{}][{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Remain {remain_time} '
        #                 'MainLoss {main_loss_meter.val:.4f} '
        #                 'Loss {loss_meter.val:.4f} '
        #                 'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs - 1, i + 1, len(train_loader),
        #                                                   batch_time=batch_time, data_time=data_time,
        #                                                   remain_time=remain_time, main_loss_meter=main_loss_meter,
        #                                                   loss_meter=loss_meter, accuracy=accuracy))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'SegLoss {main_loss_meter.val:.4f} '
                        'ManiLoss {manifold_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs - 1, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          manifold_loss_meter=manifold_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>end train<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #return main_loss_meter.avg, mIoU, mAcc, allAcc
    return main_loss_meter.avg, mIoU, mAcc, allAcc, manifold_loss_meter.avg

# 新增辅助函数
def get_palette(num_cls):
    """获取可视化调色板"""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def img_to_svg_base64(img_np):
    """
    将 numpy 图像数组转换为 base64 编码的字符串，用于嵌入 SVG。
    支持 RGB 或灰度图。
    """
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def create_svg_with_images(imgs_data, titles, save_path):
    """
    创建一个包含多张图片的 SVG 文件。
    imgs_data: list of numpy arrays (H, W, 3)
    titles: list of strings
    save_path: 输出路径
    """
    num_imgs = len(imgs_data)
    if num_imgs == 0:
        return

    h, w = imgs_data[0].shape[:2]

    # 定义SVG画布宽度和高度，设置足够大的宽度以容纳图片
    svg_width = num_imgs * (w + 20) + 20
    svg_height = h + 60

    svg_root = Element('svg')
    svg_root.set('xmlns', 'http://www.w3.org/2000/svg')
    svg_root.set('width', str(svg_width))
    svg_root.set('height', str(svg_height))

    for i, img in enumerate(imgs_data):
        x_offset = 10 + i * (w + 20)

        # 添加文本标题
        text_elem = SubElement(svg_root, 'text')
        text_elem.set('x', str(x_offset + w / 2))
        text_elem.set('y', str(h + 40))
        text_elem.set('text-anchor', 'middle')
        text_elem.set('font-size', '16')
        text_elem.set('font-family', 'Arial')
        text_elem.text = titles[i]

        # 添加图片
        image_elem = SubElement(svg_root, 'image')
        image_elem.set('x', str(x_offset))
        image_elem.set('y', '10')
        image_elem.set('width', str(w))
        image_elem.set('height', str(h))
        image_elem.set('href', 'data:image/png;base64,' + img_to_svg_base64(img))

    # 保存文件
    xml_str = minidom.parseString(tostring(svg_root)).toprettyxml(indent="   ")
    with open(save_path, 'w') as f:
        f.write(xml_str)


def extract_subcls_indices(subcls, batch_size):
    """将 DataLoader 整理后的 subcls 转成长度为 batch_size 的一维整数数组。

    兼容以下常见形式：
      1. Tensor[B]
      2. Tensor[B, shot]
      3. list[Tensor[B]]（Dataset 返回 subcls_list 时，default_collate 的结果）
      4. list[int]、list[list[int]] 或 numpy.ndarray
    """
    if batch_size <= 0:
        return np.empty((0,), dtype=np.int64)

    def _from_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 0:
            return np.empty((0,), dtype=np.int64)
        if value.ndim == 0:
            return np.full((batch_size,), int(value.item()), dtype=np.int64)
        if value.ndim == 1:
            return value.numpy().astype(np.int64, copy=False).reshape(-1)
        if value.shape[0] == batch_size:
            return value.reshape(batch_size, -1)[:, 0].numpy().astype(np.int64, copy=False)
        if value.shape[-1] == batch_size:
            return value.reshape(-1, batch_size)[0].numpy().astype(np.int64, copy=False)
        return value.reshape(-1).numpy().astype(np.int64, copy=False)

    if torch.is_tensor(subcls):
        indices = _from_tensor(subcls)
    elif isinstance(subcls, np.ndarray):
        array = np.asarray(subcls)
        if array.ndim == 0:
            indices = np.full((batch_size,), int(array.item()), dtype=np.int64)
        elif array.ndim == 1:
            indices = array.astype(np.int64, copy=False).reshape(-1)
        elif array.shape[0] == batch_size:
            indices = array.reshape(batch_size, -1)[:, 0].astype(np.int64, copy=False)
        elif array.shape[-1] == batch_size:
            indices = array.reshape(-1, batch_size)[0].astype(np.int64, copy=False)
        else:
            indices = array.astype(np.int64, copy=False).reshape(-1)
    elif isinstance(subcls, (list, tuple)):
        if len(subcls) == 0:
            indices = np.empty((0,), dtype=np.int64)
        elif torch.is_tensor(subcls[0]):
            # Dataset 返回 [cls] 或 [cls, cls, ...] 时，default_collate 会得到
            # list[Tensor[B]]。每个 shot 都属于同一查询类别，因此取第一个即可。
            indices = _from_tensor(subcls[0])
        else:
            array = np.asarray(subcls)
            if array.ndim == 0:
                indices = np.full((batch_size,), int(array.item()), dtype=np.int64)
            elif array.ndim == 1:
                indices = array.astype(np.int64, copy=False).reshape(-1)
            elif array.shape[0] == batch_size:
                indices = array.reshape(batch_size, -1)[:, 0].astype(np.int64, copy=False)
            else:
                # 兼容 shot-major 的嵌套列表，取第一个 shot 的 batch 类别。
                indices = array.reshape(array.shape[0], -1)[0].astype(np.int64, copy=False)
    else:
        indices = np.full((batch_size,), int(subcls), dtype=np.int64)

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 1 and batch_size > 1:
        indices = np.repeat(indices, batch_size)
    if indices.size < batch_size:
        raise ValueError(
            f'Cannot extract {batch_size} class indices from subcls; '
            f'type={type(subcls).__name__}, extracted={indices.tolist()}'
        )
    return indices[:batch_size]

def _normalized_centroid(mask):
    """Return foreground centroid in normalized (x, y) coordinates."""
    if torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    fg = arr == 1
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return None
    h, w = fg.shape
    x = float(xs.mean()) / max(w - 1, 1)
    y = float(ys.mean()) / max(h - 1, 1)
    return np.array([x, y], dtype=np.float64)


def _support_query_displacement(support_masks, query_mask):
    """Mean/max normalized centroid displacement across K support shots."""
    q_center = _normalized_centroid(query_mask)
    if q_center is None:
        return float('nan'), float('nan')

    if torch.is_tensor(support_masks):
        supp = support_masks.detach().cpu()
    else:
        supp = np.asarray(support_masks)

    distances = []
    shot_count = int(supp.shape[0]) if getattr(supp, 'ndim', 0) >= 3 else 1
    for shot_idx in range(shot_count):
        sm = supp[shot_idx] if shot_count > 1 else supp
        s_center = _normalized_centroid(sm)
        if s_center is None:
            continue
        # Coordinates are in [0, 1]^2; divide by sqrt(2) so the maximum
        # possible corner-to-corner displacement is 1.
        d = float(np.linalg.norm(s_center - q_center) / math.sqrt(2.0))
        distances.append(d)

    if not distances:
        return float('nan'), float('nan')
    return float(np.mean(distances)), float(np.max(distances))


def _binary_episode_iou(pred, target, ignore_label=255):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred)
    target = np.asarray(target)

    valid = target != ignore_label
    if not np.any(valid):
        return float('nan'), float('nan')

    pred = pred[valid]
    target = target[valid]

    fg_inter = np.logical_and(pred == 1, target == 1).sum()
    fg_union = np.logical_or(pred == 1, target == 1).sum()
    bg_inter = np.logical_and(pred == 0, target == 0).sum()
    bg_union = np.logical_or(pred == 0, target == 0).sum()

    fg_iou = float(fg_inter / fg_union) if fg_union > 0 else float('nan')
    bg_iou = float(bg_inter / bg_union) if bg_union > 0 else float('nan')
    fb_iou = float(np.nanmean([fg_iou, bg_iou]))
    return fg_iou, fb_iou


def _binary_auroc(scores, labels):
    """Dependency-free AUROC. Returns NaN when only one class is present."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float('nan')

    order = np.argsort(-scores, kind='mergesort')
    scores = scores[order]
    labels = labels[order]

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    distinct = np.where(np.diff(scores))[0]
    threshold_idx = np.r_[distinct, labels.size - 1]

    tpr = np.r_[0.0, tps[threshold_idx] / pos]
    fpr = np.r_[0.0, fps[threshold_idx] / neg]
    return float(np.trapz(tpr, fpr))


def _confidence_statistics(query_conf, query_target, ignore_label=255):
    """FG/BG confidence and AUROC for visualization-only query confidence."""
    if query_conf is None:
        return (float('nan'),) * 4

    if torch.is_tensor(query_conf):
        conf = query_conf.detach().float()
    else:
        conf = torch.as_tensor(query_conf, dtype=torch.float32)

    while conf.dim() > 4 and conf.size(0) == 1:
        conf = conf.squeeze(0)
    if conf.dim() == 2:
        conf = conf[None, None]
    elif conf.dim() == 3:
        conf = conf[None] if conf.size(0) != 1 else conf.unsqueeze(0)
    elif conf.dim() != 4:
        return (float('nan'),) * 4

    if torch.is_tensor(query_target):
        target = query_target.detach().cpu()
    else:
        target = torch.as_tensor(query_target)

    while target.dim() > 2:
        target = target[0]
    h, w = int(target.shape[-2]), int(target.shape[-1])
    conf = F.interpolate(
        conf.cpu(), size=(h, w), mode='bilinear', align_corners=False
    )[0, 0].numpy()

    target_np = target.numpy()
    valid = target_np != ignore_label
    fg = (target_np == 1) & valid
    bg = (target_np == 0) & valid

    fg_conf = float(conf[fg].mean()) if np.any(fg) else float('nan')
    bg_conf = float(conf[bg].mean()) if np.any(bg) else float('nan')
    gap = fg_conf - bg_conf if np.isfinite(fg_conf) and np.isfinite(bg_conf) else float('nan')
    auroc = _binary_auroc(conf[valid], (target_np[valid] == 1).astype(np.uint8))
    return fg_conf, bg_conf, gap, auroc


def _class_balanced_miou(rows):
    by_class = {}
    for row in rows:
        value = row.get('fg_iou', float('nan'))
        cls = row.get('class_idx', -1)
        if np.isfinite(value):
            by_class.setdefault(cls, []).append(float(value))
    if not by_class:
        return float('nan')
    class_means = [np.mean(v) for v in by_class.values() if len(v) > 0]
    return float(np.mean(class_means)) if class_means else float('nan')


def _write_pma_analysis(rows, summary):
    if not rows:
        return

    analysis_dir = os.path.join(args.result_path, 'pma_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    tag = str(getattr(args, 'pma_run_tag', 'pma')).replace(os.sep, '_')

    fieldnames = [
        'episode_id', 'seed', 'class_idx', 'cmad_match_mode', 'pma_mode', 'patch_size',
        'epsilon', 'sinkhorn_iter', 'balanced_iter', 'use_adapter',
        'disp_mean', 'disp_max', 'fg_iou', 'episode_fb_iou',
        'fg_conf', 'bg_conf', 'conf_gap', 'alignment_auroc'
    ]
    csv_path = os.path.join(analysis_dir, f'episodes_{tag}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in rows:
            writer_csv.writerow({k: row.get(k, '') for k in fieldnames})

    valid_disp = np.array(
        [r['disp_mean'] for r in rows if np.isfinite(r['disp_mean'])],
        dtype=np.float64
    )
    mismatch_rows = []
    if valid_disp.size >= 3:
        q1, q2 = np.quantile(valid_disp, [1.0 / 3.0, 2.0 / 3.0])
        groups = [
            ('Low', lambda d: d <= q1),
            ('Medium', lambda d: q1 < d <= q2),
            ('High', lambda d: d > q2),
        ]
        for name, predicate in groups:
            subset = [
                r for r in rows
                if np.isfinite(r['disp_mean']) and predicate(r['disp_mean'])
            ]
            if not subset:
                continue
            mismatch_rows.append({
                'group': name,
                'count': len(subset),
                'disp_lower': float(np.min([r['disp_mean'] for r in subset])),
                'disp_upper': float(np.max([r['disp_mean'] for r in subset])),
                'mean_disp': float(np.mean([r['disp_mean'] for r in subset])),
                'class_balanced_miou': _class_balanced_miou(subset),
                'mean_episode_iou': float(np.nanmean([r['fg_iou'] for r in subset])),
                'mean_episode_fb_iou': float(np.nanmean([r['episode_fb_iou'] for r in subset])),
                'mean_conf_gap': float(np.nanmean([r['conf_gap'] for r in subset])),
                'mean_alignment_auroc': float(np.nanmean([r['alignment_auroc'] for r in subset])),
            })

        mismatch_path = os.path.join(analysis_dir, f'mismatch_{tag}.csv')
        with open(mismatch_path, 'w', newline='', encoding='utf-8-sig') as f:
            names = list(mismatch_rows[0].keys()) if mismatch_rows else []
            if names:
                w = csv.DictWriter(f, fieldnames=names)
                w.writeheader()
                w.writerows(mismatch_rows)

    summary_path = os.path.join(analysis_dir, f'run_summary_{tag}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info('PMA episode analysis saved to: %s', csv_path)
    logger.info('PMA run summary saved to: %s', summary_path)

def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = len(val_loader.dataset.val_class)
    class_intersection_meter = [0.0] * split_gap
    class_union_meter = [0.0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.eval()
    end = time.time()
    val_start = end
    test_batches = len(val_loader)

    value_scale = 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * value_scale
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * value_scale
    palette = get_palette(args.classes)

    do_viz = bool(getattr(args, 'viz', False))
    do_pma_analysis = bool(getattr(args, 'pma_analysis', False))
    vis_limit = int(getattr(args, 'vis_num', 20))
    pma_rows = []
    if main_process():
        logger.info(
            'PMA/CMAD config: pre_pma=%s cmad_match=%s patch=%s eps=%s iter=%s balanced_iter=%s adapter=%s analysis=%s',
            getattr(args, 'pma_mode', 'sinkhorn'),
            getattr(args, 'cmad_match_mode', 'pma'),
            getattr(args, 'pma_patch_size', 7),
            getattr(args, 'pma_epsilon', 0.05),
            getattr(args, 'pma_sinkhorn_iter', 2),
            getattr(args, 'pma_balanced_iter', 20),
            not bool(getattr(args, 'pma_no_adapter', False)),
            do_pma_analysis,
        )
    if main_process() and do_viz:
        vis_dir = os.path.join(args.result_path, 'vis_svg')
        fdpg_dir = os.path.join(args.result_path, 'vis_fdpg')
        pma_dir = os.path.join(args.result_path, 'vis_pma')
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(fdpg_dir, exist_ok=True)
        os.makedirs(pma_dir, exist_ok=True)
        logger.info('Qualitative results will be saved under: %s', args.result_path)

    for batch_idx, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
        data_time.update(time.time() - end)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ori_label = ori_label.cuda(non_blocking=True)

        # Keep the network-resolution query mask for centroid/confidence analysis.
        # The official metric path below may resize target/logits back to the
        # original resolution when ori_resize=True.
        target_network = target.detach().clone()

        start_time = time.time()
        with torch.inference_mode():
            need_diag = bool(do_viz or do_pma_analysis)
            if need_diag:
                output_logits, vis_dict = model(
                    s_x=s_input,
                    s_y=s_mask,
                    x=input,
                    y=target,
                    return_vis=do_viz,
                    return_pma_analysis=do_pma_analysis
                )
            else:
                output_logits = model(s_x=s_input, s_y=s_mask, x=input, y=target)
                vis_dict = None
        model_time.update(time.time() - start_time)

        if args.ori_resize:
            # 将同一 batch 中不同原始尺寸统一填充到公共方形尺寸。
            original_sizes = [tuple(ori_label[idx].shape[-2:]) for idx in range(ori_label.size(0))]
            common_side = max(max(h_i, w_i) for h_i, w_i in original_sizes)
            resized_targets = []
            for sample_idx, (ori_h, ori_w) in enumerate(original_sizes):
                canvas = torch.full(
                    (common_side, common_side),
                    fill_value=args.ignore_label,
                    dtype=ori_label.dtype,
                    device=ori_label.device
                )
                canvas[:ori_h, :ori_w] = ori_label[sample_idx, :ori_h, :ori_w]
                resized_targets.append(canvas)
            target = torch.stack(resized_targets, dim=0).long()
            output_logits = F.interpolate(
                output_logits,
                size=(common_side, common_side),
                mode='bilinear',
                align_corners=True
            )

        loss = criterion(output_logits, target).mean()
        pred_mask = output_logits.argmax(dim=1)

        # Per-episode PMA analysis. The segmentation IoU is computed on the same
        # prediction/target tensors used by the official evaluator, whereas
        # displacement and confidence use the network-resolution target so they
        # remain spatially aligned with the PMA features.
        if do_pma_analysis:
            batch_size = pred_mask.size(0)
            subcls_indices_analysis = extract_subcls_indices(subcls, batch_size)
            for sample_idx in range(batch_size):
                disp_mean, disp_max = _support_query_displacement(
                    s_mask[sample_idx],
                    target_network[sample_idx]
                )
                fg_iou, episode_fb_iou = _binary_episode_iou(
                    pred_mask[sample_idx],
                    target[sample_idx],
                    args.ignore_label
                )

                fg_conf = bg_conf = conf_gap = alignment_auroc = float('nan')
                if vis_dict is not None and vis_dict.get('pma') is not None:
                    pma_sample = slice_vis_dict(
                        vis_dict['pma'], sample_idx, batch_size
                    )
                    query_conf = pma_sample.get('query_align_patch_up', None)
                    fg_conf, bg_conf, conf_gap, alignment_auroc = _confidence_statistics(
                        query_conf,
                        target_network[sample_idx],
                        args.ignore_label
                    )

                pma_rows.append({
                    'episode_id': f'{batch_idx:06d}_{sample_idx:02d}',
                    'seed': int(getattr(args, 'manual_seed', 0)),
                    'class_idx': int(subcls_indices_analysis[sample_idx]),
                    'cmad_match_mode': str(getattr(args, 'cmad_match_mode', 'pma')),
                    'pma_mode': str(getattr(args, 'pma_mode', 'sinkhorn')),
                    'patch_size': int(getattr(args, 'pma_patch_size', 7)),
                    'epsilon': float(getattr(args, 'pma_epsilon', 0.05)),
                    'sinkhorn_iter': int(getattr(args, 'pma_sinkhorn_iter', 2)),
                    'balanced_iter': int(getattr(args, 'pma_balanced_iter', 20)),
                    'use_adapter': int(not bool(getattr(args, 'pma_no_adapter', False))),
                    'disp_mean': disp_mean,
                    'disp_max': disp_max,
                    'fg_iou': fg_iou,
                    'episode_fb_iou': episode_fb_iou,
                    'fg_conf': fg_conf,
                    'bg_conf': bg_conf,
                    'conf_gap': conf_gap,
                    'alignment_auroc': alignment_auroc,
                })

        # 先保存可视化，target 仍保持为 Tensor，避免 numpy 覆盖错误。
        if main_process() and do_viz and vis_dict is not None:
            batch_size = pred_mask.size(0)
            for sample_idx in range(batch_size):
                save_idx = batch_idx * args.batch_size_val + sample_idx
                if save_idx >= vis_limit:
                    continue

                try:
                    # 原有 Query/Support/GT/Pred 总览 SVG
                    query_np = denorm_image(input[sample_idx], mean, std)
                    gt_color = colorize_mask(
                        target[sample_idx].detach().cpu().numpy(), palette, args.classes
                    )
                    pred_color = colorize_mask(
                        pred_mask[sample_idx].detach().cpu().numpy(), palette, args.classes
                    )

                    all_imgs = [query_np]
                    all_titles = ['Query']
                    for shot_idx in range(s_input.size(1)):
                        all_imgs.append(denorm_image(s_input[sample_idx, shot_idx], mean, std))
                        all_titles.append(f'Support_{shot_idx + 1}')
                        all_imgs.append(colorize_mask(
                            s_mask[sample_idx, shot_idx].detach().cpu().numpy(),
                            palette,
                            args.classes
                        ))
                        all_titles.append(f'Support_Mask_{shot_idx + 1}')
                    all_imgs.extend([gt_color, pred_color])
                    all_titles.extend(['GT', 'Prediction'])
                    create_svg_with_images(
                        all_imgs,
                        all_titles,
                        os.path.join(vis_dir, f'{save_idx:04d}.svg')
                    )

                    fdpg_sample = slice_vis_dict(vis_dict['fdpg'], sample_idx, batch_size)
                    pma_sample = slice_vis_dict(vis_dict['pma'], sample_idx, batch_size)

                    save_fdpg_figure(
                        support_img=vis_dict['support_img'][sample_idx],
                        support_mask=vis_dict['support_mask'][sample_idx],
                        fdpg_vis=fdpg_sample,
                        save_path=os.path.join(fdpg_dir, f'{save_idx:04d}_fdpg.png'),
                        mean=mean,
                        std=std,
                        palette=palette,
                        num_classes=args.classes
                    )
                    if pma_sample.get('transport_plan', None) is not None:
                        save_pma_figure(
                            query_img=input[sample_idx],
                            pred_mask=pred_mask[sample_idx],
                            gt_mask=target[sample_idx],
                            pma_vis=pma_sample,
                            save_path=os.path.join(pma_dir, f'{save_idx:04d}_pma.png'),
                            mean=mean,
                            std=std,
                            palette=palette,
                            num_classes=args.classes,
                            support_img=vis_dict['support_img'][sample_idx],
                            support_mask=vis_dict['support_mask'][sample_idx],
                            topk=6
                        )
                except Exception as exc:
                    # 单张图保存失败不应中断完整评估。
                    logger.exception('Failed to save visualization for sample %d: %s', save_idx, exc)

        # 指标计算放在最后，并且绝不覆盖 target Tensor。
        intersection_t, union_t, new_target_t = intersectionAndUnionGPU(
            pred_mask, target, args.classes, args.ignore_label
        )
        intersection = intersection_t.cpu().numpy()
        union = union_t.cpu().numpy()
        new_target = new_target_t.cpu().numpy()

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(new_target)

        # 支持 Dataset 返回 list、Tensor 或 ndarray；逐样本记录目标类别 IoU。
        subcls_indices = extract_subcls_indices(subcls, pred_mask.size(0))
        for sample_idx, cls_value in enumerate(subcls_indices):
            cls_idx = int(cls_value)
            if 0 <= cls_idx < split_gap:
                sample_intersection_t, sample_union_t, _ = intersectionAndUnionGPU(
                    pred_mask[sample_idx:sample_idx + 1],
                    target[sample_idx:sample_idx + 1],
                    args.classes,
                    args.ignore_label
                )
                sample_intersection = sample_intersection_t.cpu().numpy()
                sample_union = sample_union_t.cpu().numpy()
                class_intersection_meter[cls_idx] += float(sample_intersection[1])
                class_union_meter[cls_idx] += float(sample_union[1])
            else:
                logger.warning('subcls %s is outside [0, %d]', cls_idx, split_gap - 1)

        accuracy = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if main_process() and ((batch_idx + 1) % max(1, args.print_freq) == 0 or batch_idx + 1 == test_batches):
            logger.info(
                'Test: [%d/%d] Data %.3f (%.3f) Batch %.3f (%.3f) '
                'Loss %.4f (%.4f) Accuracy %.4f',
                batch_idx + 1,
                test_batches,
                data_time.val,
                data_time.avg,
                batch_time.val,
                batch_time.avg,
                loss_meter.val,
                loss_meter.avg,
                accuracy
            )

    val_time = time.time() - val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    for cls_idx in range(split_gap):
        cls_iou = class_intersection_meter[cls_idx] / (class_union_meter[cls_idx] + 1e-10)
        class_iou_class.append(cls_iou)
    class_miou = float(np.mean(class_iou_class)) if class_iou_class else 0.0

    logger.info('meanIoU---Val result: mIoU %.4f.', class_miou)
    for cls_idx, cls_iou in enumerate(class_iou_class):
        logger.info('Class_%d Result: iou %.4f.', cls_idx + 1, cls_iou)
    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc %.4f/%.4f/%.4f.', mIoU, mAcc, allAcc)
        for cls_idx in range(args.classes):
            logger.info(
                'Class_%d Result: iou/accuracy %.4f/%.4f.',
                cls_idx,
                iou_class[cls_idx],
                accuracy_class[cls_idx]
            )
    if do_pma_analysis and main_process():
        finite_gap = [r['conf_gap'] for r in pma_rows if np.isfinite(r['conf_gap'])]
        finite_auc = [r['alignment_auroc'] for r in pma_rows if np.isfinite(r['alignment_auroc'])]
        summary = {
            'run_tag': str(getattr(args, 'pma_run_tag', 'pma')),
            'seed': int(getattr(args, 'manual_seed', 0)),
            'cmad_match_mode': str(getattr(args, 'cmad_match_mode', 'pma')),
            'pma_mode': str(getattr(args, 'pma_mode', 'sinkhorn')),
            'patch_size': int(getattr(args, 'pma_patch_size', 7)),
            'epsilon': float(getattr(args, 'pma_epsilon', 0.05)),
            'sinkhorn_iter': int(getattr(args, 'pma_sinkhorn_iter', 2)),
            'balanced_iter': int(getattr(args, 'pma_balanced_iter', 20)),
            'use_adapter': bool(not getattr(args, 'pma_no_adapter', False)),
            'num_episodes': int(len(pma_rows)),
            # These are the official metrics already used by the original code.
            'target_class_miou': float(class_miou),
            'fb_iou': float(mIoU),
            'mAcc': float(mAcc),
            'allAcc': float(allAcc),
            'class_iou': [float(x) for x in class_iou_class],
            'mean_episode_fg_iou': float(np.nanmean([r['fg_iou'] for r in pma_rows])) if pma_rows else float('nan'),
            'mean_conf_gap': float(np.mean(finite_gap)) if finite_gap else float('nan'),
            'mean_alignment_auroc': float(np.mean(finite_auc)) if finite_auc else float('nan'),
            'avg_inference_sec_per_batch': float(model_time.avg),
        }
        _write_pma_analysis(pma_rows, summary)

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    logger.info('Total validation time: %.4f s, average inference time: %.4f s/batch',
                val_time, model_time.avg)

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, class_iou_class

if __name__ == '__main__':
    main()
