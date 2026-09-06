import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
import model.vgg as vgg_models
import numpy as np
import math

class GBM(nn.Module):
    def __init__(self, in_channels, out_channels=None, mid_ratio=8):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = max(in_channels // mid_ratio, 16)
        self.pw1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dw = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                            groups=mid_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_channels)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.skip = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        g = self.gate(x)
        out = self.relu(self.bn1(self.pw1(x)))
        out = self.relu(self.bn_dw(self.dw(out)))
        out = self.bn2(self.pw2(out))
        return self.relu(residual + out * g)

class PFE_S(nn.Module):
    def __init__(self, in_channels, out_channels=None, mid_ratio=8, img_size=224):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = max(in_channels // mid_ratio, 16)
        self.img_size = img_size
        self.pw1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dw = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_channels)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.radial_encoder = self._build_radial_encoding(img_size)
        self.radial_proj = nn.Conv2d(2, mid_channels, 1, bias=False)
        self.highlight_detector = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.Sigmoid()
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1,
                                                                                bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _build_radial_encoding(self, img_size):
        y, x = torch.meshgrid(torch.linspace(-1, 1, img_size), torch.linspace(-1, 1, img_size), indexing='ij')
        radial_dist = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0)
        radial_angle = torch.atan2(y, x).unsqueeze(0).unsqueeze(0)
        return nn.Parameter(torch.cat([radial_dist, radial_angle], dim=1), requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = self.skip(x)
        if h != self.img_size or w != self.img_size:
            radial_feat = F.interpolate(self.radial_encoder, size=(h, w), mode='bilinear', align_corners=False)
        else:
            radial_feat = self.radial_encoder
        radial_feat = self.radial_proj(radial_feat.repeat(b, 1, 1, 1))
        highlight_mask = self.highlight_detector(x)
        out = self.relu(self.bn1(self.pw1(x)))
        out = self.relu(self.bn_dw(self.dw(out)))
        out = self.bn2(self.pw2(out))
        gate_input = torch.cat([x, radial_feat], dim=1)
        g = self.gate(gate_input) * (1 - highlight_mask)
        return self.relu(residual + out * g)

        # residual = self.skip(x)
        # g = self.gate(x)
        # out = self.relu(self.bn1(self.pw1(x)))
        # out = self.relu(self.bn_dw(self.dw(out)))
        # out = self.bn2(self.pw2(out))
        # return self.relu(residual + out * g)

class PFE(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.3):
        super().__init__()
        self.embed = nn.Sequential(PFE_S(in_c, out_c), nn.Dropout2d(drop_rate))
    def forward(self, x):
        return self.embed(x)

class MCA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        c_g = channels // self.groups
        assert c_g > 0

        self.softmax = nn.Softmax(-1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c_g, c_g)
        self.conv1x1 = nn.Conv2d(c_g, c_g, 1, bias=False)

        self.dw3x3 = nn.Conv2d(c_g, c_g, 3, padding=1, groups=c_g, bias=False)
        self.pool_local = nn.AdaptiveAvgPool2d((3, 3))
        self.hf_gate = nn.Sequential(
            nn.Conv2d(c_g, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.dw3x3(group_x)
        x_local = self.pool_local(x2)  # [B*g, C/g, 3, 3]
        x_local_ch = x_local.mean(dim=[2, 3], keepdim=True)  # [B*g, C/g, 1, 1]  ★
        x11 = self.softmax(x_local_ch.reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(x_local_ch.reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        hf_mask = self.hf_gate(x2)
        final_weights = weights.sigmoid() * (0.5 + hf_mask)
        return (group_x * final_weights).reshape(b, c, h, w)

class GLCM(nn.Module):
    def __init__(self, d_model, reduction=16):
        super().__init__()
        hidden = max(d_model // reduction, 16)
        self.local_attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model)
        )
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            pool = torch.mean(x_flat, dim=1, keepdim=True)
            attn = self.sigmoid(self.local_attn(x_flat) + self.global_attn(pool))
            return x * attn.transpose(1, 2).reshape(b, c, h, w)

        pool = torch.mean(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.local_attn(x) + self.global_attn(pool))

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -harr_wav_H[0, 0]

        def make_filter(w1, w2):
            f = np.transpose(w1) * w2
            return torch.from_numpy(f).unsqueeze(0).float()

        for name, filt in [('LL', make_filter(harr_wav_L, harr_wav_L)),
                           ('LH', make_filter(harr_wav_L, harr_wav_H)),
                           ('HL', make_filter(harr_wav_H, harr_wav_L)),
                           ('HH', make_filter(harr_wav_H, harr_wav_H))]:
            conv = nn.Conv2d(in_channels, in_channels, kernel_size=2,
                             stride=2, padding=0, bias=False, groups=in_channels)
            conv.weight.data = filt.unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            conv.weight.requires_grad_(False)
            setattr(self, name, conv)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class FDPG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inter_c = channels // 2
        self.channels = channels

        self.gbm = GBM(channels, self.inter_c)
        self.wave_pool = WavePool(self.inter_c)

        self.mca_h = MCA(self.inter_c)
        self.mca_v = MCA(self.inter_c)
        self.mca_d = MCA(self.inter_c)
        self.glcm_low = GLCM(self.inter_c)

        self.dir_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inter_c * 3, 3),
            nn.Softmax(dim=1)
        )

        self.freq_fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.theta = GBM(channels, self.inter_c)
        self.phi = GBM(channels, self.inter_c)
        self.W = nn.Sequential(GBM(self.inter_c, channels), nn.BatchNorm2d(channels))
        self.glcm_global = GLCM(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_vis=False):
        b = x.size(0)
        gbm_feat = self.gbm(x)

        LL, LH, HL, HH = self.wave_pool(gbm_feat)

        hl_enh = self.mca_h(HL)
        lh_enh = self.mca_v(LH)
        hh_enh = self.mca_d(HH)

        target_size = LL.shape[2:]
        hl_up = F.interpolate(hl_enh, size=target_size, mode='bilinear', align_corners=False)
        lh_up = F.interpolate(lh_enh, size=target_size, mode='bilinear', align_corners=False)
        hh_up = F.interpolate(hh_enh, size=target_size, mode='bilinear', align_corners=False)

        dir_cat = torch.cat([hl_up, lh_up, hh_up], dim=1)
        w_dir = self.dir_weight(dir_cat)

        w_h = w_dir[:, 0:1, None, None]
        w_v = w_dir[:, 1:2, None, None]
        w_d = w_dir[:, 2:3, None, None]

        high_freq_feat = w_h * hl_up + w_v * lh_up + w_d * hh_up
        low_freq_feat = self.glcm_low(LL)

        w_high, w_low = self.freq_fusion_weight.softmax(dim=0)
        mca_feat = w_high * high_freq_feat + w_low * low_freq_feat

        original_size = x.shape[2:]
        mca_feat_up = F.interpolate(mca_feat, size=original_size, mode='bilinear', align_corners=False)

        theta_x = self.theta(x)
        phi_x = self.phi(x)

        g_flat = mca_feat_up.view(b, self.inter_c, -1).permute(0, 2, 1)
        theta_flat = theta_x.view(b, self.inter_c, -1).permute(0, 2, 1)
        phi_flat = phi_x.view(b, self.inter_c, -1)

        attn = F.softmax(torch.bmm(theta_flat, phi_flat), dim=-1)
        y = torch.bmm(attn, g_flat).permute(0, 2, 1).contiguous().view(
            b, self.inter_c, *x.shape[2:]
        )

        W_y = self.W(y)
        z = self.gamma * W_y + x
        out = self.glcm_global(z)

        if not return_vis:
            return out

        vis_dict = {
            'input_feat': x.detach(),
            'gbm_feat': gbm_feat.detach(),
            'LL': LL.detach(),
            'LH': LH.detach(),
            'HL': HL.detach(),
            'HH': HH.detach(),
            'hl_enh': hl_enh.detach(),
            'lh_enh': lh_enh.detach(),
            'hh_enh': hh_enh.detach(),
            'high_freq_feat': high_freq_feat.detach(),
            'low_freq_feat': low_freq_feat.detach(),
            'mca_feat': mca_feat.detach(),
            'mca_feat_up': mca_feat_up.detach(),
            'attn': attn.detach(),
            'prototype': out.detach(),
            'dir_weight': w_dir.detach(),
            # [B, 2]，便于验证阶段按 batch 样本安全切片
            'freq_weight': torch.stack([w_high, w_low]).view(1, 2).expand(b, -1).detach()
        }
        return out, vis_dict

class PMA(nn.Module):
    """Unified PMA ablation module.

    Supported modes:
      - concat: direct feature concatenation + GBM (no explicit correspondence)
      - cosine: patchwise cosine-similarity gating
      - softmax: one-sided softmax over the same L2 patch cost
      - sinkhorn: the original lightweight Sinkhorn-style alternating normalization
      - balanced: log-domain balanced Sinkhorn reference

    The final fusion path is shared by all explicit matching modes. When
    ``use_adapter=False`` the transport-derived scalar confidence is min-max
    normalized per sample and broadcast to all channels. This is used only for
    the "raw confidence" internal ablation.
    """

    VALID_MODES = ('concat', 'cosine', 'softmax', 'sinkhorn', 'balanced')

    def __init__(self, in_channels, mid_channels=None, sinkhorn_iter=2,
                 epsilon=0.05, patch_size=7, mode='sinkhorn',
                 use_adapter=True, balanced_iter=20):
        super().__init__()
        mid_channels = mid_channels or in_channels // 4

        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f'Unsupported PMA mode: {mode}. Valid modes: {self.VALID_MODES}')

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.balanced_iter = int(balanced_iter)
        self.epsilon = float(epsilon)
        self.patch_size = int(patch_size)
        self.mode = mode
        self.use_adapter = bool(use_adapter)

        if self.mode == 'concat':
            # Functionally comparable direct-fusion baseline.
            self.concat_fuse = GBM(in_channels * 2, in_channels)
            self.transform_q = None
            self.transform_k = None
            self.adapter = None
        else:
            # Keep the same projections and adapter across cosine/softmax/
            # sinkhorn/balanced so the comparison isolates the matching rule.
            self.transform_q = nn.Sequential(
                GBM(in_channels, mid_channels),
                nn.BatchNorm2d(mid_channels)
            )
            self.transform_k = nn.Sequential(
                GBM(in_channels, mid_channels),
                nn.BatchNorm2d(mid_channels)
            )
            self.adapter = nn.Sequential(
                nn.Conv2d(1, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            )

    @staticmethod
    def _safe_minmax(x, eps=1e-8):
        dims = tuple(range(2, x.dim()))
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _light_sinkhorn(self, K):
        """Original CFMANet normalization; kept exactly for the main PMA."""
        b, n_q, n_k = K.shape
        u = torch.ones(b, n_q, 1, device=K.device, dtype=K.dtype) / max(n_q, 1)
        v = torch.ones(b, n_k, 1, device=K.device, dtype=K.dtype) / max(n_k, 1)
        for _ in range(self.sinkhorn_iter):
            v = v / (K.transpose(1, 2) @ u + 1e-8)
            u = u / (K @ v + 1e-8)
        return u * K * v.transpose(1, 2)

    def _balanced_log_sinkhorn(self, cost):
        """Balanced Sinkhorn in the log domain with uniform marginals."""
        b, n_q, n_k = cost.shape
        log_K = -cost / max(self.epsilon, 1e-8)
        log_a = cost.new_full((b, n_q), -math.log(max(n_q, 1)))
        log_b = cost.new_full((b, n_k), -math.log(max(n_k, 1)))
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for _ in range(self.balanced_iter):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_v = log_b - torch.logsumexp(
                log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2
            )

        log_T = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        return torch.exp(log_T)

    def forward(self, base_feat, guidance_feat, return_vis=False):
        b, c, h, w = base_feat.shape
        if guidance_feat.shape[-2:] != (h, w):
            guidance_feat = F.interpolate(
                guidance_feat, (h, w), mode='bilinear', align_corners=False
            )

        if self.mode == 'concat':
            fused = self.concat_fuse(torch.cat([base_feat, guidance_feat], dim=1))
            if not return_vis:
                return fused
            return fused, {
                'mode': self.mode,
                'base_feat': base_feat.detach(),
                'guidance_feat': guidance_feat.detach(),
                'fused_feat': fused.detach(),
                'transport_plan': None,
                'query_align_patch': None,
                'query_align_patch_up': None,
                'patch_size': 0,
                'patch_grid_hw': (0, 0),
            }

        q = self.transform_q(guidance_feat)
        k = self.transform_k(base_feat)

        # Avoid an invalid pooling window when a very large patch size is used
        # at a small feature resolution.
        ps = max(1, min(self.patch_size, h, w))
        q_patch = F.avg_pool2d(q, kernel_size=ps, stride=ps)
        k_patch = F.avg_pool2d(k, kernel_size=ps, stride=ps)

        b_p, c_p, h_p, w_p = q_patch.shape
        q_flat = q_patch.flatten(2).permute(0, 2, 1)  # guidance/prototype patches
        k_flat = k_patch.flatten(2).permute(0, 2, 1)  # base/query patches

        cost_matrix = torch.cdist(q_flat, k_flat, p=2)
        # Keep the normalization used by the current CFMANet implementation.
        cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)

        if self.mode == 'cosine':
            q_norm = F.normalize(q_flat, p=2, dim=-1)
            k_norm = F.normalize(k_flat, p=2, dim=-1)
            similarity = torch.bmm(q_norm, k_norm.transpose(1, 2))
            # Direct similarity response; no probability normalization.
            transport_plan = ((similarity + 1.0) * 0.5).clamp(0.0, 1.0)

        elif self.mode == 'softmax':
            transport_plan = F.softmax(
                -cost_matrix / max(self.epsilon, 1e-8), dim=2
            )

        elif self.mode == 'balanced':
            transport_plan = self._balanced_log_sinkhorn(cost_matrix)

        else:  # sinkhorn: original lightweight PMA
            K = torch.exp(-cost_matrix / max(self.epsilon, 1e-8))
            transport_plan = self._light_sinkhorn(K)

        # Model-used confidence. The original implementation reduces the
        # query/base candidate dimension and keeps the guidance-grid response.
        align_patch, _ = transport_plan.max(dim=2)
        align_patch = align_patch.reshape(b, 1, h_p, w_p)
        align_patch_up = F.interpolate(
            align_patch, size=(h, w), mode='bilinear', align_corners=False
        )

        if self.use_adapter:
            align_weight = self.adapter(align_patch_up)
        else:
            # Deterministic raw-confidence baseline. Per-sample min-max
            # normalization prevents scale differences between matching rules
            # from trivially collapsing the gate.
            scalar_gate = self._safe_minmax(align_patch_up).clamp(0.0, 1.0)
            align_weight = scalar_gate.expand(-1, c, -1, -1)

        fused = (1 - align_weight) * base_feat + align_weight * guidance_feat

        if not return_vis:
            return fused

        # Diagnostic query-side confidence. transport_plan is indexed as
        # [guidance/prototype patch, base/query patch]. Reducing the guidance
        # axis leaves one score per query patch. This branch is visualization/
        # analysis only and never changes the forward result.
        query_align_patch, _ = transport_plan.max(dim=1)
        query_align_patch = query_align_patch.reshape(b, 1, h_p, w_p)
        query_align_patch_up = F.interpolate(
            query_align_patch, size=(h, w), mode='bilinear', align_corners=False
        )

        vis_dict = {
            'mode': self.mode,
            'base_feat': base_feat.detach(),
            'guidance_feat': guidance_feat.detach(),
            'q_patch': q_patch.detach(),
            'k_patch': k_patch.detach(),
            'cost_matrix': cost_matrix.detach(),
            'transport_plan': transport_plan.detach(),
            'align_patch': align_patch.detach(),
            'align_patch_up': align_patch_up.detach(),
            'align_weight': align_weight.detach(),
            'query_align_patch': query_align_patch.detach(),
            'query_align_patch_up': query_align_patch_up.detach(),
            'fused_feat': fused.detach(),
            'patch_size': ps,
            'patch_grid_hw': (h_p, w_p),
        }
        return fused, vis_dict


class DirectFusion(nn.Module):
    """CMAD interaction baseline without explicit correspondence modeling.

    Support/guidance information is still provided, so this is a stricter
    baseline than simply returning the query feature.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.fuse = GBM(in_channels * 2, in_channels)

    def forward(self, base_feat, guidance_feat):
        if guidance_feat.shape[-2:] != base_feat.shape[-2:]:
            guidance_feat = F.interpolate(
                guidance_feat,
                size=base_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return self.fuse(torch.cat([base_feat, guidance_feat], dim=1))


class PatchCrossAttention(nn.Module):
    """Lightweight patch cross-attention used only for the CMAD replacement ablation.

    Fairness constraints relative to PMA:
      * the same input features are used;
      * the same patch pooling window is used;
      * Q/K width matches PMA's reduced matching width;
      * no Transformer FFN/LayerNorm block is added;
      * no extra V projection is used;
      * the same 1x1-BN-Sigmoid confidence gate style is retained.

    Query tokens come from ``base_feat`` and key/value tokens come from
    ``guidance_feat``. The attention-weighted support feature is projected back
    to the query spatial grid and fused through a confidence gate.
    """
    def __init__(self, in_channels, mid_channels=None, patch_size=7):
        super().__init__()
        mid_channels = mid_channels or in_channels // 4
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.patch_size = int(patch_size)

        self.transform_q = nn.Sequential(
            GBM(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels)
        )
        self.transform_k = nn.Sequential(
            GBM(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels)
        )
        # Keep the fusion-gate capacity comparable with PMA.
        self.adapter = nn.Sequential(
            nn.Conv2d(1, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, base_feat, guidance_feat):
        b, c, h, w = base_feat.shape
        if guidance_feat.shape[-2:] != (h, w):
            guidance_feat = F.interpolate(
                guidance_feat, size=(h, w), mode='bilinear', align_corners=False
            )

        # Standard cross-attention orientation:
        # query <- base/query feature, key/value <- support-guidance feature.
        q = self.transform_q(base_feat)
        k = self.transform_k(guidance_feat)

        ps = max(1, min(self.patch_size, h, w))
        q_patch = F.avg_pool2d(q, kernel_size=ps, stride=ps)
        k_patch = F.avg_pool2d(k, kernel_size=ps, stride=ps)
        v_patch = F.avg_pool2d(guidance_feat, kernel_size=ps, stride=ps)

        _, _, h_p, w_p = q_patch.shape
        q_flat = q_patch.flatten(2).transpose(1, 2)         # [B, Nq, d]
        k_flat = k_patch.flatten(2).transpose(1, 2)         # [B, Nk, d]
        v_flat = v_patch.flatten(2).transpose(1, 2)         # [B, Nk, C]

        scale = float(max(self.mid_channels, 1)) ** -0.5
        logits = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale
        attn = F.softmax(logits, dim=2)                     # [B, Nq, Nk]

        attended = torch.bmm(attn, v_flat)                  # [B, Nq, C]
        attended = attended.transpose(1, 2).contiguous().reshape(
            b, c, h_p, w_p
        )
        attended_up = F.interpolate(
            attended, size=(h, w), mode='bilinear', align_corners=False
        )

        # Query-side confidence from the strongest support correspondence.
        confidence = attn.max(dim=2)[0].reshape(b, 1, h_p, w_p)
        confidence_up = F.interpolate(
            confidence, size=(h, w), mode='bilinear', align_corners=False
        )
        gate = self.adapter(confidence_up)

        return (1.0 - gate) * base_feat + gate * attended_up


def _build_cmad_matcher(mode, dim, pma_kwargs):
    """Construct only the CMAD interaction module; pre-alignment PMA is separate."""
    mode = str(mode).lower()
    if mode == 'pma':
        return PMA(dim, **pma_kwargs)
    if mode == 'cross_attention':
        return PatchCrossAttention(
            dim,
            mid_channels=dim // 4,
            patch_size=int(pma_kwargs.get('patch_size', 7))
        )
    if mode == 'direct':
        return DirectFusion(dim)
    raise ValueError(
        f'Unsupported cmad_match_mode={mode}. '
        "Valid values are: direct, cross_attention, pma"
    )

class SepAtrousConv(nn.Module):
    def __init__(self, in_c, out_c, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=dilation, dilation=dilation,
                      groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class CMAD(nn.Module):
    """Cross-granularity matching and alignment decoder.

    ``match_mode`` changes ONLY the two interaction stages inside CMAD.
    The pre-alignment ``CFMANet.pma_ref`` remains the original PMA, which makes
    Cross-Attention-vs-PMA a clean decoder-level replacement ablation.
    """
    VALID_MATCH_MODES = ('direct', 'cross_attention', 'pma')

    def __init__(self, dim=256, drop_rate=0.3, pma_kwargs=None, match_mode='pma'):
        super().__init__()
        pma_kwargs = dict(pma_kwargs or {})
        self.match_mode = str(match_mode).lower()
        if self.match_mode not in self.VALID_MATCH_MODES:
            raise ValueError(
                f'Unsupported CMAD match mode: {self.match_mode}. '
                f'Valid modes: {self.VALID_MATCH_MODES}'
            )

        self.mca = MCA(dim)
        self.pma_fusion = _build_cmad_matcher(self.match_mode, dim, pma_kwargs)
        self.dropout = nn.Dropout2d(drop_rate)
        self.up1 = nn.Sequential(
            GBM(dim, dim),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up2 = nn.Sequential(
            GBM(dim, dim),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.low_fusion = GBM(dim * 2, dim)
        self.pma_high = _build_cmad_matcher(self.match_mode, dim, pma_kwargs)
        self.ms_conv = nn.ModuleList([
            SepAtrousConv(dim, dim // 4, dilation=2 ** i)
            for i in range(3)
        ])
        self.fusion = GBM(dim + 3 * (dim // 4), dim)
        self.detail_branch = nn.Sequential(
            GBM(dim, dim), MCA(dim), GBM(dim, dim)
        )
        self.semantic_branch = nn.Sequential(GBM(dim, dim), GLCM(dim))
        self.head = nn.Sequential(
            GBM(dim, dim // 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim // 2, 2, 1, bias=False)
        )

    def forward(self, query_feat, support_feat, merge_feat, h, w):
        support_enhanced = self.mca(support_feat)
        fused_feat = self.pma_fusion(query_feat, support_enhanced)
        fused_feat = self.dropout(fused_feat)

        x2 = self.up1(merge_feat)
        x4 = self.up2(x2)
        x2_up = F.interpolate(
            x2, size=x4.shape[2:], mode='bilinear', align_corners=False
        )
        x_fused = self.low_fusion(torch.cat([x4, x2_up], dim=1))
        x_fused = self.pma_high(
            x_fused,
            F.interpolate(
                fused_feat,
                size=x_fused.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        )
        multi_feats = [x_fused] + [conv(x_fused) for conv in self.ms_conv]
        decode_feat = self.fusion(torch.cat(multi_feats, dim=1))
        out = self.detail_branch(decode_feat) + self.semantic_branch(decode_feat)
        seg_out = self.head(out)
        if seg_out.shape[2:] != (h, w):
            seg_out = F.interpolate(
                seg_out, size=(h, w), mode='bilinear', align_corners=True
            )
        return seg_out

class GBM_ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(GBM(in_c, out_c), GBM(out_c, out_c))
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class CFMANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        self.classes = args.classes
        self.pretrained = True
        models.BatchNorm = BatchNorm
        self.layers = args.layers

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = self._get_vgg16_layer(vgg16)
        else:
            print(f'>>>>>>>>> Using ResNet {self.layers} <<<<<<<<<')
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 512 + 256 if self.vgg else 1024 + 512

        self.pfe = PFE(fea_dim, reduce_dim)
        self.fdpg = FDPG(reduce_dim)

        # Unified PMA ablation configuration. getattr keeps backward
        # compatibility with existing YAML files/checkpoints.
        self.pma_mode = str(getattr(args, 'pma_mode', 'sinkhorn')).lower()
        self.pma_patch_size = int(getattr(args, 'pma_patch_size', 7))
        self.pma_epsilon = float(getattr(args, 'pma_epsilon', 0.05))
        self.pma_sinkhorn_iter = int(getattr(args, 'pma_sinkhorn_iter', 2))
        self.pma_balanced_iter = int(getattr(args, 'pma_balanced_iter', 20))
        self.pma_use_adapter = not bool(getattr(args, 'pma_no_adapter', False))

        pma_kwargs = dict(
            mode=self.pma_mode,
            patch_size=self.pma_patch_size,
            epsilon=self.pma_epsilon,
            sinkhorn_iter=self.pma_sinkhorn_iter,
            balanced_iter=self.pma_balanced_iter,
            use_adapter=self.pma_use_adapter,
        )
        # IMPORTANT FOR THE CMAD ABLATION:
        # pma_ref is intentionally NOT controlled by cmad_match_mode.
        # It always follows pma_mode (use sinkhorn for the fair main comparison).
        self.cmad_match_mode = str(getattr(args, 'cmad_match_mode', 'pma')).lower()
        self.cmad = CMAD(
            reduce_dim,
            pma_kwargs=pma_kwargs,
            match_mode=self.cmad_match_mode
        )
        self.pma_ref = PMA(reduce_dim, **pma_kwargs)

        self.init_merge = nn.Sequential(GBM(reduce_dim * 3 + 1, reduce_dim), nn.BatchNorm2d(reduce_dim),
                                        nn.ReLU(inplace=True))
        self.feature_enhance = nn.Sequential(GBM_ResBlock(reduce_dim, reduce_dim), MCA(reduce_dim),
                                             GBM_ResBlock(reduce_dim, reduce_dim))

        simple_in = 512 if self.vgg else 2048
        self.simple_proj = nn.Sequential(nn.Conv2d(simple_in, reduce_dim, 1, bias=False), nn.ReLU(inplace=True))
        self.simple_enhance = nn.Sequential(GBM(reduce_dim, reduce_dim), GBM(reduce_dim, reduce_dim))
        self.simple_head = nn.Conv2d(reduce_dim, self.classes, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self._freeze_backbone()

    def _freeze_backbone(self):
        for m in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters(): p.requires_grad = False

    def _get_vgg16_layer(self, model):
        ranges = [range(0, 7), range(7, 14), range(14, 24), range(24, 34), range(34, 43)]
        return tuple(nn.Sequential(*[model.features[i] for i in r]) for r in ranges)

    def get_optim(self, args, LR):
        return torch.optim.AdamW (self.parameters(), lr=LR, weight_decay=args.weight_decay)

    def _compute_spm_prior(self, q4, supp_list, mask_list, target_size):
        corr_list, eps = [], 1e-7
        for i, supp_feat in enumerate(supp_list):
            size = supp_feat.size(2)
            mask = F.interpolate(mask_list[i], size=(size, size), mode='bilinear', align_corners=True)
            masked_supp = supp_feat * mask
            b, c, sp = q4.size(0), q4.size(1), size * size
            q = q4.view(b, c, -1)
            s = masked_supp.view(b, c, -1).permute(0, 2, 1)
            q_norm = torch.norm(q, 2, 1, True)
            s_norm = torch.norm(s, 2, 2, True)
            sim = torch.bmm(s, q) / (torch.bmm(s_norm, q_norm) + eps)
            sim = sim.max(1)[0].view(b, sp)
            sim = (sim - sim.min(1, keepdim=True)[0]) / (
                    sim.max(1, keepdim=True)[0] - sim.min(1, keepdim=True)[0] + eps)
            corr = sim.view(b, 1, size, size)
            corr = F.interpolate(corr, size=target_size, mode='bilinear', align_corners=True)
            corr_list.append(corr)
        return torch.stack(corr_list).mean(dim=0)

    def patch_level_loss(self, query_feat, prototype):
        b, c, h, w = query_feat.shape
        q_flat = query_feat.flatten(2).permute(0, 2, 1)
        p_flat = prototype.flatten(2).permute(0, 2, 1)

        cost_matrix = torch.cdist(q_flat, p_flat, p=2)

        ot_weights = F.softmax(-cost_matrix / 0.1, dim=1)  # [B, HW, 1]

        w_loss = (ot_weights * cost_matrix).sum(dim=1).mean()
        return w_loss

    def forward(self, x, s_x=None, s_y=None, y=None, return_vis=False, return_pma_analysis=False):
        if s_x is None:
            s_x = torch.zeros(x.size(0), self.shot, 3, x.size(2), x.size(3), device=x.device)
            s_y = torch.zeros(x.size(0), self.shot, x.size(2), x.size(3), device=x.device)

        h, w = x.size()[2:]

        with torch.no_grad():
            q0 = self.layer0(x)
            q1 = self.layer1(q0)
            q2 = self.layer2(q1)
            q3 = self.layer3(q2)
            q4 = self.layer4(q3)

            if self.vgg:
                q2 = F.interpolate(q2, size=q3.shape[2:], mode='bilinear', align_corners=True)

            query_cat = torch.cat([q3, q2], dim=1)

        query_feat = self.pfe(query_cat)
        mask_list, proto_list, supp_high_list = [], [], []

        fdpg_vis = None
        first_support_img = None
        first_support_mask = None

        for i in range(self.shot):
            supp_gt = (s_y[:, i] == 1).float().unsqueeze(1)

            with torch.no_grad():
                s0 = self.layer0(s_x[:, i])
                s1 = self.layer1(s0)
                s2 = self.layer2(s1)
                s3 = self.layer3(s2)
                s4 = self.layer4(s3)

                if self.vgg:
                    s2 = F.interpolate(s2, size=s3.shape[2:], mode='bilinear', align_corners=True)

                mask = F.interpolate(supp_gt, size=s3.shape[2:], mode='bilinear', align_corners=True)
                supp_cat = torch.cat([s3, s2], dim=1)

                supp_high_list.append(s4)
                mask_list.append(supp_gt)

            supp_feat = self.pfe(supp_cat)

            if return_vis and i == 0:
                prototype_i, fdpg_vis = self.fdpg(supp_feat * mask, return_vis=True)
                first_support_img = s_x[:, i].detach()
                first_support_mask = supp_gt.detach()
            else:
                prototype_i = self.fdpg(supp_feat * mask)

            # Analysis needs the first support example for centroid/matching
            # metadata, but does not need the full FDPG diagnostic dictionary.
            if return_pma_analysis and i == 0 and first_support_img is None:
                first_support_img = s_x[:, i].detach()
                first_support_mask = supp_gt.detach()

            proto_list.append(prototype_i)

        prototype = torch.stack(proto_list).mean(dim=0) if self.shot > 1 else proto_list[0]

        # 5-shot 时 FDPG 首次调用只对应第一个 support。
        # 将最终平均 prototype 写回可视化字典，保证论文图与 PMA 实际输入一致。
        if return_vis and fdpg_vis is not None:
            fdpg_vis['single_shot_prototype'] = fdpg_vis['prototype']
            fdpg_vis['prototype'] = prototype.detach()

        need_pma_diag = bool(return_vis or return_pma_analysis)
        if need_pma_diag:
            query_enhanced, pma_vis = self.pma_ref(
                query_feat, prototype, return_vis=True
            )
        else:
            query_enhanced = self.pma_ref(query_feat, prototype)
            pma_vis = None

        corr_mask = self._compute_spm_prior(q4, supp_high_list, mask_list, query_feat.shape[2:])
        merge_feat = self.init_merge(torch.cat([query_enhanced, prototype, query_feat, corr_mask], dim=1))
        merge_feat = self.feature_enhance(merge_feat)
        seg_out = self.cmad(query_feat=query_enhanced, support_feat=prototype, merge_feat=merge_feat, h=h, w=w)

        if not self.training:
            seg_out = self.max_pool(seg_out)
            seg_out = -self.max_pool(-seg_out)

            if return_vis or return_pma_analysis:
                vis_dict = {
                    'query_img': x.detach(),
                    'query_feat': query_feat.detach(),
                    'prototype': prototype.detach(),
                    'query_enhanced': query_enhanced.detach(),
                    'corr_mask': corr_mask.detach(),
                    'support_img': first_support_img,
                    'support_mask': first_support_mask,
                    'fdpg': fdpg_vis,
                    'pma': pma_vis,
                    'pma_mode': self.pma_mode,
                }
                return seg_out, vis_dict

            return seg_out

        simple_feat = self.simple_enhance(self.simple_proj(q4))
        simple_out = F.interpolate(self.simple_head(simple_feat), size=(h, w), mode='bilinear', align_corners=True)

        main_loss = self.criterion(seg_out, y.long())
        aux_loss = self.criterion(simple_out, y.long())
        manifold_loss = self.patch_level_loss(query_feat, prototype)
        total_loss = main_loss + 0.3 * aux_loss + 0.1 * manifold_loss

        return seg_out.max(1)[1], total_loss, main_loss, manifold_loss

class mfanet(CFMANet): pass

if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.vgg = False;
            self.layers = 50;
            self.classes = 2;
            self.shot = 1
            self.ignore_label = 255;
            self.momentum = 0.9;
            self.weight_decay = 0.0001;
            self.base_lr = 0.001

    args = Args()
    model = CFMANet(args)
    model.pretrained = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    x = torch.randn(2, 3, 224, 224)
    s_x = torch.randn(2, 1, 3, 224, 224)
    s_y = torch.randint(0, 2, (2, 1, 224, 224)).float()
    y = torch.randint(0, 2, (2, 224, 224)).long()

    model.train()
    output, total_loss, seg_loss, m_loss = model(x=x, s_x=s_x, s_y=s_y, y=y)

    print(f"\n=== Test Results ===")
    print(f"Output shape: {output.shape}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Seg Loss:   {seg_loss.item():.4f}")
    print(f"Mani Loss:  {m_loss.item():.4f}")
    print("CFMANet Test Passed!")