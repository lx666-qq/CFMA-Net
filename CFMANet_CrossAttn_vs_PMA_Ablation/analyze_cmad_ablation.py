import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def load_episode_files(root, seed=None):
    files = glob.glob(os.path.join(root, '**', 'episodes_*.csv'), recursive=True)
    by_mode = defaultdict(list)
    for path in files:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            mode = (r.get('cmad_match_mode') or '').strip()
            if not mode:
                # Backward-compatible path inference.
                low = path.lower()
                if 'cross_attention' in low:
                    mode = 'cross_attention'
                elif 'direct' in low:
                    mode = 'direct'
                elif 'pma' in low:
                    mode = 'pma'
            rseed = int(float(r.get('seed', 0) or 0))
            if seed is not None and rseed != seed:
                continue
            r['_path'] = path
            r['_mode'] = mode
            by_mode[mode].append(r)

    # Avoid accidental duplication if eval was run more than once.
    dedup = {}
    for mode, rows in by_mode.items():
        tmp = {}
        for r in rows:
            key = (int(float(r.get('seed', 0) or 0)), r.get('episode_id', ''))
            tmp[key] = r
        dedup[mode] = list(tmp.values())
    return dedup


def load_official_summaries(root, seed=None):
    files = glob.glob(os.path.join(root, '**', 'run_summary_*.json'), recursive=True)
    out = {}
    for path in files:
        try:
            d = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            continue
        dseed = int(d.get('seed', 0))
        if seed is not None and dseed != seed:
            continue
        mode = str(d.get('cmad_match_mode', '')).strip()
        if not mode:
            low = path.lower()
            if 'cross_attention' in low:
                mode = 'cross_attention'
            elif 'direct' in low:
                mode = 'direct'
            elif 'pma' in low:
                mode = 'pma'
        d['_path'] = path
        out[mode] = d
    return out


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def bootstrap_diff(a, b, n_boot=5000, seed=2025):
    # Inputs are paired lists: a=PMA, b=Cross-Attention.
    diffs = [x - y for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float('nan'), float('nan'), float('nan'), 0
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return sum(diffs) / n, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--bootstrap', type=int, default=5000)
    args = ap.parse_args()

    episodes = load_episode_files(args.root, args.seed)
    summaries = load_official_summaries(args.root, args.seed)
    out_dir = os.path.join(args.root, 'analysis_results')
    os.makedirs(out_dir, exist_ok=True)

    # 1) Official metrics produced by the original evaluator.
    official_rows = []
    for mode in ('direct', 'cross_attention', 'pma'):
        d = summaries.get(mode)
        if not d:
            continue
        official_rows.append({
            'cmad_match_mode': mode,
            'seed': d.get('seed', ''),
            'target_class_miou': d.get('target_class_miou', ''),
            'fb_iou': d.get('fb_iou', ''),
            'mAcc': d.get('mAcc', ''),
            'allAcc': d.get('allAcc', ''),
            'num_episodes': d.get('num_episodes', ''),
            'source_json': d.get('_path', ''),
        })
    if official_rows:
        write_csv(
            os.path.join(out_dir, 'official_metrics.csv'),
            list(official_rows[0].keys()), official_rows
        )

    # 2) Shared mismatch thresholds defined by PMA episodes, then reused by all methods.
    ref = episodes.get('pma', [])
    disp = sorted(fnum(r.get('disp_mean')) for r in ref if math.isfinite(fnum(r.get('disp_mean'))))
    spatial_rows = []
    if len(disp) >= 3:
        def quantile(vals, q):
            pos = (len(vals) - 1) * q
            lo = int(math.floor(pos)); hi = int(math.ceil(pos))
            if lo == hi: return vals[lo]
            return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
        q1, q2 = quantile(disp, 1/3), quantile(disp, 2/3)
        groups = (
            ('Low', lambda x: x <= q1),
            ('Medium', lambda x: q1 < x <= q2),
            ('High', lambda x: x > q2),
        )
        for mode in ('direct', 'cross_attention', 'pma'):
            rows = episodes.get(mode, [])
            for gname, pred in groups:
                sub = [r for r in rows if math.isfinite(fnum(r.get('disp_mean'))) and pred(fnum(r.get('disp_mean')))]
                vals = [fnum(r.get('fg_iou')) for r in sub if math.isfinite(fnum(r.get('fg_iou')))]
                fb = [fnum(r.get('episode_fb_iou')) for r in sub if math.isfinite(fnum(r.get('episode_fb_iou')))]
                spatial_rows.append({
                    'cmad_match_mode': mode,
                    'group': gname,
                    'count': len(sub),
                    'shared_q1': q1,
                    'shared_q2': q2,
                    'mean_episode_fg_iou': sum(vals)/len(vals) if vals else float('nan'),
                    'mean_episode_fb_iou': sum(fb)/len(fb) if fb else float('nan'),
                })
        write_csv(
            os.path.join(out_dir, 'spatial_mismatch_shared_thresholds.csv'),
            list(spatial_rows[0].keys()), spatial_rows
        )

    # 3) Paired PMA - Cross-Attention episode-wise IoU bootstrap CI.
    pma = {(r.get('episode_id'), r.get('class_idx')): r for r in episodes.get('pma', [])}
    ca = {(r.get('episode_id'), r.get('class_idx')): r for r in episodes.get('cross_attention', [])}
    keys = sorted(set(pma).intersection(ca))
    a = [fnum(pma[k].get('fg_iou')) for k in keys]
    b = [fnum(ca[k].get('fg_iou')) for k in keys]
    mean_d, lo, hi, n = bootstrap_diff(a, b, args.bootstrap, args.seed)
    paired_rows = [{
        'comparison': 'PMA - Cross-Attention',
        'paired_episodes': n,
        'mean_episode_fg_iou_diff': mean_d,
        'bootstrap_95ci_low': lo,
        'bootstrap_95ci_high': hi,
        'bootstrap_samples': args.bootstrap,
    }]
    write_csv(
        os.path.join(out_dir, 'paired_bootstrap.csv'),
        list(paired_rows[0].keys()), paired_rows
    )

    report = os.path.join(out_dir, 'README_RESULTS.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('CMAD Cross-Attention vs PMA analysis\n')
        f.write('===================================\n\n')
        f.write('official_metrics.csv: use these official mIoU/FB-IoU values in the main paper table.\n')
        f.write('spatial_mismatch_shared_thresholds.csv: episode-level mechanism analysis using PMA-defined common tertile thresholds.\n')
        f.write('paired_bootstrap.csv: paired bootstrap CI of episode foreground IoU, not aggregate dataset mIoU.\n')
        f.write('The FG/BG confidence fields in episodes_*.csv come from the unchanged pre-PMA diagnostic; do not use them as Cross-Attention-vs-PMA CMAD evidence.\n')

    print('Analysis written to:', out_dir)
    if official_rows:
        for r in official_rows:
            print(r['cmad_match_mode'], 'mIoU=', r['target_class_miou'], 'FB-IoU=', r['fb_iou'])
    print('PMA - Cross-Attention paired episode IoU:', mean_d, '95% CI=', (lo, hi), 'n=', n)


if __name__ == '__main__':
    main()
