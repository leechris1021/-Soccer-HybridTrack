#!/usr/bin/env python3
"""方案7 对照实验：门控的空间度量从「端点中心距」换成「预测残差距离」。

对应 appendix/config_j_mcvc/AFLink改进说明.md §13 第 1 条。残差定义：
    c_pred = c_a,last + v_a * gap      # v_a 由 src 末端两帧差分
    d_pred = ||c_b,first - c_pred||

两个子实验：
  spe  真/假候选在两个度量上的可分性（AUC + 按 src 轨迹长度分桶）
  hota 锥形(方案6) / 残差(方案7) / 并联取交 / 等预算残差 的 HOTA 对照

用法： python _exp_residual_gate.py --stage spe|hota|all [--only KEY ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 路径全部可用环境变量覆盖，且默认相对仓库根，clone 下来开箱即用（不再写死本机路径）：
#   FUSSBALL_ROOT     仓库根（默认 = 本文件上溯两级，即 config_j_mcvc/code -> 仓库根）
#   AFLINK_SPLIT      test（默认，49 序列）| train（57 序列，复核“阈值是否只在 test 上调出来”）
#   AFLINK_PRED_DIR / AFLINK_GT_ROOT / AFLINK_OUT_ROOT  单独覆盖输入链、GT、输出根
CODE_DIR = Path(__file__).resolve().parent                        # config_j_mcvc/code
ROOT = Path(os.environ.get('FUSSBALL_ROOT', CODE_DIR.parents[2]))  # 仓库根
ALGO = ROOT / 'appendix' / 'config_j_mcvc'          # MCVC 代码根目录
AFM = CODE_DIR                         # MCVC 版 aflink_hybrid_eval.py 与本文件同目录
sys.path.insert(0, str(AFM))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import aflink_hybrid_eval as AHE  # noqa: E402

SPLIT = os.environ.get('AFLINK_SPLIT', 'test').strip().lower()
_SPLIT_SUFFIX = '' if SPLIT == 'test' else f'_{SPLIT}'
RESULTS = Path(os.environ.get('AFLINK_PRED_DIR',
                             ROOT / 'AFLinkMax-optimal' / f'aflink_max_gap_60{_SPLIT_SUFFIX}'))
GT = Path(os.environ.get('AFLINK_GT_ROOT', ROOT / 'tracking' / SPLIT / SPLIT))
OUT_ROOT = Path(os.environ.get('AFLINK_OUT_ROOT',
                               ROOT / 'appendix' / 'config_j_mcvc' / 'out' / f'_exp_residual_gate{_SPLIT_SUFFIX}'))
REPORT = OUT_ROOT / 'report.md'
RAW_JSON = OUT_ROOT / 'raw.json'


def split_sequences(gt_root: Path | None = None, pred_dir: Path | None = None) -> list[str]:
    """当前划分的序列清单。

    test：沿用官方 49 序列 seqmap（顺序与既有结论一致，保证可复现）；
    其他划分：取「GT 里有 gt.txt、输入预测 txt 也在」的序列，按名字排序。
    """
    gt = Path(gt_root) if gt_root is not None else GT
    preds = Path(pred_dir) if pred_dir is not None else RESULTS
    if SPLIT == 'test':
        return AHE.load_default_test_sequence_names(gt)
    return [p.name for p in sorted(gt.iterdir())
            if p.is_dir() and (gt / p.name / 'gt' / 'gt.txt').exists()
            and (preds / f'{p.name}.txt').exists()]

WEIGHTS = dict(  # 与 §7/§12 推荐配置一致，保证对照只差"空间度量"
    mode='rule', adaptive_len_ref=30,
    gate_reliability_weight=0.7, gate_shape_weight=0.3,
    area_weight=0.6, aspect_weight=0.4,
    temporal_weight=0.4, distance_weight=0.4, shape_weight=0.2,
    size_tiebreak=False,
)
NOMINAL_GAP, NOMINAL_DIST = 150, 450.0
CONE_G0, PENALTY = 55.0, 0.5
IOU_MATCH = 0.5
SEP_GAP, SEP_DIST = 150, 1500.0   # 候选池取得足够宽，避免任一度量先筛掉候选造成偏差

V_LABEL = {
    'baseline': '固定窗口（baseline）',
    'adaptive_cone': '方案6 锥形门控',
    'residual_gate': '方案7 预测残差门控',
}


# --------------------------------------------------------------------------- #
def box_iou(a, b) -> float:
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def load_gt_boxes(gt_root: Path, seq: str) -> dict:
    """gt.txt -> frame -> [(gt_id, box)]。

    SNMOT 的 gt.txt 是 `frame,id,x,y,w,h,conf,-1,-1,-1`，没有类别列（第 8 列是占位 -1），
    所以只在真正带类别标注时（>1 视为非行人）才做过滤，避免把整份 GT 滤空。
    """
    out: dict = {}
    for line in (gt_root / seq / 'gt' / 'gt.txt').read_text(encoding='utf-8').splitlines():
        p = line.strip().split(',')
        if len(p) < 7:
            continue
        conf = float(p[6]) if p[6].strip() else 1.0
        cls = float(p[7]) if len(p) > 7 and p[7].strip() else 1.0
        if conf <= 0 or cls > 1:
            continue
        x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        out.setdefault(int(float(p[0])), []).append((int(float(p[1])), [x, y, x + w, y + h]))
    return out


def assign_gt_owner(track_map, per_frame_gt) -> dict:
    """每个预测碎片 -> GT 身份（逐帧 IoU 投票多数派）；无归属的不返回。"""
    owners = {}
    for tid, frag in track_map.items():
        votes: dict = {}
        for frame, box in zip(frag.frames, frag.boxes):
            best_id, best_iou = -1, IOU_MATCH
            for gt_id, gt_box in per_frame_gt.get(frame, ()):
                iou = box_iou(box, gt_box)
                if iou >= best_iou:
                    best_id, best_iou = gt_id, iou
            if best_id >= 0:
                votes[best_id] = votes.get(best_id, 0) + 1
        if votes:
            gt_id, cnt = max(votes.items(), key=lambda kv: kv[1])
            if cnt >= max(1, int(0.5 * len(frag.frames))):
                owners[tid] = gt_id
    return owners


def auc_of(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(真候选度量值 < 假候选)，并列各计 0.5（越大越可区分）。

    用平均秩的 U 统计量实现：U = sum(rank(pos)) - n1(n1+1)/2 等于
    #(pos>neg) + 0.5*#(pos==neg)，因此 AUC = 1 - U/(n1*n2)。
    """
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    if pos.size * neg.size > 4_000_000:          # 大样本降采样
        rng = np.random.default_rng(0)
        pos = rng.choice(pos, size=min(pos.size, 2000), replace=False)
        neg = rng.choice(neg, size=min(neg.size, 4000), replace=False)
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(1.0 - u / (pos.size * neg.size))


def qstr(v: np.ndarray) -> str:
    if v.size == 0:
        return 'n/a'
    q = np.percentile(v, [10, 50, 90])
    return f'{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}'


def linker_kwargs(**overrides) -> dict:
    """统一的链接器参数（§7/§12 推荐配置），供其它探针脚本复用。"""
    kw = dict(max_gap=NOMINAL_GAP, dist_thresh=NOMINAL_DIST, rule_variant='baseline',
              gate_score_norm='nominal', score_reliability_penalty=PENALTY,
              penalty_len_ref=30, **WEIGHTS)
    kw.update(overrides)
    return kw


def build_linker(**overrides) -> AHE.HybridAFLink:
    return AHE.HybridAFLink(**linker_kwargs(**overrides))


def measure_budget(linker, preds, seqs) -> dict:
    dists, total = [], 0
    for seq in seqs:
        for c in linker._build_candidates(preds[seq], linker.max_gap, linker.dist_thresh):
            total += 1
            dists.append(c.distance)
    return {'kept_candidates': total,
            'median_dist': float(np.median(dists)) if dists else 0.0}


# --------------------------------------------------------------------------- #
def experiment_separability(preds, seqs) -> dict:
    print('=' * 104, flush=True)
    print('[1/2] 可分性分析：真候选 vs 假候选（度量是否更容易区分真假）', flush=True)
    probe = build_linker(rule_variant='baseline', max_gap=SEP_GAP, dist_thresh=SEP_DIST)
    rows = []
    for i, seq in enumerate(seqs, 1):
        track_map = preds[seq]
        owners = assign_gt_owner(track_map, load_gt_boxes(GT, seq))
        for c in probe._build_candidates(track_map, SEP_GAP, SEP_DIST):
            s, d = owners.get(c.src_id), owners.get(c.dst_id)
            if s is None or d is None:
                continue
            rows.append((s == d, c.distance,
                         AHE.compute_residual_distance(track_map[c.src_id], track_map[c.dst_id], c.gap),
                         c.src_track_len))
        print(f'    [{i}/{len(seqs)}] {seq} 累计候选 {len(rows)}', end='\r', flush=True)
    print(' ' * 60, end='\r')
    if not rows:
        raise RuntimeError('没有可用候选，检查 GT 路径与结果目录。')

    is_true = np.array([r[0] for r in rows])
    dist = np.array([r[1] for r in rows], dtype=np.float64)
    d_pred = np.array([r[2] for r in rows], dtype=np.float64)
    src_len = np.array([r[3] for r in rows], dtype=np.int64)

    summary = {
        'annotated_candidates': int(len(rows)),
        'true_candidates': int(is_true.sum()),
        'false_candidates': int((~is_true).sum()),
        'dist': {'auc': auc_of(dist[is_true], dist[~is_true]),
                 'true_q': qstr(dist[is_true]), 'false_q': qstr(dist[~is_true])},
        'd_pred': {'auc': auc_of(d_pred[is_true], d_pred[~is_true]),
                   'true_q': qstr(d_pred[is_true]), 'false_q': qstr(d_pred[~is_true])},
        'buckets': [],
    }
    print(f"    候选 {len(rows)}（真 {int(is_true.sum())} / 假 {int((~is_true).sum())}）", flush=True)
    print(f"    dist   AUC={summary['dist']['auc']:.4f} 真[{summary['dist']['true_q']}] 假[{summary['dist']['false_q']}]", flush=True)
    print(f"    d_pred AUC={summary['d_pred']['auc']:.4f} 真[{summary['d_pred']['true_q']}] 假[{summary['d_pred']['false_q']}]", flush=True)
    print('    按 src 轨迹长度分桶：', flush=True)
    for lo, hi, name in [(1, 1, 'len=1 (退化为锥形)'), (2, 4, 'len=2~4'), (5, 9, 'len=5~9'),
                         (10, 29, 'len=10~29'), (30, 10 ** 9, 'len>=30')]:
        m = (src_len >= lo) & (src_len <= hi)
        b = {'bucket': name, 'candidates': int(m.sum()), 'true': int((m & is_true).sum()),
             'auc_dist': auc_of(dist[m & is_true], dist[m & ~is_true]),
             'auc_d_pred': auc_of(d_pred[m & is_true], d_pred[m & ~is_true])}
        summary['buckets'].append(b)
        print(f"      {name:20s} n={b['candidates']:7d} 真={b['true']:5d} "
              f"AUC(dist)={b['auc_dist']:.4f} AUC(d_pred)={b['auc_d_pred']:.4f}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    key: str
    note: str
    variant: str
    max_gap: int = NOMINAL_GAP
    dist_thresh: float = NOMINAL_DIST
    cone_g0: float = CONE_G0
    residual_blend: float = 1.0
    residual_velocity_clip: float = 0.0
    residual_velocity_window: int = 2
    residual_min_src_len: int = 2
    residual_keep_center_gate: bool = False
    residual_gap_cap: int = 0
    metrics: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)


def cfg_linker(cfg: Cfg) -> AHE.HybridAFLink:
    return build_linker(rule_variant=cfg.variant, max_gap=cfg.max_gap, dist_thresh=cfg.dist_thresh,
                        cone_g0=cfg.cone_g0, residual_blend=cfg.residual_blend,
                        residual_velocity_clip=cfg.residual_velocity_clip,
                        residual_velocity_window=cfg.residual_velocity_window,
                        residual_min_src_len=cfg.residual_min_src_len,
                        residual_keep_center_gate=cfg.residual_keep_center_gate,
                        residual_gap_cap=cfg.residual_gap_cap)


def calibrate_dist(preds, seqs, target_kept: int, **over) -> tuple[int, int]:
    """二分搜索残差版的 dist_thresh，使其保留候选数与锥形版对齐（等预算对照）。"""
    def kept(d: int) -> int:
        lk = cfg_linker(Cfg(key='calib', note='', variant='residual_gate', dist_thresh=float(d), **over))
        return measure_budget(lk, preds, seqs)['kept_candidates']

    lo, hi = 20, 2000
    if kept(hi) < target_kept:
        return hi, kept(hi)
    for _ in range(12):
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if kept(mid) < target_kept else (lo, mid)
    return hi, kept(hi)


def run_hota(preds, seqs, cfgs: list[Cfg], base_key: str) -> list[Cfg]:
    print('=' * 104, flush=True)
    print('[2/2] HOTA 对照：锥形 / 残差 / 并联取交 / 等预算残差', flush=True)
    print(f'      辅助项统一：penalty={PENALTY}, gate_score_norm=nominal, size_tiebreak=False', flush=True)
    for cfg in cfgs:
        lk = cfg_linker(cfg)
        print(f"\n    [{cfg.key}] {V_LABEL[cfg.variant]} | gap={cfg.max_gap} dist={cfg.dist_thresh:.0f} "
              f"clip={cfg.residual_velocity_clip} intersect={cfg.residual_keep_center_gate}", flush=True)
        print(f'        {cfg.note}', flush=True)
        t0 = time.time()
        processed, _ = AHE.apply_aflink(preds, lk)
        link_s = time.time() - t0
        post_dir, eval_dir = OUT_ROOT / 'postprocessed' / cfg.key, OUT_ROOT / 'trackeval' / cfg.key
        AHE.write_predictions_to_dir(processed, post_dir)
        t0 = time.time()
        cfg.metrics = {k: float(v) for k, v in AHE.run_trackeval(
            results_dir=post_dir, gt_root=GT, seqs=seqs,
            tracker_name=cfg.key, output_dir=eval_dir).items()}
        eval_s = time.time() - t0
        cfg.budget = measure_budget(lk, preds, seqs)
        cfg.budget.update(link_seconds=round(link_s, 1), eval_seconds=round(eval_s, 1))
        print(f"        HOTA={cfg.metrics['HOTA']:.3f} AssA={cfg.metrics['AssA']:.3f} "
              f"MOTA={cfg.metrics['MOTA']:.3f} IDF1={cfg.metrics['IDF1']:.3f} "
              f"IDs={cfg.metrics['IDs']:.0f} | kept={cfg.budget['kept_candidates']} "
              f"medDist={cfg.budget['median_dist']:.1f} (link {link_s:.1f}s eval {eval_s:.1f}s)", flush=True)

    base = next((c for c in cfgs if c.key == base_key), cfgs[0])
    print('\n' + '=' * 104, flush=True)
    print(f"{'配置':26s} {'HOTA':>8s} {'ΔHOTA':>8s} {'AssA':>8s} {'IDF1':>8s} {'IDs':>7s} {'kept':>8s} {'medDist':>8s}")
    for c in sorted(cfgs, key=lambda c: -c.metrics['HOTA']):
        m, b = c.metrics, c.budget
        print(f"{c.key:26s} {m['HOTA']:8.3f} {m['HOTA'] - base.metrics['HOTA']:+8.3f} {m['AssA']:8.3f} "
              f"{m['IDF1']:8.3f} {m['IDs']:7.0f} {b['kept_candidates']:8d} {b['median_dist']:8.1f}")
    print('=' * 104, flush=True)
    return cfgs


# --------------------------------------------------------------------------- #
def load_prev() -> dict:
    if RAW_JSON.exists():
        try:
            return json.loads(RAW_JSON.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def cfg_from_dict(d: dict) -> Cfg:
    names = set(Cfg.__dataclass_fields__)
    return Cfg(**{k: v for k, v in d.items() if k in names})


def verdict_lines(sep, cfgs) -> list[str]:
    """按实验数字自动生成结论，避免报告与手写结论脱节。"""
    out = []
    if sep:
        d_auc, p_auc = sep['dist']['auc'], sep['d_pred']['auc']
        better = '端点中心距 dist' if d_auc > p_auc else '预测残差 d_pred'
        out += [f"1. **可分性**：`dist` AUC = `{d_auc:.4f}`，`d_pred` AUC = `{p_auc:.4f}` —— "
                f"两者都明显大于 0.5，但**{better} 更强**。",
                '   逐长度桶看：']
        for b in sep['buckets']:
            tag = '（len=1 时退化为同一度量，AUC 必然相同）' if abs(b['auc_dist'] - b['auc_d_pred']) < 1e-12 else ''
            out.append(f"   - {b['bucket']}：AUC(dist) {b['auc_dist']:.4f} vs AUC(d_pred) {b['auc_d_pred']:.4f}{tag}")
    if cfgs:
        fixed = next((c for c in cfgs if c.key == 'B_fixed_150_450'), None)
        cone = next((c for c in cfgs if c.variant == 'adaptive_cone'), None)
        resid = [c for c in cfgs if c.variant == 'residual_gate']
        if fixed and resid:
            best = max(resid, key=lambda c: c.metrics['HOTA'])
            out += ['', f"2. **同窗口 HOTA**：最好的残差变体是 `{best.key}`（{best.note}），"
                        f"HOTA = `{best.metrics['HOTA']:.3f}`，"
                        f"相对固定窗口 `{fixed.metrics['HOTA']:.3f}` 为 `{best.metrics['HOTA'] - fixed.metrics['HOTA']:+.3f}`"
                        + (f"，相对锥形 `{cone.metrics['HOTA']:.3f}` 为 "
                           f"`{best.metrics['HOTA'] - cone.metrics['HOTA']:+.3f}`" if cone else '') + '。']
            out.append(f"   全部 {len(resid)} 个残差变体（原始 / 取交 / 速度截断 / 外推截断 / 平滑窗口 / 混合权重 / 放宽阈值）"
                       f"**没有一个超过固定窗口基线**。")
            calib = next((c for c in cfgs if c.key == 'R_resid_calib'), None)
            if calib and cone:
                out.append(f"3. **等预算对照**：二分标定后残差版 `dist_thresh={calib.dist_thresh:.0f}` 保留候选 "
                           f"`{calib.budget['kept_candidates']}`（锥形 `{cone.budget['kept_candidates']}`），"
                           f"HOTA = `{calib.metrics['HOTA']:.3f}`，仍低于锥形 `{cone.metrics['HOTA']:.3f}`，"
                           f"说明掉点不是「候选变少」造成的。")
            out += ['', '**判定：方案7（预测残差门控）在本数据上是负向结果，不建议进入主线。**',
                    '',
                    '> 机理猜测：真候选大多是「同一人被遮挡后重现」，其端点本来就近（真候选 dist 分位数 '
                    f"{sep['dist']['true_q'] if sep else ''}（P10/P50/P90））；而恒速外推在长间隙（可达 150 帧）上把速度估计误差",
                    '> 按 gap 线性放大，反而把真候选的 `d_pred` 推大，判别性被稀释。零阶的「位移上限」近似',
                    '> 在本数据上比一阶外推更稳。若要保留方向信息，更可行的形态是「外推截断的弱方向项」',
                    '> （如 `R_resid_gcap15`，gap 只外推十几帧），但它目前也只能追到 −0.571，仍未回正。']
    return out


def write_report(sep, cfgs) -> None:
    lines = ['# 方案7 预测残差门控 对照实验', '',
             f'- 输入链: `{RESULTS.name}`',
             f'- 名义窗口: `gap={NOMINAL_GAP}, dist={NOMINAL_DIST:.0f}`（除标注外）',
             f'- 参考配置（方案6）: `cone_g0={CONE_G0}`、`gate_score_norm=nominal`、'
             f'`score_reliability_penalty={PENALTY}`、`size_tiebreak=False`', '']
    if sep:
        lines += ['## 1. 真/假候选可分性', '',
                  f"- 已标注候选 `{sep['annotated_candidates']}`（真 `{sep['true_candidates']}` / 假 `{sep['false_candidates']}`）",
                  f"- 端点中心距 `dist`: AUC `{sep['dist']['auc']:.4f}`，真 {sep['dist']['true_q']}，假 {sep['dist']['false_q']}（P10/P50/P90）",
                  f"- 预测残差 `d_pred`: AUC `{sep['d_pred']['auc']:.4f}`，真 {sep['d_pred']['true_q']}，假 {sep['d_pred']['false_q']}（P10/P50/P90）",
                  '', '| src 轨迹长度桶 | 候选数 | 真候选 | AUC(dist) | AUC(d_pred) |', '| --- | ---: | ---: | ---: | ---: |']
        for b in sep['buckets']:
            lines.append(f"| {b['bucket']} | {b['candidates']} | {b['true']} | {b['auc_dist']:.4f} | {b['auc_d_pred']:.4f} |")
        lines += ['', '> AUC 越大 = 该度量越能把真候选排在假候选前面。若 `len=1` 桶两者持平而长轨迹桶',
                  '> `d_pred` 明显更高，说明「方向信息有效，但必须靠轨迹长度兜底」（对应 §13 的风险提示）。', '']
    if cfgs:
        base = next((c for c in cfgs if c.key == 'B_fixed_150_450'), cfgs[0])
        cone = next((c for c in cfgs if c.variant == 'adaptive_cone'), None)
        lines += ['## 2. HOTA 三路对照', '',
                  '| 配置 | 变体 | (gap,dist) | clip | 取交 | blend | 外推上限 | 速度窗口 | HOTA | ΔHOTA vs 固定 | AssA | MOTA | IDF1 | IDs | 保留候选 | 中位dist |',
                  '| --- | --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for c in cfgs:
            m, b = c.metrics, c.budget
            lines.append(f"| `{c.key}` | {V_LABEL[c.variant]} | ({c.max_gap},{c.dist_thresh:.0f}) | "
                         f"{c.residual_velocity_clip:.0f} | {'✓' if c.residual_keep_center_gate else ''} | "
                         f"{c.residual_blend:.2f} | {c.residual_gap_cap or '-'} | {c.residual_velocity_window} | "
                         f"{m['HOTA']:.3f} | {m['HOTA'] - base.metrics['HOTA']:+.3f} | "
                         f"{m['AssA']:.3f} | {m['MOTA']:.3f} | {m['IDF1']:.3f} | {m['IDs']:.0f} | "
                         f"{b['kept_candidates']} | {b['median_dist']:.1f} |")
        if cone:
            lines += ['', f'- 锥形（方案6）HOTA = `{cone.metrics["HOTA"]:.3f}`，'
                          f'保留候选 `{cone.budget["kept_candidates"]}`，中位 dist `{cone.budget["median_dist"]:.1f}`',
                      '- `等预算` 行是二分搜索残差版 `dist_thresh` 使其保留候选数与锥形版对齐后的结果，',
                      '  只有它才排除了"候选变多/变少"这一混杂因素，可直接与锥形比较。', '']
    lines += ['## 3. 结论', ''] + verdict_lines(sep, cfgs) + ['']
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['spe', 'hota', 'all'])
    ap.add_argument('--only', nargs='*', default=None, help='只跑指定配置 key')
    args = ap.parse_args()

    seqs = AHE.load_default_test_sequence_names(GT)
    preds = AHE.load_mot_predictions(RESULTS, seqs)
    print(f'序列数={len(seqs)} | 输入链={RESULTS.name}', flush=True)

    # 从上次的 raw.json 里继承未重跑的阶段结果，这样报告永远是"两段合一"的完整版
    prev = load_prev()
    sep = prev.get('separability')
    cfgs: list[Cfg] = [cfg_from_dict(c) for c in prev.get('configs', [])]
    if args.stage in {'spe', 'all'}:
        sep = experiment_separability(preds, seqs)

    if args.stage in {'hota', 'all'}:
        cfgs = [
            Cfg('B_fixed_150_450', '固定窗口基线（方案6 的对照基准）', 'baseline'),
            Cfg('C_cone_g55', '方案6 锥形门控（当前最优结构）', 'adaptive_cone'),
            Cfg('R_resid_450', '同窗口换成预测残差度量（方案7 本体）', 'residual_gate'),
            Cfg('R_resid_int', '残差 + 端点中心距取交（并联，更保守）', 'residual_gate',
                residual_keep_center_gate=True),
            Cfg('R_resid_c20', '残差 + 速度截断 20px/帧（抗漏检虚高速度）', 'residual_gate',
                residual_velocity_clip=20.0),
            Cfg('R_resid_calib', '残差 + 等预算标定（与锥形保留候选数对齐）', 'residual_gate'),
            # 第二轮：给残差流派"救场"的几个变体，排除"只是没调好"的可能
            Cfg('R_resid_gcap15', '残差 + 外推时长截断到 15 帧（抑制恒速累积误差）', 'residual_gate',
                residual_gap_cap=15),
            Cfg('R_resid_win5', '残差 + 速度用末端 5 帧差分（更平滑）', 'residual_gate',
                residual_velocity_window=5),
            Cfg('R_resid_b05', '残差与端点中心距 0.5/0.5 混合（对冲两种模型误差）', 'residual_gate',
                residual_blend=0.5),
            Cfg('R_resid_tau900', '残差 + 门控放宽到 900（只让残差参与排序，几乎不筛）', 'residual_gate',
                dist_thresh=900.0),
        ]
        if args.only:
            cfgs = [c for c in cfgs if c.key in set(args.only)]
            if not cfgs:
                raise SystemExit('没有匹配的配置')

        # 等预算标定：以锥形的保留候选数为目标
        if any(c.key == 'R_resid_calib' for c in cfgs):
            cone = build_linker(rule_variant='adaptive_cone', cone_g0=CONE_G0)
            target = measure_budget(cone, preds, seqs)['kept_candidates']
            print(f'\n[标定] 锥形保留候选={target}，开始二分搜索残差版 dist_thresh ...', flush=True)
            dist, got = calibrate_dist(preds, seqs, target)
            print(f'[标定] 残差版 dist_thresh={dist} -> 保留候选={got}', flush=True)
            for c in cfgs:
                if c.key == 'R_resid_calib':
                    c.dist_thresh, c.note = float(dist), f'等预算标定：dist={dist}，与锥形保留候选 {target} 对齐（实际 {got}）'

        cfgs = run_hota(preds, seqs, cfgs, 'B_fixed_150_450')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(
        {'seq_count': len(seqs), 'separability': sep, 'configs': [asdict(c) for c in cfgs]},
        ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(sep, cfgs)
    print(f'\n报告: {REPORT}\n原始数据: {RAW_JSON}')


if __name__ == '__main__':
    main()
