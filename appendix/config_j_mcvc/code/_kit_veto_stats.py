"""方案8（队色否决）的**逐序列**配对统计 + 该实验全部配置的聚合对照。

用途：review4 的 Comment 4 要求「置信区间 / 均值差 / 正态性检验 / 精确或 FDR 校正 p 值」，
Comment 9 又质疑「没有新增方法成分」。要拿方案8 去回应，必须先在 49 条序列上给出配对检验，
而不是只报一个聚合 HOTA（+0.604）。本脚本就是补这个统计。

数据源：`config_j_mcvc/out/_exp_residual_gate[_train]/color_veto/trackeval/<key>/<key>/pedestrian_{summary,detailed}.*`
（由 `_probe_color_team.py --stage pixels --sweep-*` 生成；本仓库只带报告，未附这批 TrackEval 产物）
用法：python _kit_veto_stats.py            # 全量对照 + 三组配对检验（默认 test 划分）
      python _kit_veto_stats.py --ladder   # 只列全部配置的聚合指标
      python _kit_veto_stats.py --train    # 改看 train 划分（57 序列）的复核结果
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

# 划分拿法要放在路径计算之前：--train 或 AFLINK_SPLIT=train 都行（云上直接带环境变量跑）
_argv = sys.argv[1:]
SPLIT = (next((a.split('=', 1)[1] for a in _argv if a.startswith('--split=')), None)
         or ('train' if '--train' in _argv else os.environ.get('AFLINK_SPLIT', 'test'))).strip().lower()
_SUFFIX = '' if SPLIT == 'test' else f'_{SPLIT}'
ROOT = Path(os.environ.get('FUSSBALL_ROOT', Path(__file__).resolve().parents[2]))  # 仓库根
# 与 _exp_residual_gate.OUT_ROOT 的约定一致：config_j_mcvc/out/_exp_residual_gate[_train]/
TE = (Path(os.environ.get('AFLINK_OUT_ROOT',
                          ROOT / 'appendix' / 'config_j_mcvc' / 'out' / f'_exp_residual_gate{_SUFFIX}'))
      / 'color_veto' / 'trackeval')
METRIC = 'HOTA(0)'          # 逐序列行里的 alpha=0 HOTA


def _paths(key: str) -> tuple[Path, Path]:
    return TE / key / key / 'pedestrian_detailed.csv', TE / key / key / 'pedestrian_summary.txt'


def keys() -> list[str]:
    return sorted(p.name for p in TE.iterdir() if (p / p.name / 'pedestrian_summary.txt').exists())


def per_seq(key: str) -> dict[str, float]:
    """逐序列 HOTA。注意 TrackEval 的 detailed.csv 末尾还有一行 seq='COMBINED'，必须剔除。"""
    detail, _ = _paths(key)
    with detail.open(newline='', encoding='utf-8') as fh:
        return {row['seq']: float(row[METRIC]) for row in csv.DictReader(fh)
                if row['seq'] != 'COMBINED'}


def summary(key: str) -> dict[str, float]:
    """TrackEval 官方聚合值（数据集级，不是逐序列均值）。"""
    _, summ = _paths(key)
    lines = summ.read_text(encoding='utf-8').splitlines()
    return {n: float(v) for n, v in zip(lines[0].split(), lines[1].split())}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, str, int]:
    """Wilcoxon 符号秩检验（双侧，丢弃零差 = scipy 默认 zero_method='wilcox'）。"""
    try:
        from scipy.stats import wilcoxon as sw
        return float(sw(a, b).pvalue), 'scipy', int(((a - b) != 0).sum())
    except ImportError:
        pass
    d = a - b
    d = d[d != 0]                                   # 丢弃零差
    n = len(d)
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    absd = np.abs(d)[order]
    i = 0
    while i < n:                                    # 并列取平均秩
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sd = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
    z = (w_plus - mu - np.sign(w_plus - mu) * 0.5) / sd        # 连续性校正
    from math import erfc
    return erfc(abs(z) / np.sqrt(2)), 'normal-approx', n


def shapiro_p(d: np.ndarray) -> float | None:
    try:
        from scipy.stats import shapiro
        return float(shapiro(d).pvalue)
    except ImportError:
        return None


def report(name: str, base: str, new: str) -> None:
    a, b = per_seq(base), per_seq(new)
    seqs = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seqs]) * 100
    y = np.array([b[s] for s in seqs]) * 100
    d = y - x
    p, how, nz = wilcoxon(y, x)
    agg_x, agg_y = summary(base)['HOTA'], summary(new)['HOTA']   # summary.txt 已是百分数
    up, down, flat = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((np.abs(d) <= 1e-9).sum())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    sp = shapiro_p(d)
    print(f'\n【{name}】{base} -> {new}')
    print(f'  聚合 HOTA : {agg_x:.3f} -> {agg_y:.3f}  (Δ = {agg_y - agg_x:+.3f})')
    print(f'  逐序列均值差 Δ = {d.mean():+.3f} ± {se:.3f} (SE), 95% CI '
          f'[{d.mean() - 1.959964 * se:+.3f}, {d.mean() + 1.959964 * se:+.3f}]')
    print(f'  中位数差 = {float(np.median(d)):+.3f} | Cohen\'s d = {float(d.mean() / d.std(ddof=1)):+.2f}')
    print(f'  Wilcoxon 双侧 p = {p:.3e} ({how}, 非零差 {nz}/{len(d)})')
    print(f'  Shapiro-Wilk 正态性 p = ' + ('不可用(无 scipy)' if sp is None else f'{sp:.3e}'))
    try:
        from scipy.stats import ttest_rel
        print(f'  配对 t 检验 p = {float(ttest_rel(y, x).pvalue):.3e}（正态成立时的参数化对照）')
    except ImportError:
        pass
    print(f'  改善 {up} 条 / 恶化 {down} 条 / 不变 {flat} 条；最大改善 {d.max():+.3f}'
          f'（{seqs[int(d.argmax())]}），最大恶化 {d.min():+.3f}（{seqs[int(d.argmin())]}）')


def ladder() -> None:
    rows = []
    for k in keys():
        s = summary(k)
        rows.append((k, s['HOTA'], s['AssA'], s['IDF1'], int(s['IDs']), int(s['IDSW'])))
    rows.sort(key=lambda r: r[1])
    print(f"{'配置':32s}{'HOTA':>8}{'AssA':>8}{'IDF1':>8}{'IDSW':>7}{'IDs':>7}")
    for k, h, a, i, ids, idsw in rows:
        print(f'{k:32s}{h:8.3f}{a:8.3f}{i:8.3f}{idsw:7d}{ids:7d}')


if __name__ == '__main__':
    print(f'【划分】{SPLIT}  →  {TE}')
    ladder()
    if '--ladder' in sys.argv:
        raise SystemExit(0)
    have = set(keys())
    pairs = [
        ('方案8 的边际贡献（相对纯锥形门控基线）', 'B_no_veto', 'PIX_c25_out_t25m2uS'),
        ('其中「剪枝 0.15/2 → 0.25/2」单独一档', 'PIX_c25_outuS', 'PIX_c25_out_t25m2uS'),
        ('ORACLE 上界（仅参照，不可上线）', 'B_no_veto', 'ORACLE_side'),
    ]
    for name, base, new in pairs:
        if base in have and new in have:
            report(name, base, new)
        else:
            missing = base if base not in have else new
            print(f'\n【{name}】跳过：该划分没跑 {missing}')
