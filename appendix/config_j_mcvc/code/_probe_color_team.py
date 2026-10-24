"""颜色/队色先验探针：用 gameinfo.ini 的官方队别标签量出「队色否决」的理论天花板。

只用现成标签、不读像素，回答两个问题：
  1) 锥形门控后的**假候选**里，有多少是「跨队/跨侧」的 —— 即队色（jersey color）能直接否掉的部分；
  2) 真候选会不会被队色误伤 —— 同一 GT id 必然同队，理论上应为 0，用来校验标签映射没错位。

跑法：
  python _probe_color_team.py                # 只做天花板占比分析（不读像素）
  python _probe_color_team.py --stage upper  # 追加「用 GT 队别标签做完美色否决」的 HOTA 上界
  python _probe_color_team.py --stage pixels # 像素版：读图取球衣色 -> 逐序列无监督二聚类 -> HOTA
     可选：--frames-per-frag 8  每条碎片采样帧数
           --refresh          丢弃缓存重新取色
           --limit 3          只跑前 N 个序列（冒烟测试 + 估时）
           --feature abL      碎片颜色特征：abL(默认,最优) / ab / ab_lower / ab_both
           --veto-outliers    额外否决离群碎片(裁判/球/异色门将) vs 队内（实测 +0.057 HOTA）
           --sweep-outliers   链接级扫描离群剪枝强度网格（挑 trim_frac/max_trim，再跑 HOTA）
           --trim-frac 0.15   离群剪枝占比上限（越大抓到越多裁判/球）
           --max-trim 2       离群剪枝最多迭代几轮
          --out-conf 0.10    低置信（<门槛）碎片也算离群（抓被吸进队簇的裁判/门将）
          --out-unlab        取色失败的碎片也算离群（抓小球；会误标取色失败的球员）
          --out-strict       严格模式：非「高置信队内」一律算离群（实测收益最大，见下）
           --links            链接级诊断（只看被采纳的链接，HOTA 真正在乎的口径）
          --dump-labels X.json  按当前参数把队别标签表落成 JSON 后退出（供生产入口复用，见文末）

           --sweep-features   追加特征变体对比表（只出混淆表，不额外跑 HOTA）
           --no-hota          只出取色/混淆表，跳过 HOTA 对照（快）


像素版说明（不需要任何训练、不引入 Re-ID 网络）：
  1) 每条碎片采若干帧，裁出「球衣带」（框高 15%~50%，左右各内缩 12%），
     排除草地/过暗/过曝像素后取 Lab 中位数色；
  2) 逐序列对碎片的中位色做 k=2 聚类 —— 聚类标签是 left 还是 right 无所谓，
     否决只需要「同簇 / 不同簇」；
  3) 只在**两个碎片都高置信**且簇间距足够大时才否决（不对称风险：误否决真候选代价远高于漏否决）。

方案8 已**植入 AFLink 本体**（2026-09-23）：否决逻辑现在是 `aflink_hybrid_eval.HybridAFLink` 的
原生开关 `kit_veto ∈ {off, side, any, all}`（默认 off，不改变既有行为）。本脚本不再自带子类，
只负责产出队别标签表：
  - 就地实验：`make_veto_linker(lookup, mode)` 直接构造带 kit_veto 的原生链接器；
  - 生产入口：`--dump-labels` 落 JSON -> `aflink_hybrid_eval.py --kit_veto any --kit_labels <JSON>`。
等价性由 `_verify_kit_veto_native.py` 验证（与 64.122 的既定输出逐字节一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # 重定向到文件时也保证中文可读

# 其余路径由 _exp_residual_gate 决定；默认相对仓库根，可用 FUSSBALL_ROOT 覆盖
CODE_DIR = Path(__file__).resolve().parent                        # config_j_mcvc/code
ROOT = Path(os.environ.get('FUSSBALL_ROOT', CODE_DIR.parents[2]))  # 仓库根
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT))

import aflink_hybrid_eval as AHE  # noqa: E402
import _exp_residual_gate as EX  # noqa: E402

KIT_MAP = {
    'player team left': 'left',
    'player team right': 'right',
    'goalkeepers team left': 'left_gk',
    'goalkeepers team right': 'right_gk',
    'referee': 'ref',
    'ball': 'ball',
}
GAP_BUCKETS = [(1, 10), (11, 30), (31, 75), (76, 150)]


def load_kit_map(seq: str) -> dict:
    """gameinfo.ini -> gt_id -> kit 标签（left/right/left_gk/right_gk/ref/ball）。"""
    out = {}
    path = EX.GT / seq / 'gameinfo.ini'
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line.startswith('trackletID_'):
            continue
        key, val = line.split('=', 1)
        raw = val.strip().lower().split(';')[0].strip()
        out[int(key.replace('trackletID_', ''))] = KIT_MAP.get(raw, 'other:' + raw)
    return out


def side_of(kit: str | None) -> str | None:
    if kit in ('left', 'left_gk'):
        return 'left'
    if kit in ('right', 'right_gk'):
        return 'right'
    return None


def classify(kit_a, kit_b) -> str:
    if kit_a is None or kit_b is None:
        return '任一侧无 GT 归属'
    if kit_a == 'ball' or kit_b == 'ball':
        return '涉及球'
    if kit_a == 'ref' or kit_b == 'ref':
        return '涉及裁判'
    sa, sb = side_of(kit_a), side_of(kit_b)
    if sa is not None and sb is not None and sa != sb:
        return '跨侧（队色可否定）'
    if kit_a == kit_b:
        return '完全同 kit（队色无法区分）'
    return '同侧但球衣不同（球员 vs 门将，理论上可否定）'


def gap_bucket(gap: int) -> str:
    for lo, hi in GAP_BUCKETS:
        if lo <= gap <= hi:
            return f'{lo}-{hi}'
    return f'>{GAP_BUCKETS[-1][1]}'


def build_kit_map(preds, seqs) -> dict:
    """id(track_map) -> {碎片 id: kit 标签}，走 GT IoU 归属，仅用于 oracle 上界。"""
    out = {}
    for seq in seqs:
        kits = load_kit_map(seq)
        if not kits:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        out[id(preds[seq])] = {tid: kits[o] for tid, o in owners.items() if o in kits}
    return out


def make_veto_linker(kit_lookup, mode: str) -> AHE.HybridAFLink:
    """方案8 的链接器：直接用 AFLink **原生** `kit_veto`（本脚本原有的诊断子类已删除）。

    原生实现在 `aflink_hybrid_eval.HybridAFLink._apply_kit_veto`，语义与旧子类逐位一致：
    在候选生成层出口按队别标签摘掉「跨队/跨侧」与「离群 vs 队内」的候选。
    `kit_lookup` 这里用 `id(track_map)` 作键（原生同样支持用序列名作键）。
    """
    return AHE.HybridAFLink(**EX.linker_kwargs(rule_variant='adaptive_cone', cone_g0=EX.CONE_G0),
                            kit_veto=mode, kit_lookup=kit_lookup)


def run_veto_hota(preds, seqs, key, mode, kit_lookup) -> tuple[dict, dict]:
    """跑一次「锥形门控 + 色否决」的完整链并评测，返回 (metrics, veto_stats)。"""
    lk = make_veto_linker(kit_lookup, mode)
    processed, _ = AHE.apply_aflink(preds, lk)
    post_dir = EX.OUT_ROOT / 'color_veto' / key
    eval_dir = EX.OUT_ROOT / 'color_veto' / 'trackeval' / key
    AHE.write_predictions_to_dir(processed, post_dir)
    metrics = {k: float(v) for k, v in AHE.run_trackeval(
        results_dir=post_dir, gt_root=EX.GT, seqs=seqs,
        tracker_name=key, output_dir=eval_dir).items()}
    st = lk.kit_veto_totals          # 跨 49 条序列的累计口径，与旧子类的 veto_stats 等价
    print(f"        HOTA={metrics['HOTA']:.3f} AssA={metrics['AssA']:.3f} "
          f"IDF1={metrics['IDF1']:.3f} IDs={metrics['IDs']:.0f} | 候选 {st['candidates']} "
          f"否决 {st['vetoed']}（{st['vetoed'] / max(st['candidates'], 1) * 100:.1f}%）", flush=True)
    return metrics, st


def print_hota_table(results: list, notes: list) -> None:
    base = results[0][2]['HOTA']
    print('\n    ' + '-' * 92)
    print(f"    {'配置':<16}{'HOTA':>9}{'ΔHOTA':>9}{'AssA':>9}{'IDF1':>9}{'IDs':>7}{'否决候选':>10}")
    for key, _note, metrics, st in results:
        print(f"    {key:<16}{metrics['HOTA']:>9.3f}{metrics['HOTA'] - base:>+9.3f}"
              f"{metrics['AssA']:>9.3f}{metrics['IDF1']:>9.3f}{metrics['IDs']:>7.0f}{st['vetoed']:>10}")
    print('\n    ' + '-' * 92)
    for key, note in notes:
        print(f'    {key}: {note}')


def experiment_upper(preds, seqs, kit_lookup) -> None:
    print('\n' + '=' * 100)
    print('[2/2] 完美队色否决的 HOTA 上界（oracle：直接用 GT 队别标签，真候选零误伤）', flush=True)
    variants = [
        ('B_no_veto', '锥形门控，不做任何色否决（真对照）', 'off'),
        ('V_kit_side', '锥形 + 跨侧否决（只看 left/right，最易用像素实现）', 'side'),
        ('V_kit_any', '锥形 + 任意 kit 不同即否决（含门将/裁判/球，最强约束）', 'any'),
        ('SANITY_veto_all', '破坏性自检：全否决（HOTA 必须崩，否则说明钩子没生效）', 'all'),
    ]
    results = []
    for key, note, mode in variants:
        results.append((key, note) + run_veto_hota(preds, seqs, key, mode, kit_lookup))
    print_hota_table(results, [(k, n) for k, n, _ in variants])


PIX_CACHE = ROOT / 'appendix' / 'config_j_mcvc' / ('_color_cache' if EX.SPLIT == 'test' else f'_color_cache_{EX.SPLIT}')
JERSEY_BAND = (0.15, 0.50)      # 球衣带（框高比例）
LOWER_BAND = (0.50, 0.85)       # 下半身（备用带，球衣不够用时再看短裤）
BAND_INSET = 0.12               # 左右内缩比例，避开背景
MIN_BOX_H = 24                  # 框太矮不取色
PIX_CONF, PIX_MIN_SEP = 0.25, 8.0
SIL_THRS = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55)   # silhouette 门槛扫描（序列级双峰性）
CONF_THRS = (0.0, 0.25, 0.45, 0.60, 0.75, 0.85)  # 碎片级置信门槛扫描（误伤在这条通道上）
PIX_CONF_SWEEP = (0.25, 0.45, 0.60, 0.75)         # 进 HOTA 对照的几档 conf
# 离群剪枝强度网格：(剪枝占比上限, 最多剪几轮)。越宽松抓到的裁判/球越多，误剪真球员的风险也升。
OUTLIER_GRID = ((0.15, 2), (0.20, 2), (0.25, 2), (0.25, 4), (0.30, 4), (0.35, 6), (0.50, 8))
# (低置信离群门槛, 未取色也算离群, 严格模式：非高置信队内一律算离群)；None = 关闭该通道
OUT_CONF_GRID = ((None, False, False), (0.05, False, False), (0.10, False, False),
                 (0.15, False, False), (0.20, False, False),
                 (0.10, True, False), (0.15, True, False),
                 (None, True, True), (0.10, True, True))


def seq_image_spec(seq: str) -> tuple[Path, str]:
    """从 seqinfo.ini 读图像目录与扩展名。"""
    im_dir, im_ext = 'img1', '.jpg'
    path = EX.GT / seq / 'seqinfo.ini'
    if path.exists():
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if line.startswith('imDir='):
                im_dir = line.split('=', 1)[1].strip()
            elif line.startswith('imExt='):
                im_ext = line.split('=', 1)[1].strip()
    return EX.GT / seq / im_dir, im_ext


def read_bgr(path: Path) -> np.ndarray | None:
    """中文路径下 cv2.imread 返回 None（已实测），必须走 fromfile + imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def describe_patch(bgr: np.ndarray) -> dict | None:
    """一小块裁剪图 -> 球衣色描述子：排除草地/过暗/过曝后的 Lab 中位数。"""
    if bgr.shape[0] < 6 or bgr.shape[1] < 4:
        return None
    patch = cv2.resize(bgr, (24, 40), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (sat > 60) & (hue > 35) & (hue < 95)
    usable = (val > 40) & (val < 250) & (~grass)
    if usable.sum() < 10:           # 白/灰球衣可能整体低饱和，退回只排除明暗极端
        usable = (val > 40) & (val < 250)
    if usable.sum() < 5:
        return None
    l_, a_, b_ = lab[usable, 0], lab[usable, 1], lab[usable, 2]
    return {'L': float(np.median(l_)), 'a': float(np.median(a_)), 'b': float(np.median(b_)),
            'S': float(np.median(sat[usable])), 'V': float(np.median(val[usable])),
            'grass': round(float(grass.mean()), 3), 'n': int(usable.sum())}


def sample_indices(n: int, k: int) -> list[int]:
    """碎片生命周期里均匀取 k 帧，端点内缩 10% 以避开入画/出画与遮挡。"""
    if n <= k:
        return list(range(n))
    lo, hi = int(0.1 * (n - 1)), int(0.9 * (n - 1))
    return sorted({int(round(v)) for v in np.linspace(lo, hi, k)})


def extract_seq(seq: str, track_map, frames_per_frag: int) -> dict:
    """采样取色，返回 {碎片 id: [逐帧描述子 {f, j, l}, ...]}。"""
    im_dir, im_ext = seq_image_spec(seq)
    reqs = []
    for tid, frag in track_map.items():
        for i in sample_indices(len(frag.frames), frames_per_frag):
            box = frag.boxes[i]
            if box[3] - box[1] >= MIN_BOX_H:      # 太矮的框取色不可靠
                reqs.append((frag.frames[i], tid, box))
    reqs.sort(key=lambda r: r[0])                 # 按帧号排序 -> 单帧缓存，每帧只解码一次
    out: dict = defaultdict(list)
    last_frame, img = -1, None
    for frame, tid, box in reqs:
        if frame != last_frame:
            img = read_bgr(im_dir / f'{frame:06d}{im_ext}')
            last_frame = frame
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        ix1, ix2 = int(max(0, x1 + BAND_INSET * bw)), int(min(w, x2 - BAND_INSET * bw))
        rec = {'f': frame}
        for name, (top, bot) in (('j', JERSEY_BAND), ('l', LOWER_BAND)):
            iy1, iy2 = int(max(0, y1 + top * bh)), int(min(h, y1 + bot * bh))
            if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                continue
            desc = describe_patch(img[iy1:iy2, ix1:ix2])
            if desc:
                rec[name] = desc
        if len(rec) > 1:
            out[tid].append(rec)
    return dict(out)


def _extract_one(task: tuple) -> tuple:
    """多进程取色的一支：序列之间彼此独立，只做「读图 -> 描述子」。"""
    seq, track_map, frames_per_frag = task
    return seq, extract_seq(seq, track_map, frames_per_frag)


def load_color_cache(preds, seqs, frames_per_frag: int, refresh: bool, jobs: int = 1) -> dict:
    """取色缓存。jobs>1 时按序列并行（云上 128 核可把几小时的取色压到几分钟）。"""
    PIX_CACHE.mkdir(parents=True, exist_ok=True)
    cache, t0 = {}, time.time()
    todo = []
    for seq in seqs:
        path = PIX_CACHE / f'{seq}_k{frames_per_frag}.json'
        if path.exists() and not refresh:
            cache[seq] = {int(t): v for t, v in json.loads(path.read_text(encoding='utf-8')).items()}
            print(f'    [cache] {seq} 碎片 {len(cache[seq])}', flush=True)
        else:
            todo.append(seq)

    def _store(seq: str, data: dict) -> None:
        (PIX_CACHE / f'{seq}_k{frames_per_frag}.json').write_text(
            json.dumps({str(t): v for t, v in data.items()}), encoding='utf-8')
        cache[seq] = data

    n_todo = len(todo)
    if n_todo and jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, n_todo)) as ex:
            futs = [ex.submit(_extract_one, (s, preds[s], frames_per_frag)) for s in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                seq, data = fut.result()
                _store(seq, data)
                el = time.time() - t0
                print(f'    [{i}/{n_todo}] {seq} extract 碎片 {len(data)} | 累计 {el:.0f}s '
                      f'ETA {el / i * (n_todo - i):.0f}s', flush=True)
    else:
        for i, seq in enumerate(todo, 1):
            _store(seq, extract_seq(seq, preds[seq], frames_per_frag))
            el = time.time() - t0
            print(f'    [{i}/{n_todo}] {seq} extract 碎片 {len(cache[seq])} | 累计 {el:.0f}s '
                  f'ETA {el / i * (n_todo - i):.0f}s', flush=True)
    if n_todo:
        print(f'  取色完成：新算 {n_todo} 条 / 共 {len(cache)} 条，用时 {time.time() - t0:.0f}s', flush=True)
    return cache


def fragment_feature(descs, kind: str = 'ab') -> np.ndarray | None:
    """碎片级颜色特征 = 各帧中位数（对单帧噪声与部分遮挡稳健）。

    kind: ab        球衣带色度 (a*, b*)，默认
          abL       球衣带色度 + 明度（CIE76 尺度，明度对阴影敏感，作为对照）
          ab_lower  下半身（短裤）色度
          ab_both   球衣 + 短裤 4 维
    """
    def med(band: str, key: str):
        vals = [d[band][key] for d in descs if band in d]
        return float(np.median(vals)) if vals else None

    a, b = med('j', 'a'), med('j', 'b')
    if kind == 'ab':
        return None if a is None else np.array([a - 128.0, b - 128.0], dtype=np.float32)
    if kind == 'ab_lower':
        al, bl = med('l', 'a'), med('l', 'b')
        return None if al is None else np.array([al - 128.0, bl - 128.0], dtype=np.float32)
    if kind == 'abL':
        l_ = med('j', 'L')
        return None if (a is None or l_ is None) else np.array(
            [a - 128.0, b - 128.0, l_ * 100.0 / 255.0], dtype=np.float32)
    if kind == 'ab_both':
        al, bl = med('l', 'a'), med('l', 'b')
        if a is None or al is None:
            return None
        return np.array([a - 128.0, b - 128.0, al - 128.0, bl - 128.0], dtype=np.float32)
    raise ValueError(f'未知特征类型 {kind}')


FEATURE_CHOICES = ('ab', 'abL', 'ab_lower', 'ab_both')
FEATURE_DIM = {'ab': 2, 'abL': 3, 'ab_lower': 2, 'ab_both': 4}


def feature_min_sep(feature: str, base: float = PIX_MIN_SEP) -> float:
    """序列级「簇间距」门槛按特征维数缩放，保证不同特征变体可比。

    簇间距是 L2 距离，维数升高本身会让距离变大（随机噪声下 ≈ sqrt(dim)），
    所以 4 维特征的 8.0 门槛其实比 2 维宽松。按 sqrt(dim/2) 归一到 2 维尺度。
    """
    return base * float(np.sqrt(FEATURE_DIM.get(feature, 2) / 2.0))


def cluster_chroma(feats: dict, trim_frac: float = 0.15, max_trim: int = 2) -> tuple[dict, dict, float, float, set]:
    """逐序列在色度平面做 k=2 聚类：先剪离群小簇，再判定是否真的双峰。

    实测教训（SNMOT-116）：白衣队 (a*,b*)≈(128,125) 几乎无色度，
    而黄色裁判 ≈(103,201) 离得更远，于是普通 k=2 把「白衣 vs 裁判」当成两类，
    两队（白衣 vs 深红）反而没被分开，准确率掉到 48%。
    所以：反复剪掉占比 ≤ trim_frac 的离群小簇（裁判/门将/球），
    再用 silhouette 判定剩下的是否真双峰（只有一队在场/两队同色 -> silhouette 很低 -> 整条序列不否决）。
    返回 (簇标签 0/1/None, 置信度, 簇间距, silhouette, 被剪掉的离群碎片集合)。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    tids = sorted(feats)
    labels = {t: None for t in tids}
    conf = {t: 0.0 for t in tids}
    X = np.stack([feats[t] for t in tids])
    if len(tids) < 4 or np.unique(X, axis=0).shape[0] < 2:
        return labels, conf, 0.0, 0.0, set()
    keep = np.ones(len(tids), bool)
    for _ in range(max_trim):
        idx = np.flatnonzero(keep)
        if idx.size < 4:
            break
        lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X[idx])
        small = int(np.argmin(np.bincount(lab, minlength=2)))
        if int((lab == small).sum()) > trim_frac * idx.size:   # 小簇不算离群，停手
            break
        keep[idx[lab == small]] = False
    outliers = {tids[i] for i in np.flatnonzero(~keep)}
    idx = np.flatnonzero(keep)
    if idx.size < 4:
        return labels, conf, 0.0, 0.0, outliers
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[idx])
    centers = km.cluster_centers_
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    sil = float(silhouette_score(X[idx], km.labels_))
    for j, i in enumerate(idx):
        lab = int(km.labels_[j])
        d_own = float(np.linalg.norm(X[i] - centers[lab]))
        d_oth = float(np.linalg.norm(X[i] - centers[1 - lab]))
        labels[tids[i]] = lab
        conf[tids[i]] = (d_oth - d_own) / (d_oth + d_own + 1e-6)
    return labels, conf, sep, sil, outliers


def pixel_predict(cache, seqs, feature: str = 'ab', trim_frac: float = 0.15,
                  max_trim: int = 2) -> dict:
    """-> {seq: {'label': {tid: 0/1/None}, 'conf': {tid: float}, 'sep': float,
             'sil': float, 'outlier': set(tid)}}

    'outlier' = 被离群小簇剪掉的碎片（裁判/球/异色门将）。它们的颜色与两队都不同，
    所以「离群碎片 vs 队内碎片」的链接同样应该被否决 —— 这正是 oracle 靠
    side_of(裁判/球)=None 隐性拿到、而像素版此前丢掉的那部分收益。
    trim_frac/max_trim 控制剪枝强度：越宽松抓到的裁判/球越多，但误剪真球员的风险也升。
    """

    out = {}
    for seq in seqs:
        feats = {}
        for tid, descs in (cache.get(seq) or {}).items():
            vec = fragment_feature(descs, feature)
            if vec is not None:
                feats[tid] = vec
        labels, conf, sep, sil, outliers = cluster_chroma(feats, trim_frac, max_trim)
        out[seq] = {'label': labels, 'conf': conf, 'sep': sep, 'sil': sil,
                    'outlier': outliers}
    return out


def gated_labels(pix: dict, th: float, min_sep: float, sil_min: float,
                 veto_outliers: bool = False, out_conf: float | None = None) -> dict:
    """三重门后给出可用的队别标签：{seq: {tid: 'c0'/'c1'[,'cX']}}，未通过者视为未知（不否决）。

    门槛顺序 = 序列级（双峰性 silhouette、簇间距）-> 碎片级（到簇心的归一化间距）。
    刻意做成不对称：宁可漏否决，不可误否决真候选。
    veto_outliers=True 时额外把「被剪掉的离群碎片」标成 'cX'：它与两队都不同色，
    所以「离群 vs 队内」的链接应当被否决（裁判/球/异色门将不应和球员连成一条轨迹）。
    out_conf 再补一条通道：簇内**低置信**碎片也标 'cX'。裁判常被 KMeans 硬塞进某个
    队簇，但到两个簇心的归一化间距都很小（实测 conf≈0.07）—— 它本来就「不属于任一方」，
    这正是我们要的离群信号。取色失败的碎片不在这里处理（见 build_pixel_lookup）。
    """
    out = {}
    for seq, info in pix.items():
        if info['sil'] < sil_min or info['sep'] < min_sep:
            out[seq] = {}
            continue
        table = {t: f'c{lab}' for t, lab in info['label'].items()
                 if lab is not None and info['conf'][t] >= th}
        if veto_outliers:
            table.update({t: 'cX' for t in info.get('outlier', ())})
            if out_conf is not None:
                table.update({t: 'cX' for t, lab in info['label'].items()
                              if lab is not None and info['conf'][t] < out_conf})
        out[seq] = table
    return out


def build_pixel_lookup(preds, seqs, cache, th: float, sil_min: float,
                       min_sep: float | None = None, owners_by_seq=None, feature: str = 'ab',
                       veto_outliers: bool = False, trim_frac: float = 0.15,
                       max_trim: int = 2, pix: dict | None = None,
                       out_conf: float | None = None, out_unlab: bool = False,
                       out_strict: bool = False) -> dict:
    """-> {id(track_map): {tid: 'c0'/'c1'/'cX'}}，只放过三重门的碎片，其余视为未知（不否决）。

    out_unlab=True：取色失败/未分簇的碎片（压根不在 label 里）也算离群 —— 只有给了
    lookup 表项，否决才会生效；标成 'cX' 后它们与任何队内碎片的链接都会被否掉。
    out_strict=True：更极端 —— 凡「没被高置信判成某队」的碎片一律 'cX'。
    """
    if min_sep is None:
        min_sep = feature_min_sep(feature)
    if pix is None:
        pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    gated = gated_labels(pix, th, min_sep, sil_min, veto_outliers=veto_outliers,
                         out_conf=out_conf)
    out = {}
    for seq in seqs:
        table = dict(gated.get(seq, {}))
        info = pix.get(seq) or {}
        passed = bool(info) and info['sil'] >= sil_min and info['sep'] >= min_sep
        if veto_outliers and passed and (out_unlab or out_strict):
            labs = info.get('label', {})
            for tid in preds[seq]:
                if out_strict or tid not in labs:
                    table.setdefault(tid, 'cX')
        if owners_by_seq is not None:
            allowed = set(owners_by_seq.get(seq, {}))
            table = {t: v for t, v in table.items() if t in allowed}
        out[id(preds[seq])] = table
    return out


def dump_kit_labels(preds, seqs, cache, path, *, feature: str, th: float, sil_min: float,
                    veto_outliers: bool, trim_frac: float, max_trim: int,
                    out_conf: float | None, out_unlab: bool, out_strict: bool) -> None:
    """把某一组参数的队别标签表落成 JSON，供**生产入口**直接复用（免去重跑取色 + 聚类）。

    生成：python _probe_color_team.py --stage pixels --veto-outliers --out-unlab --out-strict \
              --trim-frac 0.25 --max-trim 2 --dump-labels labels.json
    使用：python aflink_hybrid_eval.py --kit_veto any --kit_labels labels.json
    """
    pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    lookup = build_pixel_lookup(preds, seqs, cache, th, sil_min, feature=feature, pix=pix,
                                veto_outliers=veto_outliers, trim_frac=trim_frac,
                                max_trim=max_trim, out_conf=out_conf,
                                out_unlab=out_unlab, out_strict=out_strict)
    table = {seq: {str(tid): lab for tid, lab in sorted((lookup.get(id(preds[seq])) or {}).items())}
             for seq in seqs}
    meta = {'feature': feature, 'conf_th': th, 'sil_min': sil_min, 'min_sep': feature_min_sep(feature),
            'veto_outliers': bool(veto_outliers), 'trim_frac': trim_frac, 'max_trim': max_trim,
            'out_conf': out_conf, 'out_unlab': bool(out_unlab), 'out_strict': bool(out_strict),
            'base': f'adaptive_cone(g0={EX.CONE_G0:.0f})', 'input_chain': EX.RESULTS.name}
    out_path = Path(path)
    out_path.write_text(json.dumps({'meta': meta, 'sequences': table}, ensure_ascii=False, indent=1),
                        encoding='utf-8')
    n_lab = sum(len(t) for t in table.values())
    n_out = sum(1 for t in table.values() for v in t.values() if v == 'cX')
    print(f'\n队别标签已写出：{out_path}')
    print(f'  {len(table)} 个序列 / {n_lab} 条碎片，其中离群 cX {n_out} 条；参数 {meta}')
    print(f'  生产入口用法：--kit_veto any --kit_labels "{out_path}"', flush=True)


HEAD_CONFUSION = (f"  {'门槛':>8}{'候选':>7}{'否决':>7}{'真候选':>7}{'误伤真候选':>11}"
                  f"{'跨侧假候选':>11}{'命中跨侧':>9}{'召回':>7}{'跨侧漏':>7}{'漏因未分类':>11}")


def print_confusion_row(tag: str, st: Counter) -> None:
    print(f"  {tag:>8}{st['候选']:>7}{st['否决']:>7}{st['真候选']:>7}{st['误伤真候选']:>11}"
          f"{st['跨侧']:>11}{st['命中跨侧']:>9}{st['命中跨侧'] / max(st['跨侧'], 1):>6.1%}"
          f"{st['跨侧漏']:>7}{st['漏因未分类']:>11}", flush=True)


def collect_cand_rows(preds, seqs, cache, linker, feature: str, pix=None) -> tuple[list, dict]:
    """构建候选级混淆所需的展平数据：(cand_rows, seq_info)。

    每个特征变体都要重跑「取色 -> 聚类 -> 建候选」，所以抽出来复用，
    避免 evaluate_pixel 的可视化输出在横向对比时刷屏。
    """
    if pix is None:
        pix = pixel_predict(cache, seqs, feature)
    cand_rows, seq_info = [], {}
    for seq in seqs:
        kits, info = load_kit_map(seq), pix.get(seq)
        if not kits or not info:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        lab, conf = info['label'], info['conf']
        seq_info[seq] = (info['sil'], info['sep'])
        for cand in linker._build_candidates(preds[seq], EX.NOMINAL_GAP, EX.NOMINAL_DIST):
            sa, sb = lab.get(cand.src_id), lab.get(cand.dst_id)
            diff_conf = None if (sa is None or sb is None or sa == sb) else min(
                conf.get(cand.src_id, 0.0), conf.get(cand.dst_id, 0.0))
            oa, ob = owners.get(cand.src_id), owners.get(cand.dst_id)
            cand_rows.append((seq, oa is not None and oa == ob,
                              classify(kits.get(oa), kits.get(ob)), diff_conf))
    return cand_rows, seq_info


def compare_features(preds, seqs, cache, linker, features, conf_thrs=(0.25, 0.45),
                     sil_min: float = 0.30) -> dict:
    """只出混淆表、不算 HOTA：横向对比颜色特征变体对「否决召回」的影响。

    瓶颈是召回（跨侧假候选命中率）而非误伤，所以看「召回 vs 误伤」两列即可；
    每档 conf 的簇距门槛按特征维数缩放，保证不同特征在同一尺子上比较。
    """
    print('\n' + '=' * 100)
    print(f'[特征变体对比] 只看候选级混淆，不跑 HOTA（sil>={sil_min:.2f}，簇距门槛按维数缩放）',
          flush=True)
    print(f"  {'特征':>10}{'维度':>5}{'已分类碎片':>11}{'conf':>6}{'候选':>7}{'真候选':>7}{'误伤':>7}"
          f"{'跨侧假候选':>12}{'命中跨侧':>9}{'召回':>7}{'跨侧漏':>7}{'漏因未分类':>11}")
    out = {}
    for feat in features:
        pix = pixel_predict(cache, seqs, feat)
        cand_rows, seq_info = collect_cand_rows(preds, seqs, cache, linker, feat, pix=pix)
        n_feat = sum(1 for info in pix.values()
                     for v in info['label'].values() if v is not None)
        for th in conf_thrs:
            st = count_confusion(cand_rows, seq_info, th, sil_min, feature_min_sep(feat))
            print(f"  {feat:>10}{FEATURE_DIM.get(feat, 2):>5}{n_feat:>11}{th:>6.2f}"
                  f"{st['候选']:>7}{st['真候选']:>7}{st['误伤真候选']:>7}{st['跨侧']:>12}"
                  f"{st['命中跨侧']:>9}{st['命中跨侧'] / max(st['跨侧'], 1):>6.1%}"
                  f"{st['跨侧漏']:>7}{st['漏因未分类']:>11}", flush=True)
            out[(feat, th)] = st
    print('  召回 = 命中的跨侧假候选 / 全部跨侧假候选；「漏因未分类」= 因为序列级门未通过而漏掉的')
    print('  误伤 = 被否决的真候选（同 GT id），代价最高，必须保持个位数')
    return out


HEAD_LINKS = (f"  {'配置':>10}{'采纳链接':>9}{'真链接':>8}{'假链接':>8}{'跨侧假链接':>11}"
              f"{'同kit假':>9}{'其他假':>8}{'真链接净损失':>13}{'消除跨侧':>9}{'跨侧消除率':>11}")


def accepted_link_stats(preds, seqs, lookup, mode: str) -> Counter:
    """只统计**真正被采纳的链接**（link_sequence 的输出），这是 HOTA 真正在乎的口径。

    候选池召回率是个失真指标：池子里的假候选绝大多数根本不会被选中，
    否掉它们对输出零影响。真正决定 HOTA 的是「被采纳的链接里有多少跨侧假链接被否掉」。
    """
    lk = make_veto_linker(lookup, mode)
    st = Counter()
    for seq in seqs:
        kits = load_kit_map(seq)
        if not kits:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        link_map, _ = lk.link_sequence(preds[seq])
        for src, dst in link_map.items():
            oa, ob = owners.get(src), owners.get(dst)
            if oa is None or ob is None:
                st['无归属链接'] += 1
                continue
            st['采纳链接'] += 1
            if oa == ob:
                st['真链接'] += 1
            else:
                st['假链接'] += 1
                kind = classify(kits.get(oa), kits.get(ob))
                st[f'kind:{kind}'] += 1
                if kind == '跨侧（队色可否定）':
                    st['跨侧假链接'] += 1
                elif kind == '完全同 kit（队色无法区分）':
                    st['同kit假链接'] += 1
                else:
                    st['其他假链接'] += 1
    return st


def print_false_breakdown(rows: list) -> None:
    """假链接按 classify() 构成明细 —— 用来确认 oracle 多清掉的那些到底属于哪一类。"""
    print('\n  假链接构成明细（被采纳的链接里，按 GT 队别关系分类）：', flush=True)
    kinds = sorted({k for _t, st in rows for k in st if k.startswith('kind:')})
    base_total = rows[0][1]['假链接'] or 1
    print('    ' + f"{'类别':<26}" + ''.join(f'{t:>12}' for t, _s in rows))
    for k in kinds:
        name = k[len('kind:'):]
        cells = ''.join(f"{st.get(k, 0):>12}" for _t, st in rows)
        print(f'    {name:<26}{cells}')
    print('    ' + f"{'(合计)':<26}" + ''.join(f"{st['假链接']:>12}" for _t, st in rows))
    print(f"    基线假链接合计 {rows[0][1]['假链接']}，其中跨侧占 "
          f"{rows[0][1]['跨侧假链接'] / base_total:.1%}", flush=True)


def print_links_row(tag: str, st: Counter, base: Counter) -> None:
    remain = st['跨侧假链接']                     # 否决后仍残留的跨侧假链接
    removed = base['跨侧假链接'] - remain         # 相对基线消除掉的跨侧假链接
    lost_true = base['真链接'] - st['真链接']      # 真链接净损失 = 误伤 - 因否决而新增的正确合并
    print(f"  {tag:>10}{st['采纳链接']:>9}{st['真链接']:>8}{st['假链接']:>8}{remain:>11}"
          f"{st['同kit假链接']:>9}{st['其他假链接']:>8}{lost_true:>13}"
          f"{removed:>9}{removed / max(base['跨侧假链接'], 1):>10.1%}", flush=True)


def diagnose_links(preds, seqs, cache, feature: str = 'abL', veto_outliers: bool = False,
                   trim_frac: float = 0.15, max_trim: int = 2) -> None:
    """对比「无否决 / oracle / 像素各档 conf」下被采纳链接的构成。"""
    print('\n' + '=' * 100)
    print('[链接级诊断] 只看被采纳的链接（HOTA 真正在乎的口径）', flush=True)
    print(HEAD_LINKS)
    base = accepted_link_stats(preds, seqs, {}, 'off')
    oracle = build_kit_map(preds, seqs)
    pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    shown = [('B_no_veto', base),
             ('ORACLE_side', accepted_link_stats(preds, seqs, oracle, 'side')),
             ('PIX_c25', accepted_link_stats(preds, seqs, build_pixel_lookup(
                 preds, seqs, cache, 0.25, 0.30, feature=feature, pix=pix), 'any'))]
    if veto_outliers:
        shown.append(('PIX_c25_out', accepted_link_stats(preds, seqs, build_pixel_lookup(
            preds, seqs, cache, 0.25, 0.30, feature=feature, veto_outliers=True, pix=pix), 'any')))
    for tag, st in shown:
        print_links_row(tag, st, base)
    print_false_breakdown(shown)
    print(f"  基线采纳跨侧假链接 {base['跨侧假链接']} 条；「跨侧消除率」= 这些链接里被否掉消失的比例")
    print('  「同kit假链接」= 队色天然无法区分的假链接（同侧不同球员），是队色路线的硬上限')
    print('  「真链接净损失」为负说明否决反而促成了更多正确合并（坏链接被摘掉后好链接才轮到）')


def _print_sweep(title: str, base: Counter, rows: list, notes: list) -> None:
    """打印一档扫描结果：链接级总表 + 假链接构成 + 每档脚注。"""
    print('\n' + '=' * 100)
    print(title, flush=True)
    print(HEAD_LINKS)
    print_links_row('B_no_veto', base, base)
    for tag, st, _info in rows:
        print_links_row(tag, st, base)
    print_false_breakdown([('B_no_veto', base)] + [(t, s) for t, s, _i in rows])
    for line in notes:
        print(f'  {line}')


def sweep_outlier_trim(preds, seqs, cache, feature: str = 'abL',
                       th: float = PIX_CONF) -> None:
    """链接级扫描离群剪枝强度（trim_frac, max_trim）。

    HOTA 太贵，先用这条便宜口径挑参数：看「涉及裁判/涉及球」的假链接被清掉多少，
    同时盯住「真链接净损失」有没有变正（= 把真球员误剪成离群，反而否决了正确链接）。
    """
    base = accepted_link_stats(preds, seqs, {}, 'off')
    rows = []
    for tf, mt in OUTLIER_GRID:
        pix = pixel_predict(cache, seqs, feature, tf, mt)
        lookup = build_pixel_lookup(preds, seqs, cache, th, 0.30, feature=feature,
                                    veto_outliers=True, trim_frac=tf, max_trim=mt, pix=pix)
        n_out = sum(len(v['outlier']) for v in pix.values())
        n_lab = sum(1 for v in pix.values() for x in v['label'].values() if x is not None)
        rows.append((f't{tf:.2f}/m{mt}', accepted_link_stats(preds, seqs, lookup, 'any'),
                     (n_out, n_lab)))
    _print_sweep(f'[离群剪枝强度扫描] 链接级口径，特征={feature}，conf>={th:.2f}', base, rows, [
        '离群碎片数 / 仍可分簇碎片数：'
        + '  '.join(f'{t}={o}/{l}' for t, _s, (o, l) in rows),
        '若某档「仍可分簇碎片数」骤降或否决归零，说明剪枝过猛把真球员也剪没了（破坏性自检）。',
        '挑参数原则：优先把「涉及裁判/涉及球」压到 oracle 附近，且「真链接净损失」不要转正。'])


def sweep_outlier_conf(preds, seqs, cache, feature: str = 'abL', th: float = PIX_CONF,
                       trim_frac: float = 0.25, max_trim: int = 2) -> None:
    """链接级扫描「低置信即离群」通道（out_conf）与「未取色也算离群」（out_unlab）。

    实测：裁判被 KMeans 硬塞进队簇，但 conf 只有 0.07（到两簇心都差不多远），
    本质是「不属于任一方」；球则常常取色失败(未分类)。两条通道分别补这两类。
    """
    base = accepted_link_stats(preds, seqs, {}, 'off')
    pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    rows = []
    for oc, ou, os_ in OUT_CONF_GRID:
        lookup = build_pixel_lookup(preds, seqs, cache, th, 0.30, feature=feature,
                                    veto_outliers=True, trim_frac=trim_frac,
                                    max_trim=max_trim, pix=pix, out_conf=oc,
                                    out_unlab=ou, out_strict=os_)
        n_cx = sum(1 for v in lookup.values() for x in v.values() if x == 'cX')
        tag = f'c{"-" if oc is None else f"{oc:.2f}"}{"+u" if ou else ""}{"+S" if os_ else ""}'
        rows.append((tag, accepted_link_stats(preds, seqs, lookup, 'any'), n_cx))
    _print_sweep(f'[低置信离群扫描] trim={trim_frac}/{max_trim}，特征={feature}，'
                 f'队别 conf>={th:.2f}；列名 c<阈值>=低置信离群门槛，+u=未取色也算离群，'
                 f'+S=严格模式（非高置信队内一律算离群）', base, rows, [
        '标为 cX 的碎片数：' + '  '.join(f'{t}={n}' for t, _s, n in rows),
        '注意：out_conf 必须小于队别 conf 门槛才有意义（否则那些碎片本来也不进否决表）。'])


def explain_missed_links(preds, seqs, cache, feature: str = 'abL', th: float = PIX_CONF,
                         sil_min: float = 0.30) -> Counter:
    """对被采纳的跨侧假链接，逐条归因「为什么没被像素否决掉」。

    判据：该链接在无否决基线里被采纳、且在像素配置下*仍然被采纳* -> 否决确实没生效。
    归因顺序 = 序列级门 -> 碎片未分类 -> 同簇（聚类错）-> 置信度不足（降门槛可救）。
    """
    min_sep = feature_min_sep(feature)
    pix = pixel_predict(cache, seqs, feature)
    base_lk, pix_lk = _linker_for({}, 'off'), _linker_for(
        build_pixel_lookup(preds, seqs, cache, th, sil_min, feature=feature), 'any')
    reasons = Counter()
    for seq in seqs:
        kits = load_kit_map(seq)
        if not kits:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        base_map, _ = base_lk.link_sequence(preds[seq])
        pix_map, _ = pix_lk.link_sequence(preds[seq])
        info = pix[seq]
        seq_gated_out = info['sil'] < sil_min or info['sep'] < min_sep
        for src, dst in base_map.items():
            oa, ob = owners.get(src), owners.get(dst)
            if oa is None or ob is None or oa == ob:
                continue
            if classify(kits.get(oa), kits.get(ob)) != '跨侧（队色可否定）':
                continue
            reasons['应否决跨侧链接'] += 1
            if pix_map.get(src) != dst:               # 该链接在像素配置下已不复存在
                reasons['已否决'] += 1
                continue
            la, lb = info['label'].get(src), info['label'].get(dst)
            if seq_gated_out:
                reasons['漏:序列级门'] += 1
            elif la is None or lb is None:
                reasons['漏:碎片未分类'] += 1
            elif la == lb:
                reasons['漏:同簇(聚类错)'] += 1
            else:                                     # 标签确实不同，只可能是 conf 没过门
                reasons['漏:置信度不足'] += 1
    return reasons


def explain_ref_ball_links(preds, seqs, cache, feature: str = 'abL',
                           trim_frac: float = 0.15, max_trim: int = 2,
                           th: float = PIX_CONF, sil_min: float = 0.30) -> None:
    """对「涉及裁判/涉及球」的被采纳假链接，查关键端点（裁判/球碎片）的像素状态。

    oracle 靠 side_of(裁判/球)=None 把这类链接全清掉；像素版只能先把裁判/球标成离群(cX)
    再靠「cX vs c0/c1」否决。这里直接摊开「裁判/球端点被判定成什么」：
      离群(cX)     -> 机制本该生效（若链接仍在，说明 veto 没落地，是 bug）
      未分类        -> 取色失败/未分簇（球太小？MIN_BOX_H 挡掉了？）-> 天然无法否决
      队内 c0/c1   -> 被错误吸进某个队簇（颜色像某队）-> 剪枝没抓到
    """
    print('\n' + '=' * 100)
    print(f'[裁判/球假链接归因] 剪枝={trim_frac}/{max_trim}，conf>={th:.2f}，特征={feature}', flush=True)
    pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    min_sep = feature_min_sep(feature)
    base_lk = _linker_for({}, 'off')
    pix_lk = _linker_for(build_pixel_lookup(
        preds, seqs, cache, th, sil_min, feature=feature, veto_outliers=True,
        trim_frac=trim_frac, max_trim=max_trim, pix=pix), 'any')
    for key_kit, kit_name in (('ref', '裁判'), ('ball', '球')):
        cases = Counter()
        worst = Counter()
        n_vetoed = 0
        for seq in seqs:
            kits = load_kit_map(seq)
            if not kits:
                continue
            owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
            base_map, _ = base_lk.link_sequence(preds[seq])
            pix_map, _ = pix_lk.link_sequence(preds[seq])
            info = pix[seq]
            for src, dst in base_map.items():
                oa, ob = owners.get(src), owners.get(dst)
                if oa is None or ob is None or oa == ob:
                    continue
                ka, kb = kits.get(oa), kits.get(ob)
                if kit_name == '裁判' and (ka == 'ref' or kb == 'ref'):
                    key, opp = (src, dst) if ka == 'ref' else (dst, src)
                elif kit_name == '球' and (ka == 'ball' or kb == 'ball'):
                    key, opp = (src, dst) if ka == 'ball' else (dst, src)
                else:
                    continue
                if pix_map.get(src) != dst and pix_map.get(dst) != src:
                    n_vetoed += 1
                    continue
                cases[(state_of(info, key, th), state_of(info, opp, th))] += 1
                if state_of(info, key, th) not in ('离群(cX)',):
                    worst[seq] += 1
        tot = sum(cases.values()) + n_vetoed
        print(f'\n  【涉及{kit_name}】基线被采纳的此类假链接 {tot} 条（已否决 {n_vetoed}），'
              f'残存链接的端点状态分布（关键端点 -> 对端）：', flush=True)
        for (ks, os_), cnt in cases.most_common():
            print(f'    {ks:>10} -> {os_:>10}{cnt:>6}', flush=True)
        if worst:
            top = '  '.join(f'{s}:{c}' for s, c in worst.most_common(8))
            print(f'    关键端点未被标离群、分布最多的序列：{top}', flush=True)


def state_of(info: dict, tid, th: float = PIX_CONF) -> str:
    """碎片在像素预测里「实际会被怎么用」的状态（含 conf 门）。

    注意不能只看 label：聚类会给每个碎片一个标签，但 conf 低于门槛的碎片根本不进
    否决表（=不否决），这才是它的真实语义。
    """
    if tid in info.get('outlier', ()):
        return '离群(cX)'
    lab = info['label'].get(tid)
    if lab is None:
        return '未分类'
    return f'队内 c{lab}' if info['conf'].get(tid, 0.0) >= th else f'低置信(c{lab})'


def audit_outlier_labels(preds, seqs, cache, feature: str = 'abL', th: float = PIX_CONF,
                         trim_frac: float = 0.15, max_trim: int = 2,
                         out_conf: float | None = None, out_unlab: bool = False,
                         out_strict: bool = False) -> None:
    """用 GT 队别标签（免费标注）审「离群检测」的精度：标成 cX 的碎片到底是什么。

    理想：标离群的几乎全是 裁判/球/异色门将/虚检（无 GT 归属），真球员极少。
    「误标真球员」= 离群通道直接吃掉的真链接来源，是当前剩下的主要损耗。
    """
    print('\n' + '=' * 100)
    print(f'[离群检测精度审计] trim={trim_frac}/{max_trim}，out_conf={out_conf}，'
          f'unlab={out_unlab}，strict={out_strict}，特征={feature}', flush=True)
    lookup = build_pixel_lookup(preds, seqs, cache, th, 0.30, feature=feature,
                                veto_outliers=True, trim_frac=trim_frac, max_trim=max_trim,
                                out_conf=out_conf, out_unlab=out_unlab, out_strict=out_strict)
    st, kinds = Counter(), Counter()
    for seq in seqs:
        kits = load_kit_map(seq)
        if not kits:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        tags = {t: v for t, v in lookup.get(id(preds[seq]), {}).items() if v == 'cX'}
        for tid in tags:
            owner = owners.get(tid)
            kit = kits.get(owner)
            if kit in ('left', 'right'):
                kinds['真球员(误标!)'] += 1
                st[seq] += 1
            elif kit in ('left_gk', 'right_gk'):
                kinds['门将(异色, 标对)'] += 1
            elif kit == 'ref':
                kinds['裁判(标对)'] += 1
            elif kit == 'ball':
                kinds['球(标对)'] += 1
            else:
                kinds['虚检/无 GT 归属(无所谓)'] += 1
    tot = sum(kinds.values())
    print(f'  共标为离群 {tot} 个碎片：', flush=True)
    for k, v in kinds.most_common():
        print(f'    {k:<24}{v:>6}  {v / max(tot, 1):6.1%}', flush=True)
    print(f'  离群检测精确率（非真球员占比）= '
          f'{1 - kinds["真球员(误标!)"] / max(tot, 1):.1%}', flush=True)
    if st:
        top = '  '.join(f'{s}:{c}' for s, c in st.most_common(8))
        print(f'    误标真球员最多的序列：{top}', flush=True)


def print_miss_reasons(reasons: Counter) -> None:
    tot = reasons['应否决跨侧链接']
    print('\n' + '=' * 100)
    print(f'[漏否决归因] 被采纳的跨侧假链接共 {tot} 条，逐条查为什么没被否掉：', flush=True)
    for key, cnt in reasons.most_common():
        if key == '应否决跨侧链接':
            continue
        print(f'  {key:<22}{cnt:>6}  {cnt / max(tot, 1) * 100:5.1f}%')
    done = reasons['已否决']
    print(f'  {"-> 已否决":<22}{done:>6}  {done / max(tot, 1) * 100:5.1f}%')


def _linker_for(lookup, mode: str) -> AHE.HybridAFLink:
    return make_veto_linker(lookup, mode)


def count_confusion(cand_rows, seq_info, th: float, sil_min: float, min_sep: float) -> Counter:
    """候选级混淆计数器（口径与 evaluate_pixel 的表格一致），供特征变体横向对比复用。"""
    st = Counter()
    for seq, same_owner, kind, diff_conf in cand_rows:
        sil, sep = seq_info.get(seq, (0.0, 0.0))
        veto = diff_conf is not None and diff_conf >= th and sil >= sil_min and sep >= min_sep
        st['候选'] += 1
        st['否决'] += veto
        if same_owner:
            st['真候选'] += 1
            st['误伤真候选'] += veto
        elif kind == '跨侧（队色可否定）':
            st['跨侧'] += 1
            st['命中跨侧'] += veto
            if not veto:
                st['跨侧漏'] += 1
                if sil < sil_min or sep < min_sep:
                    st['漏因未分类'] += 1
    return st


def evaluate_pixel(preds, seqs, cache, linker, sil_thrs, conf_th: float = PIX_CONF,
                   feature: str = 'ab') -> None:
    """像素队色分类器的准确率与候选级混淆 —— GT 队别标签正好是免费标注。"""
    pix = pixel_predict(cache, seqs, feature)
    min_sep = feature_min_sep(feature)
    seq_rows, cand_rows, n_no_color = [], [], 0
    for seq in seqs:
        kits, info = load_kit_map(seq), pix.get(seq)
        if not kits or not info:
            continue
        owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
        lab, conf = info['label'], info['conf']
        n_no_color += sum(1 for t in preds[seq] if lab.get(t) is None)
        judged = [(t, side_of(kits[o])) for t, o in owners.items()
                  if o in kits and side_of(kits[o]) and lab.get(t) is not None]
        acc = None
        if len(judged) >= 4:      # 簇标签 0/1 与 left/right 的对应关系由多数投票定，另一侧随之确定
            vote = Counter((lab[t], s) for t, s in judged)
            c0_side = 'left' if vote[(0, 'left')] >= vote[(0, 'right')] else 'right'
            acc = sum(1 for t, s in judged if (s == c0_side) == (lab[t] == 0)) / len(judged)
        seq_rows.append((seq, acc, len(judged), info['sep'], info['sil']))
        for cand in linker._build_candidates(preds[seq], EX.NOMINAL_GAP, EX.NOMINAL_DIST):
            sa, sb = lab.get(cand.src_id), lab.get(cand.dst_id)
            # diff_conf = 两碎片被判为不同簇时较低的置信度；同簇/未知 -> None（永不否决）
            diff_conf = None if (sa is None or sb is None or sa == sb) else min(
                conf.get(cand.src_id, 0.0), conf.get(cand.dst_id, 0.0))
            oa, ob = owners.get(cand.src_id), owners.get(cand.dst_id)
            cand_rows.append((seq, oa is not None and oa == ob,
                              classify(kits.get(oa), kits.get(ob)), diff_conf))

    ok = [(a, n) for _s, a, n, _sep, _sil in seq_rows if a is not None]
    if ok:
        arr = np.array([a for a, _ in ok])
        q = np.percentile(arr, [10, 50, 90])
        micro = sum(a * n for a, n in ok) / max(sum(n for _, n in ok), 1)
        print(f'\n碎片级队别准确率（仅 GT 队别可判且取色成功）：可判序列 {len(arr)}/{len(seq_rows)} | '
              f'微平均 {micro:.1%} | P10/P50/P90 {q[0]:.1%}/{q[1]:.1%}/{q[2]:.1%}', flush=True)
    print(f'未能取色/聚类的碎片累计：{n_no_color}', flush=True)

    seq_info = {s: (sil, sep) for s, _a, _n, sep, sil in seq_rows}
    print('\n序列级门控（silhouette 越高说明两队色越好分；门槛以下的序列整体不否决）：', flush=True)
    print(f"  {'sil门槛':>8}{'通过序列':>9}{'覆盖碎片':>9}{'微平均准确率':>13}")
    for s in sil_thrs:
        rows = [(a, n) for _sq, a, n, _sep, sil in seq_rows if sil >= s and a is not None]
        if not rows:
            print(f'  {s:>8.2f}{0:>9}{0:>9}{"n/a":>13}')
            continue
        tot = sum(n for _, n in rows)
        print(f'  {s:>8.2f}{len(rows):>9}{tot:>9}{sum(a * n for a, n in rows) / max(tot, 1):>12.1%}')
    worst = sorted([r for r in seq_rows if r[1] is not None], key=lambda r: r[4])[:15]
    print('\n  按 silhouette 最低的 15 条序列（准确率 = 碎片级队别准确率）：')
    print(f"    {'序列':>12}{'sil':>7}{'簇距':>7}{'准确率':>9}{'可判碎片':>9}")
    for sq, a, n, sep, sil in worst:
        print(f"    {sq:>12}{sil:>7.2f}{sep:>7.1f}{a:>8.0%}{n:>9}")

    print(f'\n候选级混淆（锥形门控候选池，特征={feature}，簇距门槛={min_sep:.1f}）：', flush=True)
    print(HEAD_CONFUSION)
    for s in sil_thrs:                      # 先扫序列级 silhouette 门
        print_confusion_row(f'sil{s:.2f}', count_confusion(cand_rows, seq_info, 0.0, s, min_sep))
    print()
    for th in CONF_THRS:                    # 再扫碎片级置信门（误伤通道就在这里）
        print_confusion_row(f'conf{th:.2f}', count_confusion(cand_rows, seq_info, th, 0.30, min_sep))


def dump_seq(preds, seq: str, cache, feature: str = 'ab') -> None:
    """调试：逐碎片打印 GT 队别 / 聚类簇 / 取到的颜色，定位聚类为何对不上队别。"""
    kits = load_kit_map(seq)
    owners = EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq))
    pix = pixel_predict(cache, [seq], feature)[seq]
    rows = cache.get(seq) or {}
    n_lab = sum(1 for v in pix['label'].values() if v is not None)
    print(f"\n[debug] {seq} 碎片 {len(preds[seq])} | 成功分簇 {n_lab} | "
          f"簇间距 {pix['sep']:.1f} | silhouette {pix['sil']:.2f}")
    print(f"  {'tid':>6}{'GT队别':>9}{'簇':>4}{'conf':>7}{'a*':>7}{'b*':>7}{'L':>7}{'S':>7}{'帧':>4}")
    for tid in sorted(preds[seq]):
        desc = [d['j'] for d in rows.get(tid, []) if 'j' in d]
        mean = (lambda k: float(np.mean([d[k] for d in desc])) if desc else float('nan'))
        lab = pix['label'].get(tid)
        print(f"  {tid:>6}{str(kits.get(owners.get(tid), '-')):>9}"
              f"{('-' if lab is None else lab):>4}{pix['conf'].get(tid, 0.0):>7.3f}"
              f"{mean('a'):>7.1f}{mean('b'):>7.1f}{mean('L'):>7.1f}{mean('S'):>7.1f}{len(desc):>4}")


def experiment_pixels(preds, seqs, cache, linker, frames_per_frag: int,
                      feature: str = 'ab', sweep_features: bool = False,
                      no_hota: bool = False, links: bool = False,
                      veto_outliers: bool = False, sweep_outliers: bool = False,
                      trim_frac: float = 0.15, max_trim: int = 2,
                      out_conf: float | None = None, out_unlab: bool = False,
                      out_strict: bool = False, only_out: bool = False) -> None:
    print('\n' + '=' * 100)
    print(f'[2/2] 像素版队色否决（每碎片最多 {frames_per_frag} 帧，逐序列 k=2 聚类，'
          f'特征={feature}，全程无训练）', flush=True)
    evaluate_pixel(preds, seqs, cache, linker, SIL_THRS, feature=feature)

    if sweep_features:
        compare_features(preds, seqs, cache, linker, FEATURE_CHOICES)
    if links:
        diagnose_links(preds, seqs, cache, feature=feature, veto_outliers=veto_outliers,
                       trim_frac=trim_frac, max_trim=max_trim)
        for th in (0.25, 0.45):
            print_miss_reasons(explain_missed_links(preds, seqs, cache, feature, th))
    if sweep_outliers:
        sweep_outlier_trim(preds, seqs, cache, feature=feature)
        sweep_outlier_conf(preds, seqs, cache, feature=feature,
                           trim_frac=trim_frac, max_trim=max_trim)
        explain_ref_ball_links(preds, seqs, cache, feature=feature,
                               trim_frac=trim_frac, max_trim=max_trim)
        for oc, ou, ost in ((out_conf, out_unlab, out_strict), (0.15, True, False),
                            (None, True, True)):
            audit_outlier_labels(preds, seqs, cache, feature=feature,
                                 trim_frac=trim_frac, max_trim=max_trim,
                                 out_conf=oc, out_unlab=ou, out_strict=ost)
    if no_hota:
        print('\n[--no-hota] 只出混淆表，跳过 HOTA 对照', flush=True)
        return

    owners_by_seq = {seq: EX.assign_gt_owner(preds[seq], EX.load_gt_boxes(EX.GT, seq)) for seq in seqs}
    pix = pixel_predict(cache, seqs, feature, trim_frac, max_trim)
    runs = []
    if not only_out:                    # --only-out：基线/conf 扫描都测过了，跳过省时间
        runs += [
            ('B_no_veto', '无否决对照', 'off', {}),
            ('ORACLE_side', 'oracle 跨侧否决（上界参考）', 'side', build_kit_map(preds, seqs)),
        ]
        for th in PIX_CONF_SWEEP:
            runs.append((f'PIX_c{int(th * 100):02d}', f'像素否决 conf>={th:.2f}, sil>=0.30', 'any',
                         build_pixel_lookup(preds, seqs, cache, th, 0.30, feature=feature, pix=pix)))
        runs.append(('PIX_c50_nosil', 'conf>=0.50 但关掉 silhouette 门（看序列级门是否必要）', 'any',
                     build_pixel_lookup(preds, seqs, cache, 0.50, 0.0, feature=feature, pix=pix)))
        runs.append(('PIX_c50_owned', 'conf>=0.50 且只否有 GT 归属碎片（消融：虚检碎片是否拖后腿）', 'any',
                     build_pixel_lookup(preds, seqs, cache, 0.50, 0.30,
                                        owners_by_seq=owners_by_seq, feature=feature, pix=pix)))
    if veto_outliers:
        # 非默认剪枝参数必须写进目录名，否则会把默认参数的既有结果覆盖掉
        tsfx = ('' if (abs(trim_frac - 0.15) < 1e-9 and max_trim == 2)
                else f'_t{int(round(trim_frac * 100)):02d}m{int(max_trim)}')
        cands = [(out_conf, out_unlab, out_strict)]
        if sweep_outliers:                      # 便宜口径已筛过，这里只跑最有希望的两档
            cands = [(0.10, True, False), (None, True, True)]
        for th in (0.25, 0.45):
            for oc, ou, ost in cands:
                tag = (f'PIX_c{int(th * 100):02d}_out' + tsfx
                       + (f'_oc{int(oc * 100):02d}' if oc is not None else '')
                       + ('u' if ou else '') + ('S' if ost else ''))
                note = (f'conf>={th:.2f} 且额外否决「离群碎片(裁判/球/异色门将) vs 队内」'
                        f'（剪枝 {trim_frac}/{max_trim}'
                        + (f'，低置信离群<{oc:.2f}' if oc is not None else '')
                        + ('，未取色也算离群' if ou else '')
                        + ('，严格模式:非高置信队内一律算离群' if ost else '') + '）')
                runs.append((tag, note, 'any',
                             build_pixel_lookup(preds, seqs, cache, th, 0.30, feature=feature,
                                                veto_outliers=True, pix=pix, out_conf=oc,
                                                out_unlab=ou, out_strict=ost)))
    print(f'\nHOTA 对照（锥形门控 g0={EX.CONE_G0:.0f}，其余同 §12 推荐配置）：', flush=True)
    results = []
    for key, note, mode, lookup in runs:
        results.append((key, note) + run_veto_hota(preds, seqs, key, mode, lookup))
    print_hota_table(results, [(k, n) for k, n, _m, _l in runs])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['ceiling', 'upper', 'pixels'], default='ceiling')
    parser.add_argument('--frames-per-frag', type=int, default=8, help='每条碎片采样帧数')
    parser.add_argument('--refresh', action='store_true', help='丢弃取色缓存重算')
    parser.add_argument('--jobs', type=int, default=1, help='取色阶段并行进程数（按序列切分，默认 1=串行）')
    parser.add_argument('--limit', type=int, default=0, help='只跑前 N 个序列（冒烟测试/估时）')
    parser.add_argument('--debug-seq', default='', help='只转储该序列的逐碎片取色详情后退出')
    parser.add_argument('--extract-only', action='store_true', help='只生成取色缓存后退出')
    parser.add_argument('--feature', choices=FEATURE_CHOICES, default='abL',
                        help='碎片颜色特征：abL(球衣色度+明度, 实测最优) / ab / ab_lower(短裤) / ab_both')
    parser.add_argument('--sweep-features', action='store_true',
                        help='追加特征变体对比表（只出混淆表，不重复跑 HOTA）')
    parser.add_argument('--no-hota', action='store_true', help='只出取色/混淆表，跳过 HOTA 对照')
    parser.add_argument('--links', action='store_true',
                        help='追加链接级诊断（只看被采纳的链接，HOTA 真正在乎的口径）')
    parser.add_argument('--veto-outliers', action='store_true',
                        help='额外否决「离群小簇碎片（裁判/球/异色门将）vs 队内碎片」的链接')
    parser.add_argument('--sweep-outliers', action='store_true',
                        help='链接级扫描离群剪枝强度网格（trim_frac, max_trim），挑完再跑 HOTA')
    parser.add_argument('--trim-frac', type=float, default=0.15,
                        help='离群剪枝的占比上限（默认 0.15，越大抓到越多裁判/球）')
    parser.add_argument('--max-trim', type=int, default=2,
                        help='离群剪枝最多迭代几轮（默认 2）')
    parser.add_argument('--out-conf', type=float, default=None,
                        help='低置信即离群的门槛（<队别 conf 门槛才有意义；默认关闭）')
    parser.add_argument('--out-unlab', action='store_true',
                        help='把「取色失败/未分类」的碎片也算离群（能抓小球，但会误标取色失败的球员）')
    parser.add_argument('--out-strict', action='store_true',
                        help='严格模式：只有「高置信判成某队」的碎片算球员，其余一律算离群')
    parser.add_argument('--only-out', action='store_true',
                        help='HOTA 只跑离群否决各档，跳过已测过的基线/conf 扫描（换剪枝参数复测时省时间）')
    parser.add_argument('--dump-labels', default=None,
                        help='按当前参数把队别标签表写成 JSON（{seq:{tid:label}}）后退出，'
                             '供生产入口 aflink_hybrid_eval.py --kit_veto/--kit_labels 复用')
    args = parser.parse_args()
    seqs = EX.split_sequences()
    if args.limit:
        seqs = seqs[:args.limit]
    preds = AHE.load_mot_predictions(EX.RESULTS, seqs)
    linker = EX.build_linker(rule_variant='adaptive_cone', cone_g0=EX.CONE_G0)
    print(f'序列数={len(seqs)} | 输入链={EX.RESULTS.name} | 门控=adaptive_cone(g0={EX.CONE_G0}) '
          f'gap={EX.NOMINAL_GAP} dist={EX.NOMINAL_DIST:.0f}', flush=True)

    n_true = n_false = 0
    false_kind = Counter()
    false_kind_by_gap = {b: Counter() for b in [f'{lo}-{hi}' for lo, hi in GAP_BUCKETS]}
    true_kind = Counter()
    unmatched_frag = Counter()
    per_seq_cross = []

    for seq in seqs:
        per_frame_gt = EX.load_gt_boxes(EX.GT, seq)
        owners = EX.assign_gt_owner(preds[seq], per_frame_gt)
        kits = load_kit_map(seq)
        if not kits:
            print(f'  [警告] {seq} 缺 gameinfo.ini，跳过', flush=True)
            continue
        for frag_owner in owners.values():
            if frag_owner not in kits:
                unmatched_frag[seq] += 1
        seq_true = 0
        seq_cross = 0
        for cand in linker._build_candidates(preds[seq], EX.NOMINAL_GAP, EX.NOMINAL_DIST):
            oa = owners.get(cand.src_id)
            ob = owners.get(cand.dst_id)
            if oa is None or ob is None:
                unmatched_frag['候选无归属'] += 1
                continue
            ka, kb = kits.get(oa), kits.get(ob)
            if oa == ob:
                n_true += 1
                seq_true += 1
                true_kind[classify(ka, kb)] += 1
            else:
                n_false += 1
                kind = classify(ka, kb)
                false_kind[kind] += 1
                bucket = gap_bucket(cand.gap)
                if bucket in false_kind_by_gap:
                    false_kind_by_gap[bucket][kind] += 1
                if kind == '跨侧（队色可否定）':
                    seq_cross += 1
        false_all = 0
        for cand in linker._build_candidates(preds[seq], EX.NOMINAL_GAP, EX.NOMINAL_DIST):
            oa, ob = owners.get(cand.src_id), owners.get(cand.dst_id)
            if oa is not None and ob is not None and oa != ob:
                false_all += 1
        if false_all:
            per_seq_cross.append(seq_cross / false_all)

    print('\n' + '=' * 100)
    print(f'候选池：真 {n_true} / 假 {n_false}（无 GT 归属的候选 {unmatched_frag["候选无归属"]} 已剔除）')
    print('\n假候选构成（队色能覆盖多少）：')
    for kind, cnt in false_kind.most_common():
        print(f'  {kind:<34} {cnt:>6}  {cnt / max(n_false, 1) * 100:5.1f}%')
    print('\n真候选构成（应为「完全同 kit」，否则说明标签映射错位）：')
    for kind, cnt in true_kind.most_common():
        print(f'  {kind:<34} {cnt:>6}  {cnt / max(n_true, 1) * 100:5.1f}%')
    print('\n按 gap 分桶的假候选（跨侧占比 = 队色可直接排除的比例）：')
    print(f'  {"gap":<8}{"假候选":>8}{"跨侧":>8}{"跨侧占比":>10}{"完全同kit":>10}')
    for bucket, counter in false_kind_by_gap.items():
        tot = sum(counter.values())
        cross = counter['跨侧（队色可否定）']
        same = counter['完全同 kit（队色无法区分）']
        print(f'  {bucket:<8}{tot:>8}{cross:>8}{(cross / tot * 100 if tot else 0):>9.1f}%{same:>10}')
    if per_seq_cross:
        per_seq_cross.sort()
        n = len(per_seq_cross)
        print(f'\n逐序列跨侧占比：min {per_seq_cross[0]:.1%} / P50 {per_seq_cross[n // 2]:.1%} / '
              f'max {per_seq_cross[-1]:.1%}（{n} 个序列）')
    if unmatched_frag:
        print(f'\n其他统计：{dict(list(unmatched_frag.items())[:6])}')

    if args.stage == 'upper':
        experiment_upper(preds, seqs, build_kit_map(preds, seqs))
    elif args.stage == 'pixels':
        cache = load_color_cache(preds, seqs, args.frames_per_frag, args.refresh, args.jobs)
        if args.extract_only:
            print(f'取色缓存完成：{len(cache)} 个序列 -> {PIX_CACHE}', flush=True)
            return
        if args.debug_seq:
            dump_seq(preds, args.debug_seq, cache, args.feature)
            return
        if args.dump_labels:
            dump_kit_labels(preds, seqs, cache, args.dump_labels, feature=args.feature,
                            th=PIX_CONF, sil_min=0.30, veto_outliers=args.veto_outliers,
                            trim_frac=args.trim_frac, max_trim=args.max_trim,
                            out_conf=args.out_conf, out_unlab=args.out_unlab,
                            out_strict=args.out_strict)
            return
        experiment_pixels(preds, seqs, cache, linker, args.frames_per_frag,
                          feature=args.feature, sweep_features=args.sweep_features,
                          no_hota=args.no_hota, links=args.links,
                          veto_outliers=args.veto_outliers,
                          sweep_outliers=args.sweep_outliers,
                          trim_frac=args.trim_frac, max_trim=args.max_trim,
                          out_conf=args.out_conf, out_unlab=args.out_unlab,
                          only_out=args.only_out,
                          out_strict=args.out_strict)


if __name__ == '__main__':
    main()
