from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable


import numpy as np

# 兼容旧版 TrackEval / numpy 别名
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import torch
import torch.nn as nn
import torch.nn.functional as F
import trackeval
from scipy.optimize import linear_sum_assignment


def resolve_project_root() -> Path:
    env_candidates = [
        os.environ.get('FUSSBALL_ROOT'),
    ]
    for raw in env_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if (candidate / 'association.py').exists():
            return candidate

    local_root = Path(__file__).resolve().parent
    if (local_root / 'association.py').exists():
        return local_root

    # 不写死任何绝对路径：从脚本所在目录逐级向上回溯，
    # 同时兼容「扁平布局 <root>/association.py」与「仓库布局 <root>/fussball/association.py」
    for candidate in [local_root, *local_root.parents]:
        if (candidate / 'association.py').exists() or (candidate / 'fussball' / 'association.py').exists():
            return candidate.resolve()

    return local_root


PROJECT_ROOT = resolve_project_root()
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'tracker_results' / 'sensitivity_aflink_local' / 'aflink_max_gap_60'
DEFAULT_GT_ROOT = PROJECT_ROOT / 'tracking' / 'test' / 'test'
DEFAULT_CACHE_PATH = PROJECT_ROOT / 'raw_pred_cache.json'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'tracker_results' / 'hybrid_aflink_eval'
DEFAULT_REPORT_PATH = PROJECT_ROOT / 'sensitivity_report.md'
DEFAULT_SEQMAP_CANDIDATES = [
    PROJECT_ROOT / 'sn-tracking-main' / 'tools' / 'SNMOT-test.txt',
    PROJECT_ROOT / 'tools' / 'SNMOT-test.txt',
]
EXPECTED_TEST_SEQUENCE_COUNT = 49
DEFAULT_WEIGHT_CANDIDATES = [
    PROJECT_ROOT / 'aflink_model_v3.pth',
    PROJECT_ROOT / 'aflinknetv2.pth',
    PROJECT_ROOT / 'aflink_net_v2.pth',
    PROJECT_ROOT / 'aflink_v2.pth',
    PROJECT_ROOT / 'aflink.pth',
    PROJECT_ROOT / 'weights' / 'aflinknetv2.pth',
]
RULE_VARIANT_CHOICES = [
    'baseline',
    'adaptive_gate',
    'tri_factor',
    'adaptive_tri',
    'gap_decay',
    'length_conf',
    'gap_length',
    'adaptive_cone',
    'residual_gate',
    'kinematic_gate',
]
# 方案8：队色否决（跨队/跨侧 与 离群碎片 的候选级硬否决）。
# 这是**正交于 rule_variant 的一道独立门**，默认 off = 完全不改变既有行为。
KIT_VETO_CHOICES = ['off', 'side', 'any', 'all']
KIT_OUTLIER_LABEL = 'cX'

# ---------------------------------------------------------------------------
# MCVC（Multi-Cue Veto Cascade，多线索否决级联）—— 取代「自适应门控」的统一组件。
#
# 设计原则（来自本项目实测结论，见 方案9_运动学门控_解决方案.md §9.15）：
#   **否决型增益的上限由「否决的精确率」决定，而不是由可用信息量决定。**
#   证据：方案8 队色否决否决率 56~59% → +0.47~+0.77；方案10 方向否决率 57.4%（几乎相同）
#   → −0.849。二者形态同为「离散否决」，唯一差别是精确率。
#
# 因此 MCVC 只保留**高精确率**的否决层，且全部「只删不增」：
#   V1 时序-几何锥形剪枝（原门控，降级为剪枝器，gate_score_norm=nominal，不参与评分标定）
#   V2 队色等价类否决（kit_veto=any，取代评分型门控成为主增益）
#   V3 一致闭包否决（本轮新增，实测负向/不触发 → 默认 off）
#       V3a 互选一致否决     --mcv_recip
#       V3b 分量纯度否决     --mcv_trans   ← 用全局一致性补上 V2 的「未知标签放行」漏洞
#   V4 离群实体否决（并入 V2 的 cX 标签）
#   V5 球-人互斥否决        --mcv_ball_excl
#
# ---------- 双划分消融结论（test49 / train57，rp=0.0，ΔHOTA，见 _mcv/report_*.md）----------
#   基线阶梯  A0 原始(30/75) → A1 宽窗(150/450) → [T1 锥形] → [T2 队色否决]
#   ------------------------------------------------------------------
#   T2 队色否决（组件核心）      +0.774 / +0.689（在名义窗口上单独开启）
#   T2 之上的 T1 锥形剪枝        −0.091 / +0.057   ← 符号翻转 ≈0，降为可选（--mcv_cone）
#   V3a 互选否决                 −0.173 / −0.028   ← 负向，默认 off
#   V3b 分量纯度                  0.000 /  0.000   ← 从不触发（死层），默认 off
#   T4 回补轮 400                +0.098 / −0.092   ← 划分不稳定，默认 off
#   全量级联（V3a+V3b+T4 全开）   +0.157 / −0.107   ← 同上，默认 off
#   ------------------------------------------------------------------
# 即：**新增的 V3/V5/T4 三个「聪明层」全部没能超过 T2 的精确率**，恰好印证上面那条
# 设计原则——信息量更大并不等于增益更大。故 MCVC 的默认形态定为「T2 单独」：
# 名义窗口 150/450 + 队色等价类否决，**原自适应门控退场**（不再参与评分与筛选）。
# 实测 64.143(test) / 64.517(train)，优于锥形版 64.052 / 64.574 的极小极大（64.143 > 64.052）。
#
# V3b 的形式化依据（设计动机，虽实测不触发，保留以备后续复用）：V2 把「队色」当成等价
# 关系，链接必须**保持等价类**。由于 V2 刻意不对称（未知标签一律放行），会出现
# s(A) → d(未知) → e(B) 这种跨类路径。该路径上**至少有一条边是假链接**。取候选图上按得分
# 递减加边的并查集：**第一条把两个不同队色连通起来的边**，就是最大可靠性路径（widest
# path）上的最弱环节，删之即可切断该不一致。这是一个**可证明的必要条件**，零参数、零训练。
MCV_RECIP_CHOICES = ['off', 'one', 'both']
MCV_TRANS_CHOICES = ['off', 'weak']


def kit_side_of(label: str | None) -> str | None:
    """队别标签 -> 侧别。left/left_gk -> 'left'，right/right_gk -> 'right'，其余 -> None。

    注意 None 在 'side' 模式下不代表「无法判断」，而代表「不属于任何一侧」：
    所以「裁判/球 vs 球员」会被判为跨侧而否决 —— 实测这部分收益比跨侧召回本身还大。
    """
    if label in ('left', 'left_gk'):
        return 'left'
    if label in ('right', 'right_gk'):
        return 'right'
    return None


def load_kit_labels(path: str | Path) -> dict[str, dict[int, str]]:
    """读队别标签表 JSON：{seq: {track_id: label}}。

    标签口径两种都支持（由提供者决定）：
      - 像素版：'c0'/'c1'/'cX'（cX = 离群碎片：裁判/球/异色门将/取色失败/非高置信）
      - oracle：'left'/'right'/'left_gk'/'right_gk'/'ref'/'ball'（仅测上界用，不可上线）
    由 `_probe_color_team.py --stage pixels --dump-labels <PATH>` 生成。
    """
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict) and 'sequences' in raw:
        raw = raw['sequences']
    if not isinstance(raw, dict):
        raise ValueError('队别标签文件顶层必须是 dict。')
    out: dict[str, dict[int, str]] = {}
    for seq_name, table in raw.items():
        if not isinstance(table, dict):
            continue
        out[str(seq_name)] = {int(tid): str(label) for tid, label in table.items()}
    return out


BEST_TUNED_CONFIG = {
    'rule_variant': 'adaptive_tri',
    'max_gap': 100,
    'dist_thresh': 200,
    'gate_reliability_weight': 0.9,
    'gate_shape_weight': 0.1,
    'temporal_weight': 0.5,
    'distance_weight': 0.3,
    'shape_weight': 0.2,
    'shape_area_weight': 0.2,
    'shape_aspect_weight': 0.8,
    'adaptive_len_ref': 30,
    'adaptive_gap_min_ratio': 0.50,
    'adaptive_dist_min_ratio': 0.50,
}


def load_default_test_sequence_names(gt_root: Path | None = None) -> list[str]:

    for seqmap_path in DEFAULT_SEQMAP_CANDIDATES:
        if seqmap_path.exists():
            seqs = [
                line.strip()
                for line in seqmap_path.read_text(encoding='utf-8').splitlines()
                if line.strip() and line.strip().lower() != 'name'
            ]
            if len(seqs) != EXPECTED_TEST_SEQUENCE_COUNT:
                raise RuntimeError(
                    f'SNMOT 测试序列清单应为 {EXPECTED_TEST_SEQUENCE_COUNT} 个，'
                    f'但在 {seqmap_path} 中读取到 {len(seqs)} 个。'
                )
            return seqs

    if gt_root is not None and gt_root.exists():
        seqs = sorted(
            p.name
            for p in gt_root.iterdir()
            if p.is_dir() and (p / 'gt' / 'gt.txt').exists() and (p / 'seqinfo.ini').exists()
        )
        if len(seqs) == EXPECTED_TEST_SEQUENCE_COUNT:
            return seqs

    raise FileNotFoundError(
        '未找到有效的 49 序列测试清单，请检查 `sn-tracking-main/tools/SNMOT-test.txt`。'
    )


@dataclass
class TrackFragment:



    track_id: int
    frames: list[int]
    boxes: list[list[float]]

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def first_box(self) -> list[float]:
        return self.boxes[0]

    @property
    def last_box(self) -> list[float]:
        return self.boxes[-1]

    @property
    def frame_set(self) -> set[int]:
        return set(self.frames)


@dataclass
class CandidatePair:
    src_id: int
    dst_id: int
    gap: int
    distance: float
    rule_score: float
    temporal_score: float = 0.0
    distance_score: float = 0.0
    shape_score: float = 1.0
    length_confidence: float = 1.0
    src_track_len: int = 0
    dst_track_len: int = 0
    neural_score: float | None = None
    center_distance: float | None = None
    pred_distance: float | None = None


    @property
    def final_score(self) -> float:
        if self.neural_score is not None:
            return float(self.neural_score)
        return float(self.rule_score)



class TemporalModule(nn.Module):
    """旧版时间建模模块：Conv1D + BiGRU + attention pooling。"""

    def __init__(self, in_channels: int = 9, conv_channels: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(1, 2)
        seq_feat, _ = self.gru(x)
        attn_weight = torch.softmax(self.attn(seq_feat).squeeze(-1), dim=1)
        pooled = torch.sum(seq_feat * attn_weight.unsqueeze(-1), dim=1)
        return pooled


class FusionModule(nn.Module):
    """旧版轨迹对融合模块。"""

    def __init__(self, feat_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        fusion_dim = feat_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([feat_a, feat_b, torch.abs(feat_a - feat_b), feat_a * feat_b], dim=1)
        return self.mlp(fusion)


class AFLinkNetV2Legacy(nn.Module):
    """旧版 AFLinkNetV2：共享 temporal + MLP fusion。"""

    def __init__(self, in_channels: int = 9, conv_channels: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.temporal = TemporalModule(
            in_channels=in_channels,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
        )
        self.fusion = FusionModule(feat_dim=hidden_dim * 2, hidden_dim=hidden_dim * 2)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.temporal(seq_a)
        feat_b = self.temporal(seq_b)
        return self.fusion(feat_a, feat_b)


class TemporalConvModuleV3(nn.Module):
    """v3 时间编码器：4 层 Conv1d 堆叠。"""

    def __init__(self, in_channels: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModuleV3(nn.Module):
    """v3 融合模块：对双分支差异特征做时域卷积融合。"""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fused = torch.abs(feat_a - feat_b)
        fused = self.conv(fused)
        fused = self.bn(fused)
        fused = torch.relu(fused)
        return fused


class AFLinkNetV2(nn.Module):
    """v3 AFLinkNetV2：双 temporal 分支 + conv fusion + classifier。"""

    def __init__(self, in_channels: int = 9):
        super().__init__()
        self.temporal_a = TemporalConvModuleV3(in_channels=in_channels)
        self.temporal_b = TemporalConvModuleV3(in_channels=in_channels)
        self.fusion = FusionModuleV3(channels=256)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.temporal_a(seq_a)
        feat_b = self.temporal_b(seq_b)
        fused = self.fusion(feat_a, feat_b)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool1d(feat_a, 1).squeeze(-1),
                F.adaptive_avg_pool1d(feat_b, 1).squeeze(-1),
                F.adaptive_avg_pool1d(fused, 1).squeeze(-1),
            ],
            dim=1,
        )
        return self.classifier(pooled)



class UnionFind:
    def __init__(self, nodes: Iterable[int]):
        self.parent = {int(node): int(node) for node in nodes}

    def find(self, x: int) -> int:
        x = int(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return dict(groups)


class HybridAFLink:
    def __init__(
        self,
        mode: str = 'hybrid',
        max_gap: int = 60,
        dist_thresh: float = 200.0,
        score_thresh: float = 0.55,
        seq_len: int = 30,
        model_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 256,
        rule_variant: str = 'baseline',
        gap_decay_alpha: float = 3.0,
        length_ref: int = 30,
        length_conf_power: float = 1.0,
        adaptive_len_ref: int = 30,
        adaptive_gap_min_ratio: float = 0.5,
        adaptive_dist_min_ratio: float = 0.5,
        adaptive_gap_max_ratio: float = 1.0,
        adaptive_dist_max_ratio: float = 1.0,
        adaptive_gate_hard_filter: bool = True,
        adaptive_gate_signal_invert: bool = False,
        gate_score_norm: str = 'effective',
        gate_signal_ref: float | None = None,
        gate_signal_gain: float = 1.0,
        gate_reliability_weight: float = 0.7,
        gate_shape_weight: float = 0.3,
        area_weight: float = 0.6,
        aspect_weight: float = 0.4,
        temporal_weight: float = 0.4,
        distance_weight: float = 0.4,
        shape_weight: float = 0.2,
        size_tiebreak: bool = True,
        assist_weight: float = 0.2,
        cone_g0: float = 60.0,
        cone_power: float = 1.0,
        cone_floor: float = 0.0,
        score_reliability_penalty: float = 0.0,
        penalty_len_ref: int = 30,
        scale_norm: str = 'none',
        kin_kappa: float = 0.0,
        kin_floor: float = 0.0,
        kin_ceil: float = 0.0,
        kin_window: int = 5,
        kin_min_len: int = 4,
        kin_pair: bool = False,
        kin_dir_tau: float = 0.0,
        kin_dir_vref: float = 1.0,
        kin_dir_pair: bool = True,
        residual_blend: float = 1.0,
        residual_velocity_clip: float = 0.0,
        residual_velocity_window: int = 2,
        residual_min_src_len: int = 2,
        residual_keep_center_gate: bool = False,
        residual_gap_cap: int = 0,
        kit_veto: str = 'off',
        kit_lookup: dict | None = None,
        mcv_recip: str = 'off',
        mcv_trans: str = 'off',
        mcv_trans_max: int = 1,
        mcv_ball_excl: bool = False,
        mcv_ball_rel_h: float = 0.45,
        mcv_ball_aspect: float = 0.75,
        mcv_recover: bool = False,
        mcv_recover_gap: int = 400,
        mcv_recover_dist: float = 1200.0,
        mcv_recover_permissive: bool = False,
    ):
        self.mode = mode
        self.max_gap = int(max_gap)
        self.dist_thresh = float(dist_thresh)
        self.score_thresh = float(score_thresh)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.rule_variant = normalize_rule_variant(rule_variant)
        self.gap_decay_alpha = float(gap_decay_alpha)
        self.length_ref = max(int(length_ref), 1)
        self.length_conf_power = max(float(length_conf_power), 1e-6)
        self.adaptive_len_ref = max(int(adaptive_len_ref), 1)
        self.adaptive_gap_min_ratio = float(np.clip(adaptive_gap_min_ratio, 1e-6, 1.0))
        self.adaptive_dist_min_ratio = float(np.clip(adaptive_dist_min_ratio, 1e-6, 1.0))
        self.adaptive_gap_max_ratio = max(float(adaptive_gap_max_ratio), self.adaptive_gap_min_ratio)
        self.adaptive_dist_max_ratio = max(float(adaptive_dist_max_ratio), self.adaptive_dist_min_ratio)
        self.adaptive_gate_hard_filter = bool(adaptive_gate_hard_filter)
        self.adaptive_gate_signal_invert = bool(adaptive_gate_signal_invert)
        self.gate_score_norm = str(gate_score_norm).strip().lower()
        if self.gate_score_norm not in {'effective', 'nominal'}:
            raise ValueError(f"gate_score_norm 只能是 'effective' 或 'nominal'，收到: {gate_score_norm!r}")
        self.gate_signal_ref = None if gate_signal_ref is None else float(gate_signal_ref)
        self.gate_signal_gain = float(gate_signal_gain)
        self.gate_reliability_weight, self.gate_shape_weight = normalize_pair_weights(
            gate_reliability_weight,
            gate_shape_weight,
        )
        self.area_weight, self.aspect_weight = normalize_pair_weights(
            area_weight,
            aspect_weight,
        )
        self.temporal_weight, self.distance_weight, self.shape_weight = normalize_rule_weights(
            temporal_weight,
            distance_weight,
            shape_weight,
        )
        self.size_tiebreak = bool(size_tiebreak)
        self.assist_weight = float(np.clip(assist_weight, 0.0, 1.0))
        # 方案9 复用锥形作为基准门限，因此也走 cone_mode 分支。
        self.cone_mode = self.rule_variant in {'adaptive_cone', 'kinematic_gate'}
        self.cone_g0 = max(float(cone_g0), 1e-6)
        self.cone_power = max(float(cone_power), 1e-6)
        self.cone_floor = float(np.clip(cone_floor, 0.0, 1.0))
        self.kin_kappa = max(float(kin_kappa), 0.0)
        self.kin_floor = max(float(kin_floor), 0.0)
        self.kin_ceil = max(float(kin_ceil), 0.0)
        self.kin_window = max(int(kin_window), 2)
        self.kin_min_len = max(int(kin_min_len), 2)
        self.kin_pair = bool(kin_pair)
        # 方案10（新）：方向一致性硬否决。方案9 的 κ 只用了速度「模长」，把方向丢掉了；
        # 这里补上方向余弦 —— 它是尺度无关量，不会被 gap 线性放大（这是方案7 残差门控失败的原因）。
        self.kin_dir_tau = float(kin_dir_tau)
        self.kin_dir_vref = max(float(kin_dir_vref), 0.0)
        self.kin_dir_pair = bool(kin_dir_pair)
        self.kin_dir_stats: dict[str, int] = {'checked': 0, 'vetoed': 0}
        self.kin_dir_totals: dict[str, int] = {'checked': 0, 'vetoed': 0}
        self.scale_norm = str(scale_norm).lower()
        self.score_reliability_penalty = float(np.clip(score_reliability_penalty, 0.0, 1.0))
        self.penalty_len_ref = max(int(penalty_len_ref), 1)
        # 方案7：预测残差门控。blend=1 表示纯残差距离，blend=0 退化为端点中心距。
        self.residual_blend = float(np.clip(residual_blend, 0.0, 1.0))
        self.residual_velocity_clip = max(float(residual_velocity_clip), 0.0)
        self.residual_velocity_window = max(int(residual_velocity_window), 2)
        self.residual_min_src_len = max(int(residual_min_src_len), 1)
        self.residual_keep_center_gate = bool(residual_keep_center_gate)
        self.residual_gap_cap = max(int(residual_gap_cap), 0)
        # 方案8：队色否决（正交开关）。off = 零影响；标签表由外部注入（见 set_kit_lookup）。
        self.kit_veto = str(kit_veto).strip().lower() or 'off'
        if self.kit_veto not in KIT_VETO_CHOICES:
            raise ValueError(f"kit_veto 只能是 {KIT_VETO_CHOICES}，收到: {kit_veto!r}")
        self.kit_lookup: dict = dict(kit_lookup) if kit_lookup else {}
        self.current_seq: str | None = None
        self.kit_veto_stats: dict[str, int] = {'candidates': 0, 'vetoed': 0}   # 单序列（每次 link_sequence 重置）
        self.kit_veto_totals: dict[str, int] = {'candidates': 0, 'vetoed': 0}  # 累计（跨序列，供整轮实验统计）
        # MCVC（多线索否决级联）：V3 一致闭包否决 + V5 球-人互斥。
        # 全部默认 off，对既有全部结果零影响。
        self.mcv_recip = str(mcv_recip).strip().lower() or 'off'
        if self.mcv_recip not in MCV_RECIP_CHOICES:
            raise ValueError(f"mcv_recip 只能是 {MCV_RECIP_CHOICES}，收到: {mcv_recip!r}")
        self.mcv_trans = str(mcv_trans).strip().lower() or 'off'
        if self.mcv_trans not in MCV_TRANS_CHOICES:
            raise ValueError(f"mcv_trans 只能是 {MCV_TRANS_CHOICES}，收到: {mcv_trans!r}")
        self.mcv_trans_max = max(int(mcv_trans_max), 0)
        self.mcv_ball_excl = bool(mcv_ball_excl)
        self.mcv_ball_rel_h = max(float(mcv_ball_rel_h), 0.0)
        self.mcv_ball_aspect = float(np.clip(mcv_ball_aspect, 0.0, 4.0))
        # T4 互补回补轮：更深窗口 + 更严标签，专门补回 T1 深窗口漏掉的真链接。
        self.mcv_recover = bool(mcv_recover)
        self.mcv_recover_gap = max(int(mcv_recover_gap), 0)
        self.mcv_recover_dist = max(float(mcv_recover_dist), 0.0)
        self.mcv_recover_permissive = bool(mcv_recover_permissive)
        self.mcv_stats: dict[str, int] = {'checked': 0, 'recip': 0, 'trans': 0, 'ball': 0, 'passed': 0, 'recover': 0}
        self.mcv_totals: dict[str, int] = {'checked': 0, 'recip': 0, 'trans': 0, 'ball': 0, 'passed': 0, 'recover': 0}

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_path = self._resolve_model_path(model_path)
        self.model = None


        if self.mode in {'hybrid', 'hybrid_greedy', 'neural', 'assist'}:
            self.model = self._load_model(self.model_path)


    @staticmethod
    def _resolve_model_path(model_path: str | Path | None) -> Path | None:
        if model_path:
            return Path(model_path).expanduser().resolve()
        for candidate in DEFAULT_WEIGHT_CANDIDATES:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _load_model(self, model_path: Path | None) -> nn.Module:
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                '当前工作区未找到 AFLinkNetV2 权重文件。\n'
                '请通过 --aflink_weights 显式传入权重，例如 aflink_model_v3.pth。'
            )

        state = torch.load(str(model_path), map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('无法解析 AFLinkNetV2 权重格式。')

        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        if any(k.startswith('temporal_a.net.') for k in cleaned):
            model = AFLinkNetV2()
            variant = 'v3'
        elif any(k.startswith('temporal.') for k in cleaned):
            model = AFLinkNetV2Legacy()
            variant = 'legacy'
        else:
            raise ValueError('无法识别 AFLink 权重结构，请检查 state_dict 键名。')

        missing, unexpected = model.load_state_dict(cleaned, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f'AFLink 权重与模型结构不匹配: missing={len(missing)}, unexpected={len(unexpected)}'
            )

        model.eval()
        model.to(self.device)
        print(f'[AFLinkNetV2] 权重结构: {variant}')
        print(f'[AFLinkNetV2] 加载权重: {model_path}')
        print(f'[AFLinkNetV2] 运行设备: {self.device}')
        return model


    def link_sequence(self, track_map: dict[int, TrackFragment]) -> tuple[dict[int, int], dict[str, float | int]]:
        empty_stats: dict[str, float | int] = {
            'raw_candidates': 0,
            'kept_candidates': 0,
            'links': 0,
            'rule_score_sum': 0.0,
            'temporal_score_sum': 0.0,
            'distance_score_sum': 0.0,
            'shape_score_sum': 0.0,
            'length_confidence_sum': 0.0,
            'kit_candidates': 0,
            'kit_vetoed': 0,
            'kin_dir_checked': 0,
            'kin_dir_vetoed': 0,
            'mcv_checked': 0,
            'mcv_recip_vetoed': 0,
            'mcv_trans_vetoed': 0,
            'mcv_ball_vetoed': 0,
            'mcv_recover_added': 0,
        }

        if len(track_map) < 2:
            return {}, empty_stats

        candidate_gap = self.max_gap
        candidate_dist = self.dist_thresh
        if self.mode == 'neural':
            candidate_gap = max(self.max_gap, int(round(self.max_gap * 1.5)))
            candidate_dist = max(self.dist_thresh, self.dist_thresh * 1.5)

        self.reset_kit_veto_stats()
        self.reset_mcv_stats()
        candidates = self._build_candidates(track_map, candidate_gap, candidate_dist)
        raw_count = len(candidates)
        kit_stats: dict[str, float | int] = {
            'kit_candidates': int(self.kit_veto_stats['candidates']),
            'kit_vetoed': int(self.kit_veto_stats['vetoed']),
            'kin_dir_checked': int(self.kin_dir_stats['checked']),
            'kin_dir_vetoed': int(self.kin_dir_stats['vetoed']),
        }
        if not candidates:
            return {}, {**empty_stats, **kit_stats}

        # MCVC V3/V5：一致闭包否决（默认全关 → 本段对既有结果零影响）。
        if self.mcv_active:
            candidates = self.mcv_apply(candidates, track_map)
        mcv_stats: dict[str, float | int] = {
            'mcv_checked': int(self.mcv_stats['checked']),
            'mcv_recip_vetoed': int(self.mcv_stats['recip']),
            'mcv_trans_vetoed': int(self.mcv_stats['trans']),
            'mcv_ball_vetoed': int(self.mcv_stats['ball']),
            'mcv_recover_added': int(self.mcv_stats.get('recover', 0)),
        }
        if not candidates:
            return {}, {**empty_stats, **kit_stats, **mcv_stats}

        post_mcv_count = len(candidates)
        candidate_stats: dict[str, float | int] = {
            'raw_candidates': raw_count,
            'rule_score_sum': float(sum(cand.rule_score for cand in candidates)),
            'temporal_score_sum': float(sum(cand.temporal_score for cand in candidates)),
            'distance_score_sum': float(sum(cand.distance_score for cand in candidates)),
            'shape_score_sum': float(sum(cand.shape_score for cand in candidates)),
            'length_confidence_sum': float(sum(cand.length_confidence for cand in candidates)),
            **kit_stats,
            **mcv_stats,
        }


        if self.mode == 'rule':
            link_map = self._greedy_match(
                candidates,
                use_neural=False,
                rule_variant=self.rule_variant,
                size_tiebreak=self.size_tiebreak,
            )
            if self.mcv_recover_active:
                link_map = self._mcv_recover_pass(link_map, track_map, candidate_gap, candidate_dist)
            return self._flatten_link_map(link_map), {
                **candidate_stats,
                'mcv_recover_added': int(self.mcv_stats.get('recover', 0)),
                'kept_candidates': post_mcv_count,
                'links': len(link_map),
            }

        if self.mode == 'rule_hungarian':
            link_map = self._hungarian_match(candidates, track_map, use_rule_score=True)
            return self._flatten_link_map(link_map), {
                **candidate_stats,
                'kept_candidates': raw_count,
                'links': len(link_map),
            }

        scored = self._score_candidates(candidates, track_map)
        filtered = [cand for cand in scored if cand.neural_score is not None and cand.neural_score >= self.score_thresh]

        if self.mode == 'assist':
            link_map = self._greedy_match(
                scored,
                use_neural='assist',
                rule_variant=self.rule_variant,
                size_tiebreak=self.size_tiebreak,
                assist_weight=self.assist_weight,
            )
            return self._flatten_link_map(link_map), {
                **candidate_stats,
                'kept_candidates': len(scored),
                'links': len(link_map),
            }

        if self.mode == 'hybrid':
            link_map = self._hungarian_match(filtered, track_map)
        elif self.mode == 'hybrid_greedy':
            link_map = self._greedy_match(
                filtered,
                use_neural=True,
                rule_variant=self.rule_variant,
                size_tiebreak=self.size_tiebreak,
            )
        elif self.mode == 'neural':
            link_map = self._union_find_match(filtered, track_map)
        else:
            raise ValueError(f'未知 mode: {self.mode}')

        return self._flatten_link_map(link_map), {
            **candidate_stats,
            'kept_candidates': len(filtered),
            'links': len(link_map),
        }


    def set_kit_lookup(self, kit_lookup: dict | None) -> None:
        """注入/替换队别标签表（方案8）。键可以是序列名 (str) 或 id(track_map) (int)。"""
        self.kit_lookup = dict(kit_lookup) if kit_lookup else {}

    def reset_kit_veto_stats(self) -> None:
        """只重置「单序列」计数；累计计数见 reset_kit_veto_totals。"""
        self.kit_veto_stats = {'candidates': 0, 'vetoed': 0}

    def reset_kit_veto_totals(self) -> None:
        """重置跨序列累计计数（一轮实验开始前调用，也可直接看 kit_veto_totals）。"""
        self.kit_veto_totals = {'candidates': 0, 'vetoed': 0}

    def _kit_table(self, track_map: dict[int, TrackFragment]) -> dict:
        """取当前序列的队别标签表：优先按 current_seq（序列名）取，退回按 id(track_map) 取。"""
        if not self.kit_lookup:
            return {}
        if self.current_seq is not None:
            table = self.kit_lookup.get(self.current_seq)
            if table is not None:
                return table
        return self.kit_lookup.get(id(track_map)) or {}

    def _should_veto_kit(self, label_a: str | None, label_b: str | None) -> bool:
        """方案8 的否决判据。

        off  : 不否决（默认，等价于本功能不存在）
        side : 跨侧否决（用 GT 风格标签 left/right/left_gk/right_gk/ref/ball 测上界）
        any  : 两侧标签都已知且不同即否决（'c0' vs 'c1'、'cX' vs 'c0'/'c1' 都命中）
        all  : 全否（破坏性自检，HOTA 必须崩，否则说明钩子没生效）

        刻意不对称：只有两侧都拿到标签才可能否决 —— 标签未知一律放行（宁可漏否决，不可误否决真候选）。
        """
        if self.kit_veto == 'off':
            return False
        if self.kit_veto == 'all':
            return True
        if label_a is None or label_b is None:
            return False
        if self.kit_veto == 'side':
            return kit_side_of(label_a) != kit_side_of(label_b)
        return label_a != label_b

    def _apply_kit_veto(
        self,
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
    ) -> list[CandidatePair]:
        """在候选生成层出口摘掉「跨队/跨侧」与「离群 vs 队内」的候选。"""
        kit = self._kit_table(track_map)
        kept: list[CandidatePair] = []
        for cand in candidates:
            self.kit_veto_stats['candidates'] += 1
            self.kit_veto_totals['candidates'] += 1
            if self._should_veto_kit(kit.get(cand.src_id), kit.get(cand.dst_id)):
                self.kit_veto_stats['vetoed'] += 1
                self.kit_veto_totals['vetoed'] += 1
            else:
                kept.append(cand)
        return kept

    # ------------------------------------------------------------------
    # MCVC（多线索否决级联）—— 取代「自适应门控」的统一组件
    # ------------------------------------------------------------------
    @property
    def mcv_active(self) -> bool:
        return self.mcv_recip != 'off' or self.mcv_trans != 'off' or self.mcv_ball_excl

    @property
    def mcv_recover_active(self) -> bool:
        return self.mcv_recover

    def reset_mcv_stats(self) -> None:
        self.mcv_stats = {'checked': 0, 'recip': 0, 'trans': 0, 'ball': 0, 'passed': 0, 'recover': 0}

    def reset_mcv_totals(self) -> None:
        self.mcv_totals = {'checked': 0, 'recip': 0, 'trans': 0, 'ball': 0, 'passed': 0, 'recover': 0}

    def mcv_apply(
        self,
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
    ) -> list[CandidatePair]:
        """V3 + V5：一致闭包否决。全部为「只删不增」的离散否决，默认全关。"""
        if not self.mcv_active or not candidates:
            return candidates
        kept = list(candidates)
        if self.mcv_ball_excl:
            kept = self._mcv_ball_exclusive_veto(kept, track_map)
        if self.mcv_recip != 'off':
            kept = self._mcv_reciprocal_veto(kept)
        if self.mcv_trans != 'off':
            kept = self._mcv_component_purity_veto(kept, track_map)
        self.mcv_stats['passed'] = len(kept)
        return kept

    def _mcv_reciprocal_veto(self, candidates: list[CandidatePair]) -> list[CandidatePair]:
        """V3a 互选一致否决。

        对每个 src 取它的最高分 dst，对每个 dst 取它的最高分 src。**双向都不把对方当首选**
        的配对在真链接中罕见（真链接的两端通常互为最近邻），在假链接中常见 → 高精确率。
        `one` = 只要一侧互选即放行（宽松）；`both` = 两侧都要互选（严格）。
        """
        best_dst: dict[int, tuple[float, int]] = {}
        best_src: dict[int, tuple[float, int]] = {}
        for cand in candidates:
            q = float(cand.rule_score)
            cur = best_dst.get(cand.src_id)
            if cur is None or q > cur[0] or (q == cur[0] and cand.dst_id < cur[1]):
                best_dst[cand.src_id] = (q, cand.dst_id)
            cur = best_src.get(cand.dst_id)
            if cur is None or q > cur[0] or (q == cur[0] and cand.src_id < cur[1]):
                best_src[cand.dst_id] = (q, cand.src_id)

        kept: list[CandidatePair] = []
        for cand in candidates:
            self.mcv_stats['checked'] += 1
            self.mcv_totals['checked'] += 1
            ok_src = best_dst.get(cand.src_id, (0.0, -1))[1] == cand.dst_id
            ok_dst = best_src.get(cand.dst_id, (0.0, -1))[1] == cand.src_id
            keep = (ok_src or ok_dst) if self.mcv_recip == 'one' else (ok_src and ok_dst)
            if keep:
                kept.append(cand)
            else:
                self.mcv_stats['recip'] += 1
                self.mcv_totals['recip'] += 1
        return kept

    def _mcv_component_purity_veto(
        self,
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
    ) -> list[CandidatePair]:
        """V3b 分量纯度否决（widest-path 最弱环节销边）。

        队色是等价关系 ⇒ 链接必须保持等价类 ⇒ 候选图的每个连通分量只能含单一已知队色。
        由于 V2 刻意「未知标签一律放行」，分量内可能出现 s(A) → d(未知) → e(B) 的跨类路径。
        按得分**递减**往图上加边：第一条把两个不同已知队色并入同一分量的边，正是它们之间
        最大可靠性路径上的**最弱环节** —— 删掉它即可切断该不一致。每序列最多删
        `mcv_trans_max` 条（默认 1，刻意保守：宁可漏否决，不可误否决）。
        """
        if self.mcv_trans_max <= 0:
            return candidates
        kit = self._kit_table(track_map)
        if not kit:
            return candidates
        ordered = sorted(
            candidates,
            key=lambda c: (-float(c.rule_score), c.gap, c.distance, c.src_id, c.dst_id),
        )
        removed: set[tuple[int, int]] = set()
        for _ in range(self.mcv_trans_max):
            parent: dict[int, int] = {}
            labels: dict[int, str | None] = {}

            def find(x: int) -> int:
                root = x
                while parent[root] != root:
                    root = parent[root]
                while parent[x] != root:
                    parent[x], x = root, parent[x]
                return root

            conflict: CandidatePair | None = None
            for cand in ordered:
                if (cand.src_id, cand.dst_id) in removed:
                    continue
                for node in (cand.src_id, cand.dst_id):
                    if node not in parent:
                        parent[node] = node
                        labels[node] = kit.get(node)
                ra, rb = find(cand.src_id), find(cand.dst_id)
                if ra == rb:
                    continue
                la, lb = labels[ra], labels[rb]
                # 不变量：每个分量内部始终只含一种已知队色，故 la/lb 各自至多一个取值。
                if la is not None and lb is not None and la != lb:
                    conflict = cand
                    break
                merged = la if la is not None else lb
                if ra <= rb:
                    parent[rb] = ra
                    labels[ra] = merged
                else:
                    parent[ra] = rb
                    labels[rb] = merged
            if conflict is None:
                break
            removed.add((conflict.src_id, conflict.dst_id))
            self.mcv_stats['trans'] += 1
            self.mcv_totals['trans'] += 1
        if not removed:
            return candidates
        return [c for c in candidates if (c.src_id, c.dst_id) not in removed]

    @staticmethod
    def _mcv_ball_like(box: list[float], ref_h: float, rel_h: float, aspect: float) -> bool:
        """球相对球员的判别：**绝对高度很小** 且 **接近方形**（球员框是细高的）。"""
        h = float(box[3] - box[1])
        w = float(box[2] - box[0])
        if h <= 0.0 or w <= 0.0 or ref_h <= 0.0:
            return False
        return h <= rel_h * ref_h and (w / h) >= aspect

    def _mcv_ball_exclusive_veto(
        self,
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
    ) -> list[CandidatePair]:
        """V5 球-人互斥否决：疑似球的碎片与疑似人的碎片之间一律否决。

        「球」与「人」在几何上是互斥类别（球不可能变成球员），故该否决的精确率极高。
        参考高度取当前序列全部碎片末帧高度的中位数，避免依赖绝对分辨率。
        """
        heights = sorted(
            float(frag.last_box[3] - frag.last_box[1])
            for frag in track_map.values()
            if frag.last_box[3] - frag.last_box[1] > 0
        )
        if not heights:
            return candidates
        ref_h = heights[len(heights) // 2]
        flag: dict[int, bool] = {}
        for tid, frag in track_map.items():
            flag[tid] = self._mcv_ball_like(frag.last_box, ref_h, self.mcv_ball_rel_h, self.mcv_ball_aspect)
        kept: list[CandidatePair] = []
        for cand in candidates:
            a = flag.get(cand.src_id, False)
            b = flag.get(cand.dst_id, False)
            if a != b:
                self.mcv_stats['ball'] += 1
                self.mcv_totals['ball'] += 1
            else:
                kept.append(cand)
        return kept

    def _mcv_recover_pass(
        self,
        link_map: dict[int, int],
        track_map: dict[int, TrackFragment],
        candidate_gap: int,
        candidate_dist: float,
    ) -> dict[int, int]:
        """T4 互补回补轮。

        方案8 的结论是「否决覆盖率已饱和（95.5%，反超 oracle 91.0%），瓶颈变成**否决之后
        拿什么来补**」——oracle 净赚 61 条真链接，像素版只赚到 31 条。原因是 T1 的宽窗口
        (150/450) 与 T2 的宽松标签（未知标签一律放行）这两个「放宽」是**同向**的：两者都
        只在「更容易通过」的方向上发力，漏掉的集合因此重叠。

        T4 把第二个通行证反过来设计 —— **更深窗口（默认 400/1200）配更严标签（两端都必须
        已知且同队）**：窗口放宽吃掉「真链接被距离门拒掉」的那批，标签收紧保证新增的链接
        精确率。两个通行证的漏检集合互补，故称「互补回补」。

        只在**第一轮留下的链头/链尾**上做（链尾=无后继的 src，链头=无前驱的 dst），因此
        不可能与第一轮的链接竞争，也不可能成环（所有候选都要求 a.end_frame < b.start_frame）。
        """
        if not self.mcv_recover or len(track_map) < 2:
            return link_map
        if self.mcv_recover_gap <= 0:
            return link_map
        has_succ = set(link_map.keys())
        has_pred = set(link_map.values())
        free_src = {tid for tid in track_map if tid not in has_succ}
        free_dst = {tid for tid in track_map if tid not in has_pred}
        if not free_src or not free_dst:
            return link_map

        kit = self._kit_table(track_map)
        # _build_candidates 会再次累计队色/方向否决计数，回补轮是「探索」，不计入主统计。
        snap_kit = dict(self.kit_veto_stats)
        snap_kit_total = dict(self.kit_veto_totals)
        snap_dir = dict(self.kin_dir_stats)
        snap_dir_total = dict(self.kin_dir_totals)
        try:
            pool = self._build_candidates(track_map, self.mcv_recover_gap, self.mcv_recover_dist)
        finally:
            self.kit_veto_stats = snap_kit
            self.kit_veto_totals = snap_kit_total
            self.kin_dir_stats = snap_dir
            self.kin_dir_totals = snap_dir_total

        strict: list[CandidatePair] = []
        for cand in pool:
            if cand.src_id not in free_src or cand.dst_id not in free_dst:
                continue
            if not self.mcv_recover_permissive:
                la = kit.get(cand.src_id)
                lb = kit.get(cand.dst_id)
                if la is None or lb is None or la != lb:
                    continue
            strict.append(cand)
        if not strict:
            return link_map
        extra = self._greedy_match(
            strict,
            use_neural=False,
            rule_variant=self.rule_variant,
            size_tiebreak=self.size_tiebreak,
        )
        if not extra:
            return link_map
        self.mcv_stats['recover'] += len(extra)
        self.mcv_totals['recover'] += len(extra)
        merged = dict(link_map)
        merged.update(extra)
        return merged

    def _build_candidates(
        self,
        track_map: dict[int, TrackFragment],
        max_gap: int,
        dist_thresh: float,
    ) -> list[CandidatePair]:
        self.kin_dir_stats = {'checked': 0, 'vetoed': 0}
        track_ids = sorted(track_map.keys(), key=lambda tid: (track_map[tid].start_frame, track_map[tid].end_frame, tid))
        track_lengths = {tid: len(track_map[tid].frames) for tid in track_ids}
        # 方案9：逐轨迹观测速度（末端用于 src / 首端用于 dst），只在实际启用时计算。
        # 取值 -1 表示轨迹太短、速度不可信，门控届时退回锥形限。
        speed_end: dict[int, float] = {}
        speed_head: dict[int, float] = {}
        vel_end: dict[int, tuple[float, float]] = {}
        vel_head: dict[int, tuple[float, float]] = {}
        if self.kin_kappa > 0.0 or self.kin_dir_tau > 0.0:
            for tid in track_ids:
                frag = track_map[tid]
                if len(frag.frames) >= self.kin_min_len:
                    vx, vy = compute_track_velocity(frag, self.kin_window, 0.0)
                    hx, hy = compute_track_head_velocity(frag, self.kin_window, 0.0)
                    speed_end[tid] = float(math.hypot(vx, vy))
                    speed_head[tid] = float(math.hypot(hx, hy))
                    vel_end[tid] = (float(vx), float(vy))
                    vel_head[tid] = (float(hx), float(hy))
                else:
                    speed_end[tid] = -1.0
                    speed_head[tid] = -1.0
        candidates: list[CandidatePair] = []

        for src_id in track_ids:
            track_a = track_map[src_id]
            src_track_len = track_lengths[src_id]
            for dst_id in track_ids:
                if src_id == dst_id:
                    continue
                track_b = track_map[dst_id]
                if track_a.end_frame >= track_b.start_frame:
                    continue
                gap = track_b.start_frame - track_a.end_frame
                if gap > max_gap:
                    continue
                if not track_a.frame_set.isdisjoint(track_b.frame_set):
                    continue
                dist_center = center_distance(track_a.last_box, track_b.first_box)
                pair_scale = 1.0
                if self.scale_norm != 'none':
                    # 方案11：距离改成「相对于候选对自身尺寸」的无量纲量，
                    # 使门限可以跨分辨率 / 跨划分共用同一套配置。
                    pair_scale = compute_pair_scale(track_a.last_box, track_b.first_box, self.scale_norm)
                    dist_center = dist_center / pair_scale
                dist_pred: float | None = None
                if uses_residual_gate_variant(self.rule_variant):
                    dist_pred = compute_residual_distance(
                        track_a,
                        track_b,
                        gap,
                        velocity_window=self.residual_velocity_window,
                        velocity_clip=self.residual_velocity_clip,
                        min_src_len=self.residual_min_src_len,
                        gap_cap=self.residual_gap_cap,
                    )
                    dist_pred = dist_pred / pair_scale
                    dist = (
                        self.residual_blend * dist_pred
                        + (1.0 - self.residual_blend) * dist_center
                    )
                    # 并联取交：残差门控之外仍要求满足端点中心距上限。
                    if self.residual_keep_center_gate and dist_center > dist_thresh:
                        continue
                else:
                    dist = dist_center
                if dist > dist_thresh:
                    continue

                dst_track_len = track_lengths[dst_id]
                shape_score = compute_shape_score(
                    track_a.last_box,
                    track_b.first_box,
                    area_weight=self.area_weight,
                    aspect_weight=self.aspect_weight,
                )
                effective_gap = max_gap
                effective_dist_thresh = dist_thresh
                if uses_adaptive_gate_variant(self.rule_variant):
                    adaptive_gap_limit, adaptive_dist_limit = compute_adaptive_gate_limits(
                        max_gap,
                        dist_thresh,
                        src_track_len,
                        dst_track_len,
                        shape_score,
                        adaptive_len_ref=self.adaptive_len_ref,
                        adaptive_gap_min_ratio=self.adaptive_gap_min_ratio,
                        adaptive_dist_min_ratio=self.adaptive_dist_min_ratio,
                        adaptive_gap_max_ratio=self.adaptive_gap_max_ratio,
                        adaptive_dist_max_ratio=self.adaptive_dist_max_ratio,
                        gate_reliability_weight=self.gate_reliability_weight,
                        gate_shape_weight=self.gate_shape_weight,
                        gate_signal_ref=self.gate_signal_ref,
                        gate_signal_gain=self.gate_signal_gain,
                        signal_invert=self.adaptive_gate_signal_invert,
                        candidate_gap=gap,
                        cone_mode=self.cone_mode,
                        cone_g0=self.cone_g0,
                        cone_power=self.cone_power,
                        cone_floor=self.cone_floor,
                        kin_kappa=self.kin_kappa,
                        kin_floor=self.kin_floor,
                        kin_ceil=self.kin_ceil,
                        kin_pair=self.kin_pair,
                        kin_speed_src=speed_end.get(src_id, -1.0),
                        kin_speed_dst=speed_head.get(dst_id, -1.0),
                    )

                    if self.adaptive_gate_hard_filter and (
                        gap > adaptive_gap_limit or dist > adaptive_dist_limit
                    ):
                        continue
                    if self.gate_score_norm == 'nominal':
                        effective_gap = max_gap
                        effective_dist_thresh = dist_thresh
                    else:
                        effective_gap = adaptive_gap_limit
                        effective_dist_thresh = adaptive_dist_limit

                # 方案10（新）：方向一致性硬否决。
                # 方案9 的 κ 只用了速度的模长；这里补上方向：真关联要求 src 末端速度方向
                # 与 src→dst 位移方向一致，且 dst 首端速度方向同向。
                # 与方案7（残差距离）的关键差别：余弦是尺度无关量，不会被 gap 线性放大。
                if self.kin_dir_tau > 0.0:
                    ax, ay = box_center(track_a.last_box)
                    bx, by = box_center(track_b.first_box)
                    dx, dy = bx - ax, by - ay
                    norm_d = math.hypot(dx, dy)
                    if norm_d > 1e-6:
                        cosines: list[float] = []
                        for vec in (vel_end.get(src_id), vel_head.get(dst_id)):
                            if vec is None:
                                continue
                            vx, vy = vec
                            sp = math.hypot(vx, vy)
                            if sp < self.kin_dir_vref:
                                continue
                            cosines.append((vx * dx + vy * dy) / (sp * norm_d))
                        if cosines:
                            self.kin_dir_stats['checked'] += 1
                            self.kin_dir_totals['checked'] += 1
                            worst = min(cosines) if self.kin_dir_pair else max(cosines)
                            if worst < self.kin_dir_tau:
                                self.kin_dir_stats['vetoed'] += 1
                                self.kin_dir_totals['vetoed'] += 1
                                continue

                temporal_score = compute_temporal_score(
                    gap,
                    effective_gap,
                    rule_variant=self.rule_variant,
                    gap_decay_alpha=self.gap_decay_alpha,
                )
                distance_score = compute_distance_score(dist, effective_dist_thresh)
                length_confidence = compute_length_confidence(
                    src_track_len,
                    dst_track_len,
                    length_ref=self.length_ref,
                    length_conf_power=self.length_conf_power,
                )
                rule_score = compute_rule_score(
                    gap,
                    effective_gap,
                    dist,
                    effective_dist_thresh,
                    rule_variant=self.rule_variant,
                    gap_decay_alpha=self.gap_decay_alpha,
                    temporal_score=temporal_score,
                    distance_score=distance_score,
                    shape_score=shape_score,
                    length_confidence=length_confidence,
                    temporal_weight=self.temporal_weight,
                    distance_weight=self.distance_weight,
                    shape_weight=self.shape_weight,
                )
                if self.score_reliability_penalty > 0.0:
                    # 方案6附加项：把「两侧轨迹都足够长」的候选排到前面，短碎片候选整体降权。
                    # 只改排序、不改筛选，实测在固定窗口与锥形门控上都能稳定小涨。
                    penalty_rel = compute_track_reliability(
                        src_track_len,
                        dst_track_len,
                        self.penalty_len_ref,
                    )
                    rule_score = float(
                        np.clip(
                            rule_score
                            * ((1.0 - self.score_reliability_penalty)
                               + self.score_reliability_penalty * penalty_rel),
                            0.0,
                            1.0,
                        )
                    )
                candidates.append(
                    CandidatePair(
                        src_id=src_id,
                        dst_id=dst_id,
                        gap=gap,
                        distance=dist,
                        rule_score=rule_score,
                        temporal_score=temporal_score,
                        distance_score=distance_score,
                        shape_score=shape_score,
                        length_confidence=length_confidence,
                        src_track_len=src_track_len,
                        dst_track_len=dst_track_len,
                        center_distance=dist_center,
                        pred_distance=dist_pred,
                    )
                )

        if self.kit_veto != 'off':
            candidates = self._apply_kit_veto(candidates, track_map)

        return candidates


    def _score_candidates(
        self,
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
    ) -> list[CandidatePair]:
        if self.model is None:
            raise RuntimeError('当前模式需要 AFLinkNetV2 权重，但模型未初始化。')

        scored: list[CandidatePair] = []
        with torch.no_grad():
            for offset in range(0, len(candidates), self.batch_size):
                batch = candidates[offset: offset + self.batch_size]
                seq_a_batch = []
                seq_b_batch = []
                for cand in batch:
                    track_a = track_map[cand.src_id]
                    track_b = track_map[cand.dst_id]
                    seq_a_batch.append(track_to_seq(track_a.boxes[-self.seq_len:], seq_len=self.seq_len))
                    seq_b_batch.append(track_to_seq(track_b.boxes[:self.seq_len], seq_len=self.seq_len))

                tensor_a = torch.from_numpy(np.stack(seq_a_batch)).to(self.device)
                tensor_b = torch.from_numpy(np.stack(seq_b_batch)).to(self.device)
                logits = self.model(tensor_a, tensor_b)
                prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

                for cand, score in zip(batch, prob.tolist()):
                    cand.neural_score = float(score)
                    scored.append(cand)
        return scored

    @staticmethod
    def _greedy_match(
        candidates: list[CandidatePair],
        use_neural: bool | str = False,
        rule_variant: str = 'baseline',
        size_tiebreak: bool = True,
        assist_weight: float = 0.2,
    ) -> dict[int, int]:
        chosen_src: set[int] = set()
        chosen_dst: set[int] = set()
        link_map: dict[int, int] = {}
        variant = normalize_rule_variant(rule_variant)

        def extra_terms(cand: CandidatePair) -> tuple[float, float, int]:
            shape_term = -cand.shape_score if uses_shape_factor_variant(variant) else 0.0
            length_term = -cand.length_confidence if uses_length_conf_variant(variant) else 0.0
            size_term = (
                -(cand.src_track_len + cand.dst_track_len)
                if (variant != 'baseline' and size_tiebreak)
                else 0
            )
            return shape_term, length_term, size_term

        if use_neural == 'assist':
            sort_key = lambda cand: (
                -((1.0 - assist_weight) * cand.rule_score + assist_weight * float(cand.neural_score)),
                -cand.rule_score,
                *extra_terms(cand),
                cand.gap,
                cand.distance,
                cand.src_id,
                cand.dst_id,
            )
        elif use_neural:
            if variant == 'baseline':
                sort_key = lambda cand: (-float(cand.neural_score), cand.gap, cand.distance, cand.src_id, cand.dst_id)
            else:
                sort_key = lambda cand: (
                    -float(cand.neural_score),
                    -cand.rule_score,
                    *extra_terms(cand),
                    cand.gap,
                    cand.distance,
                    cand.src_id,
                    cand.dst_id,
                )
        else:
            if variant == 'baseline':
                sort_key = lambda cand: (-cand.rule_score, cand.gap, cand.distance, cand.src_id, cand.dst_id)
            else:
                sort_key = lambda cand: (
                    -cand.rule_score,
                    *extra_terms(cand),
                    cand.gap,
                    cand.distance,
                    cand.src_id,
                    cand.dst_id,
                )

        sorted_candidates = sorted(candidates, key=sort_key)


        for cand in sorted_candidates:
            if cand.src_id in chosen_src or cand.dst_id in chosen_dst:
                continue
            link_map[cand.src_id] = cand.dst_id
            chosen_src.add(cand.src_id)
            chosen_dst.add(cand.dst_id)
        return link_map


    @staticmethod
    def _hungarian_match(
        candidates: list[CandidatePair],
        track_map: dict[int, TrackFragment],
        use_rule_score: bool = False,
    ) -> dict[int, int]:
        if not candidates:
            return {}
        track_ids = sorted(track_map.keys())
        index = {tid: idx for idx, tid in enumerate(track_ids)}
        cost = np.full((len(track_ids), len(track_ids)), 1e9, dtype=np.float32)

        for cand in candidates:
            score = float(cand.rule_score) if use_rule_score else float(cand.neural_score)
            cost[index[cand.src_id], index[cand.dst_id]] = 1.0 - score

        row_ind, col_ind = linear_sum_assignment(cost)
        link_map: dict[int, int] = {}
        for row, col in zip(row_ind, col_ind):
            if cost[row, col] >= 1e8:
                continue
            src_id = track_ids[row]
            dst_id = track_ids[col]
            if src_id == dst_id:
                continue
            link_map[src_id] = dst_id
        return link_map

    @staticmethod
    def _union_find_match(candidates: list[CandidatePair], track_map: dict[int, TrackFragment]) -> dict[int, int]:
        if not candidates:
            return {}

        uf = UnionFind(track_map.keys())
        sorted_candidates = sorted(
            candidates,
            key=lambda cand: (-float(cand.neural_score), cand.gap, cand.distance, cand.src_id, cand.dst_id),
        )
        for cand in sorted_candidates:
            uf.union(cand.src_id, cand.dst_id)

        components = uf.components()
        link_map: dict[int, int] = {}
        for component_ids in components.values():
            if len(component_ids) < 2:
                continue
            ordered = sorted(
                component_ids,
                key=lambda tid: (track_map[tid].start_frame, track_map[tid].end_frame, tid),
            )
            for src_id, dst_id in zip(ordered[:-1], ordered[1:]):
                if track_map[src_id].end_frame < track_map[dst_id].start_frame:
                    link_map[src_id] = dst_id
        return link_map

    @staticmethod
    def _flatten_link_map(link_map: dict[int, int]) -> dict[int, int]:
        resolved: dict[int, int] = {}
        for src_id in link_map:
            dst_id = link_map[src_id]
            visited = {src_id}
            while dst_id in link_map and dst_id not in visited:
                visited.add(dst_id)
                dst_id = link_map[dst_id]
            resolved[src_id] = dst_id
        return resolved


def box_center(box: list[float]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return float(math.hypot(ax - bx, ay - by))


def compute_pair_scale(box_a: list[float], box_b: list[float], mode: str = 'diag') -> float:
    """候选对的尺度因子，用于把像素距离无量纲化（方案11 尺度自适应门控）。

    动机：同一个固定像素门限在 test / train 上最优值相差 2.75 倍（440 vs 160 px），
    而两划分在 (150, 450) 的候选总数几乎相同（5765 vs 5817）—— 说明差异来自**目标
    成像尺寸不同**，而不是运动行为不同。把门限改成「目标自身尺寸的倍数」后，阈值
    在两个划分上应当收敛到同一个数，从而可以用**单一配置**跨划分成立。
    """
    mode = str(mode).lower()
    wa = max(abs(box_a[2] - box_a[0]), 0.0)
    ha = max(abs(box_a[3] - box_a[1]), 0.0)
    wb = max(abs(box_b[2] - box_b[0]), 0.0)
    hb = max(abs(box_b[3] - box_b[1]), 0.0)
    if mode == 'diag':
        scale = 0.5 * (math.hypot(wa, ha) + math.hypot(wb, hb))
    elif mode == 'height':
        scale = 0.5 * (ha + hb)
    elif mode == 'sqrt_area':
        scale = math.sqrt(max(0.5 * (wa * ha + wb * hb), 1e-6))
    else:
        scale = 1.0
    return float(max(scale, 1e-6))


def compute_track_velocity(
    track: TrackFragment,
    velocity_window: int = 2,
    velocity_clip: float = 0.0,
) -> tuple[float, float]:
    """用轨迹末端若干帧估计每帧中心位移，即恒速运动模型的一阶项。

    `velocity_window=k` 表示取末端第 k 帧与末帧做差分（k>=2，用两帧差分的目的是
    对检测抖动更鲁棒）。轨迹只有一帧时方向不可估，速度定义为 0，此时残差门控
    自动退化成原始的「端点中心距 + 锥形上界」，行为与方案6一致，不会额外引入噪声。
    `velocity_clip > 0` 时对速度模长做截断，抑制漏检/串轨造成的虚高速度。
    """
    boxes = track.boxes
    if len(boxes) < 2:
        return 0.0, 0.0
    k = min(max(int(velocity_window), 2), len(boxes))
    span = max(int(track.frames[-1]) - int(track.frames[-k]), 1)
    last_x, last_y = box_center(boxes[-1])
    prev_x, prev_y = box_center(boxes[-k])
    vx = (last_x - prev_x) / span
    vy = (last_y - prev_y) / span
    if velocity_clip > 0.0:
        norm = math.hypot(vx, vy)
        if norm > velocity_clip:
            scale = velocity_clip / norm
            vx *= scale
            vy *= scale
    return float(vx), float(vy)


def compute_track_head_velocity(
    track: TrackFragment,
    velocity_window: int = 5,
    velocity_clip: float = 0.0,
) -> tuple[float, float]:
    """用轨迹**首端**若干帧估计每帧中心位移（方案9 用）。

    `compute_track_velocity` 看末端（用于向后外推），这里看首端：dst 轨迹在间隙**之后**
    的运动状态由它开头几帧决定，用来和 src 末端速度一起给出门槛。
    """
    boxes = track.boxes
    if len(boxes) < 2:
        return 0.0, 0.0
    k = min(max(int(velocity_window), 2), len(boxes))
    span = max(int(track.frames[k - 1]) - int(track.frames[0]), 1)
    x0, y0 = box_center(boxes[0])
    x1, y1 = box_center(boxes[k - 1])
    vx = (x1 - x0) / span
    vy = (y1 - y0) / span
    if velocity_clip > 0.0:
        norm = math.hypot(vx, vy)
        if norm > velocity_clip:
            scale = velocity_clip / norm
            vx *= scale
            vy *= scale
    return float(vx), float(vy)


def compute_residual_distance(
    track_a: TrackFragment,
    track_b: TrackFragment,
    gap: int,
    velocity_window: int = 2,
    velocity_clip: float = 0.0,
    min_src_len: int = 2,
    gap_cap: int = 0,
) -> float:
    """预测残差距离 d_pred = ‖c_b,first − (c_a,last + v_a·gap)‖。

    与 `center_distance` 的区别在于多用了方向信息：原来的零阶模型只能表达
    「最大位移 = v_max·gap」（锥形上界），这里先按 src 末端速度外推，再比较残差，
    因此同样一个阈值下对「顺着运动方向继续跑」的候选更宽容、对「莫名跳到别处」的候选
    更严格。src 轨迹短于 `min_src_len` 时速度不可信，直接退化回端点中心距。
    `gap_cap > 0` 时把外推时长截断到 `gap_cap` 帧：恒速假设的时间越长误差越大，
    长间隙上宁可少外推一点，让残差回到「位移 + 有限外推」的混合形态。
    """
    if len(track_a.frames) < max(int(min_src_len), 2):
        return center_distance(track_a.last_box, track_b.first_box)
    eff_gap = min(int(gap), int(gap_cap)) if int(gap_cap) > 0 else int(gap)
    vx, vy = compute_track_velocity(track_a, velocity_window, velocity_clip)
    ax, ay = box_center(track_a.last_box)
    bx, by = box_center(track_b.first_box)
    return float(math.hypot(bx - (ax + vx * eff_gap), by - (ay + vy * eff_gap)))



def normalize_rule_variant(rule_variant: str) -> str:
    variant = str(rule_variant).strip().lower()
    if variant not in RULE_VARIANT_CHOICES:
        raise ValueError(f'未知 rule_variant: {rule_variant}，可选值: {RULE_VARIANT_CHOICES}')
    return variant



def uses_gap_decay_variant(rule_variant: str) -> bool:
    return normalize_rule_variant(rule_variant) in {'gap_decay', 'gap_length'}



def uses_length_conf_variant(rule_variant: str) -> bool:
    return normalize_rule_variant(rule_variant) in {'length_conf', 'gap_length'}



def uses_adaptive_gate_variant(rule_variant: str) -> bool:
    return normalize_rule_variant(rule_variant) in {
        'adaptive_gate', 'adaptive_tri', 'adaptive_cone', 'kinematic_gate',
    }



def uses_kinematic_gate_variant(rule_variant: str) -> bool:
    """方案9：把锥形的**全局**速度上限换成「逐轨迹观测速度」给出的**个体**上限。

    方案6 的锥形门控等价于一个全局速度上限 `v_max = dist_thresh / g0`（推荐参数下
    450 / 55 = 8.18 px/帧）：所有轨迹共用同一条 `dist ≤ v_max·gap` 的直线。方案9 认为
    这个上限应当逐轨迹估计 —— 慢速/静止轨迹不该享受和快速轨迹一样宽的空间容差，
    快速轨迹也不该被全局上限误剪。门控量因此变成

        dist_limit = min(base_limit, (κ·speed(track) + floor)·gap)

    其中 speed 取 src 末端（`--kin_pair` 时再并入 dst 首端）的每帧中心位移模长，
    轨迹短于 `--kin_min_len` 时速度不可信、直接退回锥形限。

    与方案7（预测残差门控）的区别：方案7 用恒速外推比较**位置**，在长间隙上把速度
    估计误差按 gap 线性放大而失效；方案9 只用速度**模长**给出上界，对方向误差免疫。

    结构意义：门槛依赖具体轨迹的观测速度，因此**两个 (gap, dist) 完全相同的候选会得到
    不同判定** —— 这是它与固定窗口族的结构性差别，可用非矩形性度量（见 `_kindiag.py`）
    直接量化：任何固定窗口都无法复现它的筛选集合。
    """
    return normalize_rule_variant(rule_variant) in {'kinematic_gate'}



def uses_shape_factor_variant(rule_variant: str) -> bool:
    return normalize_rule_variant(rule_variant) in {'tri_factor', 'adaptive_tri'}


def uses_residual_gate_variant(rule_variant: str) -> bool:
    """方案7：把门控/打分的空间度量从「端点中心距」换成「预测残差距离」。"""
    return normalize_rule_variant(rule_variant) in {'residual_gate'}



def get_rule_variant_label(rule_variant: str) -> str:
    variant = normalize_rule_variant(rule_variant)
    labels = {
        'baseline': '原始规则',
        'adaptive_gate': '方案1：自适应时空门控',
        'tri_factor': '方案2：三因子规则评分',
        'adaptive_tri': '方案1+2：自适应时空门控 + 三因子规则评分',
        'gap_decay': '方案4：非线性 gap 惩罚',
        'length_conf': '方案5：长度置信度',
        'gap_length': '方案4+5：gap 惩罚 + 长度置信度',
        'adaptive_cone': '方案6：联合锥形时空门控（dist 随 gap 线性收紧）',
        'residual_gate': '方案7：预测残差门控（末端速度外推后比较残差距离）',
        'kinematic_gate': '方案9：逐轨运动学门控（dist 上限随轨迹自身观测速度自适应）',
    }
    return labels[variant]



def compute_distance_score(distance: float, dist_thresh: float) -> float:
    return float(1.0 - min(max(distance, 0.0), max(dist_thresh, 1e-6)) / max(dist_thresh, 1e-6))



def normalize_pair_weights(primary_weight: float, secondary_weight: float) -> tuple[float, float]:
    weights = np.asarray([
        max(float(primary_weight), 1e-6),
        max(float(secondary_weight), 1e-6),
    ], dtype=np.float32)
    weights /= float(weights.sum())
    return float(weights[0]), float(weights[1])



def compute_shape_score(
    box_a: list[float],
    box_b: list[float],
    area_weight: float = 0.6,
    aspect_weight: float = 0.4,
) -> float:
    wa = max(float(box_a[2] - box_a[0]), 1.0)
    ha = max(float(box_a[3] - box_a[1]), 1.0)
    wb = max(float(box_b[2] - box_b[0]), 1.0)
    hb = max(float(box_b[3] - box_b[1]), 1.0)
    area_a = wa * ha
    area_b = wb * hb
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    aspect_delta = abs(math.log(wa / ha) - math.log(wb / hb))
    aspect_score = math.exp(-aspect_delta)
    area_weight, aspect_weight = normalize_pair_weights(area_weight, aspect_weight)
    return float(np.clip(area_weight * area_ratio + aspect_weight * aspect_score, 0.0, 1.0))



def compute_track_reliability(
    src_track_len: int,
    dst_track_len: int,
    adaptive_len_ref: int = 30,
) -> float:
    effective_len = min(max(int(src_track_len), 0), max(int(dst_track_len), 0))
    return float(np.clip(effective_len / max(int(adaptive_len_ref), 1), 0.0, 1.0))



def compute_adaptive_gate_limits(
    max_gap: int,
    dist_thresh: float,
    src_track_len: int,
    dst_track_len: int,
    shape_score: float,
    adaptive_len_ref: int = 30,
    adaptive_gap_min_ratio: float = 0.5,
    adaptive_dist_min_ratio: float = 0.5,
    adaptive_gap_max_ratio: float = 1.0,
    adaptive_dist_max_ratio: float = 1.0,
    gate_reliability_weight: float = 0.7,
    gate_shape_weight: float = 0.3,
    gate_signal_ref: float | None = None,
    gate_signal_gain: float = 1.0,
    signal_invert: bool = False,
    candidate_gap: int | None = None,
    cone_mode: bool = False,
    cone_g0: float = 60.0,
    cone_power: float = 1.0,
    cone_floor: float = 0.0,
    kin_kappa: float = 0.0,
    kin_floor: float = 0.0,
    kin_ceil: float = 0.0,
    kin_pair: bool = False,
    kin_speed_src: float = -1.0,
    kin_speed_dst: float = -1.0,
) -> tuple[int, float]:
    """自适应时空门控。

    默认（cone_mode=False）：双侧自适应，ratio 由「轨迹可靠性 + 形状」门控信号给出。
    - max_ratio == 1.0 时退化为原始单侧门控：名义窗口即上限，只能收紧、无法放宽。
    - max_ratio > 1.0 时名义窗口位于区间中部：可靠轨迹可放宽到 max_ratio 倍，
      不可靠轨迹被收紧到 min_ratio 倍，即真正"可放可收"的双向自适应。

    cone_mode=True（方案6，联合锥形门控）：认为可行域是 (gap, dist) 平面上的一个锥——
    时间间隔越小，允许的空间位移必须越小；gap 上限保持名义值不变，只把 dist 上限收紧为
        dist_limit = dist_thresh * clip((gap / g0) ** power, floor, 1)
    在 SNMOT 测试集上该结构显著优于同预算的固定窗口，也优于基于轨迹长度/形状的双侧门控
    （门控信号与"该候选是否为真"几乎不相关，见 tracker_results/_gate_sweep3）。
    """
    if cone_mode:
        span = max(int(candidate_gap if candidate_gap is not None else max_gap), 0)
        ratio = float(np.clip((span / max(float(cone_g0), 1e-6)) ** max(float(cone_power), 1e-6),
                              float(np.clip(cone_floor, 0.0, 1.0)), 1.0))
        dist_limit = float(dist_thresh) * ratio
        if kin_kappa > 0.0:
            # 方案9：在锥形限之外再叠加「该轨迹自己的速度上限」。
            # speed_src < 0 表示轨迹太短、速度不可信 → 不施加运动学约束（退回锥形）。
            speed = float(kin_speed_src) if kin_speed_src >= 0.0 else None
            if speed is not None and kin_pair and kin_speed_dst >= 0.0:
                speed = max(speed, float(kin_speed_dst))
            if speed is not None:
                v_max = float(kin_kappa) * speed + max(float(kin_floor), 0.0)
                if kin_ceil > 0.0:
                    v_max = min(v_max, float(kin_ceil))
                dist_limit = min(dist_limit, v_max * max(span, 1))
        return int(max(1, int(max_gap))), max(float(dist_limit), 1e-6)
    reliability = compute_track_reliability(src_track_len, dst_track_len, adaptive_len_ref)
    gate_reliability_weight, gate_shape_weight = normalize_pair_weights(
        gate_reliability_weight,
        gate_shape_weight,
    )
    gate_signal = float(
        np.clip(
            gate_reliability_weight * reliability + gate_shape_weight * float(shape_score),
            0.0,
            1.0,
        )
    )
    if signal_invert:
        gate_signal = 1.0 - gate_signal
    gap_max_ratio = max(float(adaptive_gap_max_ratio), float(adaptive_gap_min_ratio))
    dist_max_ratio = max(float(adaptive_dist_max_ratio), float(adaptive_dist_min_ratio))
    if gate_signal_ref is None:
        gap_ratio = adaptive_gap_min_ratio + (gap_max_ratio - adaptive_gap_min_ratio) * gate_signal
        dist_ratio = adaptive_dist_min_ratio + (dist_max_ratio - adaptive_dist_min_ratio) * gate_signal
    else:
        gain = max(float(gate_signal_gain), 0.0)
        gap_ratio = 1.0 + (gate_signal - float(gate_signal_ref)) * gain
        dist_ratio = 1.0 + (gate_signal - float(gate_signal_ref)) * gain
    gap_ratio = float(np.clip(gap_ratio, adaptive_gap_min_ratio, gap_max_ratio))
    dist_ratio = float(np.clip(dist_ratio, adaptive_dist_min_ratio, dist_max_ratio))
    gap_ceiling = max(int(round(max_gap * gap_max_ratio)), 1)
    adaptive_gap_limit = max(1, min(int(round(max_gap * gap_ratio)), gap_ceiling))
    adaptive_dist_limit = max(float(dist_thresh) * dist_ratio, 1e-6)
    return adaptive_gap_limit, adaptive_dist_limit




def normalize_rule_weights(
    temporal_weight: float,
    distance_weight: float,
    shape_weight: float,
) -> tuple[float, float, float]:
    weights = np.asarray([
        max(float(temporal_weight), 1e-6),
        max(float(distance_weight), 1e-6),
        max(float(shape_weight), 1e-6),
    ], dtype=np.float32)
    weights /= float(weights.sum())
    return float(weights[0]), float(weights[1]), float(weights[2])



def compute_temporal_score(
    gap: int,
    max_gap: int,
    rule_variant: str = 'baseline',
    gap_decay_alpha: float = 3.0,
) -> float:
    normalized_gap = min(max(gap, 0), max(max_gap, 1)) / max(max_gap, 1)
    if uses_gap_decay_variant(rule_variant):
        return float(np.clip(math.exp(-max(gap_decay_alpha, 1e-6) * normalized_gap), 0.0, 1.0))
    return float(np.clip(1.0 - normalized_gap, 0.0, 1.0))



def compute_length_confidence(
    src_track_len: int,
    dst_track_len: int,
    length_ref: int = 30,
    length_conf_power: float = 1.0,
) -> float:
    effective_len = min(max(int(src_track_len), 0), max(int(dst_track_len), 0))
    base = min(effective_len / max(int(length_ref), 1), 1.0)
    return float(np.clip(base ** max(float(length_conf_power), 1e-6), 0.0, 1.0))



def compute_rule_score(
    gap: int,
    max_gap: int,
    distance: float,
    dist_thresh: float,
    rule_variant: str = 'baseline',
    gap_decay_alpha: float = 3.0,
    temporal_score: float | None = None,
    distance_score: float | None = None,
    shape_score: float = 1.0,
    length_confidence: float = 1.0,
    temporal_weight: float = 0.4,
    distance_weight: float = 0.4,
    shape_weight: float = 0.2,
) -> float:
    variant = normalize_rule_variant(rule_variant)
    temporal_score = compute_temporal_score(gap, max_gap, variant, gap_decay_alpha) if temporal_score is None else float(temporal_score)
    distance_score = compute_distance_score(distance, dist_thresh) if distance_score is None else float(distance_score)

    if uses_shape_factor_variant(variant):
        tw, dw, sw = normalize_rule_weights(temporal_weight, distance_weight, shape_weight)
        score = tw * temporal_score + dw * distance_score + sw * float(shape_score)
    elif uses_length_conf_variant(variant):
        score = 0.4 * temporal_score + 0.4 * distance_score + 0.2 * float(length_confidence)
    else:
        score = 0.5 * temporal_score + 0.5 * distance_score
    return float(np.clip(score, 0.0, 1.0))





def track_to_seq(boxes: list[list[float]], seq_len: int = 30, image_size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """将边界框序列编码为 9×T 的运动特征矩阵。"""
    width, height = image_size
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError('track_to_seq 需要形如 Nx4 的边界框列表。')

    if arr.shape[0] == 0:
        arr = np.zeros((1, 4), dtype=np.float32)

    if arr.shape[0] >= seq_len:
        arr = arr[-seq_len:]
    else:
        pad = np.repeat(arr[:1], seq_len - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)

    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    w = np.maximum(x2 - x1, 1.0)
    h = np.maximum(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    dw = np.diff(w, prepend=w[0])
    dh = np.diff(h, prepend=h[0])
    ratio = np.log(w / h)

    features = np.stack(
        [
            cx / width,
            cy / height,
            w / width,
            h / height,
            dx / width,
            dy / height,
            dw / width,
            dh / height,
            ratio,
        ],
        axis=0,
    ).astype(np.float32)
    return features



def apply_best_tuned_config(args: argparse.Namespace) -> argparse.Namespace:
    args.rule_variant = str(BEST_TUNED_CONFIG['rule_variant'])
    args.sweep_target = 'gap_dist'
    args.max_gap_values = [int(BEST_TUNED_CONFIG['max_gap'])]
    args.dist_thresh_values = [int(BEST_TUNED_CONFIG['dist_thresh'])]
    args.weight_base_max_gap = int(BEST_TUNED_CONFIG['max_gap'])
    args.weight_base_dist_thresh = int(BEST_TUNED_CONFIG['dist_thresh'])
    args.gate_reliability_weight = float(BEST_TUNED_CONFIG['gate_reliability_weight'])
    args.gate_shape_weight = float(BEST_TUNED_CONFIG['gate_shape_weight'])
    args.area_weight = float(BEST_TUNED_CONFIG['shape_area_weight'])
    args.aspect_weight = float(BEST_TUNED_CONFIG['shape_aspect_weight'])
    args.temporal_weight = float(BEST_TUNED_CONFIG['temporal_weight'])
    args.distance_weight = float(BEST_TUNED_CONFIG['distance_weight'])
    args.shape_weight = float(BEST_TUNED_CONFIG['shape_weight'])
    args.adaptive_len_ref = int(BEST_TUNED_CONFIG['adaptive_len_ref'])
    args.adaptive_gap_min_ratio = float(BEST_TUNED_CONFIG['adaptive_gap_min_ratio'])
    args.adaptive_dist_min_ratio = float(BEST_TUNED_CONFIG['adaptive_dist_min_ratio'])
    return args



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='规则 AFLink 混合版后处理与灵敏性分析脚本。')
    parser.add_argument('--mode', type=str, default='hybrid', choices=['rule', 'rule_hungarian', 'hybrid', 'hybrid_greedy', 'neural'])
    parser.add_argument('--results_dir', type=str, default=str(DEFAULT_RESULTS_DIR), help='原始 MOT 结果目录。')
    parser.add_argument('--gt_root', type=str, default=str(DEFAULT_GT_ROOT), help='GT 根目录。')
    parser.add_argument('--cache_path', type=str, default=str(DEFAULT_CACHE_PATH), help='raw_pred_cache.json 路径。')
    parser.add_argument('--aflink_weights', type=str, default=None, help='AFLinkNetV2 权重路径。')
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_OUTPUT_ROOT), help='后处理结果与评测输出根目录。')
    parser.add_argument('--report_path', type=str, default=str(DEFAULT_REPORT_PATH), help='markdown 报告保存路径。')
    parser.add_argument('--score_thresh', type=float, default=0.55, help='神经网络打分阈值。')
    parser.add_argument('--use_best_config', action='store_true', help='直接使用灵敏性分析得到的最优 adaptive_tri 配置，并只运行这一组参数。')
    parser.add_argument(
        '--rule_variant',
        type=str,
        default='baseline',
        choices=RULE_VARIANT_CHOICES,
        help='规则变体：baseline 原始方案，adaptive_gate 为方案1，tri_factor 为方案2，adaptive_tri 为方案1+2，gap_decay 为方案4，length_conf 为方案5，gap_length 为方案4+5，'
             'adaptive_cone 为方案6锥形门控，residual_gate 为方案7预测残差门控。',
    )

    parser.add_argument('--gap_decay_alpha', type=float, default=3.0, help='方案4中 gap 非线性衰减强度，越大表示长中断降分越快。')
    parser.add_argument('--length_ref', type=int, default=30, help='方案5中长度置信度的参考长度，达到该长度后长度项饱和。')
    parser.add_argument('--length_conf_power', type=float, default=1.0, help='方案5中长度置信度幂次，>1 更保守，<1 更宽松。')
    parser.add_argument('--adaptive_len_ref', type=int, default=30, help='方案1中自适应门控的参考轨迹长度，越短的碎片门控越严格。')
    parser.add_argument('--adaptive_gap_min_ratio', type=float, default=0.5, help='方案1中 temporal gate 的最小保留比例，越小表示短轨更严格。')
    parser.add_argument('--adaptive_dist_min_ratio', type=float, default=0.5, help='方案1中 spatial gate 的最小保留比例，越小表示短轨更严格。')
    parser.add_argument('--adaptive_gap_max_ratio', type=float, default=1.0,
                        help='方案1门控 gap 上限比例。与 --gate_signal_ref 配合时才是双侧门控（可放宽到 1.5 倍窗口）。')
    parser.add_argument('--adaptive_dist_max_ratio', type=float, default=1.0,
                        help='方案1门控 dist 上限比例。与 --gate_signal_ref 配合时才是双侧门控（可放宽到 1.5 倍窗口）。')
    parser.add_argument('--gate_signal_ref', type=float, default=None,
                        help='方案1门控信号的标定基准 ref。设置后走双侧分支 ratio = 1 + (G - ref)·gain，'
                             'G=ref 时窗口等于名义窗口、高于 ref 放宽、低于 ref 收紧；不传则退回 [min,max] 线性映射。')
    parser.add_argument('--gate_signal_gain', type=float, default=1.0,
                        help='方案1门控信号的标定增益（仅在传入 --gate_signal_ref 时生效）。')
    parser.add_argument('--cone_g0', type=float, default=60.0, help='方案6联合锥形门控的基准 gap：gap<=g0 时空间门不收紧。实测 55~58 最优。')
    parser.add_argument('--cone_power', type=float, default=1.0, help='方案6锥形收紧幂次，1.0 为线性。')
    parser.add_argument('--cone_floor', type=float, default=0.0, help='方案6空间门收紧的下限比例，0 表示不设下限。')
    parser.add_argument('--score_reliability_penalty', type=float, default=0.0,
                        help='方案6附加项：按两侧轨迹长度对 rule_score 降权的幅度（0~1，只改排序不改筛选）。实测 0.5~1.0 稳定涨点。')
    parser.add_argument('--residual_blend', type=float, default=1.0,
                        help='方案7：残差距离与端点中心距的混合权重，1.0 表示纯预测残差，0.0 退化为原点距。')
    parser.add_argument('--residual_velocity_clip', type=float, default=0.0,
                        help='方案7：末端速度模长上限（px/帧），0 表示不截断。')
    parser.add_argument('--residual_velocity_window', type=int, default=2,
                        help='方案7：估计速度所用的末端帧数（>=2，越大越平滑但对转向越迟钝）。')
    parser.add_argument('--residual_min_src_len', type=int, default=2,
                        help='方案7：src 轨迹短于该值时不做外推，退化为端点中心距（抑制短碎片的速度噪声）。')
    parser.add_argument('--residual_keep_center_gate', dest='residual_keep_center_gate',
                        action='store_true', default=False,
                        help='方案7：残差门控与端点中心距门控取交（并联约束，比单用残差更保守）。')
    parser.add_argument('--residual_gap_cap', type=int, default=0,
                        help='方案7：外推时长上限（帧），0 表示按真实 gap 全额外推。长间隙上可抑制恒速假设的累积误差。')
    parser.add_argument('--penalty_len_ref', type=int, default=30, help='方案6附加项的轨迹长度参考值。')
    parser.add_argument('--kit_veto', type=str, default='off', choices=KIT_VETO_CHOICES,
                        help="方案8 队色否决（正交于 rule_variant 的一道候选级硬门）：off 关闭（默认）；"
                             "side 跨侧否决（需 left/right/ref/ball 这类标签，仅测上界用）；"
                             "any 两侧标签不同即否决（像素版 c0/c1/cX 用这一档）；all 全否（破坏性自检）。")
    parser.add_argument('--mcv', dest='mcv', action='store_true', default=False,
                        help='一键启用 MCVC（论文 Table 11「MCVC (ours)」一行）= **T2 队色等价类否决**，'
                             '并让原自适应门控退场（不参与评分/筛选）。'
                             '只补 --gate_score_norm nominal / --kit_veto any（不动 score_reliability_penalty）；'
                             'V3/T4 等新增层经双划分消融判为负向或从不触发，默认 off，需显式打开。'
                             '标签表仍需 --kit_labels 提供。')
    parser.add_argument('--mcv_cone', dest='mcv_cone', action='store_true', default=False,
                        help='在 MCVC 上追加 T1 锥形剪枝层（--rule_variant adaptive_cone --cone_g0 55）。'
                             '实测其边际代价在 test 为 −0.09、train 为 +0.06，符号翻转、无法区分于 0，故默认关闭。')
    parser.add_argument('--mcv_recip', type=str, default='off', choices=MCV_RECIP_CHOICES,
                        help='MCVC V3a 互选一致否决：off 关闭（默认）；one 一侧互选即放行；both 两侧都要互选。')
    parser.add_argument('--mcv_trans', type=str, default='off', choices=MCV_TRANS_CHOICES,
                        help='MCVC V3b 分量纯度否决：off 关闭（默认）；weak 逐序列删除最低分的「跨队色桥边」。')
    parser.add_argument('--mcv_trans_max', type=int, default=1,
                        help='MCVC V3b 每序列最多删除的边数（默认 1，刻意保守）。')
    parser.add_argument('--mcv_ball_excl', dest='mcv_ball_excl', action='store_true', default=False,
                        help='MCVC V5 球-人互斥否决：疑似球的碎片与疑似人的碎片之间一律否决。')
    parser.add_argument('--mcv_ball_rel_h', type=float, default=0.45,
                        help='MCVC V5 疑似球的相对高度上限（相对当前序列碎片高度中位数）。')
    parser.add_argument('--mcv_ball_aspect', type=float, default=0.75,
                        help='MCVC V5 疑似球的最小宽高比 w/h（球员框细高，此值越大越保守）。')
    parser.add_argument('--mcv_recover', dest='mcv_recover', action='store_true', default=False,
                        help='MCVC T4 互补回补轮：更深窗口 + 更严标签，补回第一轮漏掉的真链接。')
    parser.add_argument('--mcv_recover_gap', type=int, default=400,
                        help='MCVC T4 回补轮的窗口上限（帧），默认 400。')
    parser.add_argument('--mcv_recover_dist', type=float, default=1200.0,
                        help='MCVC T4 回补轮的端点距离上限（像素），默认 1200。')
    parser.add_argument('--mcv_recover_permissive', dest='mcv_recover_permissive',
                        action='store_true', default=False,
                        help='MCVC T4 回补轮放宽标签要求（允许未知标签放行）。默认关闭 = 严格要求两端同队。')
    parser.add_argument('--kit_labels', type=str, default=None,
                        help='方案8 队别标签表 JSON（{seq: {track_id: label}}），由 '
                             '`_probe_color_team.py --stage pixels --dump-labels <PATH>` 生成；'
                             '不传时即便 kit_veto 非 off 也不产生任何否决。')
    parser.add_argument('--scale_norm', type=str, default='none',
                        choices=['none', 'diag', 'height', 'sqrt_area'],
                        help='方案11 尺度自适应门控：像素距离除以候选对尺度，使阈值跨分辨率/划分可比。')
    parser.add_argument('--kin_kappa', type=float, default=0.0,
                        help='方案9 逐轨运动学门控：dist_limit 再受 (κ·speed+floor)·gap 约束，κ=0 关闭。')
    parser.add_argument('--kin_floor', type=float, default=0.0, help='方案9 速度下限补偿（px/帧）。')
    parser.add_argument('--kin_ceil', type=float, default=0.0, help='方案9 v_max 上限截断（px/帧），0=不截断。')
    parser.add_argument('--kin_window', type=int, default=5, help='方案9 速度估计窗口（帧）。')
    parser.add_argument('--kin_min_len', type=int, default=4, help='方案9 轨迹短于此长度时不启用速度约束。')
    parser.add_argument('--kin_pair', dest='kin_pair', action='store_true', default=False,
                        help='方案9 取 src 末端速度与 dst 首端速度的较大者。')
    parser.add_argument('--kin_dir_tau', type=float, default=0.0,
                        help='方案10 方向一致性硬否决：src 末端速度与 src→dst 位移的余弦低于该值即否决候选（0=关闭，不影响既有结果）。')
    parser.add_argument('--kin_dir_vref', type=float, default=1.0,
                        help='方案10 速度低于该值（px/帧）时方向不可信，该侧不参与否决。')
    parser.add_argument('--kin_dir_pair', dest='kin_dir_pair', action='store_true', default=True,
                        help='方案10 默认：src 与 dst 两侧方向都要通过（取 min）。')
    parser.add_argument('--no_kin_dir_pair', dest='kin_dir_pair', action='store_false',
                        help='方案10 放宽：只要有一侧方向通过即可（取 max）。')
    parser.add_argument('--gate_score_norm', type=str, default='effective', choices=['effective', 'nominal'],
                        help="门控存在时 rule_score 的归一化基准：'effective' 用收紧后的门限，'nominal' 仍用名义 (max_gap, dist_thresh)。"
                             "实测方案6必须取 'nominal'。")
    parser.add_argument('--gate_reliability_weight', type=float, default=0.7, help='门控信号中轨迹可靠性权重，最终会自动归一化。')
    parser.add_argument('--gate_shape_weight', type=float, default=0.3, help='门控信号中形态项权重，最终会自动归一化。')
    parser.add_argument('--area_weight', type=float, default=0.6, help='shape_score 中面积比例权重，最终会自动归一化。')
    parser.add_argument('--aspect_weight', type=float, default=0.4, help='shape_score 中长宽比相似度权重，最终会自动归一化。')
    parser.add_argument('--temporal_weight', type=float, default=0.4, help='方案2中时间因子权重，最终会自动归一化。')
    parser.add_argument('--distance_weight', type=float, default=0.4, help='方案2中距离因子权重，最终会自动归一化。')
    parser.add_argument('--shape_weight', type=float, default=0.2, help='方案2中形态连续性因子权重，最终会自动归一化。')
    parser.add_argument('--size_tiebreak', dest='size_tiebreak', action='store_true', default=True, help='非 baseline 变体贪心排序时附加「两条轨迹总长越大越优先」的 tiebreak（此前未在报告中记录），默认开启以保持与既有结果一致。')
    parser.add_argument('--no_size_tiebreak', dest='size_tiebreak', action='store_false', help='关闭 size tiebreak，使非 baseline 变体与 baseline 变体的排序项完全同质，便于消融归因。')

    parser.add_argument('--seq_len', type=int, default=30, help='AFLink 序列长度。')
    parser.add_argument('--sweep_target', type=str, default='gap_dist', choices=['gap_dist', 'shape_mix', 'gate_mix', 'rule_mix'], help='灵敏性分析维度：默认扫描 max_gap/dist_thresh；其余模式分别扫描 shape_score、gate_signal、rule_score 权重。')
    parser.add_argument('--max_gap_values', nargs='+', type=int, default=[40, 50, 60, 70, 80, 100])
    parser.add_argument('--dist_thresh_values', nargs='+', type=int, default=[150, 200, 250, 300, 400])
    parser.add_argument('--shape_area_weight_values', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8], help='shape_mix 模式下扫描的面积权重，aspect 权重自动取 1-area。')
    parser.add_argument('--gate_reliability_weight_values', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9], help='gate_mix 模式下扫描的可靠性权重，shape 权重自动取 1-reliability。')
    parser.add_argument('--rule_weight_triplets', nargs='+', default=['0.50,0.30,0.20', '0.40,0.40,0.20', '0.30,0.50,0.20', '0.35,0.35,0.30', '0.45,0.35,0.20', '0.35,0.45,0.20'], help='rule_mix 模式下扫描的时间/距离/形态三因子权重，格式如 0.4,0.4,0.2。')
    parser.add_argument('--weight_base_max_gap', type=int, default=100, help='权重扫描时固定使用的 max_gap。')
    parser.add_argument('--weight_base_dist_thresh', type=int, default=200, help='权重扫描时固定使用的 dist_thresh。')
    parser.add_argument('--seqs', nargs='+', default=None, help='只处理指定序列；默认按 SNMOT 测试集 49 序列全量执行。')



    parser.add_argument('--device', type=str, default=None, help='torch 设备，如 cpu/cuda:0。')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_combos', type=int, default=None, help='若设置，则只跑前 N 个参数组合。')
    parser.add_argument('--tracker_name_prefix', type=str, default='hybrid_aflink')
    args = parser.parse_args()
    # MCVC 一键开关（论文 Table 11「MCVC (ours)」）：
    #   T1 锥形剪枝 —— 把门控从「评分标定器」降级为「剪枝器」（gate_score_norm=nominal）
    #   T2 队色等价类否决 —— 主增益（kit_veto=any，取代评分型门控）
    # 只填「仍是默认值」的项，不覆盖用户显式传入的取值。
    # 双划分消融结论（test49 / train57，见 _mcv/report_*.md）：
    #   T2 队色否决   +0.604 / +0.582  ← 唯一稳健增益，纳入默认
    #   V3a 互选否决  −0.210 / −0.039  ← 负向
    #   V3b 分量纯度   0.000 /  0.000  ← 从不触发（死层）
    #   T4 回补轮     +0.078 / −0.162  ← 划分不稳
    # 故 V3/V4/T4 一律保持 off，仅留作可选层。
    if getattr(args, 'mcv', False):
        # T2 队色等价类否决 —— 组件核心，也是唯一在两个划分上都稳健的增益层。
        if args.gate_score_norm == 'effective':
            args.gate_score_norm = 'nominal'
        if args.kit_veto == 'off':
            args.kit_veto = 'any'
        # T1 锥形剪枝 —— 显式可选。实测叠加到 T2 之上：test −0.091 / train +0.057
        # （符号翻转），故**不**作为默认；需要时用 --mcv_cone 打开。
        if getattr(args, 'mcv_cone', False):
            if args.rule_variant == 'baseline':
                args.rule_variant = 'adaptive_cone'
            if args.cone_g0 == 60.0:
                args.cone_g0 = 55.0
        # 注意：**不**碰 score_reliability_penalty。该开关足以让门控符号反转
        # （方案9 §9.6），Table 11 整表口径固定为 0.0，组件不得擅自改动。
    return args




def load_json_predictions(cache_path: Path) -> dict[str, dict[int, TrackFragment]]:
    with cache_path.open('r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'sequences' in raw:
        raw = raw['sequences']

    if not isinstance(raw, dict):
        raise ValueError('raw_pred_cache.json 顶层必须是 dict。')

    parsed: dict[str, dict[int, TrackFragment]] = {}
    for seq_name, seq_payload in raw.items():
        parsed[seq_name] = normalize_sequence_payload(seq_payload)
    return parsed



def normalize_sequence_payload(seq_payload: Any) -> dict[int, TrackFragment]:
    if isinstance(seq_payload, dict) and 'tracks' in seq_payload:
        seq_payload = seq_payload['tracks']

    if isinstance(seq_payload, dict):
        maybe_track_map = {}
        for key, value in seq_payload.items():
            if isinstance(value, dict) and 'frames' in value and 'boxes' in value:
                track_id = int(value.get('track_id', key))
                maybe_track_map[track_id] = build_track_fragment(track_id, value['frames'], value['boxes'])
        if maybe_track_map:
            return maybe_track_map

        flat_records = []
        for value in seq_payload.values():
            if isinstance(value, list):
                flat_records.extend(value)
        if flat_records:
            return group_detection_records(flat_records)

    if isinstance(seq_payload, list):
        return group_detection_records(seq_payload)

    raise ValueError('无法解析 raw_pred_cache.json 中的序列结构。')



def group_detection_records(records: list[dict[str, Any]]) -> dict[int, TrackFragment]:
    grouped: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for item in records:
        if not isinstance(item, dict):
            continue
        track_id = int(item.get('track_id', item.get('id', item.get('tid', -1))))
        frame_id = int(item.get('frame', item.get('frame_id', item.get('fid', -1))))
        if track_id < 0 or frame_id < 0:
            continue
        box = extract_box_from_record(item)
        grouped[track_id].append((frame_id, box))

    track_map: dict[int, TrackFragment] = {}
    for track_id, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0])
        frames = [frame for frame, _ in pairs]
        boxes = [box for _, box in pairs]
        track_map[track_id] = build_track_fragment(track_id, frames, boxes)
    return track_map



def extract_box_from_record(item: dict[str, Any]) -> list[float]:
    for key in ('bbox', 'box', 'tlbr', 'xyxy'):
        if key in item:
            box = item[key]
            if len(box) != 4:
                raise ValueError('检测框长度必须为 4。')
            return [float(v) for v in box]
    if 'tlwh' in item:
        x, y, w, h = [float(v) for v in item['tlwh']]
        return [x, y, x + w, y + h]
    x = float(item['x'])
    y = float(item['y'])
    w = float(item['w'])
    h = float(item['h'])
    return [x, y, x + w, y + h]



def build_track_fragment(track_id: int, frames: list[int], boxes: list[list[float]]) -> TrackFragment:
    paired = sorted(zip([int(f) for f in frames], boxes), key=lambda pair: pair[0])
    unique_pairs: list[tuple[int, list[float]]] = []
    last_frame = None
    for frame, box in paired:
        box_float = [float(v) for v in box]
        if last_frame == frame:
            unique_pairs[-1] = (frame, box_float)
        else:
            unique_pairs.append((frame, box_float))
            last_frame = frame
    out_frames = [frame for frame, _ in unique_pairs]
    out_boxes = [box for _, box in unique_pairs]
    return TrackFragment(track_id=int(track_id), frames=out_frames, boxes=out_boxes)



def load_mot_predictions(results_dir: Path, seqs: list[str] | None = None) -> dict[str, dict[int, TrackFragment]]:
    if seqs is None:
        seq_files = sorted(results_dir.glob('*.txt'))
    else:
        seq_files = [results_dir / f'{seq}.txt' for seq in seqs]

    predictions: dict[str, dict[int, TrackFragment]] = {}
    for txt_file in seq_files:
        if not txt_file.exists():
            raise FileNotFoundError(f'结果文件不存在: {txt_file}')
        predictions[txt_file.stem] = parse_mot_file(txt_file)
    return predictions



def parse_mot_file(txt_file: Path) -> dict[int, TrackFragment]:
    grouped: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    with txt_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            grouped[track_id].append((frame_id, [x, y, x + w, y + h]))

    track_map: dict[int, TrackFragment] = {}
    for track_id, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0])
        frames = [frame for frame, _ in pairs]
        boxes = [box for _, box in pairs]
        track_map[track_id] = build_track_fragment(track_id, frames, boxes)
    return track_map



def collect_sequences(results_dir: Path, gt_root: Path, requested_seqs: list[str] | None = None) -> list[str]:
    if requested_seqs:
        seqs = list(requested_seqs)
    else:
        seqs = load_default_test_sequence_names(gt_root)

    if not seqs:
        raise RuntimeError('未找到可用序列。')

    missing_results = [seq for seq in seqs if not (results_dir / f'{seq}.txt').exists()]
    if missing_results:
        raise FileNotFoundError(f'以下结果文件不存在: {missing_results}')

    missing_gt = [seq for seq in seqs if not (gt_root / seq / 'gt' / 'gt.txt').exists()]
    if missing_gt:
        raise FileNotFoundError(f'以下序列缺少 GT: {missing_gt}')
    return seqs




def merge_tracks(track_map: dict[int, TrackFragment], link_map: dict[int, int]) -> tuple[dict[int, TrackFragment], dict[int, list[int]]]:
    root_groups: dict[int, list[int]] = defaultdict(list)
    for track_id in track_map:
        root_id = link_map.get(track_id, track_id)
        root_groups[root_id].append(track_id)

    merged: dict[int, TrackFragment] = {}
    lineage: dict[int, list[int]] = {}
    for root_id, track_ids in root_groups.items():
        ordered = sorted(track_ids, key=lambda tid: (track_map[tid].start_frame, track_map[tid].end_frame, tid))
        frame_to_box: dict[int, list[float]] = {}
        for tid in ordered:
            fragment = track_map[tid]
            for frame, box in zip(fragment.frames, fragment.boxes):
                frame_to_box.setdefault(frame, [float(v) for v in box])
        merged_frames = sorted(frame_to_box.keys())
        merged_boxes = [frame_to_box[frame] for frame in merged_frames]
        merged[root_id] = TrackFragment(track_id=root_id, frames=merged_frames, boxes=merged_boxes)
        lineage[root_id] = ordered
    return merged, lineage



def apply_aflink(
    predictions: dict[str, dict[int, TrackFragment]],
    linker: HybridAFLink,
) -> tuple[dict[str, dict[int, TrackFragment]], dict[str, dict[str, float | int]]]:
    processed: dict[str, dict[int, TrackFragment]] = {}
    debug_stats: dict[str, dict[str, float | int]] = {}

    for seq_name, track_map in predictions.items():
        setattr(linker, 'current_seq', seq_name)   # 方案8：让队别标签表可按序列名索引
        link_map, stats = linker.link_sequence(track_map)
        merged_tracks, lineage = merge_tracks(track_map, link_map)
        stats = dict(stats)
        stats['merged_tracks'] = sum(1 for ids in lineage.values() if len(ids) > 1)
        stats['track_count_before'] = len(track_map)
        stats['track_count_after'] = len(merged_tracks)
        processed[seq_name] = merged_tracks
        debug_stats[seq_name] = stats

    if getattr(linker, 'kin_dir_tau', 0.0) > 0.0:
        print(f"[方案10] 方向一致性否决: checked={linker.kin_dir_totals['checked']} "
              f"vetoed={linker.kin_dir_totals['vetoed']}", flush=True)
    if getattr(linker, 'kit_veto', 'off') != 'off':
        print(f"[方案8] 队色否决: candidates={linker.kit_veto_totals['candidates']} "
              f"vetoed={linker.kit_veto_totals['vetoed']}", flush=True)
    if linker.mcv_active or linker.mcv_recover_active \
            or getattr(linker, 'kit_veto', 'off') != 'off':
        mt = linker.mcv_totals
        print(f"[MCVC] T1/T2 锥形剪枝+队色等价类否决: variant={linker.rule_variant} "
              f"kit_veto={linker.kit_veto}", flush=True)
        print(f"[MCVC] T3 一致闭包否决: checked={mt['checked']} recip_vetoed={mt['recip']} "
              f"trans_vetoed={mt['trans']} ball_vetoed={mt['ball']}", flush=True)
        print(f"[MCVC] T4 互补回补: added={mt['recover']}", flush=True)
        print(f"[MCVC] 配置: recip={linker.mcv_recip} trans={linker.mcv_trans} "
              f"trans_max={linker.mcv_trans_max} ball_excl={linker.mcv_ball_excl} "
              f"recover={linker.mcv_recover}({linker.mcv_recover_gap}/{linker.mcv_recover_dist}"
              f"{'/permissive' if linker.mcv_recover_permissive else '/strict'})", flush=True)

    return processed, debug_stats



def write_predictions_to_dir(predictions: dict[str, dict[int, TrackFragment]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for seq_name, track_map in predictions.items():
        rows: list[tuple[int, int, float, float, float, float]] = []
        for track_id, fragment in track_map.items():
            for frame_id, box in zip(fragment.frames, fragment.boxes):
                x1, y1, x2, y2 = box
                rows.append((frame_id, track_id, x1, y1, x2 - x1, y2 - y1))

        rows.sort(key=lambda item: (item[0], item[1]))
        with (output_dir / f'{seq_name}.txt').open('w', encoding='utf-8') as f:
            for frame_id, track_id, x, y, w, h in rows:
                f.write(f'{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n')



def get_seq_length(gt_root: Path, seq: str) -> int:
    seqinfo_path = gt_root / seq / 'seqinfo.ini'
    for line in seqinfo_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('seqLength='):
            return int(line.split('=', 1)[1].strip())
    raise ValueError(f'在 {seqinfo_path} 中未找到 seqLength')



def build_trackeval_components(gt_root: Path, trackers_root: Path, tracker_name: str, seqs: list[str], output_dir: Path):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        'USE_PARALLEL': False,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False,
    })

    seq_info = {seq: get_seq_length(gt_root, seq) for seq in seqs}
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': str(gt_root),
        'TRACKERS_FOLDER': str(trackers_root),
        'OUTPUT_FOLDER': str(output_dir),
        'TRACKERS_TO_EVAL': [tracker_name],
        'TRACKER_DISPLAY_NAMES': [tracker_name],
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'SNMOT',
        'SPLIT_TO_EVAL': 'test',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': False,
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        'SEQ_INFO': seq_info,
        'SEQMAP_FILE': None,
        'SKIP_SPLIT_FOL': True,
    })

    metrics_config = {'THRESHOLD': 0.5, 'PRINT_CONFIG': False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    return evaluator, dataset, metrics_list



def run_trackeval(
    results_dir: Path,
    gt_root: Path,
    seqs: list[str],
    tracker_name: str,
    output_dir: Path,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix='trackeval_local_') as temp_dir:
        temp_root = Path(temp_dir)
        tracker_dir = temp_root / tracker_name / 'data'
        tracker_dir.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            copy2(results_dir / f'{seq}.txt', tracker_dir / f'{seq}.txt')

        evaluator, dataset, metrics_list = build_trackeval_components(
            gt_root=gt_root,
            trackers_root=temp_root,
            tracker_name=tracker_name,
            seqs=seqs,
            output_dir=output_dir,
        )
        output_res, output_msg = evaluator.evaluate([dataset], metrics_list)
        dataset_name = dataset.get_name()
        if output_msg[dataset_name][tracker_name] != 'Success':
            raise RuntimeError(output_msg[dataset_name][tracker_name])

        combined = output_res[dataset_name][tracker_name]['COMBINED_SEQ']['pedestrian']
        return {
            'HOTA': 100.0 * float(np.mean(combined['HOTA']['HOTA'])),
            'DetA': 100.0 * float(np.mean(combined['HOTA']['DetA'])),
            'AssA': 100.0 * float(np.mean(combined['HOTA']['AssA'])),
            'MOTA': 100.0 * float(combined['CLEAR']['MOTA']),
            'IDF1': 100.0 * float(combined['Identity']['IDF1']),
            'IDs': float(combined['CLEAR']['IDSW']),
        }



TABLE_PARAM_COLUMNS = [
    ('sweep_target', 'sweep'),
    ('max_gap', 'max_gap'),
    ('dist_thresh', 'dist_thresh'),
    ('shape_area_weight', 'area_w'),
    ('shape_aspect_weight', 'aspect_w'),
    ('gate_reliability_weight', 'gate_rel_w'),
    ('gate_shape_weight', 'gate_shape_w'),
    ('temporal_weight', 'temp_w'),
    ('distance_weight', 'dist_w'),
    ('shape_weight', 'shape_w'),
]



def get_active_table_param_columns(records: list[dict[str, float | int | str]]) -> list[tuple[str, str]]:
    active: list[tuple[str, str]] = []
    for key, label in TABLE_PARAM_COLUMNS:
        if not any(key in item for item in records):
            continue
        values = [item.get(key) for item in records]
        unique = {repr(value) for value in values}
        if key in {'sweep_target', 'max_gap', 'dist_thresh'} or len(unique) > 1:
            active.append((key, label))
    return active



def format_table_value(key: str, value: float | int | str | None) -> str:
    if value is None:
        return ''
    if key == 'sweep_target':
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)



def format_markdown_table(records: list[dict[str, float | int | str]]) -> str:
    param_columns = get_active_table_param_columns(records)
    metric_columns = [
        ('HOTA', 'HOTA'),
        ('DetA', 'DetA'),
        ('AssA', 'AssA'),
        ('MOTA', 'MOTA'),
        ('IDF1', 'IDF1'),
        ('IDs', 'IDs'),
    ]
    headers = ['Rank', 'mode', 'variant', *[label for _, label in param_columns], *[label for _, label in metric_columns]]
    alignments = ['---', '---', '---', *(['---:' for _ in param_columns]), '---:', '---:', '---:', '---:', '---:', '---:']
    rows = [f"| {' | '.join(headers)} |", f"| {' | '.join(alignments)} |"]

    for idx, item in enumerate(records, 1):
        cells = [str(idx), str(item['mode']), str(item['rule_variant'])]
        for key, _ in param_columns:
            cells.append(format_table_value(key, item.get(key)))
        for key, _ in metric_columns:
            value = item[key]
            if key == 'IDs':
                cells.append(f'{float(value):.0f}')
            else:
                cells.append(f'{float(value):.3f}')
        rows.append(f"| {' | '.join(cells)} |")
    return '\n'.join(rows)




def build_rule_config_section(item: dict[str, float | int | str]) -> list[str]:
    variant = str(item['rule_variant'])
    lines = [
        '### 当前规则配置',
        '',
        f"- mode: `{item['mode']}`",
        f"- variant: `{variant}`（{get_rule_variant_label(variant)}）",
    ]
    if uses_adaptive_gate_variant(variant):
        lines.append(f"- adaptive_len_ref: `{int(item['adaptive_len_ref'])}`")
        lines.append(f"- adaptive_gap_min_ratio: `{float(item['adaptive_gap_min_ratio']):.2f}`")
        lines.append(f"- adaptive_dist_min_ratio: `{float(item['adaptive_dist_min_ratio']):.2f}`")
        lines.append(f"- adaptive_gap_max_ratio: `{float(item.get('adaptive_gap_max_ratio', 1.0)):.2f}`")
        lines.append(f"- adaptive_dist_max_ratio: `{float(item.get('adaptive_dist_max_ratio', 1.0)):.2f}`")
        _ref = item.get('gate_signal_ref')
        lines.append(f"- gate_signal_ref: `{'None（单侧映射）' if _ref is None else '%.4f' % float(_ref)}`")
        lines.append(f"- gate_signal_gain: `{float(item.get('gate_signal_gain', 1.0)):.3f}`")
        grw, gsw = normalize_pair_weights(item['gate_reliability_weight'], item['gate_shape_weight'])
        lines.append(f"- gate_reliability_weight: `{grw:.3f}`")
        lines.append(f"- gate_shape_weight: `{gsw:.3f}`")
    if uses_adaptive_gate_variant(variant) or uses_shape_factor_variant(variant):
        aw, apw = normalize_pair_weights(item['shape_area_weight'], item['shape_aspect_weight'])
        lines.append(f"- shape_area_weight: `{aw:.3f}`")
        lines.append(f"- shape_aspect_weight: `{apw:.3f}`")
    if uses_shape_factor_variant(variant):
        tw, dw, sw = normalize_rule_weights(item['temporal_weight'], item['distance_weight'], item['shape_weight'])
        lines.append(f"- temporal_weight: `{tw:.3f}`")
        lines.append(f"- distance_weight: `{dw:.3f}`")
        lines.append(f"- shape_weight: `{sw:.3f}`")

    if uses_gap_decay_variant(variant):
        lines.append(f"- gap_decay_alpha: `{float(item['gap_decay_alpha']):.2f}`")
    if uses_length_conf_variant(variant):
        lines.append(f"- length_ref: `{int(item['length_ref'])}`")
        lines.append(f"- length_conf_power: `{float(item['length_conf_power']):.2f}`")
    if uses_residual_gate_variant(variant):
        lines.append(f"- residual_blend: `{float(item.get('residual_blend', 1.0)):.3f}`")
        lines.append(f"- residual_velocity_clip: `{float(item.get('residual_velocity_clip', 0.0)):.1f}`")
        lines.append(f"- residual_velocity_window: `{int(item.get('residual_velocity_window', 2))}`")
        lines.append(f"- residual_min_src_len: `{int(item.get('residual_min_src_len', 2))}`")
        lines.append(f"- residual_keep_center_gate: `{bool(item.get('residual_keep_center_gate', False))}`")
        lines.append(f"- residual_gap_cap: `{int(item.get('residual_gap_cap', 0))}`")
    lines.extend(
        [
            f"- avg_rule_score: `{float(item['avg_rule_score']):.3f}`",
            f"- avg_temporal_score: `{float(item['avg_temporal_score']):.3f}`",
            f"- avg_distance_score: `{float(item['avg_distance_score']):.3f}`",
            f"- avg_shape_score: `{float(item['avg_shape_score']):.3f}`",
            f"- avg_length_confidence: `{float(item['avg_length_confidence']):.3f}`",
            '',
        ]
    )
    return lines




def analyze_best_config(
    best: dict[str, float | int | str],
    gap_values: list[int],
    dist_values: list[int],
    source_desc: str,
    combo_note: str | None,
    sweep_desc: str,
) -> str:

    gap_values = sorted(gap_values)
    dist_values = sorted(dist_values)
    best_gap = int(best['max_gap'])
    best_dist = int(best['dist_thresh'])
    best_variant = str(best['rule_variant'])

    def describe_position(value: int, candidates: list[int], axis_name: str) -> str:
        if len(candidates) <= 1:
            return f'{axis_name} 固定为 {value}'
        idx = candidates.index(value)
        if idx == 0:
            return f'{axis_name} 取最严格端'
        if idx == len(candidates) - 1:
            return f'{axis_name} 取最宽松端'

        if idx in {len(candidates) // 2, max(len(candidates) // 2 - 1, 0)}:
            return f'{axis_name} 落在中间区间'
        if idx < len(candidates) // 2:
            return f'{axis_name} 偏保守'
        return f'{axis_name} 偏激进'

    explanation = [
        f"本次扫描目标为 **{sweep_desc}**。",
        f"最佳配置为 **mode={best['mode']} / variant={best_variant} / max_gap={best_gap} / dist_thresh={best_dist}**。",
        f"当前规则变体为 **{get_rule_variant_label(best_variant)}**。",
        f"从搜索区间看，{describe_position(best_gap, gap_values, 'max_gap')}，{describe_position(best_dist, dist_values, 'dist_thresh')}，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。",
        f"候选层平均规则分数为 **{float(best['avg_rule_score']):.3f}**，平均时间项为 **{float(best['avg_temporal_score']):.3f}**，平均距离项为 **{float(best['avg_distance_score']):.3f}**，平均形态项为 **{float(best['avg_shape_score']):.3f}**，平均长度置信度为 **{float(best['avg_length_confidence']):.3f}**。",
    ]

    if uses_adaptive_gate_variant(best_variant):
        grw, gsw = normalize_pair_weights(best['gate_reliability_weight'], best['gate_shape_weight'])
        explanation.append(
            f"候选筛选采用自适应时空门控，`adaptive_len_ref={int(best['adaptive_len_ref'])}`、`adaptive_gap_min_ratio={float(best['adaptive_gap_min_ratio']):.2f}`、`adaptive_dist_min_ratio={float(best['adaptive_dist_min_ratio']):.2f}`；门控信号中的可靠性/形态权重分别为 `{grw:.3f}` / `{gsw:.3f}`，短碎片与形态不连续候选会被更严格地过滤，从而优先改善 **AssA / IDF1**。"
        )
    if uses_adaptive_gate_variant(best_variant) or uses_shape_factor_variant(best_variant):
        aw, apw = normalize_pair_weights(best['shape_area_weight'], best['shape_aspect_weight'])
        explanation.append(
            f"形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `{aw:.3f}` / `{apw:.3f}`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。"
        )
    if uses_shape_factor_variant(best_variant):
        tw, dw, sw = normalize_rule_weights(best['temporal_weight'], best['distance_weight'], best['shape_weight'])
        explanation.append(
            f"规则排序采用三因子加权，时间/距离/形态权重分别为 `{tw:.3f}` / `{dw:.3f}` / `{sw:.3f}`；这会让形态连续性直接参与链接优先级，降低跨目标误配。"
        )

    if uses_gap_decay_variant(best_variant):
        explanation.append(
            f"时间项采用非线性衰减，`gap_decay_alpha={float(best['gap_decay_alpha']):.2f}`；这会对短时遮挡更宽容、对长时间中断更快降分。"
        )
    elif not uses_adaptive_gate_variant(best_variant):
        explanation.append('当前仍使用原始线性 gap 项，因此结果可直接作为旧版规则 AFLink 的 baseline。')
    if uses_length_conf_variant(best_variant):
        explanation.append(
            f"长度项使用轨迹长度置信度，`length_ref={int(best['length_ref'])}`、`length_conf_power={float(best['length_conf_power']):.2f}`；较短碎片会被更保守地排序，从而降低误连接风险。"
        )
    if uses_residual_gate_variant(best_variant):
        explanation.append(
            f"空间度量采用**预测残差距离**：先用 src 末端速度外推 `c_a,last + v_a·gap`，再与 `c_b,first` 比较"
            f"（blend=`{float(best.get('residual_blend', 1.0)):.2f}`，速度截断=`{float(best.get('residual_velocity_clip', 0.0)):.1f}` px/帧，"
            f"src 最短外推长度=`{int(best.get('residual_min_src_len', 2))}`，与端点中心距取交=`{bool(best.get('residual_keep_center_gate', False))}`）。"
            "这把原来的零阶「位移上限 = v_max·gap」升级为一阶恒速模型，在相同窗口下对顺运动方向的候选更宽容、对方向突变的候选更严格。"
        )
    explanation.append(f"本次分析的预测来源为 **{source_desc}**。")

    if combo_note:
        explanation.append(combo_note)
    return '\n\n'.join(explanation)



def save_report(
    report_path: Path,
    records: list[dict[str, float | int | str]],
    best: dict[str, float | int | str],
    gap_values: list[int],
    dist_values: list[int],
    source_desc: str,
    combo_note: str | None,
    sweep_desc: str,
) -> None:
    table = format_markdown_table(records)
    analysis = analyze_best_config(best, gap_values, dist_values, source_desc, combo_note, sweep_desc)

    content = '\n'.join(
        [
            '## 规则 AFLink 混合版灵敏性分析报告',
            '',
            *build_rule_config_section(best),
            '### 排序结果（按 HOTA 降序）',
            '',
            table,
            '',
            '### 最佳配置分析',
            '',
            analysis,
            '',
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding='utf-8')




def load_prediction_source(
    cache_path: Path,
    results_dir: Path,
    seqs: list[str],
) -> tuple[dict[str, dict[int, TrackFragment]], str]:
    if cache_path.exists():
        predictions = load_json_predictions(cache_path)
        predictions = {seq: predictions[seq] for seq in seqs if seq in predictions}
        missing = [seq for seq in seqs if seq not in predictions]
        if missing:
            raise FileNotFoundError(f'raw_pred_cache.json 中缺少序列: {missing}')
        return predictions, f'raw_pred_cache.json ({cache_path})'

    predictions = load_mot_predictions(results_dir, seqs)
    return predictions, f'MOT 结果目录回退 ({results_dir})'



def parse_rule_weight_triplet(spec: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(spec).split(',') if part.strip()]
    if len(parts) != 3:
        raise ValueError(f'规则权重格式错误: {spec}，应为 time,distance,shape。')
    return float(parts[0]), float(parts[1]), float(parts[2])



def build_sweep_grid(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], list[int], list[int], str]:
    target = str(args.sweep_target)
    base_gap = int(args.weight_base_max_gap)
    base_dist = int(args.weight_base_dist_thresh)

    if target == 'gap_dist':
        grid = [
            {
                'sweep_target': target,
                'max_gap': int(max_gap),
                'dist_thresh': int(dist_thresh),
                'shape_area_weight': float(args.area_weight),
                'shape_aspect_weight': float(args.aspect_weight),
                'gate_reliability_weight': float(args.gate_reliability_weight),
                'gate_shape_weight': float(args.gate_shape_weight),
                'temporal_weight': float(args.temporal_weight),
                'distance_weight': float(args.distance_weight),
                'shape_weight': float(args.shape_weight),
            }
            for max_gap, dist_thresh in itertools.product(args.max_gap_values, args.dist_thresh_values)
        ]
        return grid, list(args.max_gap_values), list(args.dist_thresh_values), 'max_gap × dist_thresh 网格搜索'

    if target == 'shape_mix':
        grid = []
        for area_weight in args.shape_area_weight_values:
            area_weight = float(area_weight)
            grid.append(
                {
                    'sweep_target': target,
                    'max_gap': base_gap,
                    'dist_thresh': base_dist,
                    'shape_area_weight': area_weight,
                    'shape_aspect_weight': 1.0 - area_weight,
                    'gate_reliability_weight': float(args.gate_reliability_weight),
                    'gate_shape_weight': float(args.gate_shape_weight),
                    'temporal_weight': float(args.temporal_weight),
                    'distance_weight': float(args.distance_weight),
                    'shape_weight': float(args.shape_weight),
                }
            )
        return grid, [base_gap], [base_dist], 'shape_score 权重扫描（面积比例 × 长宽比）'

    if target == 'gate_mix':
        grid = []
        for reliability_weight in args.gate_reliability_weight_values:
            reliability_weight = float(reliability_weight)
            grid.append(
                {
                    'sweep_target': target,
                    'max_gap': base_gap,
                    'dist_thresh': base_dist,
                    'shape_area_weight': float(args.area_weight),
                    'shape_aspect_weight': float(args.aspect_weight),
                    'gate_reliability_weight': reliability_weight,
                    'gate_shape_weight': 1.0 - reliability_weight,
                    'temporal_weight': float(args.temporal_weight),
                    'distance_weight': float(args.distance_weight),
                    'shape_weight': float(args.shape_weight),
                }
            )
        return grid, [base_gap], [base_dist], 'gate_signal 权重扫描（可靠性 × 形态）'

    if target == 'rule_mix':
        grid = []
        for triplet in args.rule_weight_triplets:
            temporal_weight, distance_weight, shape_weight = parse_rule_weight_triplet(triplet)
            grid.append(
                {
                    'sweep_target': target,
                    'max_gap': base_gap,
                    'dist_thresh': base_dist,
                    'shape_area_weight': float(args.area_weight),
                    'shape_aspect_weight': float(args.aspect_weight),
                    'gate_reliability_weight': float(args.gate_reliability_weight),
                    'gate_shape_weight': float(args.gate_shape_weight),
                    'temporal_weight': temporal_weight,
                    'distance_weight': distance_weight,
                    'shape_weight': shape_weight,
                }
            )
        return grid, [base_gap], [base_dist], 'rule_score 权重扫描（时间 × 距离 × 形态）'

    raise ValueError(f'未知 sweep_target: {target}')



def sensitivity_grid(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], str, str | None, list[int], list[int], str]:

    results_dir = Path(args.results_dir).resolve()
    gt_root = Path(args.gt_root).resolve()
    cache_path = Path(args.cache_path).resolve()
    output_root = Path(args.output_root).resolve()

    seqs = collect_sequences(results_dir, gt_root, args.seqs)
    predictions, source_desc = load_prediction_source(cache_path, results_dir, seqs)

    grid, gap_values, dist_values, sweep_desc = build_sweep_grid(args)
    combo_note = None
    if args.max_combos is not None and args.max_combos < len(grid):
        grid = grid[:args.max_combos]
        combo_note = (
            f'由于运行预算限制，本次仅评估了前 {len(grid)} 组参数组合，'
            '因此结论属于当前子集上的最优结果。'
        )

    print('=' * 80)
    print('规则 AFLink 混合版灵敏性分析')
    print('=' * 80)
    print(f'模式: {args.mode}')
    print(f'规则变体: {normalize_rule_variant(args.rule_variant)} ({get_rule_variant_label(args.rule_variant)})')
    print(f'扫描目标: {sweep_desc}')
    print(f'序列数: {len(seqs)}')
    print(f'预测来源: {source_desc}')
    print(f'参数组合数: {len(grid)}')
    print('=' * 80)

    records: list[dict[str, float | int | str]] = []
    variant_tag = normalize_rule_variant(args.rule_variant)
    for idx, config in enumerate(grid, 1):
        max_gap = int(config['max_gap'])
        dist_thresh = int(config['dist_thresh'])
        tracker_name = (
            f"{args.tracker_name_prefix}_{args.mode}_{variant_tag}_{config['sweep_target']}"
            f"_gap{max_gap}_dist{dist_thresh}"
            f"_aw{int(round(float(config['shape_area_weight']) * 100))}"
            f"_grw{int(round(float(config['gate_reliability_weight']) * 100))}"
            f"_tw{int(round(float(config['temporal_weight']) * 100))}"
            f"_dw{int(round(float(config['distance_weight']) * 100))}"
            f"_sw{int(round(float(config['shape_weight']) * 100))}"
        )
        combo_results_dir = output_root / 'postprocessed' / tracker_name
        combo_eval_dir = output_root / 'trackeval' / tracker_name

        print(
            f"\n[{idx}/{len(grid)}] target={config['sweep_target']}, max_gap={max_gap}, dist_thresh={dist_thresh}, "
            f"area/aspect={float(config['shape_area_weight']):.2f}/{float(config['shape_aspect_weight']):.2f}, "
            f"gate={float(config['gate_reliability_weight']):.2f}/{float(config['gate_shape_weight']):.2f}, "
            f"rule={float(config['temporal_weight']):.2f}/{float(config['distance_weight']):.2f}/{float(config['shape_weight']):.2f}"
        )
        linker = HybridAFLink(
            mode=args.mode,
            max_gap=max_gap,
            dist_thresh=dist_thresh,
            score_thresh=args.score_thresh,
            seq_len=args.seq_len,
            model_path=args.aflink_weights,
            device=args.device,
            batch_size=args.batch_size,
            rule_variant=args.rule_variant,
            gap_decay_alpha=args.gap_decay_alpha,
            length_ref=args.length_ref,
            length_conf_power=args.length_conf_power,
            adaptive_len_ref=args.adaptive_len_ref,
            adaptive_gap_min_ratio=args.adaptive_gap_min_ratio,
            adaptive_dist_min_ratio=args.adaptive_dist_min_ratio,
            adaptive_gap_max_ratio=float(getattr(args, 'adaptive_gap_max_ratio', 1.0)),
            adaptive_dist_max_ratio=float(getattr(args, 'adaptive_dist_max_ratio', 1.0)),
            gate_signal_ref=(None if getattr(args, 'gate_signal_ref', None) is None
                             else float(args.gate_signal_ref)),
            gate_signal_gain=float(getattr(args, 'gate_signal_gain', 1.0)),
            cone_g0=float(args.cone_g0),
            cone_power=float(args.cone_power),
            cone_floor=float(args.cone_floor),
            score_reliability_penalty=float(args.score_reliability_penalty),
            penalty_len_ref=int(args.penalty_len_ref),
            kin_kappa=float(getattr(args, 'kin_kappa', 0.0)),
            kin_floor=float(getattr(args, 'kin_floor', 0.0)),
            kin_ceil=float(getattr(args, 'kin_ceil', 0.0)),
            kin_window=int(getattr(args, 'kin_window', 5)),
            kin_min_len=int(getattr(args, 'kin_min_len', 4)),
            kin_pair=bool(getattr(args, 'kin_pair', False)),
            kin_dir_tau=float(getattr(args, 'kin_dir_tau', 0.0)),
            kin_dir_vref=float(getattr(args, 'kin_dir_vref', 1.0)),
            kin_dir_pair=bool(getattr(args, 'kin_dir_pair', True)),
            scale_norm=str(getattr(args, 'scale_norm', 'none')),
            residual_blend=float(getattr(args, 'residual_blend', 1.0)),
            residual_velocity_clip=float(getattr(args, 'residual_velocity_clip', 0.0)),
            residual_velocity_window=int(getattr(args, 'residual_velocity_window', 2)),
            residual_min_src_len=int(getattr(args, 'residual_min_src_len', 2)),
            residual_keep_center_gate=bool(getattr(args, 'residual_keep_center_gate', False)),
            residual_gap_cap=int(getattr(args, 'residual_gap_cap', 0)),
            kit_veto=str(getattr(args, 'kit_veto', 'off')),
            kit_lookup=(load_kit_labels(args.kit_labels) if getattr(args, 'kit_labels', None) else None),
            mcv_recip=str(getattr(args, 'mcv_recip', 'off')),
            mcv_trans=str(getattr(args, 'mcv_trans', 'off')),
            mcv_trans_max=int(getattr(args, 'mcv_trans_max', 1)),
            mcv_ball_excl=bool(getattr(args, 'mcv_ball_excl', False)),
            mcv_ball_rel_h=float(getattr(args, 'mcv_ball_rel_h', 0.45)),
            mcv_ball_aspect=float(getattr(args, 'mcv_ball_aspect', 0.75)),
            mcv_recover=bool(getattr(args, 'mcv_recover', False)),
            mcv_recover_gap=int(getattr(args, 'mcv_recover_gap', 400)),
            mcv_recover_dist=float(getattr(args, 'mcv_recover_dist', 1200.0)),
            mcv_recover_permissive=bool(getattr(args, 'mcv_recover_permissive', False)),
            gate_score_norm=str(args.gate_score_norm),
            gate_reliability_weight=float(config['gate_reliability_weight']),
            gate_shape_weight=float(config['gate_shape_weight']),
            area_weight=float(config['shape_area_weight']),
            aspect_weight=float(config['shape_aspect_weight']),
            temporal_weight=float(config['temporal_weight']),
            distance_weight=float(config['distance_weight']),
            shape_weight=float(config['shape_weight']),
            size_tiebreak=bool(getattr(args, 'size_tiebreak', True)),
        )

        processed, stats = apply_aflink(predictions, linker)
        write_predictions_to_dir(processed, combo_results_dir)
        metrics = run_trackeval(
            results_dir=combo_results_dir,
            gt_root=gt_root,
            seqs=seqs,
            tracker_name=tracker_name,
            output_dir=combo_eval_dir,
        )

        candidate_total = int(sum(seq_stat['raw_candidates'] for seq_stat in stats.values()))
        kept_total = int(sum(seq_stat['kept_candidates'] for seq_stat in stats.values()))
        merged_total = int(sum(seq_stat['merged_tracks'] for seq_stat in stats.values()))
        rule_score_sum = float(sum(seq_stat.get('rule_score_sum', 0.0) for seq_stat in stats.values()))
        temporal_score_sum = float(sum(seq_stat.get('temporal_score_sum', 0.0) for seq_stat in stats.values()))
        distance_score_sum = float(sum(seq_stat.get('distance_score_sum', 0.0) for seq_stat in stats.values()))
        shape_score_sum = float(sum(seq_stat.get('shape_score_sum', 0.0) for seq_stat in stats.values()))
        length_confidence_sum = float(sum(seq_stat.get('length_confidence_sum', 0.0) for seq_stat in stats.values()))
        avg_rule_score = rule_score_sum / candidate_total if candidate_total > 0 else 0.0
        avg_temporal_score = temporal_score_sum / candidate_total if candidate_total > 0 else 0.0
        avg_distance_score = distance_score_sum / candidate_total if candidate_total > 0 else 0.0
        avg_shape_score = shape_score_sum / candidate_total if candidate_total > 0 else 0.0
        avg_length_confidence = length_confidence_sum / candidate_total if candidate_total > 0 else 0.0

        record: dict[str, float | int | str] = {
            'sweep_target': str(config['sweep_target']),
            'mode': args.mode,
            'rule_variant': normalize_rule_variant(args.rule_variant),
            'gap_decay_alpha': float(args.gap_decay_alpha),
            'length_ref': int(args.length_ref),
            'length_conf_power': float(args.length_conf_power),
            'adaptive_len_ref': int(args.adaptive_len_ref),
            'adaptive_gap_min_ratio': float(args.adaptive_gap_min_ratio),
            'adaptive_dist_min_ratio': float(args.adaptive_dist_min_ratio),
            'adaptive_gap_max_ratio': float(getattr(args, 'adaptive_gap_max_ratio', 1.0)),
            'adaptive_dist_max_ratio': float(getattr(args, 'adaptive_dist_max_ratio', 1.0)),
            'gate_signal_ref': (None if getattr(args, 'gate_signal_ref', None) is None
                                else float(args.gate_signal_ref)),
            'gate_signal_gain': float(getattr(args, 'gate_signal_gain', 1.0)),
            'gate_reliability_weight': float(config['gate_reliability_weight']),
            'gate_shape_weight': float(config['gate_shape_weight']),
            'shape_area_weight': float(config['shape_area_weight']),
            'shape_aspect_weight': float(config['shape_aspect_weight']),
            'temporal_weight': float(config['temporal_weight']),
            'distance_weight': float(config['distance_weight']),
            'shape_weight': float(config['shape_weight']),
            'max_gap': max_gap,
            'dist_thresh': dist_thresh,
            **metrics,
            'raw_candidates': candidate_total,
            'kept_candidates': kept_total,
            'merged_tracks': merged_total,
            'avg_rule_score': float(avg_rule_score),
            'avg_temporal_score': float(avg_temporal_score),
            'avg_distance_score': float(avg_distance_score),
            'avg_shape_score': float(avg_shape_score),
            'avg_length_confidence': float(avg_length_confidence),
        }

        records.append(record)
        print(
            f"  HOTA={record['HOTA']:.3f} | DetA={record['DetA']:.3f} | AssA={record['AssA']:.3f} | "
            f"MOTA={record['MOTA']:.3f} | IDF1={record['IDF1']:.3f} | IDs={record['IDs']:.0f}"
        )
        print(f"  candidates={candidate_total} -> kept={kept_total} -> merged_tracks={merged_total}")

    records.sort(key=lambda item: float(item['HOTA']), reverse=True)
    return records, source_desc, combo_note, gap_values, dist_values, sweep_desc




def print_summary(records: list[dict[str, float | int | str]]) -> None:
    print('\n' + '=' * 80)
    print('灵敏性分析结果（按 HOTA 降序）')
    print('=' * 80)
    print(format_markdown_table(records))
    print('=' * 80)



def main() -> None:
    args = parse_args()
    if args.use_best_config:
        args = apply_best_tuned_config(args)
        print('[BestConfig] 已启用灵敏性分析最优参数：adaptive_tri + 单组参数运行。')
    records, source_desc, combo_note, gap_values, dist_values, sweep_desc = sensitivity_grid(args)

    best = records[0]
    print_summary(records)

    report_path = Path(args.report_path).resolve()
    save_report(
        report_path=report_path,
        records=records,
        best=best,
        gap_values=gap_values,
        dist_values=dist_values,
        source_desc=source_desc,
        combo_note=combo_note,
        sweep_desc=sweep_desc,
    )


    print('\n最佳配置:')
    print(
        f"  mode={best['mode']}, variant={best['rule_variant']} ({get_rule_variant_label(str(best['rule_variant']))}), "
        f"max_gap={best['max_gap']}, dist_thresh={best['dist_thresh']}, "
        f"HOTA={best['HOTA']:.3f}, DetA={best['DetA']:.3f}, AssA={best['AssA']:.3f}, "
        f"MOTA={best['MOTA']:.3f}, IDF1={best['IDF1']:.3f}, IDs={best['IDs']:.0f}"
    )
    print(
        f"  avg_rule_score={float(best['avg_rule_score']):.3f}, avg_temporal={float(best['avg_temporal_score']):.3f}, "
        f"avg_distance={float(best['avg_distance_score']):.3f}, avg_shape={float(best['avg_shape_score']):.3f}, "
        f"avg_length_conf={float(best['avg_length_confidence']):.3f}"
    )

    print(f'报告已保存: {report_path}')



if __name__ == '__main__':
    main()
