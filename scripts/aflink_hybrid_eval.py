from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
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

HELP_REQUESTED = any(arg in {'-h', '--help'} for arg in sys.argv[1:])

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None
    F = None
    if not HELP_REQUESTED:
        raise

    class _NNModuleStub:
        pass

    class _NNStub:
        Module = _NNModuleStub

    nn = _NNStub()
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import trackeval
except ModuleNotFoundError as exc:
    trackeval = None
    _TRACKEVAL_IMPORT_ERROR = exc
    if not HELP_REQUESTED:
        pass
else:
    _TRACKEVAL_IMPORT_ERROR = None

from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fussball.project import (
    DEFAULT_AFLINK_WEIGHTS,
    DEFAULT_RAW_PRED_CACHE,
    DEFAULT_TEST_DATA_ROOT,
    EXPECTED_TEST_SEQUENCE_COUNT,
    TRACKER_RESULTS_ROOT,
    WEIGHTS_ROOT,
    load_default_test_sequence_names,
)

DEFAULT_RESULTS_DIR = TRACKER_RESULTS_ROOT / 'sensitivity_aflink_local' / 'aflink_max_gap_60'
DEFAULT_GT_ROOT = DEFAULT_TEST_DATA_ROOT
DEFAULT_CACHE_PATH = DEFAULT_RAW_PRED_CACHE
DEFAULT_OUTPUT_ROOT = TRACKER_RESULTS_ROOT / 'hybrid_aflink_eval'
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / 'sensitivity_report.md'
DEFAULT_WEIGHT_CANDIDATES = [
    DEFAULT_AFLINK_WEIGHTS,
    WEIGHTS_ROOT / 'aflinknetv2.pth',
    WEIGHTS_ROOT / 'aflink_net_v2.pth',
    WEIGHTS_ROOT / 'aflink_v2.pth',
    WEIGHTS_ROOT / 'aflink.pth',
]

RULE_VARIANT_CHOICES = [
    'baseline',
    'adaptive_gate',
    'tri_factor',
    'adaptive_tri',
    'gap_decay',
    'length_conf',
    'gap_length',
]
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
        gate_reliability_weight: float = 0.7,
        gate_shape_weight: float = 0.3,
        area_weight: float = 0.6,
        aspect_weight: float = 0.4,
        temporal_weight: float = 0.4,
        distance_weight: float = 0.4,
        shape_weight: float = 0.2,
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

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_path = self._resolve_model_path(model_path)
        self.model = None


        if self.mode in {'hybrid', 'hybrid_greedy', 'neural'}:
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
        ensure_torch_available('AFLink 权重加载')
        if model_path is None or not model_path.exists():

            raise FileNotFoundError(
                '当前工作区未找到 AFLinkNetV2 权重文件。\n'
                '请通过 --aflink_weights 显式传入权重，例如 weights/aflink_model_v3.pth。'
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
        }

        if len(track_map) < 2:
            return {}, empty_stats

        candidate_gap = self.max_gap
        candidate_dist = self.dist_thresh
        if self.mode == 'neural':
            candidate_gap = max(self.max_gap, int(round(self.max_gap * 1.5)))
            candidate_dist = max(self.dist_thresh, self.dist_thresh * 1.5)

        candidates = self._build_candidates(track_map, candidate_gap, candidate_dist)
        raw_count = len(candidates)
        if not candidates:
            return {}, empty_stats

        candidate_stats: dict[str, float | int] = {
            'raw_candidates': raw_count,
            'rule_score_sum': float(sum(cand.rule_score for cand in candidates)),
            'temporal_score_sum': float(sum(cand.temporal_score for cand in candidates)),
            'distance_score_sum': float(sum(cand.distance_score for cand in candidates)),
            'shape_score_sum': float(sum(cand.shape_score for cand in candidates)),
            'length_confidence_sum': float(sum(cand.length_confidence for cand in candidates)),
        }


        if self.mode == 'rule':
            link_map = self._greedy_match(candidates, use_neural=False, rule_variant=self.rule_variant)
            return self._flatten_link_map(link_map), {
                **candidate_stats,
                'kept_candidates': raw_count,
                'links': len(link_map),
            }

        scored = self._score_candidates(candidates, track_map)
        filtered = [cand for cand in scored if cand.neural_score is not None and cand.neural_score >= self.score_thresh]

        if self.mode == 'hybrid':
            link_map = self._hungarian_match(filtered, track_map)
        elif self.mode == 'hybrid_greedy':
            link_map = self._greedy_match(filtered, use_neural=True, rule_variant=self.rule_variant)
        elif self.mode == 'neural':
            link_map = self._union_find_match(filtered, track_map)
        else:
            raise ValueError(f'未知 mode: {self.mode}')

        return self._flatten_link_map(link_map), {
            **candidate_stats,
            'kept_candidates': len(filtered),
            'links': len(link_map),
        }


    def _build_candidates(
        self,
        track_map: dict[int, TrackFragment],
        max_gap: int,
        dist_thresh: float,
    ) -> list[CandidatePair]:
        track_ids = sorted(track_map.keys(), key=lambda tid: (track_map[tid].start_frame, track_map[tid].end_frame, tid))
        track_lengths = {tid: len(track_map[tid].frames) for tid in track_ids}
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
                dist = center_distance(track_a.last_box, track_b.first_box)
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
                        gate_reliability_weight=self.gate_reliability_weight,
                        gate_shape_weight=self.gate_shape_weight,
                    )

                    if gap > adaptive_gap_limit or dist > adaptive_dist_limit:
                        continue
                    effective_gap = adaptive_gap_limit
                    effective_dist_thresh = adaptive_dist_limit

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
                    )
                )

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
    def _greedy_match(candidates: list[CandidatePair], use_neural: bool, rule_variant: str = 'baseline') -> dict[int, int]:
        chosen_src: set[int] = set()
        chosen_dst: set[int] = set()
        link_map: dict[int, int] = {}
        variant = normalize_rule_variant(rule_variant)

        def extra_terms(cand: CandidatePair) -> tuple[float, float, int]:
            shape_term = -cand.shape_score if uses_shape_factor_variant(variant) else 0.0
            length_term = -cand.length_confidence if uses_length_conf_variant(variant) else 0.0
            size_term = -(cand.src_track_len + cand.dst_track_len) if variant != 'baseline' else 0
            return shape_term, length_term, size_term

        if use_neural:
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
    def _hungarian_match(candidates: list[CandidatePair], track_map: dict[int, TrackFragment]) -> dict[int, int]:
        if not candidates:
            return {}
        track_ids = sorted(track_map.keys())
        index = {tid: idx for idx, tid in enumerate(track_ids)}
        cost = np.full((len(track_ids), len(track_ids)), 1e9, dtype=np.float32)

        for cand in candidates:
            cost[index[cand.src_id], index[cand.dst_id]] = 1.0 - float(cand.neural_score)

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


def center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax = 0.5 * (box_a[0] + box_a[2])
    ay = 0.5 * (box_a[1] + box_a[3])
    bx = 0.5 * (box_b[0] + box_b[2])
    by = 0.5 * (box_b[1] + box_b[3])
    return float(math.hypot(ax - bx, ay - by))



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
    return normalize_rule_variant(rule_variant) in {'adaptive_gate', 'adaptive_tri'}



def uses_shape_factor_variant(rule_variant: str) -> bool:
    return normalize_rule_variant(rule_variant) in {'tri_factor', 'adaptive_tri'}



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
    gate_reliability_weight: float = 0.7,
    gate_shape_weight: float = 0.3,
) -> tuple[int, float]:
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
    gap_ratio = float(
        np.clip(
            adaptive_gap_min_ratio + (1.0 - adaptive_gap_min_ratio) * gate_signal,
            adaptive_gap_min_ratio,
            1.0,
        )
    )
    dist_ratio = float(
        np.clip(
            adaptive_dist_min_ratio + (1.0 - adaptive_dist_min_ratio) * gate_signal,
            adaptive_dist_min_ratio,
            1.0,
        )
    )
    adaptive_gap_limit = max(1, min(int(round(max_gap * gap_ratio)), max(int(max_gap), 1)))
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
    parser.add_argument('--mode', type=str, default='hybrid', choices=['rule', 'hybrid', 'hybrid_greedy', 'neural'])
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
        help='规则变体：baseline 原始方案，adaptive_gate 为方案1，tri_factor 为方案2，adaptive_tri 为方案1+2，gap_decay 为方案4，length_conf 为方案5，gap_length 为方案4+5。',
    )

    parser.add_argument('--gap_decay_alpha', type=float, default=3.0, help='方案4中 gap 非线性衰减强度，越大表示长中断降分越快。')
    parser.add_argument('--length_ref', type=int, default=30, help='方案5中长度置信度的参考长度，达到该长度后长度项饱和。')
    parser.add_argument('--length_conf_power', type=float, default=1.0, help='方案5中长度置信度幂次，>1 更保守，<1 更宽松。')
    parser.add_argument('--adaptive_len_ref', type=int, default=30, help='方案1中自适应门控的参考轨迹长度，越短的碎片门控越严格。')
    parser.add_argument('--adaptive_gap_min_ratio', type=float, default=0.5, help='方案1中 temporal gate 的最小保留比例，越小表示短轨更严格。')
    parser.add_argument('--adaptive_dist_min_ratio', type=float, default=0.5, help='方案1中 spatial gate 的最小保留比例，越小表示短轨更严格。')
    parser.add_argument('--gate_reliability_weight', type=float, default=0.7, help='门控信号中轨迹可靠性权重，最终会自动归一化。')
    parser.add_argument('--gate_shape_weight', type=float, default=0.3, help='门控信号中形态项权重，最终会自动归一化。')
    parser.add_argument('--area_weight', type=float, default=0.6, help='shape_score 中面积比例权重，最终会自动归一化。')
    parser.add_argument('--aspect_weight', type=float, default=0.4, help='shape_score 中长宽比相似度权重，最终会自动归一化。')
    parser.add_argument('--temporal_weight', type=float, default=0.4, help='方案2中时间因子权重，最终会自动归一化。')
    parser.add_argument('--distance_weight', type=float, default=0.4, help='方案2中距离因子权重，最终会自动归一化。')
    parser.add_argument('--shape_weight', type=float, default=0.2, help='方案2中形态连续性因子权重，最终会自动归一化。')

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
    return parser.parse_args()




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
        link_map, stats = linker.link_sequence(track_map)
        merged_tracks, lineage = merge_tracks(track_map, link_map)
        stats = dict(stats)
        stats['merged_tracks'] = sum(1 for ids in lineage.values() if len(ids) > 1)
        stats['track_count_before'] = len(track_map)
        stats['track_count_after'] = len(merged_tracks)
        processed[seq_name] = merged_tracks
        debug_stats[seq_name] = stats

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
            gate_reliability_weight=float(config['gate_reliability_weight']),
            gate_shape_weight=float(config['gate_shape_weight']),
            area_weight=float(config['shape_area_weight']),
            aspect_weight=float(config['shape_aspect_weight']),
            temporal_weight=float(config['temporal_weight']),
            distance_weight=float(config['distance_weight']),
            shape_weight=float(config['shape_weight']),
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
