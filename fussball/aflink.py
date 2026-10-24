"""
AFLink 轨迹碎片合并模块 —— 神经网络版 V3
=========================================
网络结构：
  - TemporalModule: 4层 Conv1d (9→32→64→128→256) + BN + ReLU，对每段轨迹独立编码
  - FusionModule:   将两段轨迹特征做 1×2 卷积融合，输出 256 维
  - Classifier:     全连接 768→64→2，输出二分类 logits（是否属于同一目标）

输入特征（9维）：cx, cy, w, h, frame, vx, vy, ax, ay（位置/速度/加速度 + 帧号）
权重文件：aflink_model_v3.pth（训练集精度 92.2%）
配置文件：aflink_config_v3.json

用法：
    from fussball.aflink import AFLink
    linker = AFLink(model_path='path/to/aflink_model_v3.pth')
    merge_map = linker.link(tracks_data)
    # tracks_data: list of {'track_id': int, 'frames': [...], 'boxes': [[x1,y1,x2,y2],...]}
    # 返回 {old_id: new_id, ...}，未合并的轨迹不在字典中
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

# ─── 默认权重路径 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH  = _HERE / 'aflink_model_v3.pth'
DEFAULT_CONFIG_PATH = _HERE / 'aflink_config_v3.json'

FEAT_DIM = 9  # cx,cy,w,h,frame,vx,vy,ax,ay


# ══════════════════════════════════════════════════════════════════════════════
#  网络定义（与 train_aflink_net_v3.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
class TemporalModule(nn.Module):
    def __init__(self, in_ch=FEAT_DIM):
        super().__init__()
        channels = [32, 64, 128, 256]
        layers = []; ch = in_ch
        for out_ch in channels:
            layers += [
                nn.Conv1d(ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   # (B, 256, SEQ_LEN)


class FusionModule(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, a_feat, b_feat):
        x = torch.stack([a_feat, b_feat], dim=2)   # (B, 256, 2)
        x = self.relu(self.bn(self.conv(x)))
        return x.mean(dim=2)                         # (B, 256)


class AFLinkNetV3(nn.Module):
    """V3：fusion 输出正确拼入 classifier（256+256+256=768 维）"""
    def __init__(self):
        super().__init__()
        self.temporal_a = TemporalModule(in_ch=FEAT_DIM)
        self.temporal_b = TemporalModule(in_ch=FEAT_DIM)
        self.fusion     = FusionModule(ch=256)
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, seq_a, seq_b):
        fa = self.temporal_a(seq_a).mean(dim=2)   # (B, 256)
        fb = self.temporal_b(seq_b).mean(dim=2)   # (B, 256)
        ff = self.fusion(fa, fb)                   # (B, 256)
        feat = torch.cat([fa, fb, ff], dim=1)      # (B, 768)
        return self.classifier(feat)               # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  轨迹特征提取
# ══════════════════════════════════════════════════════════════════════════════
def _extract_features(frames, boxes, seq_len):
    """
    将一段轨迹提取为 (FEAT_DIM, seq_len) 的张量。
    特征：cx, cy, w, h, frame(归一化), vx, vy, ax, ay
    """
    n = len(frames)
    frames = np.array(frames, dtype=np.float32)
    boxes  = np.array(boxes,  dtype=np.float32)   # (n, 4) x1y1x2y2

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w  =  boxes[:, 2] - boxes[:, 0]
    h  =  boxes[:, 3] - boxes[:, 1]

    # 帧号归一化到 [0, 1]（相对序列自身）
    t = (frames - frames[0]) / max(frames[-1] - frames[0], 1.0)

    # 速度（前向差分）
    vx = np.gradient(cx)
    vy = np.gradient(cy)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    feat = np.stack([cx, cy, w, h, t, vx, vy, ax, ay], axis=1)  # (n, 9)

    # 归一化每个维度到 [-1, 1]
    for d in range(feat.shape[1]):
        rng = feat[:, d].max() - feat[:, d].min()
        if rng > 1e-6:
            feat[:, d] = (feat[:, d] - feat[:, d].min()) / rng * 2 - 1

    # 截取/填充到 seq_len
    if n >= seq_len:
        # 取末尾 seq_len 帧
        feat = feat[-seq_len:]
    else:
        pad = np.zeros((seq_len - n, FEAT_DIM), dtype=np.float32)
        feat = np.concatenate([pad, feat], axis=0)

    return feat.T.astype(np.float32)   # (9, seq_len)


# ══════════════════════════════════════════════════════════════════════════════
#  AFLink 主类
# ══════════════════════════════════════════════════════════════════════════════
class AFLink:
    """
    神经网络版 AFLink V3。
    兼容原接口：AFLink().link(tracks_data) → merge_map
    """

    def __init__(self,
                 model_path=None,
                 config_path=None,
                 device=None):
        # 加载配置
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = {}

        self.seq_len      = cfg.get('seq_len',      30)
        self.max_gap      = cfg.get('max_gap',       60)
        self.dist_thresh  = cfg.get('dist_thresh',  200)
        self.score_thresh = cfg.get('score_thresh', 0.80)

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        mdl_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.net = None
        if mdl_path.exists():
            self.net = AFLinkNetV3().to(self.device)
            state = torch.load(str(mdl_path), map_location=self.device, weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            self.net.load_state_dict(state)
            self.net.eval()
            print(f"[AFLink] 神经网络 V3 权重加载完成: {mdl_path}")
        else:
            print(f"[AFLink] 警告：未找到权重 {mdl_path}，回退到规则版本")

    # ------------------------------------------------------------------
    def link(self, tracks_data):
        """
        合并断裂的轨迹片段。
        tracks_data: list of dicts
            - track_id: int
            - frames:   list[int]
            - boxes:    list[[x1, y1, x2, y2]]
        返回: {old_track_id: new_track_id, ...}
        """
        if len(tracks_data) < 2:
            return {}

        if self.net is not None:
            return self._link_neural(tracks_data)
        else:
            return self._link_rule(tracks_data)

    # ------------------------------------------------------------------
    def _link_neural(self, tracks_data):
        n = len(tracks_data)

        # 预提取所有轨迹特征
        feats = []
        for td in tracks_data:
            feats.append(_extract_features(td['frames'], td['boxes'], self.seq_len))
        feats = np.stack(feats, axis=0)   # (n, 9, seq_len)

        cost_matrix = np.ones((n, n)) * 1e9

        # 候选对筛选（时间/空间预过滤，减少推理量）
        cand_pairs = []
        for i, ta in enumerate(tracks_data):
            for j, tb in enumerate(tracks_data):
                if i == j:
                    continue
                gap = tb['frames'][0] - ta['frames'][-1]
                if gap <= 0 or gap > self.max_gap:
                    continue
                # 空间距离预过滤
                ba = ta['boxes'][-1]
                bb = tb['boxes'][0]
                cx_a = (ba[0] + ba[2]) / 2; cy_a = (ba[1] + ba[3]) / 2
                cx_b = (bb[0] + bb[2]) / 2; cy_b = (bb[1] + bb[3]) / 2
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                if dist <= self.dist_thresh:
                    cand_pairs.append((i, j))

        if not cand_pairs:
            return {}

        # 批量推理
        BATCH = 512
        scores = {}
        with torch.no_grad():
            for start in range(0, len(cand_pairs), BATCH):
                batch = cand_pairs[start:start + BATCH]
                fa = torch.tensor(feats[[p[0] for p in batch]], device=self.device)
                fb = torch.tensor(feats[[p[1] for p in batch]], device=self.device)
                logits = self.net(fa, fb)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for (i, j), p in zip(batch, probs):
                    scores[(i, j)] = float(p)

        # 填代价矩阵
        for (i, j), p in scores.items():
            if p >= self.score_thresh:
                cost_matrix[i, j] = 1.0 - p

        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        merge_map = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1.0 - self.score_thresh:
                merge_map[tracks_data[r]['track_id']] = tracks_data[c]['track_id']

        return merge_map

    # ------------------------------------------------------------------
    def _link_rule(self, tracks_data):
        """规则版回退（无权重时使用）"""
        n = len(tracks_data)
        cost_matrix = np.ones((n, n)) * 1e9

        for i, ta in enumerate(tracks_data):
            for j, tb in enumerate(tracks_data):
                if i == j:
                    continue
                gap = tb['frames'][0] - ta['frames'][-1]
                if gap <= 0 or gap > self.max_gap:
                    continue
                ba = ta['boxes'][-1]; bb = tb['boxes'][0]
                cx_a = (ba[0] + ba[2]) / 2; cy_a = (ba[1] + ba[3]) / 2
                cx_b = (bb[0] + bb[2]) / 2; cy_b = (bb[1] + bb[3]) / 2
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                if dist > self.dist_thresh:
                    continue
                boxes_a = ta['boxes']
                if len(boxes_a) >= 2:
                    v_list = []
                    for k in range(min(5, len(boxes_a) - 1)):
                        b1 = boxes_a[-2 - k]; b2 = boxes_a[-1 - k]
                        v_list.append([b2[0] - b1[0], b2[1] - b1[1]])
                    v_a = np.mean(v_list, axis=0)
                else:
                    v_a = np.array([0, 0])
                pred_x = ba[0] + v_a[0] * gap
                pred_y = ba[1] + v_a[1] * gap
                d2 = np.sqrt((pred_x - bb[0]) ** 2 + (pred_y - bb[1]) ** 2)
                sim = np.exp(-d2 / 100)
                if sim >= 0.7:
                    cost_matrix[i, j] = 1.0 - sim

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        merge_map = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.3:
                merge_map[tracks_data[r]['track_id']] = tracks_data[c]['track_id']
        return merge_map
