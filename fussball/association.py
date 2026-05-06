"""
自适应关联代价矩阵与匈牙利匹配
核心创新：
  1. 根据目标遮挡帧数动态调整外观/运动/时空三类特征权重
  2. 空间距离门控惩罚：超过最大合理位移的匹配对代价强制拉高，
     抑制同色球衣造成的跨区域误匹配（方案C）
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
#  IOU 工具
# ------------------------------------------------------------------ #
def iou_batch(atlbr, btlbr):
    """
    计算两组边界框之间的 IOU 矩阵。
    atlbr: (M, 4) [x1,y1,x2,y2]
    btlbr: (N, 4) [x1,y1,x2,y2]
    返回: (M, N)
    """
    atlbr = np.expand_dims(atlbr, 1)   # (M, 1, 4)
    btlbr = np.expand_dims(btlbr, 0)   # (1, N, 4)

    xx1 = np.maximum(atlbr[:, :, 0], btlbr[:, :, 0])
    yy1 = np.maximum(atlbr[:, :, 1], btlbr[:, :, 1])
    xx2 = np.minimum(atlbr[:, :, 2], btlbr[:, :, 2])
    yy2 = np.minimum(atlbr[:, :, 3], btlbr[:, :, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h

    area_a = (atlbr[:, :, 2] - atlbr[:, :, 0]) * (atlbr[:, :, 3] - atlbr[:, :, 1])
    area_b = (btlbr[:, :, 2] - btlbr[:, :, 0]) * (btlbr[:, :, 3] - btlbr[:, :, 1])

    iou = inter / (area_a + area_b - inter + 1e-9)
    return iou




# ------------------------------------------------------------------ #
#  空间距离门控
# ------------------------------------------------------------------ #
def spatial_gate_penalty(track_tlbr, det_tlbr_arr, max_dist=200.0, penalty=1.0):
    """
    计算轨迹中心与所有检测框中心的欧式距离，超过 max_dist 的匹配对
    附加 penalty 惩罚（直接加到代价矩阵上，使其超过匹配阈值）。

    track_tlbr : (4,)  [x1,y1,x2,y2]
    det_tlbr_arr: (N,4)
    返回: (N,) 惩罚向量，0 或 penalty
    """
    tcx = (track_tlbr[0] + track_tlbr[2]) / 2.0
    tcy = (track_tlbr[1] + track_tlbr[3]) / 2.0
    dcx = (det_tlbr_arr[:, 0] + det_tlbr_arr[:, 2]) / 2.0
    dcy = (det_tlbr_arr[:, 1] + det_tlbr_arr[:, 3]) / 2.0
    dist = np.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2)
    return np.where(dist > max_dist, penalty, 0.0).astype(np.float32)



def cosine_distance(a, b):
    """
    a: (M, D), b: (N, D), 均已 L2 归一化
    返回余弦距离矩阵 (M, N)，值域 [0, 2]
    """
    sim = a @ b.T   # (M, N)
    return 1.0 - sim


# ------------------------------------------------------------------ #
#  自适应权重计算
# ------------------------------------------------------------------ #
def adaptive_weights(occ_frames,
                     w_app0=0.5, w_mot0=0.3, w_st0=0.2,
                     gamma=0.95, alpha=0.05):
    """
    根据遮挡帧数动态计算三类特征权重，满足归一化约束。

    occ_frames: 遮挡持续帧数 (0 表示正常跟踪)
    返回: (w_app, w_mot, w_st)
    """
    if occ_frames == 0:
        return w_app0, w_mot0, w_st0

    t = occ_frames
    w_app = w_app0 * (gamma ** t)
    w_st = w_st0 * (1.0 + alpha * t)
    w_mot = w_mot0  # 运动权重保持不变

    total = w_app + w_mot + w_st
    return w_app / total, w_mot / total, w_st / total


# ------------------------------------------------------------------ #
#  自适应代价矩阵
# ------------------------------------------------------------------ #
def adaptive_cost_matrix(tracks, detections, kf,
                         app_threshold=0.7,
                         mot_threshold=9.4877,   # chi2(4, 0.95)
                         iou_threshold=0.3,
                         spatial_max_dist=200.0): # 方案C：空间门控距离阈值（像素）
    """
    构建自适应关联代价矩阵。
    返回: cost_matrix (M, N), M=tracks, N=detections
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)))

    # 检测框位置
    det_tlbr = np.array([d.bbox for d in detections])          # (N, 4)
    det_xyah = np.array([d.to_xyah() for d in detections])     # (N, 4)

    # 检测外观特征（动态推断维度，避免 PCB 128 维与硬编码 2048 不一致）
    _valid_feats = [d.feature for d in detections if d.feature is not None]
    _feat_dim = _valid_feats[0].shape[0] if _valid_feats else 128
    det_features = np.array([
        d.feature if d.feature is not None else np.zeros(_feat_dim)
        for d in detections
    ])  # (N, feat_dim)

    M, N = len(tracks), len(detections)
    cost_matrix = np.zeros((M, N), dtype=np.float32)

    for i, track in enumerate(tracks):
        occ = track.occ_frames
        w_app, w_mot, w_st = adaptive_weights(occ)

        # --- 1. 外观代价 (余弦距离) ---
        trk_feat = track.get_mean_feature()
        if (trk_feat is not None and det_features.shape[0] > 0
                and trk_feat.shape[0] == det_features.shape[1]):
            c_app = cosine_distance(trk_feat[None], det_features)[0]  # (N,)
            c_app = np.clip(c_app, 0, 1)
        else:
            c_app = np.ones(N)

        # --- 2. 运动代价 (归一化马氏距离) ---
        maha = kf.gating_distance(track.mean, track.covariance, det_xyah)  # (N,)
        c_mot = np.clip(maha / mot_threshold, 0, 1)

        # --- 3. 时空代价 (1 - IoU) ---
        trk_tlbr = track.to_tlbr()[None]            # (1, 4)
        iou = iou_batch(trk_tlbr, det_tlbr)[0]      # (N,)
        c_st = 1.0 - iou

        # --- 自适应融合 ---
        cost = w_app * c_app + w_mot * c_mot + w_st * c_st

        # --- 方案C：空间距离门控惩罚 ---
        # 超过 spatial_max_dist 像素的匹配对，代价强制 +1（超过任何匹配阈值）
        sp = spatial_gate_penalty(track.to_tlbr(), det_tlbr,
                                  max_dist=spatial_max_dist, penalty=1.0)
        cost = cost + sp

        cost_matrix[i] = cost

    return cost_matrix


# ------------------------------------------------------------------ #
#  匈牙利匹配
# ------------------------------------------------------------------ #
def hungarian_match(cost_matrix, threshold=1.0):
    """
    使用匈牙利算法进行最优匹配。
    返回:
        matches: list of (track_idx, det_idx)
        unmatched_tracks: list of track_idx
        unmatched_dets: list of det_idx
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    matched_set = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            unmatched_tracks.append(r)
            unmatched_dets.append(c)
        else:
            matches.append((r, c))
            matched_set.add((r, c))

    for r in range(cost_matrix.shape[0]):
        if r not in [m[0] for m in matches]:
            unmatched_tracks.append(r)

    for c in range(cost_matrix.shape[1]):
        if c not in [m[1] for m in matches]:
            unmatched_dets.append(c)

    return matches, list(set(unmatched_tracks)), list(set(unmatched_dets))
