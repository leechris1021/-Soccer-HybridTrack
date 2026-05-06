"""
Track 跟踪器状态管理
状态: TENTATIVE(新生) / CONFIRMED(确认) / OCCLUDED(遮挡) / DELETED(删除)

改进（v3）：
  1. EMA 特征更新：feat_ema = α*feat_old + (1-α)*feat_new，最新帧权重更高
  2. 遮挡快照（occ_snapshot）：进入遮挡前保存最后一帧干净特征，
     遮挡期间不更新 EMA，重出现时优先用快照初始化
  3. 外观可信度 app_trust：随遮挡帧数线性衰减，供代价策略动态降权
"""
import numpy as np
from collections import deque
from enum import Enum

# EMA 衰减系数：0.9 → 最新帧权重 10%，历史保留 90%
#   调小（如 0.7）→ 追新更快但更容易被噪声污染
EMA_ALPHA = 0.9


class TrackState(Enum):
    TENTATIVE = 1   # 新生，尚未稳定
    CONFIRMED = 2   # 正常跟踪
    OCCLUDED = 3    # 遮挡中
    DELETED = 4     # 已删除


class Track:
    """
    单个目标的跟踪器。
    """
    _id_counter = 0

    def __init__(self, mean, covariance, detection_class, feature,
                 n_init=3, max_age=30, max_feature_queue=5, max_trajectory=30,
                 ema_alpha=EMA_ALPHA):
        Track._id_counter += 1
        self.track_id = Track._id_counter

        self.mean = mean
        self.covariance = covariance
        self.det_class = detection_class

        # 状态
        self.state = TrackState.TENTATIVE
        self.hits = 1           # 连续匹配次数
        self.age = 1            # 总帧数
        self.time_since_update = 0  # 距上次匹配的帧数
        self.occ_frames = 0     # 连续遮挡帧数

        # 帧号记录
        self.frame_first = 0
        self.frame_last = 0

        # ── 外观特征（保留 deque 供向后兼容，同时新增 EMA）──────────────────
        self.features = deque(maxlen=max_feature_queue)
        self.ema_alpha = ema_alpha

        # EMA 特征向量（归一化，None 表示尚未初始化）
        self._ema_feature = None
        # 遮挡前快照（进入 OCCLUDED 时保存，用于重出现时初始化 EMA）
        self._occ_snapshot = None

        if feature is not None:
            self.features.append(feature)
            self._ema_feature = feature.copy()

        # 历史轨迹队列 (最近 L 帧 bbox [cx,cy,w,h])
        self.trajectory = deque(maxlen=max_trajectory)
        self.trajectory.append(mean[:4].copy())

        # 运动信息
        self.velocity = np.zeros(4)  # [vx, vy, vw, vh]

        # 配置
        self.n_init = n_init
        self.max_age = max_age  # 最大遮挡帧数

    # ------------------------------------------------------------------ #
    #  属性
    # ------------------------------------------------------------------ #
    @property
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self):
        return self.state == TrackState.TENTATIVE

    @property
    def is_occluded(self):
        return self.state == TrackState.OCCLUDED

    @property
    def is_deleted(self):
        return self.state == TrackState.DELETED

    def to_tlwh(self):
        """返回 [x1, y1, w, h]（左上角）"""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def to_tlbr(self):
        """返回 [x1, y1, x2, y2]"""
        tlwh = self.to_tlwh()
        return np.array([tlwh[0], tlwh[1],
                         tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])

    # ------------------------------------------------------------------ #
    #  外观特征接口
    # ------------------------------------------------------------------ #
    def _update_ema(self, feature, alpha=None):
        """
        用新特征更新 EMA。遮挡期间不调用（防污染）。

        alpha : 本帧使用的衰减系数。
                None → 使用 self.ema_alpha（固定值，向后兼容）。
                传入动态值时（如由置信度/年龄计算得到），覆盖默认值。
                值域 [0, 1]：越大表示越保留历史，越小表示越追新。
        """
        if feature is None:
            return
        a = self.ema_alpha if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        feat_norm = feature / (np.linalg.norm(feature) + 1e-9)
        if self._ema_feature is None:
            self._ema_feature = feat_norm.copy()
        else:
            self._ema_feature = (a * self._ema_feature
                                 + (1.0 - a) * feat_norm)
            norm = np.linalg.norm(self._ema_feature)
            self._ema_feature /= (norm + 1e-9)

    def get_mean_feature(self):
        """
        返回当前最优外观特征（优先 EMA，向后兼容 deque 均值）。
        外部代码（AFLink、已注册策略）通过此接口访问，无需改动。
        """
        if self._ema_feature is not None:
            return self._ema_feature.copy()
        # fallback：deque 均值（原逻辑）
        if len(self.features) == 0:
            return None
        feats = np.stack(self.features)
        mean_f = feats.mean(axis=0)
        norm = np.linalg.norm(mean_f)
        return mean_f / (norm + 1e-9)

    def get_occ_aware_feature(self):
        """
        遮挡感知特征：
          - 正常跟踪 → 返回 EMA（最新+历史加权）
          - 遮挡期间 → 返回遮挡前快照（不受遮挡期噪声污染）
        """
        if self.state == TrackState.OCCLUDED and self._occ_snapshot is not None:
            return self._occ_snapshot.copy()
        return self.get_mean_feature()

    def get_app_trust(self, max_occ=10):
        """
        外观可信度 ∈ [0, 1]：
          - 非遮挡 → 1.0
          - 遮挡 k 帧 → max(0, 1 - k/max_occ)
        供代价策略动态降权外观项。
        """
        if self.state != TrackState.OCCLUDED:
            return 1.0
        return float(max(0.0, 1.0 - self.occ_frames / max(max_occ, 1)))

    # ------------------------------------------------------------------ #
    #  状态更新
    # ------------------------------------------------------------------ #
    def predict(self, kf):
        """卡尔曼预测"""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        # 更新速度估计
        self.velocity = self.mean[4:8].copy()

    @staticmethod
    def _calc_dynamic_alpha(det_conf, hits,
                            alpha_base=0.9,
                            alpha_min=0.5,
                            alpha_max=0.95):
        """
        根据检测置信度和轨迹年龄动态计算 EMA 衰减系数。

        设计逻辑
        --------
        · det_conf 低（特征噪声大）→ alpha 大（多保留历史）
          det_conf 高（特征可信）  → alpha 小（更快追新）
          贡献：delta_conf = alpha_base * (1 - det_conf)  ∈ [0, alpha_base]

        · hits 小（新生轨迹，历史少）→ alpha 小（快速建立外观）
          hits 大（成熟轨迹）        → 不惩罚（已有稳定 EMA）
          贡献：hits_factor = min(hits / 10, 1.0)，对 alpha 线性加权

        最终：alpha = clip(alpha_base * hits_factor
                           + alpha_base * (1 - det_conf) * (1 - hits_factor),
                           alpha_min, alpha_max)

        直觉举例：
          hits=1,  conf=0.9 → alpha ≈ 0.50（新生高置信，快速追新）
          hits=1,  conf=0.4 → alpha ≈ 0.60（新生低置信，略保守）
          hits=20, conf=0.9 → alpha ≈ 0.90（成熟高置信，正常）
          hits=20, conf=0.4 → alpha ≈ 0.95（成熟低置信，非常保守）
        """
        conf = float(np.clip(det_conf, 0.0, 1.0))
        hits_factor = float(min(hits / 10.0, 1.0))
        alpha = (alpha_base * hits_factor
                 + alpha_base * (1.0 - conf) * (1.0 - hits_factor))
        return float(np.clip(alpha, alpha_min, alpha_max))

    def update(self, kf, detection, frame_id, use_dynamic_ema=False):
        """
        匹配到检测框后更新状态。

        use_dynamic_ema : True → 根据 det_conf + hits 动态计算 EMA alpha；
                          False → 使用固定 self.ema_alpha（默认，向后兼容）。
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        if detection.feature is not None:
            # 保留 deque（供 AFLink 等后处理）
            self.features.append(detection.feature)

            # 计算本帧 alpha
            if use_dynamic_ema:
                det_conf = getattr(detection, 'confidence', 1.0) or 1.0
                dyn_alpha = self._calc_dynamic_alpha(det_conf, self.hits)
            else:
                dyn_alpha = None  # → _update_ema 内部用固定值

            # EMA 更新（只在非遮挡期更新，防止遮挡噪声污染）
            # 注意：此处 state 尚未切换，OCCLUDED→CONFIRMED 在下方才改，
            # 因此重出现首帧也用快照恢复而不是直接 EMA 更新。
            if self.state != TrackState.OCCLUDED:
                self._update_ema(detection.feature, alpha=dyn_alpha)
            else:
                # 遮挡后重出现：用快照重初始化 EMA，再用新帧微调一次
                if self._occ_snapshot is not None:
                    self._ema_feature = self._occ_snapshot.copy()
                self._update_ema(detection.feature, alpha=dyn_alpha)
                self._occ_snapshot = None   # 快照已消费，清空

        self.hits += 1
        self.time_since_update = 0
        self.occ_frames = 0
        self.frame_last = frame_id
        self.trajectory.append(self.mean[:4].copy())

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def mark_occluded(self):
        """标记为遮挡"""
        # 第一次进入遮挡时保存外观快照
        if self.state != TrackState.OCCLUDED and self._ema_feature is not None:
            self._occ_snapshot = self._ema_feature.copy()

        self.occ_frames += 1
        if self.occ_frames > self.max_age:
            self.state = TrackState.DELETED
        else:
            self.state = TrackState.OCCLUDED
            # 继续预测轨迹
            self.trajectory.append(self.mean[:4].copy())

    def mark_missed(self):
        """标记未匹配（新生阶段直接删除）"""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        else:
            self.mark_occluded()
