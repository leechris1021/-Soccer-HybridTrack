"""
AdaptiveDeepSORT 跟踪器主逻辑
整合 YOLOv8 检测 + 特征提取 + 自适应关联代价矩阵
支持 ECC 运动补偿和 AFLink 轨迹合并
"""
import numpy as np
import cv2

from .kalman_filter import ExtendedKalmanFilter
from .track import Track, TrackState
from .detection import Detection
from .association import adaptive_cost_matrix, hungarian_match, iou_batch
from .appearance import AppearanceExtractor


class AdaptiveDeepSORT:
    """
    基于 YOLOv8 + 自适应关联代价矩阵的多目标跟踪器。
    针对 DFL Bundesliga 数据集（1920x1080 @ 25fps）进行了适配：
      - max_age=25  对应1秒@25fps
      - imgsz=1280  保留高分辨率，改善球场远景小目标检测
      - max_trajectory=25  轨迹队列对应1秒
    """

    def __init__(self,
                 yolo_model_path='yolov8n.pt',
                 conf_threshold=0.3,
                 iou_nms=0.5,
                 imgsz=1280,            # DFL 1080p 视频推理分辨率
                 target_classes=None,   # None 表示跟踪所有类别
                 match_threshold=0.7,
                 max_age=25,            # DFL @25fps，1秒=25帧
                 n_init=3,
                 max_feature_queue=5,
                 max_trajectory=25,     # DFL @25fps，显示1秒轨迹
                 occ_iou_threshold=0.1,
                 reid_threshold=0.6,
                 appearance_model='osnet',  # 'osnet' 或 'resnet50'
                 appearance_weights=None,
                 device=None,
                 # ECC 和 AFLink 相关参数（由子类使用）
                 use_ecc=False,
                 use_aflink=False):

        # YOLOv8 检测器
        from ultralytics import YOLO
        self.detector = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.iou_nms = iou_nms
        self.imgsz = imgsz
        self.target_classes = target_classes  # e.g. [0] for 'person'

        # 外观特征提取器（支持 OSNet 和 ResNet50）
        self.appearance = AppearanceExtractor(
            model_name=appearance_model,
            device=device,
            weights_path=appearance_weights
        )

        # 卡尔曼滤波器
        self.kf = ExtendedKalmanFilter()

        # 跟踪参数
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.n_init = n_init
        self.max_feature_queue = max_feature_queue
        self.max_trajectory = max_trajectory
        self.occ_iou_threshold = occ_iou_threshold
        self.reid_threshold = reid_threshold

        # ECC 和 AFLink（默认关闭，由子类启用）
        self.use_ecc = use_ecc
        self.use_aflink = use_aflink

        # 跟踪器列表
        self.tracks = []
        self.frame_id = 0

        # 统计
        self.total_ids = 0

    def reset(self):
        """重置跟踪器（新视频开始时调用）"""
        self.tracks = []
        self.frame_id = 0
        Track._id_counter = 0

    # ------------------------------------------------------------------ #
    #  主接口：处理单帧
    # ------------------------------------------------------------------ #
    def update(self, frame_bgr):
        """
        处理一帧图像，返回当前帧的跟踪结果。
        frame_bgr: BGR 图像 (H, W, 3)
        返回: list of dict {
            'id': int,
            'bbox': [x1, y1, x2, y2],
            'class': str,
            'state': str,          # 'confirmed' / 'tentative' / 'occluded'
            'velocity': [vx, vy],
            'trajectory': list of [cx, cy]
        }
        """
        self.frame_id += 1

        # --- Step 1: 目标检测 ---
        detections = self._detect(frame_bgr)

        # --- Step 2: 卡尔曼预测 ---
        for track in self.tracks:
            track.predict(self.kf)

        # --- Step 3: 两阶段关联 ---
        # 第一阶段：对已确认和遮挡轨迹进行关联
        confirmed_tracks = [t for t in self.tracks
                            if t.state in (TrackState.CONFIRMED, TrackState.OCCLUDED)]
        tentative_tracks = [t for t in self.tracks
                            if t.state == TrackState.TENTATIVE]

        matches1, unm_trk1, unm_det1 = self._associate(confirmed_tracks, detections)

        # 第二阶段：对新生轨迹用 IOU 进行关联
        remaining_dets = [detections[i] for i in unm_det1]
        matches2, unm_trk2, unm_det2 = self._associate_iou(tentative_tracks, remaining_dets)

        # --- Step 4: 更新已匹配轨迹 ---
        for trk_idx, det_idx in matches1:
            confirmed_tracks[trk_idx].update(self.kf, detections[det_idx], self.frame_id)

        for trk_idx, det_idx in matches2:
            tentative_tracks[trk_idx].update(self.kf, remaining_dets[det_idx], self.frame_id)

        # --- Step 5: 处理未匹配轨迹 ---
        for trk_idx in unm_trk1:
            confirmed_tracks[trk_idx].mark_missed()

        for trk_idx in unm_trk2:
            tentative_tracks[trk_idx].mark_missed()

        # --- Step 6: 初始化新轨迹 ---
        # remaining_dets 中还未匹配的
        final_unm_dets = [remaining_dets[i] for i in unm_det2]
        for det in final_unm_dets:
            self._init_track(det)

        # --- Step 7: 清除已删除轨迹 ---
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # --- Step 8: 构建输出 ---
        results = []
        for track in self.tracks:
            if track.is_deleted:
                continue
            state_str = {
                TrackState.CONFIRMED: 'confirmed',
                TrackState.TENTATIVE: 'tentative',
                TrackState.OCCLUDED: 'occluded',
            }.get(track.state, 'unknown')

            tlbr = track.to_tlbr()
            vel = track.velocity[:2].tolist()
            traj = [[p[0], p[1]] for p in track.trajectory]

            results.append({
                'id': track.track_id,
                'bbox': tlbr.tolist(),
                'class': str(track.det_class),
                'state': state_str,
                'velocity': vel,
                'trajectory': traj,
                'occ_frames': track.occ_frames,
            })

        return results

    # ------------------------------------------------------------------ #
    #  内部方法
    # ------------------------------------------------------------------ #
    def _detect(self, frame_bgr):
        """运行 YOLOv8 检测并提取外观特征"""
        results = self.detector(frame_bgr,
                                conf=self.conf_threshold,
                                iou=self.iou_nms,
                                imgsz=self.imgsz,
                                verbose=False)[0]

        bboxes, classes, confs = [], [], []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if self.target_classes is not None and cls_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bboxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
                confs.append(float(box.conf[0]))

        if len(bboxes) == 0:
            return []

        # 批量提取外观特征
        features = self.appearance.extract(frame_bgr, bboxes)

        detections = []
        for i, (bbox, cls, conf) in enumerate(zip(bboxes, classes, confs)):
            feat = features[i] if features.shape[0] > i else None
            detections.append(Detection(bbox, conf, cls, feat))

        return detections

    def _associate(self, tracks, detections):
        """使用自适应代价矩阵关联"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = adaptive_cost_matrix(tracks, detections, self.kf)
        return hungarian_match(cost, threshold=self.match_threshold)

    def _associate_iou(self, tracks, detections):
        """纯 IOU 关联（用于新生轨迹）"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        trk_tlbr = np.array([t.to_tlbr() for t in tracks])
        det_tlbr = np.array([d.bbox for d in detections])
        iou_mat = iou_batch(trk_tlbr, det_tlbr)
        cost = 1.0 - iou_mat
        return hungarian_match(cost, threshold=0.7)

    def _init_track(self, detection):
        """初始化新轨迹"""
        mean, cov = self.kf.initiate(detection.to_xyah())
        track = Track(
            mean=mean,
            covariance=cov,
            detection_class=detection.det_class,
            feature=detection.feature,
            n_init=self.n_init,
            max_age=self.max_age,
            max_feature_queue=self.max_feature_queue,
            max_trajectory=self.max_trajectory,
        )
        track.frame_first = self.frame_id
        track.frame_last = self.frame_id
        # 确保 track_id 唯一
        if track.track_id == 0:
            self.total_ids += 1
            track.track_id = self.total_ids
        self.tracks.append(track)

    # ------------------------------------------------------------------ #
    #  可视化
    # ------------------------------------------------------------------ #
    def visualize(self, frame_bgr, results):
        """
        在帧上绘制跟踪结果。
        返回: 带标注的 BGR 图像
        """
        vis = frame_bgr.copy()

        for obj in results:
            tid = obj['id']
            x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
            state = obj['state']
            vel = obj['velocity']
            traj = obj['trajectory']

            # 颜色：基于 ID 映射到 HSV 色环
            hue = (tid * 37) % 180
            color_hsv = np.uint8([[[hue, 220, 220]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

            # 绘制边界框
            if state == 'confirmed':
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)
            elif state == 'occluded':
                self._draw_dashed_rect(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            else:
                self._draw_dashed_rect(vis, (x1, y1), (x2, y2), (255, 180, 0), 1)

            # 文字标签
            speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
            label = f"ID:{tid}  {speed:.1f}px/f"
            cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

            # 绘制轨迹
            pts = [(int(p[0]), int(p[1])) for p in traj]
            for k in range(1, len(pts)):
                if state == 'occluded' and k == len(pts) - 1:
                    cv2.line(vis, pts[k - 1], pts[k], color_bgr, 1, cv2.LINE_AA)
                else:
                    cv2.line(vis, pts[k - 1], pts[k], color_bgr, 2, cv2.LINE_AA)

        return vis

    @staticmethod
    def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len=8):
        """绘制虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        for (a, b) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = max(1, int((dx ** 2 + dy ** 2) ** 0.5))
            for start in range(0, length, dash_len * 2):
                end = min(start + dash_len, length)
                sx = int(a[0] + dx * start / length)
                sy = int(a[1] + dy * start / length)
                ex = int(a[0] + dx * end / length)
                ey = int(a[1] + dy * end / length)
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)