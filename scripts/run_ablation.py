from __future__ import annotations

"""
消融实验脚本：支持 ECC、AFLink、空间距离限制的开关
用法:
    python scripts/run_ablation.py --use_ecc                    # 启用 ECC
    python scripts/run_ablation.py --no-use_ecc                 # 禁用 ECC
    python scripts/run_ablation.py --use_spatial_limit          # 启用空间距离限制
    python scripts/run_ablation.py --model_name osnet           # 选择外观模型
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fussball.project import DEFAULT_TEST_DATA_ROOT, DEFAULT_YOLO_WEIGHTS, TRACKER_RESULTS_ROOT, load_default_test_sequence_names
from fussball.track import TrackState
from fussball.adaptive_deepsort import AdaptiveDeepSORT
from fussball.aflink import AFLink
from fussball.detection import Detection
from fussball.ecc import ECCMotionCompensation
from fussball.association import adaptive_cost_matrix, hungarian_match


# ============================================================================
# 消融实验追踪器（继承 AdaptiveDeepSORT）
# ============================================================================
class TrackerAblation(AdaptiveDeepSORT):
    """消融实验追踪器：支持 ECC、AFLink、空间距离限制的灵活配置"""

    def __init__(self,
                 use_ecc=True,
                 use_aflink=True,
                 use_spatial_limit=False,
                 spatial_max_dist=200,
                 aflink_max_gap=60,
                 aflink_sim=0.70,
                 aflink_dist=200,
                 ecc_scale=0.75,
                 **kwargs):
        """
        初始化消融实验追踪器
        
        Args:
            use_ecc: 是否启用 ECC 运动补偿
            use_aflink: 是否启用 AFLink 轨迹合并
            use_spatial_limit: 是否在关联代价矩阵中加入空间距离限制
            spatial_max_dist: 空间距离阈值（像素）
            ecc_scale: ECC 计算时的图像缩放比例
            **kwargs: 传递给父类 AdaptiveDeepSORT 的参数
        """
        super().__init__(**kwargs)
        
        # ECC 配置
        self.use_ecc = use_ecc
        self.ecc = ECCMotionCompensation(scale=ecc_scale) if use_ecc else None
        self.prev_frame = None
        self._ecc_H = np.eye(3)
        
        # AFLink 配置
        self.use_aflink = use_aflink
        self.aflink_max_gap = aflink_max_gap
        self.aflink_sim = aflink_sim
        self.aflink_dist = aflink_dist
        self.aflink = AFLink(max_gap=aflink_max_gap, sim_thresh=aflink_sim, dist_thresh=aflink_dist) if use_aflink else None
        
        # 空间距离限制配置
        self.use_spatial_limit = use_spatial_limit
        self.spatial_max_dist = spatial_max_dist
        
        # 用于存储轨迹历史（供 AFLink 使用）
        self.track_history = defaultdict(lambda: {'frames': [], 'boxes': []})
        self._id_mapping = {}

    def reset(self):
        """重置跟踪器状态"""
        super().reset()
        if self.ecc:
            self.ecc.reset()
        self.prev_frame = None
        self._ecc_H = np.eye(3)
        self.track_history.clear()
        self._id_mapping.clear()

    # ------------------------------------------------------------------------ #
    #  核心方法重写
    # ------------------------------------------------------------------------ #
    def update(self, frame_bgr):
        """
        处理单帧图像（带 ECC 运动补偿）
        
        Args:
            frame_bgr: BGR 图像
            
        Returns:
            results: 跟踪结果列表
        """
        self.frame_id += 1
        curr_frame = frame_bgr
        
        # --- Step 1: ECC 运动补偿（计算变换矩阵）---
        if self.use_ecc and self.ecc is not None and self.prev_frame is not None:
            self._ecc_H = self.ecc.compensate(curr_frame)
        else:
            self._ecc_H = np.eye(3)
        self.prev_frame = curr_frame
        
        # --- Step 2: 目标检测 ---
        detections = self._detect(frame_bgr)
        
        # --- Step 3: 卡尔曼预测 + ECC 修正 ---
        for track in self.tracks:
            track.predict(self.kf)
            # 如果启用 ECC，将预测位置变换到当前帧坐标系
            if self.use_ecc and not np.allclose(self._ecc_H, np.eye(3)):
                xy = track.mean[:2].reshape(1, 2)
                xy_c = self.ecc.apply(xy)
                track.mean[:2] = xy_c.flatten()
        
        # --- Step 4: 两阶段关联 ---
        confirmed_tracks = [t for t in self.tracks 
                           if t.state in (TrackState.CONFIRMED, TrackState.OCCLUDED)]
        tentative_tracks = [t for t in self.tracks 
                           if t.state == TrackState.TENTATIVE]
        
        # 第一阶段：自适应代价矩阵关联（已确认轨迹）
        matches1, unm_trk1, unm_det1 = self._associate(confirmed_tracks, detections)
        
        # 第二阶段：IOU 关联（新生轨迹）
        remaining_dets = [detections[i] for i in unm_det1]
        matches2, unm_trk2, unm_det2 = self._associate_iou(tentative_tracks, remaining_dets)
        
        # --- Step 5: 更新轨迹 ---
        for trk_idx, det_idx in matches1:
            confirmed_tracks[trk_idx].update(self.kf, detections[det_idx], self.frame_id)
            
        for trk_idx, det_idx in matches2:
            tentative_tracks[trk_idx].update(self.kf, remaining_dets[det_idx], self.frame_id)
            
        # 标记未匹配轨迹
        for trk_idx in unm_trk1:
            confirmed_tracks[trk_idx].mark_missed()
        for trk_idx in unm_trk2:
            tentative_tracks[trk_idx].mark_missed()
        
        # --- Step 6: 初始化新轨迹 ---
        final_unm_dets = [remaining_dets[i] for i in unm_det2]
        for det in final_unm_dets:
            self._init_track(det)
        
        # --- Step 7: 清理已删除轨迹 ---
        self.tracks = [t for t in self.tracks if not t.is_deleted]
        
        # --- Step 8: 记录轨迹历史（供 AFLink 使用）---
        self._record_track_history()
        
        # --- Step 9: 构建输出 ---
        results = self._build_results()
        
        return results

    def _detect(self, frame_bgr):
        results = self.detector(frame_bgr, imgsz=self.imgsz, conf=self.conf_threshold, iou=self.iou_nms, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            valid_idx = [i for i, cls in enumerate(classes) if self.target_classes is None or int(cls) in self.target_classes]
            if valid_idx:
                valid_boxes = [boxes[i].tolist() for i in valid_idx]
                features = self.appearance.extract(frame_bgr, valid_boxes)
                for k, i in enumerate(valid_idx):
                    feat = features[k] if features.shape[0] > k else None
                    detections.append(Detection(valid_boxes[k], float(confs[i]), int(classes[i]), feat))
        return detections


    def _associate(self, tracks, detections):
        """使用可配置空间门控的自适应代价矩阵关联"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        spatial_max_dist = self.spatial_max_dist if self.use_spatial_limit else 1e9
        cost = adaptive_cost_matrix(
            tracks,
            detections,
            self.kf,
            spatial_max_dist=spatial_max_dist,
        )
        return hungarian_match(cost, threshold=self.match_threshold)

    def _associate_iou(self, tracks, detections):
        """纯 IOU 关联（用于新生轨迹）"""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        from scipy.optimize import linear_sum_assignment
        
        # 计算 IOU 矩阵
        iou_mat = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            trk_tlbr = track.to_tlbr()
            for j, det in enumerate(detections):
                iou_mat[i, j] = self._compute_iou(trk_tlbr, det.bbox)
        
        # 匈牙利匹配（最大化 IOU）
        cost_mat = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        
        # 阈值过滤
        matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= 0.3]
        
        matched_t = [r for r, _ in matched]
        matched_d = [c for _, c in matched]
        
        unm_t = [i for i in range(len(tracks)) if i not in matched_t]
        unm_d = [j for j in range(len(detections)) if j not in matched_d]
        
        return matched, unm_t, unm_d
    
    @staticmethod
    def _compute_iou(box1, box2):
        """计算两个边界框的 IOU"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0

    def _apply_ecc_to_box(self, box, H):
        """将 ECC 变换矩阵应用到边界框"""
        if len(box) != 4:
            return box
        
        x1, y1, x2, y2 = box
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        
        p1_t = H @ p1
        p2_t = H @ p2
        
        p1_t = p1_t / p1_t[2]
        p2_t = p2_t / p2_t[2]
        
        return np.array([p1_t[0], p1_t[1], p2_t[0], p2_t[1]], dtype=np.float32)

    def _record_track_history(self):
        """记录轨迹历史（供 AFLink 使用）"""
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy, w, h = track.mean[:4]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                
                self.track_history[track.track_id]['frames'].append(self.frame_id)
                self.track_history[track.track_id]['boxes'].append([x1, y1, x2, y2])
    
    def _build_results(self):
        """构建输出结果（可选应用 AFLink 轨迹合并）"""
        # 如果启用 AFLink，进行轨迹合并
        if self.use_aflink and self.aflink is not None and self.frame_id % 50 == 0:
            self._apply_aflink()
        
        # 构建结果（使用合并后的 ID）
        # 注意：AFLink 在线映射后，可能出现两个活跃轨迹映射到同一个最终 ID。
        # 为避免 MOT/TrackEval 出现“同一帧同一 ID 多次出现”的非法结果，
        # 这里按 final_id 去重，优先保留“原生 final_id 轨迹”，其次保留 hits 更多、框面积更大的目标。
        best_by_final_id = {}
        for track in self.tracks:
            if not track.is_confirmed:
                continue

            final_id = self._get_final_track_id(track.track_id)
            cx, cy, w, h = track.mean[:4]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            area = max(float(w), 0.0) * max(float(h), 0.0)
            score = (
                1 if track.track_id == final_id else 0,
                int(track.hits),
                area,
            )
            result = {
                'id': final_id,
                'bbox': [x1, y1, x2, y2],
                'class': str(track.det_class),
                'state': 'confirmed',
                'velocity': track.velocity[:2].tolist(),
                'trajectory': [[p[0], p[1]] for p in track.trajectory],
                'occ_frames': track.occ_frames,
            }

            existing = best_by_final_id.get(final_id)
            if existing is None or score > existing['score']:
                best_by_final_id[final_id] = {'score': score, 'result': result}

        return [best_by_final_id[track_id]['result'] for track_id in sorted(best_by_final_id)]
    
    def _apply_aflink(self):
        """应用 AFLink 合并断裂轨迹"""
        if len(self.track_history) < 2:
            return
        
        tracks_data = []
        for track_id, history in self.track_history.items():
            if len(history['frames']) > 0:
                tracks_data.append({
                    'track_id': track_id,
                    'frames': history['frames'],
                    'boxes': history['boxes'],
                })
        
        if len(tracks_data) >= 2:
            merge_map = self.aflink.link(tracks_data)
            
            # 存储 ID 映射
            if not hasattr(self, '_id_mapping'):
                self._id_mapping = {}
            
            for old_id, new_id in merge_map.items():
                # 递归查找最终 ID
                while new_id in merge_map:
                    new_id = merge_map[new_id]
                self._id_mapping[old_id] = new_id
    
    def _get_final_track_id(self, track_id):
        """获取 AFLink 合并后的最终 ID"""
        if hasattr(self, '_id_mapping') and track_id in self._id_mapping:
            return self._id_mapping[track_id]
        return track_id


# ============================================================================
# 序列处理函数
# ============================================================================
def process_sequence(tracker_class, img_dir, output_file, tracker_kwargs):
    """
    处理单个视频序列
    
    Args:
        tracker_class: 追踪器类
        img_dir: 图像目录路径
        output_file: 输出文件路径
        tracker_kwargs: 追踪器初始化参数
    
    Returns:
        total_records: 生成的记录总数
    """
    tracker = tracker_class(**tracker_kwargs)
    tracker.reset()
    
    images = sorted(img_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if not images:
        print(f"  ⚠️ 未找到图像: {img_dir}")
        return 0
    
    results = []
    
    for img_path in tqdm(images, desc=f"  {img_dir.parent.name}", leave=False):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        frame_id = int(img_path.stem)
        detections = tracker.update(frame)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            # 格式: frame_id, track_id, x, y, w, h, conf, -1, -1, -1
            results.append(f"{frame_id},{det['id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n")
    
    # 按帧号排序
    results.sort(key=lambda x: int(x.split(',')[0]))
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(results)
    
    return len(results)


def resolve_seq_root(data_root: Path) -> Path:
    """解析数据根目录结构"""
    # 允许直接传单个序列目录，例如 /root/autodl-tmp/Fussball/tracking/test/test/SNMOT-116
    if (data_root / 'img1').is_dir():
        return data_root

    # SNMOT 数据集结构: data_root/images/test/SNMOT-xxx/img1/
    legacy_root = data_root / 'images' / 'test'
    if legacy_root.is_dir():
        return legacy_root
    return data_root


def collect_sequence_dirs(seq_root: Path) -> list[Path]:
    if (seq_root / 'img1').is_dir():
        return [seq_root]
    return sorted([d for d in seq_root.iterdir() if d.is_dir() and (d / 'img1').exists()])


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='消融实验：控制 ECC/AFLink/空间距离限制')
    
    # 消融实验开关
    parser.add_argument('--use_ecc', action='store_true', default=True,
                        help='是否启用 ECC（默认开启）')
    parser.add_argument('--no-use_ecc', action='store_false', dest='use_ecc',
                        help='禁用 ECC')
    parser.add_argument('--use_aflink', action='store_true', default=True,
                        help='是否启用 AFLink（默认开启）')
    parser.add_argument('--no-use_aflink', action='store_false', dest='use_aflink',
                        help='禁用 AFLink')
    parser.add_argument('--use_spatial_limit', action='store_true', default=False,
                        help='是否启用空间距离限制（默认关闭）')
    parser.add_argument('--spatial_max_dist', type=float, default=200,
                        help='空间距离阈值（像素），默认 200')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='osnet',
                        choices=['osnet', 'resnet50'],
                        help='外观模型类型（默认 osnet）')
    parser.add_argument('--appearance_weights', type=str, default=None,
                        help='外观模型权重路径')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, 
                        default=str(DEFAULT_TEST_DATA_ROOT),
                        help='测试集根目录')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLO 权重路径')
    parser.add_argument('--seqs', nargs='+', default=None,
                        help='指定要处理的序列名；若不指定则处理当前目录下全部可用序列')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录')
    
    # 跟踪参数
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='检测置信度阈值')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='推理分辨率')
    parser.add_argument('--match_threshold', type=float, default=0.70,
                        help='匈牙利匹配阈值，默认 0.70')
    parser.add_argument('--max_age', type=int, default=25,
                        help='轨迹最大丢失帧数，默认 25')
    parser.add_argument('--n_init', type=int, default=3,
                        help='轨迹确认所需命中次数，默认 3')
    parser.add_argument('--max_feature_queue', type=int, default=5,
                        help='特征队列长度，默认 5')
    parser.add_argument('--aflink_max_gap', type=int, default=60,
                        help='AFLink 最大时间间隔，默认 60')
    parser.add_argument('--aflink_sim', type=float, default=0.70,
                        help='AFLink 相似度阈值，默认 0.70')
    parser.add_argument('--aflink_dist', type=float, default=200,
                        help='AFLink 空间距离阈值，默认 200')
    
    args = parser.parse_args()
    
    # 设置默认路径
    data_root = Path(args.data_root)
    seq_root = resolve_seq_root(data_root)
    
    if args.model is None:
        # 默认 YOLO 权重路径
        args.model = str(DEFAULT_YOLO_WEIGHTS)
    model_path = Path(args.model)
    
    if args.output_dir is None:
        # 根据配置生成输出目录名
        components = ['ablation']
        components.append('ecc' if args.use_ecc else 'noecc')
        components.append('aflink' if args.use_aflink else 'noaflink')
        if args.use_spatial_limit:
            components.append(f'spatial_{args.spatial_max_dist}')
        components.append(args.model_name)
        output_dir_name = '_'.join(components)
        args.output_dir = TRACKER_RESULTS_ROOT / output_dir_name
    else:
        args.output_dir = Path(args.output_dir)
    
    # 验证路径
    if not seq_root.exists():
        raise FileNotFoundError(f'数据目录不存在: {seq_root}')
    if not model_path.exists():
        raise FileNotFoundError(f'YOLO 权重不存在: {model_path}')
    
    # 获取序列列表
    available_seqs = collect_sequence_dirs(seq_root)
    seq_map = {seq.name: seq for seq in available_seqs}

    if args.seqs:
        missing = [name for name in args.seqs if name not in seq_map]
        if missing:
            raise FileNotFoundError(f'以下序列不存在: {missing}')
        seqs = [seq_map[name] for name in args.seqs]
    else:
        default_seq_names = load_default_test_sequence_names(seq_root)
        missing = [name for name in default_seq_names if name not in seq_map]
        if missing:
            raise FileNotFoundError(f'默认 49 序列中以下目录不存在: {missing}')
        seqs = [seq_map[name] for name in default_seq_names]
    
    # 打印配置信息
    print("=" * 60)
    print("消融实验配置")
    print("=" * 60)
    print(f"  ECC: {'✓ 启用' if args.use_ecc else '✗ 禁用'}")
    print(f"  AFLink: {'✓ 启用' if args.use_aflink else '✗ 禁用'}")
    print(f"  空间距离限制: {'✓ 启用' if args.use_spatial_limit else '✗ 禁用'}")
    if args.use_spatial_limit:
        print(f"    距离阈值: {args.spatial_max_dist}px")
    print(f"  外观模型: {args.model_name}")
    print(f"  match_threshold: {args.match_threshold}")
    print(f"  n_init: {args.n_init}")
    print(f"  max_age: {args.max_age}")
    print(f"  max_feature_queue: {args.max_feature_queue}")
    print(f"  aflink_max_gap: {args.aflink_max_gap}")
    print(f"  aflink_sim: {args.aflink_sim}")
    print(f"  aflink_dist: {args.aflink_dist}")
    print(f"  YOLO 权重: {model_path}")
    print(f"  数据目录: {seq_root}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  处理序列数: {len(seqs)}")
    print("=" * 60)
    
    # 清理并创建输出目录
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 追踪器参数
    tracker_kwargs = {
        'yolo_model_path': str(model_path),
        'conf_threshold': args.conf_threshold,
        'imgsz': args.imgsz,
        'target_classes': [0],  # 只跟踪 person
        'appearance_model': args.model_name,
        'appearance_weights': args.appearance_weights,
        'use_ecc': args.use_ecc,
        'use_aflink': args.use_aflink,
        'use_spatial_limit': args.use_spatial_limit,
        'spatial_max_dist': args.spatial_max_dist,
        'ecc_scale': 0.75,
        'match_threshold': args.match_threshold,
        'n_init': args.n_init,
        'max_age': args.max_age,
        'max_feature_queue': args.max_feature_queue,
        'aflink_max_gap': args.aflink_max_gap,
        'aflink_sim': args.aflink_sim,
        'aflink_dist': args.aflink_dist,
    }
    
    # 处理所有序列
    total_records = 0
    for i, seq in enumerate(seqs, 1):
        img_dir = seq / 'img1'
        output_file = args.output_dir / f'{seq.name}.txt'
        
        print(f"\n[{i}/{len(seqs)}] 处理序列: {seq.name}")
        
        count = process_sequence(
            TrackerAblation,
            img_dir,
            output_file,
            tracker_kwargs
        )
        
        total_records += count
        print(f"  ✅ 生成 {count} 条记录")
    
    print("\n" + "=" * 60)
    print(f"完成！总记录数: {total_records}")
    print(f"结果保存至: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()