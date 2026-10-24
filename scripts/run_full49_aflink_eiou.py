"""
全样本实验：ECC + AFLink（完全后处理）+ EIoU，跑全部 49 个序列
AFLink 在每条序列全部帧处理完后统一调用一次，与论文原版一致。
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent          # = <repo>/scripts
REPO_ROOT = PROJECT_ROOT.parent                          # = <repo>
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fussball.adaptive_deepsort import AdaptiveDeepSORT
from fussball.aflink import AFLink
from fussball.detection import Detection
from fussball.ecc import ECCMotionCompensation
from fussball.track import TrackState


# ============================================================
# 追踪器：与 run_ablation.py 基本相同，但去掉每50帧触发逻辑
# ============================================================
class TrackerPostAFLink(AdaptiveDeepSORT):
    """ECC + EIoU + AFLink 完全后处理版"""

    def __init__(self, use_ecc=True, ecc_scale=0.75, use_eiou=True, **kwargs):
        super().__init__(use_eiou=use_eiou, **kwargs)
        self.use_ecc = use_ecc
        self.ecc = ECCMotionCompensation(scale=ecc_scale) if use_ecc else None
        self.prev_frame = None
        self._ecc_H = np.eye(3)
        self.track_history = defaultdict(lambda: {'frames': [], 'boxes': []})

    def reset(self):
        super().reset()
        if self.ecc:
            self.ecc.reset()
        self.prev_frame = None
        self._ecc_H = np.eye(3)
        self.track_history.clear()

    def update(self, frame_bgr):
        self.frame_id += 1
        curr_frame = frame_bgr.copy()

        if self.use_ecc and self.ecc is not None and self.prev_frame is not None:
            self._ecc_H = self.ecc.compensate(curr_frame)
        else:
            self._ecc_H = np.eye(3)
        self.prev_frame = curr_frame

        detections = self._detect(frame_bgr)

        for track in self.tracks:
            track.predict(self.kf)
            if self.use_ecc and not np.allclose(self._ecc_H, np.eye(3)):
                xy = track.mean[:2].reshape(1, 2)
                xy_c = self.ecc.apply(xy)
                track.mean[:2] = xy_c.flatten()

        confirmed = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.OCCLUDED)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        matches1, unm_trk1, unm_det1 = self._associate(confirmed, detections)
        remaining = [detections[i] for i in unm_det1]
        matches2, unm_trk2, unm_det2 = self._associate_iou(tentative, remaining)

        for ti, di in matches1:
            confirmed[ti].update(self.kf, detections[di], self.frame_id)
        for ti, di in matches2:
            tentative[ti].update(self.kf, remaining[di], self.frame_id)
        for ti in unm_trk1:
            confirmed[ti].mark_missed()
        for ti in unm_trk2:
            tentative[ti].mark_missed()

        for det in [remaining[i] for i in unm_det2]:
            self._init_track(det)

        self.tracks = [t for t in self.tracks if not t.is_deleted]
        self._record_track_history()

        # 构建结果（不做 AFLink，直接用原始 ID）
        results = []
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy, w, h = track.mean[:4]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                results.append({
                    'id': track.track_id,
                    'bbox': [x1, y1, x2, y2],
                    'frame_id': self.frame_id,
                })
        return results

    def _detect(self, frame_bgr):
        res = self.detector(frame_bgr, imgsz=self.imgsz, conf=self.conf_threshold, iou=self.iou_nms, verbose=False)[0]
        detections = []
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            valid_idx = [i for i, cls in enumerate(classes) if self.target_classes is None or int(cls) in self.target_classes]
            if valid_idx:
                valid_boxes = [boxes[i].tolist() for i in valid_idx]
                features = self.appearance.extract(frame_bgr, valid_boxes)
                for k, i in enumerate(valid_idx):
                    feat = features[k] if features.shape[0] > k else None
                    detections.append(Detection(valid_boxes[k], float(confs[i]), int(classes[i]), feat))
        return detections

    def _associate_iou(self, tracks, detections):
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        from scipy.optimize import linear_sum_assignment
        iou_mat = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            trk_tlbr = track.to_tlbr()
            for j, det in enumerate(detections):
                x1, y1, x2, y2 = trk_tlbr
                x1g, y1g, x2g, y2g = det.bbox
                xi1, yi1 = max(x1, x1g), max(y1, y1g)
                xi2, yi2 = min(x2, x2g), min(y2, y2g)
                inter = max(0, xi2-xi1) * max(0, yi2-yi1)
                union = (x2-x1)*(y2-y1) + (x2g-x1g)*(y2g-y1g) - inter
                iou_mat[i, j] = inter/union if union > 0 else 0
        ri, ci = linear_sum_assignment(1.0 - iou_mat)
        matched = [(r, c) for r, c in zip(ri, ci) if iou_mat[r, c] >= 0.3]
        unm_t = [i for i in range(len(tracks)) if i not in [r for r, _ in matched]]
        unm_d = [j for j in range(len(detections)) if j not in [c for _, c in matched]]
        return matched, unm_t, unm_d

    def _record_track_history(self):
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy, w, h = track.mean[:4]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                self.track_history[track.track_id]['frames'].append(self.frame_id)
                self.track_history[track.track_id]['boxes'].append([x1, y1, x2, y2])


# ============================================================
# 后处理 AFLink：序列全部帧跑完后统一调用
# ============================================================
def apply_aflink_postprocess(all_results, track_history, aflinker):
    """
    all_results: list of {frame_id, id, bbox}
    track_history: dict {track_id: {frames, boxes}}
    返回重新映射 ID 后的 results
    """
    if len(track_history) < 2:
        return all_results

    tracks_data = [
        {'track_id': tid, 'frames': h['frames'], 'boxes': h['boxes']}
        for tid, h in track_history.items() if len(h['frames']) > 0
    ]

    merge_map = aflinker.link(tracks_data)

    # 递归解析最终 ID
    def resolve(tid):
        visited = set()
        while tid in merge_map and tid not in visited:
            visited.add(tid)
            tid = merge_map[tid]
        return tid

    # 重映射
    new_results = []
    for r in all_results:
        r2 = dict(r)
        r2['id'] = resolve(r['id'])
        new_results.append(r2)
    return new_results


# ============================================================
# 处理单条序列
# ============================================================
def process_sequence(img_dir, output_file, tracker_kwargs, aflinker):
    tracker = TrackerPostAFLink(**tracker_kwargs)
    tracker.reset()

    images = sorted(img_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if not images:
        print(f"  警告: 未找到图像 {img_dir}")
        return 0

    all_results = []
    for img_path in tqdm(images, desc=f"  {img_dir.parent.name}", leave=False):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        dets = tracker.update(frame)
        all_results.extend(dets)

    # 后处理 AFLink（整条序列统一做一次）
    all_results = apply_aflink_postprocess(all_results, tracker.track_history, aflinker)

    # 写出
    lines = []
    for r in all_results:
        x1, y1, x2, y2 = r['bbox']
        w, h = x2 - x1, y2 - y1
        lines.append(f"{r['frame_id']},{r['id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n")
    lines.sort(key=lambda x: int(x.split(',')[0]))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return len(lines)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=str(REPO_ROOT / 'tracking' / 'test' / 'test'),
                        help='SNMOT 数据集根目录（默认 <repo>/tracking/test/test）')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['osnet', 'resnet50'])
    parser.add_argument('--appearance_weights', type=str, default=None)
    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--seqs', nargs='+', default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    seq_root  = data_root / 'images' / 'test'
    if not seq_root.exists():
        seq_root = data_root

    if args.model is None:
        args.model = str(REPO_ROOT / 'weights' / 'best.pt')

    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / 'outputs' / 'tracker_results' / 'full49' /
                              f'full49_ecc_aflink_post_eiou_{args.model_name}')
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("全样本实验：ECC + AFLink后处理 + EIoU")
    print(f"  外观模型: {args.model_name}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 AFLink（共享，不重复加载权重）
    aflinker = AFLink()

    all_seqs = sorted([d for d in seq_root.iterdir() if d.is_dir() and (d / 'img1').exists()])
    if args.seqs:
        seq_map = {d.name: d for d in all_seqs}
        seqs = [seq_map[s] for s in args.seqs if s in seq_map]
    else:
        seqs = all_seqs  # 全部49个

    print(f"  共 {len(seqs)} 个序列")

    tracker_kwargs = {
        'yolo_model_path': args.model,
        'conf_threshold': args.conf_threshold,
        'imgsz': args.imgsz,
        'target_classes': [0],
        'appearance_model': args.model_name,
        'appearance_weights': args.appearance_weights,
        'use_eiou': True,
        'ecc_scale': 0.75,
        'match_threshold': 0.7,
        'n_init': 3,
        'max_age': 25,
    }

    total = 0
    for i, seq in enumerate(seqs, 1):
        img_dir = seq / 'img1'
        out_file = output_dir / f'{seq.name}.txt'

        print(f"\n[{i}/{len(seqs)}] {seq.name}")
        if out_file.exists() and out_file.stat().st_size > 0:
            cnt = sum(1 for _ in open(out_file, encoding='utf-8'))
            total += cnt
            print(f"  [SKIP] 已存在（{cnt} 条）")
            continue

        cnt = process_sequence(img_dir, out_file, tracker_kwargs, aflinker)
        total += cnt
        print(f"  生成 {cnt} 条记录")

    print("\n" + "=" * 60)
    print(f"完成！总记录: {total}，输出: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
