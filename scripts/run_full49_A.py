import sys, cv2, numpy as np, argparse
from pathlib import Path
from tqdm import tqdm
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent          # = <repo>/scripts
REPO_ROOT = PROJECT_ROOT.parent                          # = <repo>

parser = argparse.ArgumentParser(description='实验A：纯运动基线（IoU关联）')
parser.add_argument('--data_root', type=str, default=str(REPO_ROOT / 'tracking' / 'test' / 'test'),
                    help='SNMOT 数据集根目录（默认 <repo>/tracking/test/test）')
parser.add_argument('--model', type=str,
                    default=str(REPO_ROOT / 'weights' / 'best.pt'),
                    help='YOLO 权重路径')
parser.add_argument('--output_dir', type=str, default=None,
                    help='结果输出目录（默认 tracker_results/full49/A_baseline）')
parser.add_argument('--use_eiou', action='store_true', default=False,
                    help='用 EIoU 替换 IoU 做关联（默认关闭）')
parser.add_argument('--max_seqs', type=int, default=10,
                    help='最多处理多少个序列（默认前10个，-1表示全部）')
parser.add_argument('--seqs', nargs='+', default=None,
                    help='指定序列名列表（优先于 --max_seqs）')
args = parser.parse_args()

MODEL_PATH = args.model
DATA_ROOT = Path(args.data_root)

if args.output_dir is None:
    suffix = 'A_baseline_eiou' if args.use_eiou else 'A_baseline'
    output_dir = REPO_ROOT / 'outputs' / 'tracker_results' / 'full49' / suffix
else:
    output_dir = Path(args.output_dir)

sys.path.insert(0, str(REPO_ROOT))

# 不清空目录，支持断点续跑
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================
# 纯运动追踪器：不加载任何外观模型
# ============================================
from fussball.kalman_filter import ExtendedKalmanFilter
from fussball.track import Track, TrackState
from fussball.detection import Detection
from fussball.association import iou_batch, eiou_batch
from ultralytics import YOLO

class TrackerPureMotion:
    def __init__(self):
        self.detector = YOLO(MODEL_PATH)
        self.kf = ExtendedKalmanFilter()
        self.tracks = []
        self.frame_id = 0
        self.match_threshold = 0.7
        self.max_age = 25
        self.n_init = 3
        self.total_ids = 0
        Track._id_counter = 0

    def reset(self):
        self.tracks = []
        self.frame_id = 0
        Track._id_counter = 0

    def update(self, frame_bgr):
        self.frame_id += 1
        
        # YOLO 检测
        results = self.detector(frame_bgr, imgsz=1280, conf=0.25, iou=0.5, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
        classes = results.boxes.cls.cpu().numpy() if results.boxes is not None else np.array([])
        
        detections = []
        for box, conf, cls in zip(boxes, confs, classes):
            if int(cls) == 0:  # only person
                detections.append(Detection(box.tolist(), conf, int(cls), None))
        
        # 预测
        for track in self.tracks:
            track.predict(self.kf)
        
        # IoU 关联
        matched, unmatched_dets, unmatched_trks = self._iou_associate(detections)
        
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(self.kf, detections[det_idx], self.frame_id)
        
        for det_idx in unmatched_dets:
            self.total_ids += 1
            det = detections[det_idx]
            mean, cov = self.kf.initiate(det.to_xyah())
            t = Track(mean, cov, 0, None, self.n_init, self.max_age, 0, 30)
            t.track_id = self.total_ids
            self.tracks.append(t)
        
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        self.tracks = [t for t in self.tracks if not t.is_deleted]
        
        results = []
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy, w, h = track.mean[:4]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                results.append({
                    'id': track.track_id,
                    'bbox': [x1, y1, x2, y2],
                })
        return results

    def _iou_associate(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        from scipy.optimize import linear_sum_assignment
        trk_tlbr = np.array([t.to_tlbr() for t in self.tracks])   # (M,4)
        det_tlbr = np.array([d.bbox for d in detections])          # (N,4)
        if args.use_eiou:
            sim = eiou_batch(trk_tlbr, det_tlbr)   # (M,N), EIoU∈[-1,1]
            iou_mat = np.clip(0.5 * (1.0 + sim), 0.0, 1.0)
        else:
            iou_mat = iou_batch(trk_tlbr, det_tlbr)  # (M,N)
        ri, ci = linear_sum_assignment(-iou_mat)
        matched = [(r, c) for r, c in zip(ri, ci) if iou_mat[r, c] >= self.match_threshold]
        unm_d = [j for j in range(len(detections)) if j not in [c for _, c in matched]]
        unm_t = [i for i in range(len(self.tracks)) if i not in [r for r, _ in matched]]
        return matched, unm_d, unm_t

# ============================================
# 运行序列（默认前10个）
# ============================================
all_seqs = sorted([d for d in (DATA_ROOT / 'images' / 'test').iterdir() if d.is_dir()])

if args.seqs:
    seq_map = {d.name: d for d in all_seqs}
    seqs = [seq_map[s] for s in args.seqs if s in seq_map]
elif args.max_seqs > 0:
    seqs = all_seqs[:args.max_seqs]
else:
    seqs = all_seqs

print(f"[INFO] 共 {len(all_seqs)} 个序列，本次处理 {len(seqs)} 个")
print("[INFO] 初始化纯运动追踪器（无外观模型）")
total = 0

for i, seq in enumerate(seqs, 1):  # type: ignore[assignment]
    img_dir = seq / "img1"
    if not img_dir.exists(): continue
    
    out_file = output_dir / f"{seq.name}.txt"
    
    # 断点续跑：已存在结果则跳过
    if out_file.exists() and out_file.stat().st_size > 0:
        count = sum(1 for _ in open(out_file))
        total += count
        print(f"[{i}/{len(seqs)}] {seq.name}: [SKIP] 已存在（{count} 条）")
        continue
    
    tracker = TrackerPureMotion()
    tracker.reset()
    
    images = sorted(img_dir.glob("*.jpg"), key=lambda x: int(x.stem))
    results = []
    
    for img_path in tqdm(images, desc=f"  {seq.name}", leave=False):
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        frame_id = int(img_path.stem)
        detections = tracker.update(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            results.append(f"{frame_id},{det['id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n")
    
    with open(out_file, 'w') as f: f.writelines(results)
    count = len(results); total += count
    print(f"[{i}/{len(seqs)}] {seq.name}: {count}")

print(f"\n[INFO] 完成！总记录数: {total}")
