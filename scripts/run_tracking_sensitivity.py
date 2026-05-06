from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fussball.project import (
    CONFIGS_ROOT,
    DEFAULT_OSNET_WEIGHTS,
    DEFAULT_TEST_DATA_ROOT,
    DEFAULT_YOLO_WEIGHTS,
    TRACKER_RESULTS_ROOT,
    load_default_test_sequence_names,
)
from scripts.evaluate_trackeval_metrics import (
    build_eval_components,
    extract_summary_scores,
    prepare_tracker_layout,
    save_csv as save_trackeval_csv,
)
from scripts.run_ablation import TrackerAblation, resolve_seq_root


DEFAULT_PLAN_PATH = CONFIGS_ROOT / '灵敏性分析.txt'
DEFAULT_RESULTS_ROOT = TRACKER_RESULTS_ROOT / 'sensitivity_nonrule'
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_ROOT / 'sensitivity_report.md'
DEFAULT_GT_ROOT = DEFAULT_TEST_DATA_ROOT
DEFAULT_MODEL_PATH = DEFAULT_YOLO_WEIGHTS
DEFAULT_OSNET_PATH = DEFAULT_OSNET_WEIGHTS

PARAM_ALIASES = {
    'match_thresh': 'match_threshold',
    'match_threshold': 'match_threshold',
    'n_init': 'n_init',
    'spatial_max_dist': 'spatial_max_dist',
    'aflink_max_gap': 'aflink_max_gap',
    'aflink_sim': 'aflink_sim',
    'aflink_dist': 'aflink_dist',
    'max_age': 'max_age',
    'max_feature_queue': 'max_feature_queue',
    'conf': 'conf_threshold',
    'conf_threshold': 'conf_threshold',
    'imgsz': 'imgsz',
}

PARAM_CASTERS = {
    'match_threshold': float,
    'n_init': int,
    'spatial_max_dist': float,
    'aflink_max_gap': int,
    'aflink_sim': float,
    'aflink_dist': float,
    'max_age': int,
    'max_feature_queue': int,
    'conf_threshold': float,
    'imgsz': int,
}

DEFAULT_SWEEP_CONFIG = {
    'conf_threshold': 0.25,
    'imgsz': 1280,
    'max_feature_queue': 5,
    'max_age': 25,
    'n_init': 3,
    'match_threshold': 0.70,
    'spatial_max_dist': 200.0,
    'aflink_max_gap': 60,
    'aflink_sim': 0.70,
    'aflink_dist': 200.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='对非规则 YOLO + OSNet + AFLink + ECC 链路做单变量灵敏性分析。')
    parser.add_argument('--plan', type=str, default=str(DEFAULT_PLAN_PATH), help='灵敏性分析配置表路径。')
    parser.add_argument('--data_root', type=str, default=str(DEFAULT_GT_ROOT), help='测试集根目录。')
    parser.add_argument('--gt_root', type=str, default=str(DEFAULT_GT_ROOT), help='TrackEval GT 根目录。')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH), help='YOLO 权重路径。')
    parser.add_argument('--appearance_weights', type=str, default=str(DEFAULT_OSNET_PATH), help='OSNet 权重路径。')
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_RESULTS_ROOT), help='灵敏性分析输出根目录。')
    parser.add_argument('--report_path', type=str, default=str(DEFAULT_REPORT_PATH), help='Markdown 报告保存路径。')
    parser.add_argument('--seqs', nargs='+', default=None, help='可选：仅处理指定序列；默认处理全部 49 个序列。')
    parser.add_argument('--device', type=str, default=None, help='可选：reid 使用的设备。')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='默认开启断点续跑：已完成序列不重跑，已有评测直接复用。')
    parser.add_argument('--no-skip_existing', action='store_false', dest='skip_existing', help='忽略已有结果，强制从头重跑当前灵敏性实验。')
    return parser.parse_args()



def parse_plan(plan_path: Path) -> OrderedDict[str, list[float | int]]:
    if not plan_path.exists():
        raise FileNotFoundError(f'灵敏性分析配置文件不存在: {plan_path}')

    plan: OrderedDict[str, list[float | int]] = OrderedDict()
    lines = plan_path.read_text(encoding='utf-8').splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('参数'):
            continue

        parts = re.split(r'\s+', line)
        if len(parts) < 2:
            continue

        raw_name, raw_value = parts[0], parts[1]
        param_name = PARAM_ALIASES.get(raw_name)
        if param_name is None:
            raise ValueError(f'不支持的灵敏性分析参数: {raw_name}')

        caster = PARAM_CASTERS[param_name]
        value = caster(raw_value)
        plan.setdefault(param_name, [])
        if value not in plan[param_name]:
            plan[param_name].append(value)

    if not plan:
        raise RuntimeError(f'未能从 {plan_path} 解析出任何参数组合。')
    return plan


def collect_tracking_sequences(data_root: Path, requested_seqs: list[str] | None = None) -> list[Path]:
    seq_root = resolve_seq_root(data_root)
    if not seq_root.exists():
        raise FileNotFoundError(f'数据目录不存在: {seq_root}')

    if (seq_root / 'img1').is_dir():
        available = [seq_root]
    else:
        available = sorted([p for p in seq_root.iterdir() if p.is_dir() and (p / 'img1').exists()], key=lambda p: p.name)

    seq_map = {p.name: p for p in available}
    if requested_seqs:
        missing = [name for name in requested_seqs if name not in seq_map]
        if missing:
            raise FileNotFoundError(f'以下序列不存在: {missing}')
        return [seq_map[name] for name in requested_seqs]

    default_seq_names = load_default_test_sequence_names(seq_root)
    missing = [name for name in default_seq_names if name not in seq_map]
    if missing:
        raise FileNotFoundError(f'默认 49 序列中以下目录不存在: {missing}')
    return [seq_map[name] for name in default_seq_names]




def sanitize_value(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace('.', 'p')


def process_sequence_with_tracker(tracker: TrackerAblation, img_dir: Path, output_file: Path) -> int:
    tracker.reset()
    images = sorted(img_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if not images:
        print(f'  ⚠️ 未找到图像: {img_dir}')
        return 0

    rows: list[str] = []
    for img_path in tqdm(images, desc=f'  {img_dir.parent.name}', leave=False):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame_id = int(img_path.stem)
        detections = tracker.update(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            rows.append(f'{frame_id},{det["id"]},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.00,-1,-1,-1\n')

    rows.sort(key=lambda x: (int(x.split(',', 1)[0]), int(x.split(',')[1])))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(''.join(rows), encoding='utf-8')
    return len(rows)


def run_trackeval(results_dir: Path, gt_root: Path, tracker_name: str, seqs: list[str], output_dir: Path) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix='trackeval_local_') as temp_dir:
        temp_root = Path(temp_dir)
        prepare_tracker_layout(results_dir, tracker_name, seqs, temp_root)
        evaluator, dataset, metrics_list = build_eval_components(
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
        combined_scores, per_sequence_scores = extract_summary_scores(output_res, dataset_name, tracker_name)
        save_trackeval_csv(per_sequence_scores, combined_scores, output_dir)
        return combined_scores


def save_summary_csv(records: list[dict[str, str | float | int]], csv_path: Path) -> None:

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['parameter', 'value', 'HOTA', 'AssA', 'DetA', 'MOTA', 'results_dir'])
        writer.writeheader()
        for record in records:
            writer.writerow({
                'parameter': record['parameter'],
                'value': record['value'],
                'HOTA': f"{record['HOTA']:.3f}",
                'AssA': f"{record['AssA']:.3f}",
                'DetA': f"{record['DetA']:.3f}",
                'MOTA': f"{record['MOTA']:.3f}",
                'results_dir': record['results_dir'],
            })


def validate_gt_sequences(gt_root: Path, seq_names: list[str]) -> None:
    missing = []
    for seq in seq_names:
        seq_dir = gt_root / seq
        if not (seq_dir / 'gt' / 'gt.txt').exists() or not (seq_dir / 'seqinfo.ini').exists():
            missing.append(seq)
    if missing:
        raise FileNotFoundError(f'以下序列缺少 GT 或 seqinfo.ini: {missing}')


def has_completed_sequence_output(output_file: Path) -> bool:
    return output_file.exists() and output_file.stat().st_size > 0


def all_sequences_completed(results_dir: Path, seq_names: list[str]) -> bool:
    return all(has_completed_sequence_output(results_dir / f'{seq_name}.txt') for seq_name in seq_names)


def sanitize_mot_output_file(output_file: Path) -> int:
    if not has_completed_sequence_output(output_file):
        return 0

    best_rows: dict[tuple[int, int], tuple[tuple[float, int], str]] = {}
    duplicate_rows = 0
    for idx, raw_line in enumerate(output_file.read_text(encoding='utf-8').splitlines()):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) < 6:
            continue

        frame_id = int(float(parts[0]))
        track_id = int(float(parts[1]))
        width = float(parts[4])
        height = float(parts[5])
        area = max(width, 0.0) * max(height, 0.0)
        key = (frame_id, track_id)
        score = (area, idx)

        if key in best_rows:
            duplicate_rows += 1
        if key not in best_rows or score > best_rows[key][0]:
            best_rows[key] = (score, line)

    if duplicate_rows:
        sanitized_lines = [best_rows[key][1] + '\n' for key in sorted(best_rows, key=lambda item: (item[0], item[1]))]
        output_file.write_text(''.join(sanitized_lines), encoding='utf-8')

    return duplicate_rows


def sanitize_results_dir(results_dir: Path, seq_names: list[str]) -> tuple[list[str], int]:
    cleaned_sequences: list[str] = []
    duplicate_rows = 0
    for seq_name in seq_names:
        cleaned = sanitize_mot_output_file(results_dir / f'{seq_name}.txt')
        if cleaned:
            cleaned_sequences.append(seq_name)
            duplicate_rows += cleaned
    return cleaned_sequences, duplicate_rows


def load_combined_metrics_from_csv(csv_path: Path) -> dict[str, float]:

    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('sequence') != 'COMBINED':
                continue
            return {
                'HOTA': float(row['HOTA']),
                'AssA': float(row['AssA']),
                'DetA': float(row['DetA']),
                'MOTA': float(row['MOTA']),
            }
    raise ValueError(f'未能从评测 CSV 中找到 COMBINED 行: {csv_path}')


def update_progress_outputs(
    records: Iterable[dict[str, str | float | int]],
    summary_csv: Path,
    report_path: Path,
    seq_count: int,
    base_config: dict[str, float | int],
) -> None:
    materialized = list(records)
    save_summary_csv(materialized, summary_csv)
    report_content = format_markdown(materialized, seq_count, base_config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding='utf-8')


def format_markdown(records: Iterable[dict[str, str | float | int]], seq_count: int, base_config: dict[str, float | int]) -> str:


    lines = [
        '## 非规则 YOLO + OSNet + AFLink + ECC 灵敏性分析',
        '',
        f'- **评测序列数**: {seq_count}',
        f'- **固定参数**: conf={base_config["conf_threshold"]}, imgsz={base_config["imgsz"]}, max_feature_queue={base_config["max_feature_queue"]}, max_age={base_config["max_age"]}, n_init={base_config["n_init"]}, match_threshold={base_config["match_threshold"]}, spatial_max_dist={base_config["spatial_max_dist"]}, aflink_max_gap={base_config["aflink_max_gap"]}, aflink_sim={base_config["aflink_sim"]}, aflink_dist={base_config["aflink_dist"]}',
        '',
        '| 参数 | 值 | HOTA | AssA | DetA | MOTA |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for record in records:
        lines.append(
            f"| {record['parameter']} | {record['value']} | {record['HOTA']:.3f} | {record['AssA']:.3f} | {record['DetA']:.3f} | {record['MOTA']:.3f} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    data_root = Path(args.data_root).resolve()
    gt_root = Path(args.gt_root).resolve()
    model_path = Path(args.model).resolve()
    appearance_weights = Path(args.appearance_weights).resolve()
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.report_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f'YOLO 权重不存在: {model_path}')
    if not appearance_weights.exists():
        raise FileNotFoundError(f'OSNet 权重不存在: {appearance_weights}')
    if not gt_root.exists():
        raise FileNotFoundError(f'GT 根目录不存在: {gt_root}')

    plan = parse_plan(plan_path)
    seq_dirs = collect_tracking_sequences(data_root, args.seqs)
    seq_names = [p.name for p in seq_dirs]
    validate_gt_sequences(gt_root, seq_names)


    base_config = dict(DEFAULT_SWEEP_CONFIG)
    tracker_base_kwargs = {
        'yolo_model_path': str(model_path),
        'conf_threshold': base_config['conf_threshold'],
        'imgsz': base_config['imgsz'],
        'target_classes': [0],
        'appearance_model': 'osnet',
        'appearance_weights': str(appearance_weights),
        'device': args.device,
        'use_ecc': True,
        'use_aflink': True,
        'use_spatial_limit': True,
        'ecc_scale': 0.75,
        'match_threshold': base_config['match_threshold'],
        'n_init': base_config['n_init'],
        'max_age': base_config['max_age'],
        'max_feature_queue': base_config['max_feature_queue'],
        'spatial_max_dist': base_config['spatial_max_dist'],
        'aflink_max_gap': base_config['aflink_max_gap'],
        'aflink_sim': base_config['aflink_sim'],
        'aflink_dist': base_config['aflink_dist'],
    }

    print('=' * 80)
    print('非规则 YOLO + OSNet + AFLink + ECC 灵敏性分析')
    print('=' * 80)
    print(f'计划文件: {plan_path}')
    print(f'序列数: {len(seq_dirs)}')
    print(f'输出根目录: {output_root}')
    print('=' * 80)

    summary_csv = output_root / 'sensitivity_summary.csv'
    records_by_run: OrderedDict[str, dict[str, str | float | int]] = OrderedDict()
    total_runs = sum(len(values) for values in plan.values())
    run_idx = 0

    for param_name, values in plan.items():
        for value in values:
            run_idx += 1
            tracker_kwargs = dict(tracker_base_kwargs)
            tracker_kwargs[param_name] = value
            display_value = int(value) if isinstance(value, float) and value.is_integer() else value
            run_name = f'{param_name}_{sanitize_value(value)}'
            results_dir = output_root / run_name
            eval_dir = output_root / 'trackeval' / run_name
            metrics_csv = eval_dir / 'metrics_summary.csv'

            print(f'\n[{run_idx}/{total_runs}] {param_name} = {display_value}')
            print(f'  输出结果目录: {results_dir}')

            results_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and metrics_csv.exists():
                print('  检测到已有评测结果，直接读取指标。')
                metrics = load_combined_metrics_from_csv(metrics_csv)
            else:
                if args.skip_existing and all_sequences_completed(results_dir, seq_names):
                    print('  当前参数下所有序列结果已存在，直接计算指标。')
                else:
                    tracker = TrackerAblation(**tracker_kwargs)
                    added_records = 0
                    skipped_sequences = 0
                    for seq_idx, seq_dir in enumerate(seq_dirs, 1):
                        output_file = results_dir / f'{seq_dir.name}.txt'
                        if args.skip_existing and has_completed_sequence_output(output_file):
                            skipped_sequences += 1
                            print(f'  [{seq_idx}/{len(seq_dirs)}] 跳过已完成序列: {seq_dir.name}')
                            continue

                        print(f'  [{seq_idx}/{len(seq_dirs)}] 处理序列: {seq_dir.name}')
                        added_records += process_sequence_with_tracker(tracker, seq_dir / 'img1', output_file)

                    if skipped_sequences:
                        print(f'  已复用序列: {skipped_sequences}/{len(seq_dirs)}')
                    print(f'  本次新增记录数: {added_records}')

                missing_sequences = [
                    seq_name for seq_name in seq_names
                    if not has_completed_sequence_output(results_dir / f'{seq_name}.txt')
                ]
                if missing_sequences:
                    raise RuntimeError(f'以下序列结果仍未完成，无法评测: {missing_sequences}')

                cleaned_sequences, duplicate_rows = sanitize_results_dir(results_dir, seq_names)
                if duplicate_rows:
                    print(f'  已清理重复 ID 行: {duplicate_rows} 行，涉及 {len(cleaned_sequences)} 个序列')

                metrics = run_trackeval(
                    results_dir=results_dir,
                    gt_root=gt_root,
                    tracker_name=run_name,
                    seqs=seq_names,
                    output_dir=eval_dir,
                )


            record = {
                'parameter': param_name,
                'value': display_value,
                'HOTA': metrics['HOTA'],
                'AssA': metrics['AssA'],
                'DetA': metrics['DetA'],
                'MOTA': metrics['MOTA'],
                'results_dir': str(results_dir),
            }
            records_by_run[run_name] = record
            update_progress_outputs(records_by_run.values(), summary_csv, report_path, len(seq_dirs), base_config)
            print(
                f"  HOTA={metrics['HOTA']:.3f} | AssA={metrics['AssA']:.3f} | "
                f"DetA={metrics['DetA']:.3f} | MOTA={metrics['MOTA']:.3f}"
            )
            print(f'  已完成实验数: {len(records_by_run)}/{total_runs}')
            print(f'  汇总 CSV 已更新: {summary_csv}')

    records = list(records_by_run.values())

    print('\n' + '=' * 80)
    print('灵敏性分析完成')
    print('=' * 80)
    for record in records:
        print(
            f"{record['parameter']}={record['value']} -> "
            f"HOTA={record['HOTA']:.3f}, AssA={record['AssA']:.3f}, "
            f"DetA={record['DetA']:.3f}, MOTA={record['MOTA']:.3f}"
        )
    print('-' * 80)
    print(f'汇总 CSV: {summary_csv}')
    print(f'Markdown 报告: {report_path}')
    print('=' * 80)



if __name__ == '__main__':
    main()
