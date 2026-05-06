from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from shutil import copy2

import numpy as np


# 兼容旧版 TrackEval 对已废弃 numpy 类型别名的使用
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

HELP_REQUESTED = any(arg in {'-h', '--help'} for arg in sys.argv[1:])

try:
    import trackeval
except ModuleNotFoundError as exc:
    trackeval = None
    _TRACKEVAL_IMPORT_ERROR = exc
    if not HELP_REQUESTED:
        pass
else:
    _TRACKEVAL_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fussball.project import DEFAULT_TEST_DATA_ROOT, TRACKER_RESULTS_ROOT, load_default_test_sequence_names

DEFAULT_RESULTS_DIR = TRACKER_RESULTS_ROOT / 'sensitivity_aflink_local' / 'aflink_max_gap_60'
DEFAULT_GT_ROOT = DEFAULT_TEST_DATA_ROOT


def ensure_trackeval_available(context: str = '该功能') -> None:
    if _TRACKEVAL_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f'{context} 需要安装 `trackeval`。请先执行 `pip install -r requirements.txt`。'
        ) from _TRACKEVAL_IMPORT_ERROR


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='使用 TrackEval 评估跟踪结果，并输出 HOTA / DetA / AssA / MOTA。'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help='跟踪结果目录，内部应为每个序列一个 txt 文件。',
    )
    parser.add_argument(
        '--gt_root',
        type=str,
        default=str(DEFAULT_GT_ROOT),
        help='GT 根目录，结构应为 <gt_root>/<seq>/gt/gt.txt。',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='TrackEval 输出目录，默认写入 tracker_results/trackeval_<results_dir名>。',
    )
    parser.add_argument(
        '--tracker_name',
        type=str,
        default=None,
        help='TrackEval 中显示的 tracker 名称，默认使用结果目录名。',
    )
    parser.add_argument(
        '--seqs',
        nargs='+',
        default=None,
        help='仅评估指定序列；默认按 SNMOT 测试集 49 序列全量评估。',

    )
    return parser.parse_args()



def get_seq_length(gt_root: Path, seq: str) -> int:
    seqinfo_path = gt_root / seq / 'seqinfo.ini'
    if not seqinfo_path.exists():
        raise FileNotFoundError(f'缺少 seqinfo.ini: {seqinfo_path}')

    for line in seqinfo_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('seqLength='):
            return int(line.split('=', 1)[1].strip())

    raise ValueError(f'在 {seqinfo_path} 中未找到 seqLength')



def collect_sequences(results_dir: Path, gt_root: Path, requested_seqs: list[str] | None) -> list[str]:
    result_map = {p.stem: p for p in sorted(results_dir.glob('*.txt'))}
    if not result_map:
        raise FileNotFoundError(f'结果目录中未找到 txt 文件: {results_dir}')

    if requested_seqs:
        seqs = requested_seqs
    else:
        seqs = load_default_test_sequence_names(gt_root)


    valid_seqs: list[str] = []
    missing_results: list[str] = []
    missing_gt: list[str] = []

    for seq in seqs:
        if seq not in result_map:
            missing_results.append(seq)
            continue
        if not (gt_root / seq / 'gt' / 'gt.txt').exists():
            missing_gt.append(seq)
            continue
        if not (gt_root / seq / 'seqinfo.ini').exists():
            missing_gt.append(seq)
            continue
        valid_seqs.append(seq)

    if missing_results:
        raise FileNotFoundError(f'以下结果文件不存在: {missing_results}')
    if missing_gt:
        raise FileNotFoundError(f'以下序列缺少 GT 或 seqinfo.ini: {missing_gt}')
    if not valid_seqs:
        raise RuntimeError('没有可评估的序列。')

    return valid_seqs



def prepare_tracker_layout(results_dir: Path, tracker_name: str, seqs: list[str], temp_root: Path) -> Path:
    tracker_data_dir = temp_root / tracker_name / 'data'
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    for seq in seqs:
        src = results_dir / f'{seq}.txt'
        dst = tracker_data_dir / f'{seq}.txt'
        copy2(src, dst)

    return tracker_data_dir



def write_seqmap(seqs: list[str], seqmap_path: Path) -> None:
    lines = ['name', *seqs]
    seqmap_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')



def build_eval_components(gt_root: Path, trackers_root: Path, tracker_name: str, seqs: list[str], output_dir: Path):
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



def extract_summary_scores(output_res: dict, dataset_name: str, tracker_name: str) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    tracker_res = output_res[dataset_name][tracker_name]

    combined = tracker_res['COMBINED_SEQ']['pedestrian']
    combined_scores = {
        'HOTA': 100.0 * float(np.mean(combined['HOTA']['HOTA'])),
        'DetA': 100.0 * float(np.mean(combined['HOTA']['DetA'])),
        'AssA': 100.0 * float(np.mean(combined['HOTA']['AssA'])),
        'MOTA': 100.0 * float(combined['CLEAR']['MOTA']),
    }

    per_sequence_scores: list[dict[str, float | str]] = []
    for seq, seq_res in tracker_res.items():
        if seq == 'COMBINED_SEQ':
            continue
        per_sequence_scores.append({
            'sequence': seq,
            'HOTA': 100.0 * float(np.mean(seq_res['pedestrian']['HOTA']['HOTA'])),
            'DetA': 100.0 * float(np.mean(seq_res['pedestrian']['HOTA']['DetA'])),
            'AssA': 100.0 * float(np.mean(seq_res['pedestrian']['HOTA']['AssA'])),
            'MOTA': 100.0 * float(seq_res['pedestrian']['CLEAR']['MOTA']),
        })

    per_sequence_scores.sort(key=lambda item: str(item['sequence']))
    return combined_scores, per_sequence_scores



def save_csv(per_sequence_scores: list[dict[str, float | str]], combined_scores: dict[str, float], output_dir: Path) -> Path:
    csv_path = output_dir / 'metrics_summary.csv'
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'HOTA', 'DetA', 'AssA', 'MOTA'])
        writer.writeheader()
        for row in per_sequence_scores:
            writer.writerow({
                'sequence': row['sequence'],
                'HOTA': f"{row['HOTA']:.3f}",
                'DetA': f"{row['DetA']:.3f}",
                'AssA': f"{row['AssA']:.3f}",
                'MOTA': f"{row['MOTA']:.3f}",
            })
        writer.writerow({
            'sequence': 'COMBINED',
            'HOTA': f"{combined_scores['HOTA']:.3f}",
            'DetA': f"{combined_scores['DetA']:.3f}",
            'AssA': f"{combined_scores['AssA']:.3f}",
            'MOTA': f"{combined_scores['MOTA']:.3f}",
        })

    return csv_path



def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir).resolve()
    gt_root = Path(args.gt_root).resolve()
    tracker_name = args.tracker_name or results_dir.name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (TRACKER_RESULTS_ROOT / f'trackeval_{results_dir.name}')

    if not results_dir.exists():
        raise FileNotFoundError(f'结果目录不存在: {results_dir}')
    if not gt_root.exists():
        raise FileNotFoundError(f'GT 根目录不存在: {gt_root}')

    seqs = collect_sequences(results_dir, gt_root, args.seqs)

    with tempfile.TemporaryDirectory(prefix='trackeval_local_') as temp_dir:
        temp_root = Path(temp_dir)
        prepare_tracker_layout(results_dir, tracker_name, seqs, temp_root)
        seqmap_path = temp_root / 'seqmap.txt'
        write_seqmap(seqs, seqmap_path)

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
        csv_path = save_csv(per_sequence_scores, combined_scores, output_dir)

    print('=' * 60)
    print(f'Tracker: {tracker_name}')
    print(f'评估序列数: {len(seqs)}')
    print(f'结果目录: {results_dir}')
    print(f'GT 目录: {gt_root}')
    print('-' * 60)
    print(f"HOTA : {combined_scores['HOTA']:.3f}")
    print(f"DetA : {combined_scores['DetA']:.3f}")
    print(f"AssA : {combined_scores['AssA']:.3f}")
    print(f"MOTA : {combined_scores['MOTA']:.3f}")
    print('-' * 60)
    print(f'CSV 已保存: {csv_path}')
    print(f'TrackEval 明细目录: {output_dir / tracker_name}')
    print('=' * 60)


if __name__ == '__main__':
    main()
