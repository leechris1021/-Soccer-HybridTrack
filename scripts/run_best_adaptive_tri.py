from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fussball.project import TRACKER_RESULTS_ROOT
from scripts.aflink_hybrid_eval import (
    BEST_TUNED_CONFIG,
    DEFAULT_CACHE_PATH,
    DEFAULT_GT_ROOT,
    DEFAULT_RESULTS_DIR,
    HybridAFLink,
    apply_aflink,
    collect_sequences,
    get_rule_variant_label,
    load_prediction_source,
    run_trackeval,
    write_predictions_to_dir,
)

DEFAULT_OUTPUT_ROOT = TRACKER_RESULTS_ROOT / 'best_adaptive_tri_rule'
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / 'best_adaptive_tri_report.md'
DEFAULT_TRACKER_NAME = 'best_adaptive_tri_rule'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='单独运行最优参数的自适应门控 + 三因子规则 AFLink。')
    parser.add_argument('--results_dir', type=str, default=str(DEFAULT_RESULTS_DIR), help='原始 MOT 结果目录。')
    parser.add_argument('--gt_root', type=str, default=str(DEFAULT_GT_ROOT), help='GT 根目录。')
    parser.add_argument('--cache_path', type=str, default=str(DEFAULT_CACHE_PATH), help='raw_pred_cache.json 路径。')
    parser.add_argument('--output_root', type=str, default=str(DEFAULT_OUTPUT_ROOT), help='后处理结果与评测输出根目录。')
    parser.add_argument('--report_path', type=str, default=str(DEFAULT_REPORT_PATH), help='Markdown 报告保存路径。')
    parser.add_argument('--tracker_name', type=str, default=DEFAULT_TRACKER_NAME, help='TrackEval 中显示的 tracker 名称。')
    parser.add_argument('--seqs', nargs='+', default=None, help='只处理指定序列；默认按 49 个测试序列全量执行。')
    return parser.parse_args()



def summarize_stats(stats: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    candidate_total = int(sum(seq_stat.get('raw_candidates', 0) for seq_stat in stats.values()))
    kept_total = int(sum(seq_stat.get('kept_candidates', 0) for seq_stat in stats.values()))
    merged_total = int(sum(seq_stat.get('merged_tracks', 0) for seq_stat in stats.values()))
    rule_score_sum = float(sum(seq_stat.get('rule_score_sum', 0.0) for seq_stat in stats.values()))
    temporal_score_sum = float(sum(seq_stat.get('temporal_score_sum', 0.0) for seq_stat in stats.values()))
    distance_score_sum = float(sum(seq_stat.get('distance_score_sum', 0.0) for seq_stat in stats.values()))
    shape_score_sum = float(sum(seq_stat.get('shape_score_sum', 0.0) for seq_stat in stats.values()))
    length_confidence_sum = float(sum(seq_stat.get('length_confidence_sum', 0.0) for seq_stat in stats.values()))

    return {
        'raw_candidates': candidate_total,
        'kept_candidates': kept_total,
        'merged_tracks': merged_total,
        'avg_rule_score': rule_score_sum / candidate_total if candidate_total > 0 else 0.0,
        'avg_temporal_score': temporal_score_sum / candidate_total if candidate_total > 0 else 0.0,
        'avg_distance_score': distance_score_sum / candidate_total if candidate_total > 0 else 0.0,
        'avg_shape_score': shape_score_sum / candidate_total if candidate_total > 0 else 0.0,
        'avg_length_confidence': length_confidence_sum / candidate_total if candidate_total > 0 else 0.0,
    }



def build_linker() -> HybridAFLink:
    return HybridAFLink(
        mode='rule',
        max_gap=int(BEST_TUNED_CONFIG['max_gap']),
        dist_thresh=float(BEST_TUNED_CONFIG['dist_thresh']),
        rule_variant=str(BEST_TUNED_CONFIG['rule_variant']),
        adaptive_len_ref=int(BEST_TUNED_CONFIG['adaptive_len_ref']),
        adaptive_gap_min_ratio=float(BEST_TUNED_CONFIG['adaptive_gap_min_ratio']),
        adaptive_dist_min_ratio=float(BEST_TUNED_CONFIG['adaptive_dist_min_ratio']),
        gate_reliability_weight=float(BEST_TUNED_CONFIG['gate_reliability_weight']),
        gate_shape_weight=float(BEST_TUNED_CONFIG['gate_shape_weight']),
        area_weight=float(BEST_TUNED_CONFIG['shape_area_weight']),
        aspect_weight=float(BEST_TUNED_CONFIG['shape_aspect_weight']),
        temporal_weight=float(BEST_TUNED_CONFIG['temporal_weight']),
        distance_weight=float(BEST_TUNED_CONFIG['distance_weight']),
        shape_weight=float(BEST_TUNED_CONFIG['shape_weight']),
    )



def write_report(
    report_path: Path,
    tracker_name: str,
    seq_count: int,
    source_desc: str,
    metrics: dict[str, float],
    summary: dict[str, float | int],
) -> None:
    cfg = BEST_TUNED_CONFIG
    content = '\n'.join(
        [
            '## 最优参数 Adaptive-Tri AFLink 运行报告',
            '',
            '### 运行配置',
            '',
            f'- tracker_name: `{tracker_name}`',
            f'- sequence_count: `{seq_count}`',
            f'- source: `{source_desc}`',
            f"- rule_variant: `{cfg['rule_variant']}`（{get_rule_variant_label(str(cfg['rule_variant']))}）",
            f"- max_gap: `{int(cfg['max_gap'])}`",
            f"- dist_thresh: `{int(cfg['dist_thresh'])}`",
            f"- adaptive_len_ref: `{int(cfg['adaptive_len_ref'])}`",
            f"- adaptive_gap_min_ratio: `{float(cfg['adaptive_gap_min_ratio']):.2f}`",
            f"- adaptive_dist_min_ratio: `{float(cfg['adaptive_dist_min_ratio']):.2f}`",
            f"- gate_reliability_weight: `{float(cfg['gate_reliability_weight']):.3f}`",
            f"- gate_shape_weight: `{float(cfg['gate_shape_weight']):.3f}`",
            f"- temporal_weight: `{float(cfg['temporal_weight']):.3f}`",
            f"- distance_weight: `{float(cfg['distance_weight']):.3f}`",
            f"- shape_weight: `{float(cfg['shape_weight']):.3f}`",
            f"- shape_area_weight: `{float(cfg['shape_area_weight']):.3f}`",
            f"- shape_aspect_weight: `{float(cfg['shape_aspect_weight']):.3f}`",
            '',
            '### 评测结果',
            '',
            f"- HOTA: `{float(metrics['HOTA']):.3f}`",
            f"- DetA: `{float(metrics['DetA']):.3f}`",
            f"- AssA: `{float(metrics['AssA']):.3f}`",
            f"- MOTA: `{float(metrics['MOTA']):.3f}`",
            f"- IDF1: `{float(metrics['IDF1']):.3f}`",
            f"- IDs: `{float(metrics['IDs']):.0f}`",
            '',
            '### 候选统计',
            '',
            f"- raw_candidates: `{int(summary['raw_candidates'])}`",
            f"- kept_candidates: `{int(summary['kept_candidates'])}`",
            f"- merged_tracks: `{int(summary['merged_tracks'])}`",
            f"- avg_rule_score: `{float(summary['avg_rule_score']):.3f}`",
            f"- avg_temporal_score: `{float(summary['avg_temporal_score']):.3f}`",
            f"- avg_distance_score: `{float(summary['avg_distance_score']):.3f}`",
            f"- avg_shape_score: `{float(summary['avg_shape_score']):.3f}`",
            f"- avg_length_confidence: `{float(summary['avg_length_confidence']):.3f}`",
            '',
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding='utf-8')



def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    gt_root = Path(args.gt_root).resolve()
    cache_path = Path(args.cache_path).resolve()
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.report_path).resolve()
    tracker_name = str(args.tracker_name)

    print('=' * 80)
    print('最优参数 Adaptive-Tri AFLink 单独运行')
    print('=' * 80)
    print(f"tracker_name: {tracker_name}")
    print(f"rule_variant: {BEST_TUNED_CONFIG['rule_variant']} ({get_rule_variant_label(str(BEST_TUNED_CONFIG['rule_variant']))})")
    print(f"max_gap: {BEST_TUNED_CONFIG['max_gap']} | dist_thresh: {BEST_TUNED_CONFIG['dist_thresh']}")
    print(
        'gate: '
        f"{BEST_TUNED_CONFIG['gate_reliability_weight']:.2f}/{BEST_TUNED_CONFIG['gate_shape_weight']:.2f} | "
        'rule: '
        f"{BEST_TUNED_CONFIG['temporal_weight']:.2f}/{BEST_TUNED_CONFIG['distance_weight']:.2f}/{BEST_TUNED_CONFIG['shape_weight']:.2f} | "
        'shape: '
        f"{BEST_TUNED_CONFIG['shape_area_weight']:.2f}/{BEST_TUNED_CONFIG['shape_aspect_weight']:.2f}"
    )
    print('=' * 80)

    seqs = collect_sequences(results_dir, gt_root, args.seqs)
    predictions, source_desc = load_prediction_source(cache_path, results_dir, seqs)

    linker = build_linker()
    processed, stats = apply_aflink(predictions, linker)

    post_dir = output_root / 'postprocessed' / tracker_name
    eval_dir = output_root / 'trackeval' / tracker_name
    write_predictions_to_dir(processed, post_dir)

    metrics = run_trackeval(
        results_dir=post_dir,
        gt_root=gt_root,
        seqs=seqs,
        tracker_name=tracker_name,
        output_dir=eval_dir,
    )
    summary = summarize_stats(stats)
    write_report(report_path, tracker_name, len(seqs), source_desc, metrics, summary)

    print('\n运行完成：')
    print(
        f"HOTA={metrics['HOTA']:.3f} | DetA={metrics['DetA']:.3f} | AssA={metrics['AssA']:.3f} | "
        f"MOTA={metrics['MOTA']:.3f} | IDF1={metrics['IDF1']:.3f} | IDs={metrics['IDs']:.0f}"
    )
    print(
        f"candidates={summary['raw_candidates']} -> kept={summary['kept_candidates']} -> merged_tracks={summary['merged_tracks']}"
    )
    print(f'后处理结果目录: {post_dir}')
    print(f'评测输出目录: {eval_dir}')
    print(f'报告路径: {report_path}')


if __name__ == '__main__':
    main()
