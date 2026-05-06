## 最优参数 Adaptive-Tri AFLink 运行报告

### 运行配置

- tracker_name: `best_adaptive_tri_rule`
- sequence_count: `49`
- source: `MOT 结果目录回退 (E:\大创\Fussball\Fussball\tracker_results\sensitivity_aflink_local\aflink_max_gap_60)`
- rule_variant: `adaptive_tri`（方案1+2：自适应时空门控 + 三因子规则评分）
- max_gap: `100`
- dist_thresh: `200`
- adaptive_len_ref: `30`
- adaptive_gap_min_ratio: `0.50`
- adaptive_dist_min_ratio: `0.50`
- gate_reliability_weight: `0.900`
- gate_shape_weight: `0.100`
- temporal_weight: `0.500`
- distance_weight: `0.300`
- shape_weight: `0.200`
- shape_area_weight: `0.200`
- shape_aspect_weight: `0.800`

### 评测结果

- HOTA: `63.004`
- DetA: `70.140`
- AssA: `56.720`
- MOTA: `85.148`
- IDF1: `75.095`
- IDs: `1960`

### 候选统计

- raw_candidates: `1487`
- kept_candidates: `1487`
- merged_tracks: `411`
- avg_rule_score: `0.482`
- avg_temporal_score: `0.417`
- avg_distance_score: `0.535`
- avg_shape_score: `0.565`
- avg_length_confidence: `0.846`
