## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `adaptive_tri`（方案1+2：自适应时空门控 + 三因子规则评分）
- adaptive_len_ref: `30`
- adaptive_gap_min_ratio: `0.50`
- adaptive_dist_min_ratio: `0.50`
- gate_reliability_weight: `0.700`
- gate_shape_weight: `0.300`
- shape_area_weight: `0.200`
- shape_aspect_weight: `0.800`
- temporal_weight: `0.400`
- distance_weight: `0.400`
- shape_weight: `0.200`
- avg_rule_score: `0.498`
- avg_temporal_score: `0.419`
- avg_distance_score: `0.536`
- avg_shape_score: `0.581`
- avg_length_confidence: `0.817`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | area_w | aspect_w | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | adaptive_tri | shape_mix | 100 | 200 | 0.200 | 0.800 | 62.911 | 70.133 | 56.557 | 85.147 | 74.916 | 1965 |
| 2 | rule | adaptive_tri | shape_mix | 100 | 200 | 0.400 | 0.600 | 62.866 | 70.124 | 56.485 | 85.146 | 74.855 | 1969 |
| 3 | rule | adaptive_tri | shape_mix | 100 | 200 | 0.600 | 0.400 | 62.826 | 70.119 | 56.419 | 85.145 | 74.807 | 1974 |
| 4 | rule | adaptive_tri | shape_mix | 100 | 200 | 0.800 | 0.200 | 62.787 | 70.119 | 56.349 | 85.145 | 74.741 | 1976 |

### 最佳配置分析

本次扫描目标为 **shape_score 权重扫描（面积比例 × 长宽比）**。

最佳配置为 **mode=rule / variant=adaptive_tri / max_gap=100 / dist_thresh=200**。

当前规则变体为 **方案1+2：自适应时空门控 + 三因子规则评分**。

从搜索区间看，max_gap 固定为 100，dist_thresh 固定为 200，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.498**，平均时间项为 **0.419**，平均距离项为 **0.536**，平均形态项为 **0.581**，平均长度置信度为 **0.817**。

候选筛选采用自适应时空门控，`adaptive_len_ref=30`、`adaptive_gap_min_ratio=0.50`、`adaptive_dist_min_ratio=0.50`；门控信号中的可靠性/形态权重分别为 `0.700` / `0.300`，短碎片与形态不连续候选会被更严格地过滤，从而优先改善 **AssA / IDF1**。

形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `0.200` / `0.800`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。

规则排序采用三因子加权，时间/距离/形态权重分别为 `0.400` / `0.400` / `0.200`；这会让形态连续性直接参与链接优先级，降低跨目标误配。

本次分析的预测来源为 **MOT 结果目录回退 (E:\大创\Fussball\Fussball\tracker_results\sensitivity_aflink_local\aflink_max_gap_60)**。
