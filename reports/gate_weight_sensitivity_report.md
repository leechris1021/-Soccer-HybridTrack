## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `adaptive_tri`（方案1+2：自适应时空门控 + 三因子规则评分）
- adaptive_len_ref: `30`
- adaptive_gap_min_ratio: `0.50`
- adaptive_dist_min_ratio: `0.50`
- gate_reliability_weight: `0.900`
- gate_shape_weight: `0.100`
- shape_area_weight: `0.600`
- shape_aspect_weight: `0.400`
- temporal_weight: `0.400`
- distance_weight: `0.400`
- shape_weight: `0.200`
- avg_rule_score: `0.489`
- avg_temporal_score: `0.416`
- avg_distance_score: `0.536`
- avg_shape_score: `0.540`
- avg_length_confidence: `0.848`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | gate_rel_w | gate_shape_w | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | adaptive_tri | gate_mix | 100 | 200 | 0.900 | 0.100 | 62.850 | 70.119 | 56.461 | 85.145 | 74.896 | 1972 |
| 2 | rule | adaptive_tri | gate_mix | 100 | 200 | 0.700 | 0.300 | 62.826 | 70.119 | 56.419 | 85.145 | 74.807 | 1974 |
| 3 | rule | adaptive_tri | gate_mix | 100 | 200 | 0.500 | 0.500 | 62.812 | 70.120 | 56.393 | 85.145 | 74.736 | 1976 |
| 4 | rule | adaptive_tri | gate_mix | 100 | 200 | 0.300 | 0.700 | 62.726 | 70.127 | 56.233 | 85.144 | 74.550 | 1980 |
| 5 | rule | adaptive_tri | gate_mix | 100 | 200 | 0.100 | 0.900 | 62.677 | 70.144 | 56.133 | 85.143 | 74.387 | 1988 |

### 最佳配置分析

本次扫描目标为 **gate_signal 权重扫描（可靠性 × 形态）**。

最佳配置为 **mode=rule / variant=adaptive_tri / max_gap=100 / dist_thresh=200**。

当前规则变体为 **方案1+2：自适应时空门控 + 三因子规则评分**。

从搜索区间看，max_gap 固定为 100，dist_thresh 固定为 200，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.489**，平均时间项为 **0.416**，平均距离项为 **0.536**，平均形态项为 **0.540**，平均长度置信度为 **0.848**。

候选筛选采用自适应时空门控，`adaptive_len_ref=30`、`adaptive_gap_min_ratio=0.50`、`adaptive_dist_min_ratio=0.50`；门控信号中的可靠性/形态权重分别为 `0.900` / `0.100`，短碎片与形态不连续候选会被更严格地过滤，从而优先改善 **AssA / IDF1**。

形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `0.600` / `0.400`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。

规则排序采用三因子加权，时间/距离/形态权重分别为 `0.400` / `0.400` / `0.200`；这会让形态连续性直接参与链接优先级，降低跨目标误配。

本次分析的预测来源为 **MOT 结果目录回退 (E:\大创\Fussball\Fussball\tracker_results\sensitivity_aflink_local\aflink_max_gap_60)**。
