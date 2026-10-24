## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `adaptive_gate`（方案1：自适应时空门控）
- adaptive_len_ref: `30`
- adaptive_gap_min_ratio: `0.50`
- adaptive_dist_min_ratio: `0.50`
- adaptive_gap_max_ratio: `1.00`
- adaptive_dist_max_ratio: `1.00`
- gate_signal_ref: `None（单侧映射）`
- gate_signal_gain: `1.000`
- gate_reliability_weight: `0.300`
- gate_shape_weight: `0.700`
- shape_area_weight: `0.600`
- shape_aspect_weight: `0.400`
- avg_rule_score: `0.610`
- avg_temporal_score: `0.579`
- avg_distance_score: `0.642`
- avg_shape_score: `0.553`
- avg_length_confidence: `0.793`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | gate_rel_w | gate_shape_w | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | adaptive_gate | gate_mix | 250 | 450 | 0.300 | 0.700 | 63.115 | 70.135 | 56.919 | 85.172 | 76.375 | 1823 |
| 2 | rule | adaptive_gate | gate_mix | 250 | 450 | 0.100 | 0.900 | 63.099 | 70.136 | 56.890 | 85.172 | 76.280 | 1824 |
| 3 | rule | adaptive_gate | gate_mix | 250 | 450 | 0.500 | 0.500 | 63.074 | 70.134 | 56.845 | 85.172 | 76.353 | 1820 |
| 4 | rule | adaptive_gate | gate_mix | 250 | 450 | 0.900 | 0.100 | 63.056 | 70.134 | 56.814 | 85.173 | 76.374 | 1818 |
| 5 | rule | adaptive_gate | gate_mix | 250 | 450 | 0.700 | 0.300 | 63.036 | 70.134 | 56.777 | 85.172 | 76.268 | 1820 |

### 最佳配置分析

本次扫描目标为 **gate_signal 权重扫描（可靠性 × 形态）**。

最佳配置为 **mode=rule / variant=adaptive_gate / max_gap=250 / dist_thresh=450**。

当前规则变体为 **方案1：自适应时空门控**。

从搜索区间看，max_gap 固定为 250，dist_thresh 固定为 450，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.610**，平均时间项为 **0.579**，平均距离项为 **0.642**，平均形态项为 **0.553**，平均长度置信度为 **0.793**。

候选筛选采用自适应时空门控，`adaptive_len_ref=30`、`adaptive_gap_min_ratio=0.50`、`adaptive_dist_min_ratio=0.50`；门控信号中的可靠性/形态权重分别为 `0.300` / `0.700`，短碎片与形态不连续候选会被更严格地过滤，从而优先改善 **AssA / IDF1**。

形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `0.600` / `0.400`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。

本次分析的预测来源为 **MOT 结果目录回退 (outputs/tracker_results/aflink_max_gap_60)**。
