## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `adaptive_cone`（方案6：联合锥形时空门控（dist 随 gap 线性收紧））
- adaptive_len_ref: `30`
- adaptive_gap_min_ratio: `0.50`
- adaptive_dist_min_ratio: `0.50`
- adaptive_gap_max_ratio: `1.00`
- adaptive_dist_max_ratio: `1.00`
- gate_signal_ref: `None（单侧映射）`
- gate_signal_gain: `1.000`
- gate_reliability_weight: `0.700`
- gate_shape_weight: `0.300`
- shape_area_weight: `0.600`
- shape_aspect_weight: `0.400`
- avg_rule_score: `0.541`
- avg_temporal_score: `0.423`
- avg_distance_score: `0.660`
- avg_shape_score: `0.543`
- avg_length_confidence: `0.745`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | adaptive_cone | gap_dist | 150 | 450 | 64.516 | 71.512 | 58.285 | 88.048 | 79.763 | 2225 |

### 最佳配置分析

本次扫描目标为 **max_gap × dist_thresh 网格搜索**。

最佳配置为 **mode=rule / variant=adaptive_cone / max_gap=150 / dist_thresh=450**。

当前规则变体为 **方案6：联合锥形时空门控（dist 随 gap 线性收紧）**。

从搜索区间看，max_gap 固定为 150，dist_thresh 固定为 450，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.541**，平均时间项为 **0.423**，平均距离项为 **0.660**，平均形态项为 **0.543**，平均长度置信度为 **0.745**。

候选筛选采用自适应时空门控，`adaptive_len_ref=30`、`adaptive_gap_min_ratio=0.50`、`adaptive_dist_min_ratio=0.50`；门控信号中的可靠性/形态权重分别为 `0.700` / `0.300`，短碎片与形态不连续候选会被更严格地过滤，从而优先改善 **AssA / IDF1**。

形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `0.600` / `0.400`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。

本次分析的预测来源为 **MOT 结果目录回退 (outputs/tracker_results/aflink_max_gap_60_train)**。
