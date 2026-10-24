## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `tri_factor`（方案2：三因子规则评分）
- shape_area_weight: `0.600`
- shape_aspect_weight: `0.400`
- temporal_weight: `0.400`
- distance_weight: `0.400`
- shape_weight: `0.200`
- avg_rule_score: `0.525`
- avg_temporal_score: `0.499`
- avg_distance_score: `0.559`
- avg_shape_score: `0.509`
- avg_length_confidence: `0.734`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | tri_factor | gap_dist | 250 | 450 | 62.534 | 70.123 | 55.888 | 85.171 | 75.781 | 1826 |

### 最佳配置分析

本次扫描目标为 **max_gap × dist_thresh 网格搜索**。

最佳配置为 **mode=rule / variant=tri_factor / max_gap=250 / dist_thresh=450**。

当前规则变体为 **方案2：三因子规则评分**。

从搜索区间看，max_gap 固定为 250，dist_thresh 固定为 450，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.525**，平均时间项为 **0.499**，平均距离项为 **0.559**，平均形态项为 **0.509**，平均长度置信度为 **0.734**。

形态连续性由面积比例与长宽比相似度共同构成，其权重分别为 `0.600` / `0.400`；这决定了 `shape_score` 更偏向尺寸一致还是外形一致。

规则排序采用三因子加权，时间/距离/形态权重分别为 `0.400` / `0.400` / `0.200`；这会让形态连续性直接参与链接优先级，降低跨目标误配。

当前仍使用原始线性 gap 项，因此结果可直接作为旧版规则 AFLink 的 baseline。

本次分析的预测来源为 **MOT 结果目录回退 (outputs/tracker_results/aflink_max_gap_60)**。
