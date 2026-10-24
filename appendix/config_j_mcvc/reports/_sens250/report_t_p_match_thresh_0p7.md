## 规则 AFLink 混合版灵敏性分析报告

### 当前规则配置

- mode: `rule`
- variant: `baseline`（原始规则）
- avg_rule_score: `0.544`
- avg_temporal_score: `0.515`
- avg_distance_score: `0.573`
- avg_shape_score: `0.522`
- avg_length_confidence: `0.751`

### 排序结果（按 HOTA 降序）

| Rank | mode | variant | sweep | max_gap | dist_thresh | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rule | baseline | gap_dist | 250 | 450 | 64.306 | 70.134 | 59.082 | 85.181 | 78.191 | 1770 |

### 最佳配置分析

本次扫描目标为 **max_gap × dist_thresh 网格搜索**。

最佳配置为 **mode=rule / variant=baseline / max_gap=250 / dist_thresh=450**。

当前规则变体为 **原始规则**。

从搜索区间看，max_gap 固定为 250，dist_thresh 固定为 450，说明它在**召回较长时间中断**与**抑制跨目标误连接**之间取得了更均衡的折中。

候选层平均规则分数为 **0.544**，平均时间项为 **0.515**，平均距离项为 **0.573**，平均形态项为 **0.522**，平均长度置信度为 **0.751**。

当前仍使用原始线性 gap 项，因此结果可直接作为旧版规则 AFLink 的 baseline。

本次分析的预测来源为 **MOT 结果目录回退 (outputs/tracker_results/table346_sweep_optimal_test/match_thresh_0p7)**。
