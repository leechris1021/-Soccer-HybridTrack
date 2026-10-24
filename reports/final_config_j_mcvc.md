# 最终方案评测汇总 —— Config J / MCVC

> 本文件是**论文最终结果在 `reports/` 的入口**。
> 原始逐配置报告与敏感性扫描见 `appendix/config_j_mcvc/reports/_sens250/sens250.md`（论文 Table 3-9）
> 与 `appendix/config_j_mcvc/reports/_sens250/mcv_boot.md`（层入选判定）。

## 1. 方案

**Config J = MCVC（multi-cue veto cascade）**：名义窗口由 `150/450` 放宽到 `250/450`，叠加**队色等价类否决**（`kit_veto=any`）；自适应门控不显著，**退役**。

权威定义：`appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py --mcv`
（= `--gate_score_norm nominal` + `--kit_veto any`）；可读快照 `configs/optimal_config.json::paper_final`。

| 项 | 值 |
|---|---|
| 关联规则 | `rule_variant='baseline'`：两因子时间/距离评分 + 贪心，**无神经网络** |
| 名义窗口 | `max_gap=250` / `dist_thresh=450` |
| 分量权重 | `temporal 0.4` / `distance 0.4` / `shape 0.2` |
| 门控 | `gate_score_norm='nominal'`（自适应门控退场） |
| 队色否决 | `kit_veto='any'`，跨队/跨侧与离群碎片 `cX` 直接否决 |
| 锥形 / 互选层 | `mcv_cone=false` / `mcv_recip='off'`（均未入选） |
| 队别标签表 | 由 `_probe_color_team.py --stage pixels --dump-labels <PATH>` 生成；不提供时队色否决不生效 |

## 2. 最终指标

| 划分 | 序列数 | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
|---|---|---|---|---|---|---|---|
| test | 49 | **64.306** | 70.134 | 59.082 | 85.181 | 78.191 | 1770 |
| train | 57 | **64.872** | 71.517 | 58.927 | 88.043 | 79.603 | 2268 |

论文写作 `64.31` / `64.87`。

**口径**：deploy 取值只用 **train** 排序决定，test 仅事后报告（`sens250.md`）；层入选判定用逐序列配对 bootstrap，`NBOOT=20000`（`_sens250/mcv_boot.md`）。

## 3. 增益分解

- 窗口 `30/75 → 250/450`：约 `+0.92` HOTA
- 队色否决：`63.745 → 64.872`（train）/ `63.005 → 64.306`（test）
- 自适应门控：仅 `+0.03`，**退役**

## 4. 层入选判定（论文 Table 2 口径，窗口 250/450）

| 配置 | 层数 | train HOTA | test HOTA |
|---|---|---|---|
| A1 名义窗口 250/450 无否决 | 0 | 63.745 | 63.005 |
| T1 锥形剪枝单独 | 1 | 63.851 | 63.222 |
| **T2 队色否决单独（入选）** | 1 | **64.872** | **64.306** |
| T1+T2 锥形叠在队色之上 | 2 | 64.903 | 64.231 |
| T2 + V3a 互选一致 | 2 | 64.871 | 64.165 |
| T2 + V3b 分量纯度 | 2 | 64.903 | 64.231 |
| T2 + T4 回补 200 | 2 | 64.763 | 64.174 |
| T2 + T4 回补 400 | 2 | 64.720 | 64.167 |
| T2 + T4 回补 600 | 2 | 64.690 | 64.072 |
| 全量级联 | 5 | 64.638 | 64.188 |

即在宽窗 `250/450` 下，**只用队色否决这一层**即取到最优 test（论文 Config J）；更高层数的配置 train 略高但在 test 上回落，且增益落在 bootstrap SE 容差内，故按「层数最少」入选。

## 5. 复现

```bash
# 最终方案（Config J / MCVC）后处理 + TrackEval
python appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py --mcv \
    --kit_labels <labels.json> --results_dir <MOT 目录> --gt_root <GT 根>
# MCVC 本地等价性验证
python appendix/config_j_mcvc/code/_verify_kit_veto_native.py
```

## 6. 相关文件

| 文件 | 内容 |
|---|---|
| `appendix/config_j_mcvc/reports/_sens250/sens250.md` | 全量敏感性扫描（论文 Table 3-9），本表数字的原始出处 |
| `appendix/config_j_mcvc/reports/_sens250/mcv_boot.md` | 层入选判定与逐层边际 ΔHOTA / 配对检验 |
| `appendix/config_j_mcvc/reports/_mcv/mcv_boot.md` | 早期（150/450 基准窗）层选择，历史对照 |
| `configs/optimal_config.json::paper_final` | 最终配置快照 |
| `reports/eval6groups_report.md` | 早期选型记录（外观模型 / AFLink 变体），**非最终结果** |
| `reports/eval_full49_iou_osnet_report.md` | 早期选型记录（IoU vs eIoU 对照），**非最终结果** |
