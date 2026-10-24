# Soccer-HybridTrack-v2

足球多目标跟踪实验仓库。核心链路：**YOLO 检测 + AdaptiveDeepSORT + ECC 运动补偿 + AFLink 片段关联 + MCVC 队色否决级联**；论文最终方案即「宽窗 AFLink + MCVC」。

## 论文最终结果（口径锚点）

最终方案 **Config J = MCVC（multi-cue veto cascade）**：名义窗口由 `150/450` 放宽到 **`250/450`**，叠加**队色等价类否决**（`kit_veto=any`）；自适应门控不显著，已退役。

| 划分 | 序列数 | HOTA | DetA | AssA | MOTA | IDF1 | IDs |
|---|---|---|---|---|---|---|---|
| test | 49 | **64.306** | 70.134 | 59.082 | 85.181 | 78.191 | 1770 |
| train | 57 | **64.872** | 71.517 | 58.927 | 88.043 | 79.603 | 2268 |

论文写作 `64.31` / `64.87`，与 `appendix/config_j_mcvc/reports/_sens250/sens250.md` 的 `T2 队色否决单独（入选）` 行逐位一致。
增益分解：窗口 `30/75→250/450` 约 `+0.92` HOTA；队色否决 `63.745→64.872`(train) / `63.005→64.306`(test)；自适应门控仅 `+0.03`，退场。

## 仓库结构

```text
fussball/      核心源码包（BoT-SORT + AFLink + 统一路径常量 project.py）
scripts/       实验入口（消融、全量 49 序列、TrackEval 评测）
configs/       论文最终配置（Config J / MCVC）+ AFLink 基础配置 + 敏感性说明
reports/       最终结果入口 + 早期选型评测记录
tools/         序列清单（SNMOT-test.txt）
third_party/   第三方基线（SoccerNet DevKit）
weights/       模型权重    data/ 数据集放置    outputs/ 运行产物
appendix/      论文最终方案的实现与原始报告，见 appendix/README.md
```

## 配置

权威定义：`appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py --mcv`（= `--gate_score_norm nominal` + `--kit_veto any`）；可读快照见 `configs/optimal_config.json::paper_final`。

- 名义窗口 `max_gap=250` / `dist_thresh=450`
- `rule_variant='baseline'`：两因子时间/距离评分 + 贪心，**无神经网络**
- 自适应门控退场（`gate_score_norm='nominal'`）
- `kit_veto='any'`：跨队/跨侧与离群碎片 `cX` 直接否决
- 队别标签表：`_probe_color_team.py --stage pixels --dump-labels <PATH>` 生成

## 常用命令

```bash
pip install -r requirements.txt
# 最终方案后处理 + TrackEval
python appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py --mcv --kit_labels <labels.json> --results_dir <MOT 目录> --gt_root <GT 根>
# MCVC 本地等价性验证
python appendix/config_j_mcvc/code/_verify_kit_veto_native.py
# 消融 / 全量评测
python scripts/run_ablation.py
python scripts/run_full49_A.py
python scripts/run_full49_aflink_eiou.py
python scripts/eval_10seq_trackeval.py
```

## 复现准备

1. **数据**：`./tracking/{train/train,test/test}/SNMOT-*`（见 `data/README.md`）。
2. **权重**：见 `weights/README.md`；规则路径（最终方案 `mode='rule'`）**不加载权重**，仅重跑检测才需要。
3. **路径**：默认由 `fussball/project.py` 推导；`run_ablation.py` / `run_full49_*.py` / `eval_10seq_trackeval.py` **不读环境变量**，用各自命令行参数（如 `--data_root`）覆盖。

## 报告

| 文件 | 内容 |
|---|---|
| `reports/final_config_j_mcvc.md` | **最终结果入口**：Config J / MCVC 配置、指标、增益分解、层入选判定 |
| `appendix/config_j_mcvc/AFLink改进说明.md` | 改进版 AFLink 原理 / 分层 / 参数速查 / 已否方向（最终口径算法说明，行号指向 `aflink_hybrid_eval_v4.py`） |
| `appendix/config_j_mcvc/reports/_sens250/sens250.md` | MCVC @250/450 全量敏感性扫描（论文 Table 3-9），最终数字原始出处 |
| `appendix/config_j_mcvc/reports/_mcv/mcv_boot.md` | MCVC 层选择（150/450 基准窗，历史对照） |
| `appendix/config_j_mcvc/reports/_sens250/mcv_boot.md` | MCVC 层入选判定与逐层边际 ΔHOTA / 配对检验（250/450） |
| `reports/eval6groups_report.md` | 分档（6 组）评测记录（早期选型，非最终） |
| `reports/eval_full49_iou_osnet_report.md` | 全量 49 序列 IoU / OSNet 评测记录（早期选型，非最终） |

## 备注

- Python ≥ 3.9；`torch` 仅为神经 AFLink 分支 import，规则路径 CPU 版即可。
- TrackEval：`pip install trackeval`，或用 `third_party/` 自带副本。

## 许可证

学术研究用途发布。SoccerNet 数据集、检测器权重与第三方基线遵循各自许可，详见 `LICENSE`。
