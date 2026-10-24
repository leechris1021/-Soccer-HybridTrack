## configs

本目录存放实验配置和参数扫描计划。

### 当前文件

- `optimal_config.json`：**论文最终配置 Config J（MCVC）**的完整展开 + 复现锚点——宽标称窗口 `250/450` + 队色等价类否决 `kit_veto=any`，自适应门控退场；来源 `appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py` 的 `--mcv`；test `64.306` / train `64.872`。
- `aflink_config_v3.json`：AFLink 基础配置（与 `fussball/aflink_config_v3.json` 同步一份）
- `灵敏性分析.txt`：参数敏感性扫描范围说明

### 使用说明

- 论文最终配置（Config J / MCVC）的**权威定义**在 `appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py`（`--mcv`）；`optimal_config.json` 的 `paper_final` 段是它的一份可读快照。
- 路径常量统一由 `fussball/project.py` 维护（`CONFIGS_ROOT = <repo>/configs`）。
