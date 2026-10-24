# appendix —— 论文最终方案 Config J（MCVC）

本目录保存**论文最终方案 Config J（MCVC）**的完整实现与全部原始报告。

## `config_j_mcvc/` —— 论文最终方案 Config J（MCVC）

论文的最终方案即为 **Config J**：把名义窗口从 `150/450` 放宽到 `250/450`，并叠加**队色等价类否决**（`kit_veto=any`）；自适应门控不显著，**在最终配置中退役**。论文中写作 **MCVC**（multi-cue veto cascade）。

```text
code/                 实验脚本（6 个 py）：
                      aflink_hybrid_eval{,_v4}.py  含 kit_veto / load_kit_labels 的 MCVC 版实现
                                                   （v4 提供 --mcv 一键开关）
                      _verify_kit_veto_native.py   本地复现 kit-veto 校验
                      _exp_residual_gate.py / _kit_veto_stats.py / _probe_color_team.py
reports/_mcv/         MCVC 层选择主实验报告（T1/T2/T3… 逐层入选判定）
reports/_sens250/     以 250/450 为基准窗口的全量敏感性扫描报告（论文 Table 3-9 口径）
_verify_kit_veto_native/   本地复现 kit-veto 所需的 49 个既定参考 txt
                           （JSON 落盘路径的产物由脚本运行时写到 out/，不入库）
```

关键结论（`reports/_sens250/mcv_boot.md`、`sens250.md`）：

| 配置 | 层数 | train HOTA | test HOTA |
|---|---|---|---|
| A1 名义窗口 250/450，无否决 | 0 | 63.745 | 63.005 |
| **T2 队色否决单独（入选）** | 1 | **64.872** | **64.306** |
| T1+T2 锥形叠在队色之上 | 2 | 64.903 | 64.231 |
| T2 + V3a/V3b（互选一致 / 分量纯度） | 2 | 64.871 / 64.903 | 64.165 / 64.231 |

即在 `250/450` 宽窗下，**只用队色否决这一层**即达到最好（论文 Table 2 的 Config J）。

## 算法说明

改进版 AFLink 的原理、分层、参数速查与已否方向见 [`config_j_mcvc/AFLink改进说明.md`](config_j_mcvc/AFLink改进说明.md)。该文档以 **250/450 + 队色否决** 的最终口径撰写，所有行号指向 `config_j_mcvc/code/aflink_hybrid_eval_v4.py`，结论以它为准（任何推荐 `adaptive_cone` / 自适应门控的旧版说明均为上一代口径，已被其取代）。

## 复现

```bash
cd code
# 一键 MCVC 后处理 + 评测（需 --kit_labels 提供队别标签表）
python aflink_hybrid_eval_v4.py --mcv --kit_labels <labels.json> \
    --results_dir <BoT-SORT MOT 输出目录> --gt_root <GT 根>
# 本地 kit-veto 等价性验证
python _verify_kit_veto_native.py
```

队别标签表由 `_probe_color_team.py --stage pixels --dump-labels <PATH>` 生成。

## 备注

- 复现论文最终方案（Config J / MCVC），请使用本目录 `config_j_mcvc/code/aflink_hybrid_eval_v4.py`（`--mcv`）配合队别标签表。
- `config_j_mcvc/code/_exp_residual_gate.py` 仍保留旧目录名 `AFLinkMax-optimal` 的历史路径常量（未修正），仅供查阅。
