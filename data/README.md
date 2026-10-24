## data

数据集放置说明。**SoccerNet / SNMOT 跟踪数据集不随本仓库发布**，请前往 SoccerNet 官网获取：https://www.soccer-net.org/

### 默认数据集根

代码默认从**仓库根目录**的 `./tracking/` 读取（常量 `fussball/project.py::DEFAULT_TRACKING_ROOT`）：

```text
tracking/                          # 默认 <repo>/tracking
├─ train/train/<SEQ_NAME>/         # 57 序列
├─ test/test/<SEQ_NAME>/           # 49 序列（SNMOT-116~150、187~200）
└─ challenge/challenge/<SEQ_NAME>/ # 挑战集（可选，无脚本默认使用）
```

> 该目录已在 `.gitignore` 中排除，**不随仓库发布**。
> ⚠️ `fussball/project.py` **不读** `SOCCERNET_ROOT`，它只响应 `FUSSBALL_ROOT`。

### 各脚本引用的目录与覆盖方式

| 脚本 | 数据目录 | 覆盖方式 |
|---|---|---|
| `appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py` | BoT-SORT MOT 输出目录 + GT 根 | `--results_dir` / `--gt_root` |
| `scripts/run_ablation.py` | 测试集根 | `--data_root`；默认 `<repo>/tracking/test/test`（**不读**环境变量） |
| `scripts/run_full49_A.py`、`scripts/run_full49_aflink_eiou.py`、`scripts/eval_10seq_trackeval.py` | 测试集根 | `--data_root`；默认 `<repo>/tracking/test/test`（三者都不读环境变量） |

### 其他说明

- 49 个官方测试序列清单在 `tools/SNMOT-test.txt`（由 `fussball/project.py` 的 `load_default_test_sequence_names()` 加载），不依赖本目录结构。
- `tracking-2023/`（2023 版数据集）不属于本仓库内容。

### GitHub 上传建议

公开仓库中只保留这个 `README.md`，不要提交数据集本身。
