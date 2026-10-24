## outputs

这个目录用于存放**所有运行产物**。`outputs/**` 已在 `.gitignore` 中排除，只有本 `README.md` 会被提交。

### 运行后生成的子目录

| 子目录 | 生成者 |
|---|---|
| `tracker_results/` | `run_ablation.py`、`run_full49_A.py`、`run_full49_aflink_eiou.py`、`eval_10seq_trackeval.py` |
| `tmp/` | `eval_10seq_trackeval.py` 的临时 TrackEval 工作目录 |
| `runs/` | ultralytics 的 YOLO 推理 / 训练运行目录 |

默认输出根由 `fussball/project.py` 给出：

```text
OUTPUTS_ROOT         = <repo>/outputs
TRACKER_RESULTS_ROOT = <repo>/outputs/tracker_results
```

### 说明

- 本目录当前**只有这一个说明文件**，不含任何实验输出。
- `appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py` 的输出目录由 `--output_root` / `--report_path` 指定，需要时可显式指到 `outputs/` 下。

### GitHub 上传建议

公开仓库中只保留本 `README.md`，不提交具体实验输出。
