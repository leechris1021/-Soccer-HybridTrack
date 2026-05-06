## scripts

这个目录存放**可直接运行的实验入口脚本**。

### 文件说明

- `run_ablation.py`：基础 YOLO + AdaptiveDeepSORT + ECC + AFLink 消融实验
- `run_tracking_sensitivity.py`：非规则链路的单变量灵敏性分析
- `aflink_hybrid_eval.py`：规则 AFLink / Hybrid AFLink / 权重灵敏性分析主脚本
- `run_best_adaptive_tri.py`：只跑“自适应门控 + 三因子规则评分”最优参数
- `evaluate_trackeval_metrics.py`：对已有结果目录做 TrackEval 评测

### 推荐执行顺序

1. 先确认 `data/` 与 `weights/` 已按要求准备好
2. 先跑 `run_ablation.py` 验证基础链路
3. 再跑灵敏性分析或规则 AFLink 脚本

### 常用命令

```bash
python scripts/run_ablation.py
python scripts/run_tracking_sensitivity.py
python scripts/run_best_adaptive_tri.py
python scripts/aflink_hybrid_eval.py --help
python scripts/evaluate_trackeval_metrics.py --help
```

### 输出位置

所有脚本默认把运行结果写到：

```text
outputs/tracker_results/
```

### 说明

这些脚本已经统一改成依赖 `fussball/` 包和共享路径配置，不再假设源码直接散落在仓库根目录。 
