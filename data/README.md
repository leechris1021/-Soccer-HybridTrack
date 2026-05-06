## data

这个目录用于存放 **SoccerNet / SNMOT 跟踪数据集**。本仓库不提供实际数据，请前往SoccerNet官网获取数据：https://www.soccer-net.org/

### 推荐目录结构

```text
data/
├─ tracking/
│  ├─ train/train/<SEQ_NAME>/
│  ├─ test/test/<SEQ_NAME>/
│  └─ challenge/challenge/<SEQ_NAME>/
└─ tracking-2023/
   ├─ train/train/<SEQ_NAME>/
   └─ test/test/<SEQ_NAME>/
```

### 当前代码默认使用

- 测试集：`data/tracking/test/test`
- 训练集：`data/tracking/train/train`
- 挑战集：`data/tracking/challenge/challenge`

### 复现提示

- `run_ablation.py` 默认读取测试集目录
- `evaluate_trackeval_metrics.py` 与 `aflink_hybrid_eval.py` 默认把 `data/tracking/test/test` 作为 GT 根目录
- `tracking-2023/` 更像历史备份或旧版本数据，不是当前主链路必须目录

### GitHub 上传建议

在公开仓库中只保留这个 `README.md`。 
