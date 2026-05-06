## outputs

这个目录用于存放**所有运行产物**。

### 当前常见内容

- `tracker_results/`：跟踪结果、后处理结果、TrackEval 输出
- `runs/`：YOLO 推理或训练运行目录
- `smoke_log.txt`：临时日志
- `raw_pred_cache.json`：规则 AFLink 可能会使用的预测缓存（如果生成）

### 说明

大多数 `scripts/` 脚本默认都把结果写到：

```text
outputs/tracker_results/
```

### GitHub 上传建议

公开仓库中只保留目录说明，不提交具体实验输出。 
