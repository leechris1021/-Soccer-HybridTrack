# weights —— 模型权重

本目录存放检测与外观特征权重。**规则路径（论文最终方案 Config J / MCVC 的 `mode='rule'`）不加载任何权重**，只有重跑检测链路（`run_ablation.py` / `run_full49_*.py` 等）才需要。

## 文件清单

| 文件 | 大小 | 用途 | 必需场景 |
|---|---|---|---|
| `best.pt` | ~103 MB | YOLO 检测器权重 | 重跑检测阶段 |
| `osnet_x075_soccernet_best.pth` | ~5.1 MB | OSNet ReID 外观特征权重 | 重跑检测阶段 |
| `aflink_model_v3.pth` | ~3.5 MB | 原版 AFLink 神经网络权重 | 仅神经 AFLink 分支；**规则路径不加载** |

## 获取方式

- `best.pt`：约 103 MB（108,183,612 B），**超过 GitHub 单文件 100 MB 上限，请勿 `git add` 提交**。本机工作副本里已有该文件；公开发布需走发布渠道，或自行训练 YOLOv8m soccer 检测器。
- `fussball/aflink_model_v3.pth` 与本目录的 `aflink_model_v3.pth` 是**同一个文件**（MD5 一致），各留一份是为了让 `fussball/aflink.py` 能独立定位权重。
- `osnet_x075_soccernet_best.pth`：SoccerNet ReID 预训练权重。
- `aflink_model_v3.pth`：原版 AFLink 发布权重（`fussball/aflink.py` 的神经分支使用）。

## 路径覆盖

默认值由 `fussball/project.py` 统一维护，均指向本目录：

```text
DEFAULT_YOLO_WEIGHTS   = <repo>/weights/best.pt
DEFAULT_OSNET_WEIGHTS  = <repo>/weights/osnet_x075_soccernet_best.pth
DEFAULT_AFLINK_WEIGHTS = <repo>/weights/aflink_model_v3.pth
```

检测链路里的 `--appearance_weights`（`run_ablation.py` 等）默认取 `DEFAULT_OSNET_WEIGHTS`，可用对应命令行参数覆盖。

> **没有 `AFLINK_WEIGHTS` 这个环境变量。** 神经 AFLink 的权重由 `fussball/aflink.py` 的
> `DEFAULT_MODEL_PATH`（= `fussball/aflink_model_v3.pth`）决定，可用其命令行 `--model_path` 覆盖。
