## Fussball Tracking Experiments

这是一个围绕 **YOLO + AdaptiveDeepSORT + ECC + AFLink / Hybrid AFLink** 的足球多目标跟踪实验仓库。当前仓库已经按“源码 / 脚本 / 配置 / 数据 / 权重 / 输出 / 报告 / 第三方依赖”重新整理，目标是方便上传到 GitHub，并让其他人可以按目录说明直接复现实验。

### 仓库结构

- `fussball/`：核心源码包
- `scripts/`：实验入口脚本
- `configs/`：实验配置与扫描表
- `reports/`：实验报告与结果说明
- `tools/`：序列清单等辅助文件
- `third_party/`：第三方基线与原始工具
- `weights/`：模型权重放置目录
- `data/`：数据集放置目录
- `outputs/`：实验输出目录

### 推荐环境

- Python 3.10+
- Windows / Linux 均可
- 建议使用独立虚拟环境

安装依赖：

```bash
pip install -r requirements.txt
```

### 复现前需要准备的内容

#### 1. 数据集
将数据集按如下结构放到 `data/` 下：

```text
data/
└─ tracking/
   ├─ train/train/...
   ├─ test/test/...
   └─ challenge/challenge/...
```

完整目录说明见 `data/README.md`。

#### 2. 权重
将实验所需权重放到 `weights/` 下。完整列表见 `weights/README.md`。

至少涉及：
- `weights/best.pt`
- `weights/osnet_x075_soccernet_best.pth`
- `weights/aflink_model_v3.pth`

### 最常用命令

#### 运行基础消融链路

```bash
python scripts/run_ablation.py
```

#### 运行非规则链路灵敏性分析

```bash
python scripts/run_tracking_sensitivity.py
```

#### 运行最优参数的自适应门控 + 三因子规则 AFLink

```bash
python scripts/run_best_adaptive_tri.py
```

#### 运行 Hybrid AFLink / 权重灵敏性分析

```bash
python scripts/aflink_hybrid_eval.py --help
```

#### 单独做 TrackEval 评测

```bash
python scripts/evaluate_trackeval_metrics.py --help
```

### 复现顺序建议

- **基础链路复现**：先跑 `scripts/run_ablation.py`
- **非规则参数分析**：再跑 `scripts/run_tracking_sensitivity.py`
- **规则 + 门控最优结果**：跑 `scripts/run_best_adaptive_tri.py`
- **规则后处理完整扫描**：跑 `scripts/aflink_hybrid_eval.py`

### 目录说明入口

- `scripts/README.md`
- `fussball/README.md`
- `configs/README.md`
- `data/README.md`
- `weights/README.md`
- `outputs/README.md`
- `reports/README.md`
- `third_party/README.md`
- `tools/README.md`

