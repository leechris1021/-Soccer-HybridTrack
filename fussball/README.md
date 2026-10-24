## fussball

核心源码包：跟踪、关联、运动补偿、后处理与统一路径配置。

### 文件说明

- `project.py`：统一维护仓库根、数据/权重/输出目录、序列清单等共享常量
- `adaptive_deepsort.py`：AdaptiveDeepSORT 主体逻辑
- `track.py`：轨迹状态管理
- `kalman_filter.py`：扩展卡尔曼滤波器
- `detection.py`：检测框数据结构
- `association.py`：自适应代价矩阵与匈牙利匹配
- `appearance.py`：OSNet-x0.75 / ResNet50 外观特征提取
- `ecc.py`：ECC 摄像机运动补偿
- `aflink.py`：AFLink 轨迹片段合并模块
- `aflink_config_v3.json`：AFLink 基础配置（同步一份在 `configs/`）
- `aflink_model_v3.pth`：原版 AFLink 网络权重；规则路径（论文最终方案 Config J 的 `mode='rule'`）**不加载**它

### 与 scripts/ 的关系

`scripts/` 下的入口脚本通过 `from fussball.xxx import ...` 复用本包实现（如 `run_ablation.py`），不再假设源码散落在仓库根目录。

论文最终方案（Config J / MCVC）的实现**不依赖本包**：它是纯规则路径，见 `../appendix/config_j_mcvc/code/aflink_hybrid_eval_v4.py`。

### 路径规范

所有默认路径尽量通过 `project.py` 统一维护，避免在多个脚本里重复写死。
