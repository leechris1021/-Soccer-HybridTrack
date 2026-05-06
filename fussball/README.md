## fussball

这个目录是项目的**核心源码包**，主要包含跟踪、关联、后处理和路径配置逻辑。

### 文件说明

- `project.py`：统一维护仓库根路径、数据目录、权重目录、输出目录、序列清单等共享常量
- `adaptive_deepsort.py`：AdaptiveDeepSORT 主体逻辑
- `track.py`：轨迹状态管理
- `kalman_filter.py`：扩展卡尔曼滤波器
- `detection.py`：检测框数据结构
- `association.py`：自适应代价矩阵和匈牙利匹配
- `appearance.py`：OSNet / ResNet 外观特征提取
- `ecc.py`：ECC 摄像机运动补偿
- `aflink.py`：基础 AFLink 轨迹片段合并模块

### 设计说明

当前已经改成标准 Python 包结构，`scripts/` 中的入口脚本统一从这里导入核心模块。这样做的好处是：

- 更适合上传 GitHub
- 更容易维护依赖关系
- 更容易在不同机器上复现
- 不再依赖“源码散落在根目录”的导入方式

### 路径规范

所有默认路径都尽量通过 `project.py` 统一维护，避免在多个脚本里重复写死路径。 
