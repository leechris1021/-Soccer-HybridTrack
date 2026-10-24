## IoU + ECC + AFLink + OSNet 评估报告

> 定位：**早期选型阶段记录（IoU vs eIoU 对照），非论文最终结果**；最终结果见 `reports/final_config_j_mcvc.md`。

> 数据集：SNMOT test，49 序列，IoU 阈值 0.5
> 来源：`eval_full49_iou_osnet.log`（运行日志，未随仓库发布）
> 结果目录：`full49_ecc_aflink_iou_osnet`

### 实验配置

| 项目 | 配置 |
|------|------|
| 时空关联 | IoU（标准） |
| ECC | ✓ 启用 |
| AFLink | ✓ 启用（神经网络 v3） |
| 外观模型 | OSNet-x0.75 |
| 评估序列 | 49 序列全量 |

### 评测结果

| HOTA | DetA | AssA | MOTA |
|-----:|-----:|-----:|-----:|
| **61.65** | **70.20** | **54.28** | **85.11** |

（原始值：HOTA=0.6165, DetA=0.7020, AssA=0.5428, MOTA=0.8511）

### 与 eIoU 版本对比

| 版本 | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|------|-----:|-----:|-----:|-----:|-----:|-----:|
| **IoU** + ECC + AFLink(v3) + OSNet（本报告） | 61.65 | 70.20 | 54.28 | 85.11 | — | — |
| **eIoU** + ECC + AFLink(v3) + OSNet（G_os_v3） | 61.697 | 70.171 | 54.384 | 84.904 | 72.109 | 2317 |

两者结果非常接近，IoU 与 eIoU 差异极小（HOTA 差 ~0.05）。
