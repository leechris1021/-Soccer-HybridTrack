## 六组实验评估报告（eval6.log）

> 定位：**早期选型阶段记录（外观模型 / AFLink 变体选型），非论文最终结果**；最终结果见 `reports/final_config_j_mcvc.md`。

> 数据集：SNMOT test，49 序列，IoU 阈值 0.5
> 来源：`eval6.log`（运行日志，未随仓库发布）

### 实验说明

| 标签 | 目录名 | 说明 |
|------|--------|------|
| E_rn50_v3 | full49_ecc_aflink_post_eiou_resnet50 | ECC + AFLink(v3) + eIoU + ResNet50 |
| G_os_v3 | full49_ecc_aflink_post_eiou_osnet | ECC + AFLink(v3) + eIoU + OSNet |
| H_rn50_orig | full49_ecc_origAFLink_post_eiou_resnet50 | ECC + 原版AFLink + eIoU + ResNet50 |
| I_os_orig | full49_ecc_origAFLink_post_eiou_osnet | ECC + 原版AFLink + eIoU + OSNet |
| H2_rn50_snmot | full49_snmotAFLink_post_eiou_resnet50 | snmot AFLink + eIoU + ResNet50 |
| I2_os_snmot | full49_snmotAFLink_post_eiou_osnet | snmot AFLink + eIoU + OSNet |

### 评测结果

| 实验 | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDs |
|------|-----:|-----:|-----:|-----:|-----:|-----:|----:|
| E_rn50_v3 | 58.601 | 69.919 | 49.274 | 84.555 | 67.625 | 2699 | 1211 |
| **G_os_v3** | **61.697** | **70.171** | **54.384** | **84.904** | **72.109** | **2317** | **1211** |
| H_rn50_orig | 58.633 | 69.920 | 49.327 | 84.557 | 67.735 | 2689 | 1211 |
| I_os_orig | 61.795 | 70.167 | 54.560 | 84.907 | 72.321 | 2303 | 1211 |
| H2_rn50_snmot | 58.681 | 69.920 | 49.407 | 84.559 | 67.800 | 2679 | 1211 |
| I2_os_snmot | 61.861 | 70.168 | 54.675 | 84.909 | 72.447 | 2295 | 1211 |

### 结论

- OSNet 特征（G/I/I2）明显优于 ResNet50（E/H/H2），HOTA 高约 **+3.2**
- 三种 AFLink 变体（v3 / origAFLink / snmotAFLink）在 OSNet 组别下差异很小（≤0.16 HOTA）
- **I2_os_snmot** 综合最优（HOTA=61.861），**G_os_v3** 次之（HOTA=61.697）
