## weights

这个目录用于存放实验所需模型权重。权重文件通常较大，建议通过 **Git LFS**、**网盘链接** 或 **GitHub Release** 提供，不建议直接普通提交。

### 主要权重

- `best.pt`：YOLO 检测模型权重
- `osnet_x075_soccernet_best.pth`：OSNet ReID 权重
- `aflink_model_v3.pth`：Hybrid AFLink 神经网络权重

### 当前主链路所需最少权重

#### 基础跟踪 / 消融实验

需要：
- `weights/best.pt`
- `weights/osnet_x075_soccernet_best.pth`

#### 规则 AFLink 最优参数实验

如果只跑 `scripts/run_best_adaptive_tri.py`，主要依赖 AFLink 后处理和已有预测结果；如需完整生成预测，仍建议准备上面的检测与 ReID 权重。

#### Hybrid AFLink 神经网络模式

额外需要：
- `weights/aflink_model_v3.pth`

### 历史 / 备用权重

- `pcb_pyramid_r101.pth`
- `pcb_pyramid_r101.pdparams`

这两个文件更像历史遗留或第三方 DeepSORT 相关权重，当前主链路代码没有直接依赖。

### 建议

发布 GitHub 时可以：
- 仓库里保留这个 `README.md`
- 在 Release 或外部链接中提供下载地址
- 在下载后把权重放回本目录对应文件名位置 
