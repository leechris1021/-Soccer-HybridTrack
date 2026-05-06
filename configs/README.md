## configs

这个目录存放实验配置和扫描计划文件。

### 当前文件

- `aflink_config_v3.json`：AFLink 相关配置文件

### 使用说明

- `scripts/run_tracking_sensitivity.py` 默认会读取 `configs/灵敏性分析.txt`
- 如果你修改参数扫描范围，建议直接编辑这个文件

### 建议

如果后续增加更多实验设置，统一放到这个目录，避免配置散落在仓库根目录。 
