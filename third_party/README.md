## third_party

这个目录存放第三方仓库或外部基线代码。

### 当前内容

- `sn-tracking-main/`：SoccerNet Tracking Development Kit / Benchmark 相关代码

### 使用原则

- 主实验代码尽量放在 `fussball/` 和 `scripts/`
- 第三方原仓库尽量少改，避免后续升级或对比时混乱
- 如果某个实验依赖第三方文件，优先在 README 中说明，而不是把第三方代码混进主源码目录

### 说明

当前仓库里的 `tools/SNMOT-test.txt` 已单独复制出来，方便主脚本直接使用，不再强依赖第三方目录结构。 
