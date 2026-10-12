# 实际执行记录

- 在 Windows 11 / Python 3.12.6 上建立独立 `.venv`；没有修改全局 PyTorch/CUDA。
- 从 PyTorch 官方 cu126 索引安装 torch 2.6.0+cu126；从 PyPI 安装 mace-torch 0.3.16 和固定 ASE 3.26.0；`pip check` 实际通过。
- 从 ACEsuit 官方 release 下载模型，79,472,952 bytes，实际 SHA-256 与预期一致。首次下载耗时 83.200 s。
- 首次 smoke 初始化因 head 大小写触发项目保护：`RuntimeError: MACE selected head 'default', requested 'Default'`。MACE 提示可用 heads 为 `['default']`，配置改为精确的小写 `default`。错误发生于推理前，无原始计算数据；初始身份记录保存在 `results/smoke_initialization_failed/`。正式结果不复用该目录。
- MACE 导入设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`，PyTorch 会提示旧模型使用 pickle 兼容加载。只加载已核验哈希的官方权重。
- 可选 cuequivariance 未安装；使用 MACE 默认 e3nn CUDA 路径，未启用该可选加速。CUDA 参数设备和输入张量设备仍逐次验证。
- NumPy 2.5 对 ASE 3.26 数组 shape 设置有弃用提示；不影响本次测试结果。完整固定依赖版本见锁文件。
- Si 首次生产扫描完整保存 42 个点与压力差分点后，验证 JSON 保存遇到 `TypeError: Object of type bool is not JSON serializable`（NumPy 标量布尔类型）。已修复验证结果到 Python 标量的转换，并通过 `--resume` 继续；模型、dtype、原始推理实现未改变。原始点保持不变，失败和恢复来源保留于日志及 attempts。

正式计算及最终验证结论以 [报告](report_zh.md) 与各运行 `status.json` 为准。未运行的平台、额外 DFT、有限温度和 CPU 性能对照不计为已完成。
