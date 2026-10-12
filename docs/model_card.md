# 模型记录：MACE-MP-0b3 medium

本项目定位为预训练机器学习力场的材料性质应用与验证。固定使用官方已发布的无机材料模型，便于复现；不宣称这是最新、最强或对所有性质最优的 MACE 模型。

| 条目 | 本项目选择 |
|---|---|
| 模型名称 | `MACE-MP-0b3-medium` |
| 官方发布标识 | `mace_mp_0b3` |
| 文件名 | `mace-mp-0b3-medium.model` |
| 官方来源 | [ACEsuit / mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mp_0b3) |
| 下载地址 | [官方 release 模型文件](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model) |
| 固定 SHA256 | `2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012` |
| 训练体系与方法标签 | MPTrj，无机材料，DFT PBE(+U) |
| 官方元素覆盖 | 89 个元素，包括 Si、Al |
| 官方模型许可 | MIT |
| 本项目设备 | 显式 `device="cuda"`，不自动回退 CPU |
| 正式精度 | `float64`；`float32` 作为独立数值对照 |
| head | `default`（小写，实际检查点唯一 head）；运行元数据同时记录可用和所选值 |
| 实测参数数量 | 9,063,204 |
| 附加色散修正 | 不启用 |

下载后的本地文件必须通过上述 SHA256 验证才允许加载。本次实际 smoke 的 [model.json](../results/smoke_Si/model.json) 记录了匹配哈希、79,472,952 字节文件及 83.2002 秒原始下载耗时；[calculator.json](../results/smoke_Si/calculator.json) 记录参数和实际输入张量均为 `cuda:0`、参数精度为 `torch.float64`、唯一 head 为小写 `default`。这些是实际执行记录，不是仅从配置推断。下载权重保存在被 Git 忽略的 `.cache/models/`，不随项目上传。

## 选择依据

官方模型表将 MP-0b3 列为面向材料、基于 MPTrj 的模型，说明该版本相对于 0b2 修正了部分声子问题；同一仓库同时提醒后续版本并不保证对所有问题都优于早期版本。`medium` 是当前 8 GB 显存、小晶胞和精度验证之间的务实选择，其可运行性以本机实际 CUDA smoke 与 production 记录为依据。[官方模型说明](https://github.com/ACEsuit/mace-foundations/blob/main/README.md)

MACE 官方文档区分材料模型和有机分子模型。MACE-OFF23 的覆盖元素不含 Si 或 Al，不适用于这里的任务。文档还说明 `mace_mp()` 的默认模型曾发生变化，所以本项目不依赖会变化的默认别名，而是固定 release、文件名及哈希。[官方预训练模型文档](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)

本项目直接使用 `MACECalculator` 加载固定文件，记录实际参数数量、dtype、参数所在设备、输入张量所在设备及支持的属性。`default` 是该单 head 检查点的计算选择，不是在 PBE、r2SCAN 等多个方法 head 之间混选。每次恢复运行仍须校验模型和配置一致，不能拼接不同 head、dtype 或模型的数据。

Si 与 Al 的计算配置保持相同；初始结构、扫描范围以及每胞原子数属于材料相关参数。主流程保持理想分数坐标，研究固定相附近的静态能量曲率，不预设高压相、缺陷、熔化或反应的可靠性。

## 适用范围与结论边界

PBE(+U) 是训练数据体系的整体方法标签，不表示本项目又执行了一次 PBE DFT，也不表示对 Si 或 Al 额外施加了某个 Hubbard U。该模型的任何系统偏差、训练覆盖不足和能量面非光滑性都可能影响曲率，体模量通常比最低点位置更敏感。

模型覆盖某种元素不等于覆盖该元素的全部相态、配位、温压范围或化学环境。Si 和 Al 是常见训练材料候选；在没有逐条核查训练集及数据划分前，不把本项目称为训练集外测试。这里的两种晶相结果不能推出普遍的 DFT 精度。

这里没有模型训练、微调、独立 DFT 计算、零点振动修正或有限温度自由能计算。拟合参数的数值敏感性也不等同于模型预测的物理不确定度。若要扩展研究，可加入同一结构上的收敛 DFT 标签、声子稳定性、应变弹性常数、热力学修正和严格的数据泄漏审查。

## 引用

- 架构论文：[Batatia et al., MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2206.07697)。
- 基础模型论文：[Batatia et al., A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096)。
- [MACE 官方软件仓库](https://github.com/ACEsuit/mace) 和 [本项目固定模型的发布页](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mp_0b3)。

以上官方网页于 2026-10-12 核查。网页可能随时间更新，项目复现身份以固定 release 和经实际计算验证的 SHA256 为准。
