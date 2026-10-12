# 参考值与比较条件

文献数据位于 [`references/reference_values.csv`](../references/reference_values.csv)，与 `results/` 中本项目的 MACE 计算严格分开。检索日期：2026-10-12（Asia/Shanghai）。空白字段代表本来源未提供或本次未核实，不能当作零。

## 主要比较：静态晶格 PBE DFT

采用原作者维护的 [DeltaCodesDFT 数据仓库](https://github.com/molmod/DeltaCodesDFT/tree/f9b6693763dea92d43ee48f172c02bc506876f06)，具体数值来自 [`WIEN2k.txt`](https://raw.githubusercontent.com/molmod/DeltaCodesDFT/f9b6693763dea92d43ee48f172c02bc506876f06/WIEN2k.txt)。文件标识为 **WIEN2k 13.1、calcDelta 3.0**。这是全电子 APW+lo / PBE 的零外压静态晶格 EOS 数据，无零点振动与有限温度修正；仓库说明未包含自旋轨道耦合。

| 材料及晶相 | V₀ (Å³/atom) | 常规立方 a₀ (Å) | B₀ (GPa) | B′₀ |
|---|---:|---:|---:|---:|
| Si，diamond | 20.4530 | 5.4695173 | 88.545 | 4.31 |
| Al，fcc | 16.4796 | 4.0395739 | 78.077 | 4.57 |

原始数据直接给出 V₀、B₀、B′₀；表中 a₀ 是本项目按 `a₀=(n V₀)^(1/3)` 换算，Si 的 n=8，Al 的 n=4。保留的多位数字便于重算，并不代表相同位数的物理精度。不能将目录中旧 CIF 的晶格参数与本表不同版本的 EOS 参数混用。

方法来源：K. Lejaeghere, V. Van Speybroeck, G. Van Oost, S. Cottenier, *Error estimates for solid-state density-functional theory predictions: an overview by means of the ground-state elemental crystals*, Critical Reviews in Solid State and Materials Sciences **39**, 1–24 (2014), [DOI: 10.1080/10408436.2013.772503](https://doi.org/10.1080/10408436.2013.772503)，[作者公开稿](https://arxiv.org/abs/1204.2733)。其 II.1、II.2 节说明 EOS 定义及静态结果与实验比较的温度、零点效应问题。

本地保存了未经修改的 [`WIEN2k.txt`](../references/sources/WIEN2k.txt) 与上游 [`LGPL 2.1 许可证`](../references/sources/DeltaCodesDFT_LICENSE.txt)。固定仓库提交为 `f9b6693763dea92d43ee48f172c02bc506876f06`。原始 WIEN2k 文件 SHA256：`b9702aebe1063eca970f7a9bd2f25879584ab342bfc51a8c646ef7b18c8debaa`。上游程序并未复制到本项目。

## 补充实验数据

| 材料及晶相 | 数值 | 温度与测量条件 | 原始来源 |
|---|---|---|---|
| Si，diamond | a = 5.431144 Å；扩展不确定度 ±0.000080 Å，k=2 | 22.5 °C = 295.65 K；NIST SRM 640f 硅粉末 X 射线衍射 | [2019-02-26 NIST 官方证书，第 1 页](https://tsapps.nist.gov/srmext/certificates/archives/640f.pdf)；[Black et al. 2020，DOI: 10.1017/S0885715620000366](https://doi.org/10.1017/S0885715620000366) |
| Al，fcc | B(0 K 外推) = 79.38 GPa | 4.2–300 K 单晶超声脉冲回波测量外推至 0 K；绝热弹性常数；未去除零点效应 | [Kamm & Alers 1964，DOI: 10.1063/1.1713309](https://doi.org/10.1063/1.1713309)；[AIP 提交给 Crossref 的原论文摘要元数据](https://api.crossref.org/works/10.1063/1.1713309) |

Al 的 B 由原论文摘要所列 C₁₁=11.430、C₁₂=6.192（单位 10¹¹ dyn/cm²）换算：1 dyn/cm²=0.1 Pa，因此 C₁₁=114.30 GPa、C₁₂=61.92 GPa，立方晶体 `B=(C₁₁+2C₁₂)/3=79.38 GPa`。本次成功读取的是出版商存入 Crossref 的摘要；AIP 全文访问失败，故 CSV 明确标为 `verified_abstract`。不声称已检查全文表格、样品细节或误差预算。

Si 实验体模量、Al 实验晶格常数：**本次原始来源核实缺失**，不填入未经确认的常见手册数字。两种材料的 a₀ 和 B₀ 主比较均已有同条件 PBE DFT 数据；以上实验项仅补充背景。

## 如何解释差异

本项目的 E(V) 是静态势能曲线。室温实验包含热膨胀、声子等贡献；低温外推实验仍包含零点运动，因此上述实验值都不能直接称为“无零点效应的静态真值”。超声得到的绝热模量与有限温度的等温模量也需区分。

以 `100*(MACE值/参考值-1)` 报告每种材料相对其自身参考的偏差。PBE DFT 比较更接近本项目热力学条件，但模型训练标签与 WIEN2k 的基组、赝势/全电子实现、收敛设置可能不同；本项目没有另跑 DFT 收敛计算。只对 Si 和 Al 两个晶体相的局部平衡性质进行应用验证，不能由接近个别参考值推断模型普遍准确。

预训练数据中是否含有相同或相近结构尚未逐条排查，因此这不是严格的未见材料泛化测试。扫描范围、点数、EOS 和浮点精度导致的变化是数值敏感性；它们和文献实验不确定度、模型物理误差是不同概念。
