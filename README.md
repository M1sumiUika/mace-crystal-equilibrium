# mace-crystal-equilibrium

**基于 MACE 机器学习力场的晶体平衡性质计算与验证**

使用 MACE 预训练机器学习力场与 NVIDIA GPU，计算硅和铝的静态晶格平衡性质，包含能量—体积扫描、状态方程拟合、压力与精度验证，以及可复现代码、结果图表和中文报告。

实际使用官方 `MACE-MP-0b3 medium` 模型，研究金刚石 Si 与面心立方 Al，拟合平衡体积、晶格常数和体模量，并验证压力、周期性及浮点精度。

这是预训练 MLFF 的应用与验证项目。研究对象是**固定晶相、静态晶格、近似零温条件**，没有零点振动或有限温度修正；不应把结果称为室温性质。本项目没有训练 MACE、没有执行新的 DFT 计算，也不把拟合残差当作模型的物理预测不确定度。

[English](README_EN.md) · [中文实际计算报告](docs/report_zh.md) · [中文原理说明](docs/principles_zh.md) · [模型记录](docs/model_card.md) · [参考文献数据](references/) · [GitHub 上传步骤](docs/github_upload.md)

公开仓库：[M1sumiUika/mace-crystal-equilibrium](https://github.com/M1sumiUika/mace-crystal-equilibrium)。本地实际测试 **18 passed**（包含 CUDA 端到端 smoke），详见 [测试输出](results/test_summary.txt)。

## 已实际完成的结果

2026-10-12 在本机 NVIDIA GPU 上依次完成 Si 和 Al 的正式扫描及全部默认验证。下表来自各材料 `analysis.json` 的细扫描 Birch–Murnaghan 主拟合，正式推理均为 float64：

| 材料和晶相 | a₀ / Å | v₀ / Å³·atom⁻¹ | B₀ / GPa | 默认数值验证 |
|---|---:|---:|---:|---|
| Si，diamond | 5.470994 | 20.469569 | 81.0858 | 全部通过 |
| Al，fcc | 4.063224 | 16.770748 | 58.4536 | 全部通过 |

这里的“通过”指压力导数、独立优化、周期性和精度满足记录的数值阈值；不代表与所有实验或 DFT 数据一致。物理参考对照和偏差解释见[实际报告](docs/report_zh.md)。初始化 head 大小写修复、Si 保存中断和实际恢复过程见[执行记录](docs/execution_notes.md)，失败没有被隐藏或记成成功。

相对本次核实的静态 PBE/WIEN2k 参考，Si 体模量低 **8.42%**，Al 低 **25.13%**；这一偏差明显大于已测试的数值敏感性，是本项目的实质性结论与局限。

| Si：相对能量与 EOS | Al：相对能量与 EOS |
|---|---|
| ![Si energy-volume curve](results/production/Si/figures/energy_volume.png) | ![Al energy-volume curve](results/production/Al/figures/energy_volume.png) |

## 从哪里开始

1. 先看[实际计算报告](docs/report_zh.md)，理解结果及验证的适用范围。
2. 按下方步骤建立独立 `.venv` 环境，然后运行 `check`。
3. 运行少量点的 `smoke`，确认当前机器上的 GPU、模型下载和输出正常。
4. 运行 Si 的 `production` 并检查验证；随后用同样配置计算 Al。
5. 修改分析时先使用 `refit`，无需重新进行 GPU 推理。

正式结果保存在 [Si 运行目录](results/production/Si/) 与 [Al 运行目录](results/production/Al/)。每个目录以 `status.json` 记录真实完成情况；`smoke` 结果明确标记 `formal_results: false`，不能作为正式 EOS 结论。未运行或主动跳过的验证不会记为通过。

## 当前运行环境

本项目的实际执行平台为 Windows 11、Python 3.12.6、NVIDIA GeForce RTX 4060 Laptop GPU（约 8 GB 显存），驱动 616.64。项目环境采用 CUDA 版 PyTorch 2.6.0、CUDA 12.6 运行时、MACE 0.3.16、ASE 3.26.0。具体依赖、显存和设备以各运行目录的 `environment.json`、`requirements-freeze.txt` 和 `calculator.json` 为准。

这里的安装与运行已针对本机原生 Windows 进行工作。Linux、WSL2 和其他 GPU 的流程尚未在本项目实测，不承诺直接运行成功。CUDA 驱动可支持的最高版本与 `torch.version.cuda` 显示的 PyTorch 自带运行时是不同概念；无需为了这一项目替换系统已有 PyTorch。

## Windows 安装

在项目根目录打开 PowerShell。建议使用 Python 3.12，以尽量接近本次环境。所有安装进入项目内的 `.venv`，不会修改全局 Python 的包。PyTorch 安装源与版本对应关系可查 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

也可执行同样步骤的脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

脚本只为当前进程设置策略，不改系统执行策略。已有 `.venv` 会复用。若希望更严格复现已保存的依赖，在 CUDA PyTorch 安装之后应用锁文件；它记录本机环境，面向其他平台仍需核查兼容性。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

后续命令都显式调用 `.venv` 的 Python，因此无需激活环境。若激活 `.venv\Scripts\Activate.ps1`，也可以把 `.\.venv\Scripts\python.exe -m mace_crystal_equilibrium` 简写为 `mce`。

## 检查 GPU 并运行

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium check --output results/environment_check.json
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode smoke --output results/my_smoke_Si
```

`check` 输出操作系统、Python、PyTorch、CUDA 运行时、GPU 型号、显存及实际 CUDA float64 运算探针。正式运行还核查模型参数和输入张量确实位于 CUDA。GPU 不可用时停止计算，不自动换 CPU。首次运行从官方 GitHub release 下载模型，并验证固定 SHA256；约 8 GB 的可用容量并非显存占用承诺，运行时以实际空闲显存为准。

分别执行两个材料，便于先审阅 Si 的验证：

```powershell
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --output results/my_production/Si
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Al --mode production --output results/my_production/Al
```

也可顺序运行全部材料；程序先完成 Si，若 Si 的验证失败则停止，不继续 Al：

```powershell
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material all --mode production --output results/my_production_all
```

示例使用新的 `my_...` 目录以保留仓库内的实际交付结果。省略 `--output` 时会创建带时间戳的独立目录。已经非空的目录必须明确使用 `--resume`，避免意外覆盖。

## 扫描与验证配置

[production 配置](configs/production.json) 使用常规立方胞（Si 8 原子、Al 4 原子），参考体积 ±10% 的 21 个粗扫描点，自动检查最低点是否被包围；必要时扩大粗扫描范围。随后在粗拟合最低点附近 ±4% 体积范围取 21 点密集扫描。体积比例的立方根才是晶格常数比例；每个点保存结构、能量、最大力、完整应力、压力及同步计时。

[smoke 配置](configs/smoke.json) 仅取 5 点，检查数据流程，不进行正式 EOS 拟合。正式主结果使用 float64、第三阶 Birch–Murnaghan EOS，并比较 Vinet、范围和点数的敏感性。默认验证包括：

| 配置名称 | 用途 |
|---|---|
| `pressure` | 应力压力与能量中心差分、EOS 平衡压力 |
| `optimize` | 用应力梯度独立优化唯一的立方晶格自由度 |
| `periodicity` | 原胞、常规胞、2×2×2 常规超胞的一致性 |
| `precision` | 相同细扫描点的 float32/float64 及拟合结果对照 |
| `timing` | 预热后清除缓存并同步的 5 次实际推理计时 |

最大原子力检查始终保留。可显式选择或跳过验证；跳过项会标记为未执行，此类运行不构成默认完整验证：

```powershell
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --validations pressure,optimize --output results/selected_checks
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --skip-validation timing --output results/no_timing
```

`--skip-validation` 可重复提供；`--validations none` 跳过所有可选验证。`--dtype float32` 可改变主流程精度，其对照会使用 float64。自定义配置时复制现有 JSON，再使用 `--config configs/my_config.json --mode production`；模式必须一致，模型与 CUDA 设备固定，生产扫描至少需要 15 个粗点和 9 个细点。

## 恢复中断与重新分析

恢复时必须使用原来的材料、模式、输出目录和全部选项：

```powershell
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --output results/my_production/Si --resume
```

程序逐点保存数据，恢复前比对配置、模型 SHA256 和依赖版本。不一致时拒绝拼接，应创建新目录。已经完成的点会复用；未完整结束的一项独立验证可能从该项开始重新执行。如果最初运行指定了 `--config`、`--dtype` 或验证选项，恢复命令也必须提供相同内容。

从已有 **production** 原始数据重新拟合并生成图，不调用模型、不需要 GPU 推理：

```powershell
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium refit results/production/Si
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium refit results/production/Al
```

`refit` 更新该目录下的分析 JSON、拟合 CSV 和图像；保留原始数据。它不会重跑验证，也不会自动更新项目级中文报告。若需要保存不同分析版本，请先复制运行目录。smoke 数据会被拒绝用于正式重拟合。

## 输出文件怎么读

| 文件或目录 | 内容 |
|---|---|
| `config.json`、`identity.json` | 实际参数及恢复一致性指纹 |
| `environment.json`、`requirements-freeze.txt` | GPU、驱动、软件环境 |
| `model.json`、`calculator.json` | 模型来源/hash、head、dtype、CUDA 证据、初始化与预热 |
| `raw_points.csv`、`raw_points.json` | 每个主扫描及辅助点的原始计算；JSON 另含 3×3 应力 |
| `structures/` | 各扫描点和验证结构的 extended XYZ 文件 |
| `analysis.json`、`fits.csv` | 主 EOS、其他拟合设置及残差、敏感性 |
| `validations.json` | 实测指标、阈值、通过/失败或跳过状态 |
| `optimization.csv`、`precision.csv` | 独立零压优化和精度对照数据 |
| `figures/` | 晶体结构、能量曲线/放大、残差、压力及敏感性图 |
| `status.json`、`run.log` | 完成/中断/验证问题状态及运行日志 |

能量与体积拟合统一采用每原子量；晶格常数由每原子体积及常规胞原子数换算。能量图分别减去本材料最低值，不跨元素比较绝对能量。应力采用 ASE 拉伸为正约定，压力 `P = -trace(stress)/3`，转换常数为 `1 eV/Å³ = 160.2176634 GPa`。

原始计算结果与 [references/](references/) 的文献数据分别保存。参考条件和来源必须一起读；静态 DFT、低温实验和室温实验不能混为一谈。没有逐条核查预训练集，所以本项目不声称严格的未见材料泛化测试。

## 测试

不调用模型的测试覆盖晶胞原子数、体积与晶格常数关系、分数坐标保持、单位换算，以及具有已知参数的合成 EOS。合成数据只存在于测试中，不写入正式结果。

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not gpu" -q
```

真实 GPU 端到端 smoke 必须显式开启；未开启会清楚显示 `SKIPPED`。开启后 GPU 不可用会失败，不会改用 CPU：

```powershell
$env:MCE_GPU_TEST = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_smoke.py -q
Remove-Item Env:MCE_GPU_TEST
```

GitHub Actions 配置仅运行 CPU 逻辑测试；它不提供生产材料结果或 GPU 验证。工作流在 GitHub 的实际状态以该仓库的 Actions 页面为准。

## 常见问题

| 现象 | 具体处理 |
|---|---|
| `nvidia-smi` 不存在或报驱动错误 | 先安装适合该 GPU 的 NVIDIA 官方驱动并重启，再确认系统可识别 GPU；不要继续正式计算。 |
| `torch.cuda.is_available()` 为 `False` | 确认调用 `.venv` 的 Python；检查 `torch.__version__` 与 `torch.version.cuda`，若误装 CPU 版，按下方修复命令强制重装 CUDA wheel，再运行 `check`。 |
| CUDA out of memory | 关闭占显存程序并用 `nvidia-smi` 检查空闲显存，重新运行或恢复；不要静默降为 CPU 或混用精度数据。 |
| 模型下载失败 | 查看真实网络报错，检查是否可访问官方 release 链接；恢复网络后重试。不要用未知镜像替换权重。 |
| SHA256 不一致 | 停止加载；确认文件来自固定官方 release，移走损坏缓存后重新下载。不要跳过哈希检查。 |
| PyTorch 权重加载提示或 warning | 只加载本项目固定来源且通过哈希的官方模型；兼容性以锁定环境与真实 smoke 为准。未知文件不能作为模型替换。 |
| `ModuleNotFoundError` | 确认已在 `.venv` 内执行 `pip install -e ".[test]"`，检查 `pip check`，不要向系统环境盲目安装。 |
| Resume rejected | 找回原配置和依赖，或使用新输出目录；不要编辑指纹文件绕过一致性检查。 |
| 拟合/验证失败 | 读取 `run.log`、`status.json`、`validations.json`；失败仍会保留真实数据，不应把失败项描述为通过。 |

仅在确认项目环境误装了同版本 CPU wheel 时执行以下修复。`--force-reinstall` 可避免 pip 把相同基础版本判定为“已经安装”；`--no-deps` 保留项目其余已安装依赖。

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium check
```

更详细的物理解释见[中文原理说明](docs/principles_zh.md)。代码和文档供复现与学习使用；模型许可、适用范围及原始论文见[模型记录](docs/model_card.md)。
