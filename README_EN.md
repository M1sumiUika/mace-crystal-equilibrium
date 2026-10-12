# mace-crystal-equilibrium

Private repository: [M1sumiUika/mace-crystal-equilibrium](https://github.com/M1sumiUika/mace-crystal-equilibrium). Actual local test result: **18 passed**, including the CUDA end-to-end smoke test; see [test output](results/test_summary.txt).

Against the verified static PBE/WIEN2k references, predicted bulk moduli are **8.42% lower for Si** and **25.13% lower for Al**. These differences exceed the tested numerical sensitivity; consistency checks do not establish physical accuracy.

CUDA inference with the official **MACE-MP-0b3 medium** pretrained force field to estimate and validate equilibrium properties of diamond Si and fcc Al. This is a pretrained MLFF application, not MACE training or a new DFT study.

The scope is a fixed crystalline phase at the **static-lattice, approximately zero-temperature limit**, without zero-point or finite-temperature corrections. Predictions must not be described as room-temperature properties or assumed to have DFT accuracy.

[中文说明](README.md) · [Actual calculation report, Chinese](docs/report_zh.md) · [Model provenance](docs/model_card.md) · [Physical principles, Chinese](docs/principles_zh.md) · [Literature references](references/) · [GitHub upload](docs/github_upload.md)

## Completed calculations

Si and Al production calculations were completed sequentially on the local NVIDIA GPU on 2026-10-12. These values come from the fine-scan third-order Birch–Murnaghan fits with float64 inference:

| Phase | a₀ / Å | v₀ / Å³ per atom | B₀ / GPa | Default numerical validation |
|---|---:|---:|---:|---|
| Diamond Si | 5.470994 | 20.469569 | 81.0858 | All passed |
| fcc Al | 4.063224 | 16.770748 | 58.4536 | All passed |

Passing means numerical consistency within recorded thresholds, not general agreement with DFT or experiment. See the [actual report](docs/report_zh.md) for reference comparisons and [execution notes](docs/execution_notes.md) for the initial head-name fix and the genuine interrupted/resumed Si run.

![Si energy-volume scan and EOS fits](results/production/Si/figures/energy_volume.png)

## Environment and installation

The actual execution platform is native Windows 11, Python 3.12.6, NVIDIA GeForce RTX 4060 Laptop GPU with approximately 8 GB VRAM, driver 616.64, PyTorch 2.6.0 with CUDA 12.6 runtime, MACE 0.3.16 and ASE 3.26.0. Each run saves detailed environment and dependency records. Linux, WSL2 and other GPUs have not been tested in this project.

Run PowerShell from the repository root. The project environment does not modify globally installed packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Alternatively run `powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1`. To reproduce the recorded Windows dependencies more closely, install CUDA PyTorch first, then `pip install -r requirements-lock.txt` and `pip install --no-deps -e .` inside `.venv`.

## Run

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium check --output results/environment_check.json
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode smoke --output results/my_smoke_Si
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --output results/my_production/Si
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Al --mode production --output results/my_production/Al
```

Inspect Si validation before running Al. `--material all --output results/my_production_all` runs Si then Al, stopping if Si validation fails. New output directories preserve the delivered [Si](results/production/Si/) and [Al](results/production/Al/) results. The activated environment also exposes the equivalent `mce` entry point.

GPU inference always requests `device="cuda"` and verifies parameter and input tensor devices. Missing CUDA stops inference; there is no CPU fallback. The official model is downloaded into ignored `.cache/models/` and checked against a fixed SHA256. Download, initialization, warmup and synchronized inference times are recorded separately.

## Workflow and options

Production uses conventional cubic cells (8 atoms for Si, 4 for Al), 21 initial volume points over ±10%, automatic coarse-range expansion if needed, and 21 fine points over ±4% near the fitted minimum. Fractional coordinates stay fixed under isotropic volume changes. Main predictions use float64. Third-order Birch–Murnaghan is the primary EOS; Vinet and alternate ranges/point counts quantify numerical sensitivity.

Default checks compare stress pressure with energy finite differences, verify pressure near the fitted minimum, optimize the cubic cell independently using stress gradients, compare primitive/conventional/supercells, and repeat the fine scan in float32 for a precision comparison. Five synchronized, uncached inference repeats provide timing context. CPU speedup is not claimed.

```powershell
# Select checks, or omit one check; use a new output directory.
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --validations pressure,optimize --output results/selected_checks
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --skip-validation timing --output results/no_timing
# Continue with exactly the original model, configuration and dependencies.
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium run --material Si --mode production --output results/my_production/Si --resume
# Refit and plot recorded production data without model inference.
.\.venv\Scripts\python.exe -m mace_crystal_equilibrium refit results/production/Si
```

Validation names are `pressure,optimize,periodicity,precision,timing`; `--validations none` omits optional checks and `--skip-validation` is repeatable. Omitted checks are reported as skipped, not passed. `--dtype float32` changes the primary precision and compares against float64. `--config path.json` loads a copied configuration; its mode must match `--mode`. Resume rejects mismatched configuration, model hash or recorded dependency versions. Refit overwrites analysis and plots, preserves raw data, and does not rerun checks or regenerate the project report.

Smoke uses five points and explicitly records `formal_results: false`; it cannot be used for formal EOS refitting. Minimum production counts are 15 coarse and 9 fine points.

## Outputs and interpretation

Each run contains `config.json`, `identity.json`, `environment.json`, `requirements-freeze.txt`, `model.json`, `calculator.json`, raw CSV/JSON data, extended XYZ structures, `analysis.json`, `fits.csv`, `validations.json`, plots, `status.json` and a log. Optimization and precision checks have separate CSV files. Read `status.json` and each validation's measured values and thresholds before citing a result.

Fits consistently use per-atom energy and volume. The conventional lattice parameter is recovered as `a = (N * v)^(1/3)`. ASE tensile stress is positive; pressure is `-trace(stress)/3`. Both routes use `1 eV/Å³ = 160.2176634 GPa`. Energies are plotted relative to each material's own minimum, never compared across elements as absolute energies.

Literature data are stored separately with phase, method/temperature and original sources. Numerical fit sensitivity is not model uncertainty. These common materials have not been audited against the complete training set, so this is not claimed as a strict unseen-material generalization test.

## Tests and troubleshooting

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not gpu" -q
$env:MCE_GPU_TEST = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_smoke.py -q
Remove-Item Env:MCE_GPU_TEST
```

CPU tests cover atom counts, cell scaling, units and synthetic EOS recovery. Synthetic data are used only in tests. GPU testing is opt-in; after opting in, missing CUDA fails rather than skips. The GitHub Actions configuration runs CPU logic tests only and does not imply a hosted GPU or a completed remote CI run.

For unavailable CUDA, first check `nvidia-smi`, then the `.venv` interpreter, `torch.__version__` and `torch.version.cuda`. Install the CUDA wheel in `.venv` if a CPU wheel was installed. For out-of-memory errors, free GPU memory and resume with the same configuration. For failed downloads, restore access to the pinned official release; do not substitute unknown mirrors or bypass the hash. Read retained logs for dependency, fit or validation failures. Detailed Chinese guidance is in [README.md](README.md).
