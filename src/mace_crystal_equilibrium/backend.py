"""GPU 检查、官方模型获取和带同步的真实推理。禁止 CPU 回退。"""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

MODEL_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model"
MODEL_SHA256 = "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012"
MODEL_NAME = "MACE-MP-0b3-medium"
ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(require_cuda=True):
    import torch
    versions = {}
    for pkg in ["mace-torch", "torch", "ase", "numpy", "scipy", "matplotlib", "e3nn"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        smi = str(exc)
    info = {"timestamp": now(), "os": platform.platform(), "python": sys.version,
            "versions": versions, "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "nvidia_smi": smi}
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info.update(gpu_name=props.name, total_memory_bytes=props.total_memory,
                    compute_capability=list(torch.cuda.get_device_capability(0)))
        # 实际分配和执行 CUDA 算子，而不只查询布尔标志。
        probe = torch.ones(4, device="cuda", dtype=torch.float64)
        info["cuda_float64_probe"] = float((probe @ probe).item())
        torch.cuda.synchronize()
    elif require_cuda:
        raise RuntimeError("CUDA unavailable: formal calculation stopped. Run nvidia-smi; install CUDA-enabled PyTorch in .venv using https://download.pytorch.org/whl/cu126, then mce check. CPU fallback is disabled.")
    return info


def fetch_model():
    path = ROOT / ".cache" / "models" / "mace-mp-0b3-medium.model"
    path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0
    downloaded = False
    if not path.exists():
        start = time.perf_counter()
        temp = path.with_suffix(".download")
        print(f"Downloading official model: {MODEL_URL}", flush=True)
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temp.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        elapsed = time.perf_counter() - start
        if sha256(temp) != MODEL_SHA256:
            raise RuntimeError("Downloaded model SHA256 mismatch; refusing to load")
        temp.replace(path)
        downloaded = True
        save_json(path.with_suffix(".download.json"), {"download_seconds": elapsed, "timestamp": now(), "url": MODEL_URL, "sha256": MODEL_SHA256})
    actual_hash = sha256(path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError("Cached model SHA256 mismatch; refusing to load")
    meta = {"name": MODEL_NAME, "release": "mace_mp_0b3", "size": "medium", "source": MODEL_URL,
            "sha256": actual_hash, "size_bytes": path.stat().st_size,
            "downloaded_this_process": downloaded, "download_seconds_this_process": elapsed,
            "scope": "periodic materials, MPTrj PBE(+U), 89 elements", "dispersion": False}
    history = path.with_suffix(".download.json")
    if history.exists():
        meta["original_download"] = json.loads(history.read_text(encoding="utf-8"))
    return path, meta


def calculator(model_path, dtype="float64", head="default"):
    import torch
    from mace.calculators import MACECalculator
    environment(require_cuda=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    start = time.perf_counter()
    calc = MACECalculator(model_paths=str(model_path), device="cuda", default_dtype=dtype, head=head)
    torch.cuda.synchronize()
    if getattr(calc, "head", None) != head:
        raise RuntimeError(f"MACE selected head {getattr(calc, 'head', None)!r}, requested {head!r}")
    devices = sorted({str(p.device) for model in calc.models for p in model.parameters()})
    if not devices or any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"Model is not completely on CUDA: {devices}")
    if "stress" not in calc.implemented_properties:
        raise RuntimeError("This MACE/ASE calculator does not implement stress")
    info = {"initialization_seconds": time.perf_counter() - start,
            "parameter_devices": devices, "dtype": dtype, "head": calc.head,
            "parameter_dtypes": sorted({str(p.dtype) for model in calc.models for p in model.parameters()}),
            "available_heads": list(getattr(calc.models[0], "heads", [])),
            "parameters": sum(p.numel() for p in calc.models[0].parameters()),
            "implemented_properties": list(calc.implemented_properties),
            "input_tensor_devices": []}
    def verify_input(module, args):
        if args and isinstance(args[0], dict):
            found = sorted({str(v.device) for v in args[0].values() if isinstance(v, torch.Tensor)})
            info["input_tensor_devices"] = found
            if any(not d.startswith("cuda") for d in found):
                raise RuntimeError(f"Non-CUDA input tensor: {found}")
    calc.models[0].register_forward_pre_hook(verify_input)
    return calc, info


def evaluate(atoms, calc):
    import numpy as np
    import torch
    atoms.calc = calc
    # 重置 ASE 缓存，保证重复计时确实执行模型。
    calc.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    from .units import EV_A3_TO_GPA
    if stress.shape != (3, 3) or not np.isfinite(stress).all():
        raise RuntimeError("Stress tensor missing or non-finite")
    volume, n = atoms.get_volume(), len(atoms)
    row = {"n_atoms": n, "a_A": float((volume * (8 if atoms.symbols[0] == "Si" else 4) / n) ** (1 / 3)),
           "volume_A3_cell": volume, "volume_A3_atom": volume / n,
           "energy_eV_cell": float(energy), "energy_eV_atom": float(energy / n),
           "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
           "pressure_GPa": float(-np.trace(stress) / 3 * EV_A3_TO_GPA),
           "seconds": seconds, "stress_eV_A3": stress.tolist()}
    for i, x in enumerate("xyz"):
        for j, y in enumerate("xyz"):
            row[f"stress_{x}{y}_eV_A3"] = float(stress[i, j])
    if not np.isfinite([energy, volume, row["max_force_eV_A"]]).all():
        raise RuntimeError("Non-finite inference output")
    return row
