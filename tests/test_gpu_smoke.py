"""显式选择运行的真实 CUDA 端到端检查；不使用合成能量。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mace_crystal_equilibrium.units import EV_A3_TO_GPA


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MCE_GPU_TEST") != "1",
    reason="真实 GPU smoke 未运行；设置 MCE_GPU_TEST=1 后执行此测试。",
)
def test_cuda_cli_smoke(tmp_path):
    import torch

    # 用户显式开启 GPU 测试后，没有 CUDA 应明确失败，不能静默跳过或回退 CPU。
    assert torch.cuda.is_available(), "GPU smoke requested, but CUDA is unavailable"
    output = tmp_path / "gpu_smoke"
    completed = subprocess.run(
        [sys.executable, "-m", "mace_crystal_equilibrium", "run",
         "--material", "Si", "--mode", "smoke", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert output.exists(), "CLI did not create the requested run directory"
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "calculator.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "raw_points.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["mode"] == "smoke" and status["formal_results"] is False
    assert status["cuda_verified"] is True
    assert metadata["parameter_devices"] and metadata["input_tensor_devices"]
    assert all(device.startswith("cuda") for device in metadata["parameter_devices"])
    assert all(device.startswith("cuda") for device in metadata["input_tensor_devices"])
    assert "stress" in metadata["implemented_properties"]
    assert metadata["dtype"] == "float64"
    assert len(rows) >= 3
    assert (output / "raw_points.csv").is_file()
    for row in rows:
        assert row["n_atoms"] == 8
        assert np.isfinite([row["energy_eV_atom"], row["pressure_GPa"], row["max_force_eV_A"]]).all()
        stress = np.asarray(row["stress_eV_A3"])
        assert stress.shape == (3, 3) and np.isfinite(stress).all()
        assert row["pressure_GPa"] == pytest.approx(-np.trace(stress) / 3 * EV_A3_TO_GPA)
        assert row["seconds"] > 0
        assert (output / row["structure"]).is_file()
