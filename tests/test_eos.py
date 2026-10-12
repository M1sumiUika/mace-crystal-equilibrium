"""纯合成数据测试：这里没有 MACE 预测或材料研究结果。"""

import numpy as np
import pytest
from ase.eos import vinet as ase_vinet

from mace_crystal_equilibrium.eos import (
    EV_A3_TO_GPA, analyze_rows, evaluate_eos, fit_eos, pressure_eos,
)


def synthetic_parameters(eos):
    return {
        "eos": eos, "E0_eV_atom": -5.4, "V0_A3_atom": 20.3,
        "B0_eV_A3": 0.62, "Bprime": 4.35,
    }


@pytest.mark.parametrize("eos", ["birchmurnaghan", "vinet"])
def test_recovers_known_synthetic_parameters(eos):
    known = synthetic_parameters(eos)
    volume = known["V0_A3_atom"] * np.linspace(0.90, 1.10, 25)
    fit = fit_eos(volume, evaluate_eos(volume, known), 8, eos=eos)
    for key in ("E0_eV_atom", "V0_A3_atom", "B0_eV_A3", "Bprime"):
        assert fit[key] == pytest.approx(known[key], rel=2e-7, abs=2e-8)
    assert fit["a0_A"] == pytest.approx((8 * known["V0_A3_atom"]) ** (1 / 3))
    assert fit["rmse_meV_atom"] < 1e-6
    assert fit["minimum_bracketed"] and fit["positive_curvature"]


@pytest.mark.parametrize("eos", ["birchmurnaghan", "vinet"])
def test_pressure_is_negative_energy_derivative(eos):
    known = synthetic_parameters(eos)
    volume = np.array([18.9, 20.3, 21.5])
    h = 2e-4
    numerical = -(evaluate_eos(volume + h, known) - evaluate_eos(volume - h, known)) / (2 * h)
    assert pressure_eos(volume, known) == pytest.approx(numerical * EV_A3_TO_GPA, abs=1e-7)
    assert pressure_eos(np.array([known["V0_A3_atom"]]), known)[0] == pytest.approx(0, abs=1e-12)
    assert pressure_eos(volume, known)[0] > 0
    assert pressure_eos(volume, known)[-1] < 0


def test_vinet_matches_ase_and_has_continuous_bp_one_limit():
    known = synthetic_parameters("vinet")
    volume = np.linspace(18, 23, 17)
    expected = ase_vinet(volume, known["E0_eV_atom"], known["B0_eV_A3"], known["Bprime"], known["V0_A3_atom"])
    assert evaluate_eos(volume, known) == pytest.approx(expected, abs=1e-12)
    known["Bprime"] = 1.0
    assert np.all(np.isfinite(evaluate_eos(volume, known)))
    assert evaluate_eos(20.3, known) == pytest.approx(known["E0_eV_atom"])


def test_energy_volume_normalization_and_pressure_conversion():
    known = synthetic_parameters("birchmurnaghan")
    volume = np.linspace(18.5, 22, 17)
    energy = evaluate_eos(volume, known)
    fit = fit_eos(volume, energy, 8)
    assert fit["B0_GPa"] == pytest.approx(0.62 * 160.2176634, rel=1e-7)
    assert fit["V0_A3_cell"] == pytest.approx(8 * fit["V0_A3_atom"])
    # 同步将 E 与 V 换成整胞归一化，体模量仍保持不变。
    total_fit = fit_eos(volume * 8, energy * 8, 1)
    assert total_fit["B0_GPa"] == pytest.approx(fit["B0_GPa"], rel=1e-7)


def test_analysis_sensitivity_and_independent_pressure_on_synthetic_rows():
    known = synthetic_parameters("birchmurnaghan")
    rows = []
    for stage, ratios in (("coarse", np.linspace(.90, 1.10, 21)), ("fine", np.linspace(.97, 1.03, 25))):
        for volume in known["V0_A3_atom"] * ratios:
            rows.append({"stage": stage, "volume_A3_atom": float(volume),
                         "energy_eV_atom": float(evaluate_eos(volume, known)),
                         "pressure_GPa": float(pressure_eos(volume, known))})
    analysis = analyze_rows(rows, 8)
    assert analysis["primary"]["stage"] == "fine"
    assert len(analysis["fits"]) >= 6
    assert analysis["pressure_comparison"]["max_abs_difference_GPa"] < 1e-5
    assert analysis["pressure_comparison"]["n_interior_points"] == 21
    assert any(r["is_boundary_stencil"] for r in analysis["pressure_comparison"]["rows"])


def test_invalid_inputs_and_unbracketed_scan_are_explicit():
    with pytest.raises(ValueError, match="at least 5"):
        fit_eos([1, 2, 3], [3, 2, 1], 8)
    with pytest.raises(ValueError, match="Duplicate"):
        fit_eos([1, 2, 3, 3, 4], [4, 3, 2, 2, 1], 8)
    known = synthetic_parameters("birchmurnaghan")
    volume = np.linspace(21, 23, 11)
    fit = fit_eos(volume, evaluate_eos(volume, known), 8)
    assert not fit["minimum_bracketed"]
    assert not fit["sampled_minimum_interior"]
    assert fit["fit_warnings"]
