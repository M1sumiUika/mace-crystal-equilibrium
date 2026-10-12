"""每原子能量—体积的状态方程与独立压力检查。

所有拟合统一使用 eV/atom 和 Å³/atom，因而曲率给出 eV/Å³。
Birch–Murnaghan 的实现来自 ASE；Vinet 使用等价的数值稳定写法。
公式来源：https://docs.ase-lib.org/_modules/ase/eos.html
本模块不把拟合残差、参数敏感性或协方差解释为模型物理不确定度。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from ase.eos import birchmurnaghan
from scipy.optimize import least_squares

# 由 SI 定义的电子伏特与 Å³ 转换，1 eV/Å³ = 160.2176634 GPa。
EV_A3_TO_GPA = 160.2176634


def _vinet(volume, e0, b0, bp, v0):
    """Vinet 能量，包含 B'=1 的连续极限，避免接近平衡处相消。"""
    x = np.cbrt(np.asarray(volume, dtype=float) / v0)
    y = 1.0 - x
    b = 1.5 * (bp - 1.0)
    z = b * y
    # [1 + (z-1) exp(z)] / z² 在 z=0 处趋于 1/2。
    small = np.abs(z) < 1.0e-3
    factor = np.empty_like(z)
    factor[small] = (
        0.5 + z[small] / 3.0 + z[small] ** 2 / 8.0
        + z[small] ** 3 / 30.0 + z[small] ** 4 / 144.0
    )
    factor[~small] = (
        z[~small] * np.exp(z[~small]) - np.expm1(z[~small])
    ) / z[~small] ** 2
    return e0 + 9.0 * b0 * v0 * y**2 * factor


_FUNCTIONS = {"birchmurnaghan": birchmurnaghan, "vinet": _vinet}


def fit_eos(volumes_per_atom, energies_per_atom, natoms_conventional, eos="birchmurnaghan"):
    """拟合静态 EOS，返回可直接保存为 JSON 的参数与数值诊断。

    ``natoms_conventional`` 仅用于 a₀=(N V₀)^(1/3)，Si 为 8，Al 为 4。
    四参数非线性拟合至少需要 5 个不同体积点。拟合使用能量偏移，
    降低较大绝对能量对停止条件的影响；不改变任何能量差。
    """
    if eos not in _FUNCTIONS:
        raise ValueError(f"Unsupported EOS {eos!r}; choose {list(_FUNCTIONS)}")
    v = np.asarray(volumes_per_atom, dtype=float)
    e = np.asarray(energies_per_atom, dtype=float)
    if v.ndim != 1 or e.shape != v.shape or len(v) < 5:
        raise ValueError("EOS fitting needs matching 1-D arrays with at least 5 points")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(e)) or np.any(v <= 0):
        raise ValueError("Volumes must be positive; volumes and energies must be finite")
    if len(np.unique(v)) != len(v):
        raise ValueError("Duplicate volumes must be resolved before fitting")
    if int(natoms_conventional) != natoms_conventional or natoms_conventional <= 0:
        raise ValueError("natoms_conventional must be a positive integer")
    order = np.argsort(v)
    v, e = v[order], e[order]
    offset = float(e.min())
    shifted = e - offset
    center = float(v.mean())
    quadratic = np.polyfit(v - center, shifted, 2)
    minimum_index = int(np.argmin(shifted))
    v_guess = float(v[minimum_index])
    if quadratic[0] > 0:
        stationary = center - quadratic[1] / (2.0 * quadratic[0])
        if v[0] < stationary < v[-1]:
            v_guess = float(stationary)
    b_guess = float(np.clip(2.0 * quadratic[0] * v_guess, 0.01, 5.0))
    lower = np.array([-np.inf, 1.0e-7, -20.0, 0.5 * v[0]])
    upper = np.array([np.inf, 20.0, 30.0, 1.5 * v[-1]])
    function = _FUNCTIONS[eos]
    # 用 meV 量级残差改善优化器绝对停止阈值；输出仍为 eV/atom。
    solution = least_squares(
        lambda p: 1000.0 * (function(v, *p) - shifted),
        x0=[float(shifted.min()), b_guess, 4.0, v_guess],
        bounds=(lower, upper), x_scale="jac", ftol=1e-13, xtol=1e-13,
        gtol=1e-11, max_nfev=10000,
    )
    if not solution.success:
        raise RuntimeError(f"{eos} fit failed: {solution.message}")
    e0_shift, b0, bp, v0 = map(float, solution.x)
    residual = (function(v, *solution.x) - shifted) * 1000.0
    bracketed = bool(v[0] < v0 < v[-1])
    measured_minimum_interior = bool(0 < minimum_index < len(v) - 1)
    warnings = []
    if not bracketed:
        warnings.append("Fitted minimum lies outside sampled volume interval")
    if not measured_minimum_interior:
        warnings.append("Lowest sampled energy is at a scan boundary")
    if not 0.0 < bp < 12.0:
        warnings.append("Bprime outside broad diagnostic interval (0, 12); inspect fit")
    if np.any(solution.active_mask):
        warnings.append("At least one fit parameter reached an optimization bound")
    curvature = b0 / v0
    return {
        "eos": eos, "E0_eV_atom": e0_shift + offset,
        "V0_A3_atom": v0, "V0_A3_cell": v0 * int(natoms_conventional),
        "B0_eV_A3": b0, "B0_GPa": b0 * EV_A3_TO_GPA,
        "Bprime": bp, "a0_A": (v0 * int(natoms_conventional)) ** (1.0 / 3.0),
        "natoms_conventional": int(natoms_conventional), "npoints": int(len(v)),
        "volume_min_A3_atom": float(v[0]), "volume_max_A3_atom": float(v[-1]),
        "rmse_meV_atom": float(np.sqrt(np.mean(residual**2))),
        "max_residual_meV_atom": float(np.max(np.abs(residual))),
        "curvature_eV_per_A6_atom": curvature,
        "minimum_bracketed": bracketed,
        "sampled_minimum_interior": measured_minimum_interior,
        "positive_curvature": bool(curvature > 0),
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "jacobian_condition_number": float(np.linalg.cond(solution.jac)),
        "fit_warnings": warnings,
        "numerical_diagnostics_note": (
            "Fit residuals and EOS/range/point-count sensitivity are numerical diagnostics, "
            "not physical uncertainty estimates for the pretrained model."
        ),
    }


def evaluate_eos(volumes, fit):
    """给定每原子体积 Å³，返回每原子能量 eV。"""
    return _FUNCTIONS[fit["eos"]](
        np.asarray(volumes, dtype=float), fit["E0_eV_atom"],
        fit["B0_eV_A3"], fit["Bprime"], fit["V0_A3_atom"],
    )


def pressure_eos(volumes, fit):
    """解析 -dE/dV，返回 GPa；压缩为正、拉伸为负。"""
    v = np.asarray(volumes, dtype=float)
    if np.any(v <= 0):
        raise ValueError("Volumes must be positive")
    v0, b0, bp = fit["V0_A3_atom"], fit["B0_eV_A3"], fit["Bprime"]
    if fit["eos"] == "birchmurnaghan":
        eta = np.cbrt(v0 / v)
        pressure = 1.5 * b0 * (eta**7 - eta**5) * (1 + 0.75 * (bp - 4) * (eta**2 - 1))
    elif fit["eos"] == "vinet":
        x = np.cbrt(v / v0)
        pressure = 3 * b0 * (1 - x) / x**2 * np.exp(1.5 * (bp - 1) * (1 - x))
    else:
        raise ValueError(f"Unsupported EOS {fit['eos']!r}")
    return pressure * EV_A3_TO_GPA


def _stage_rows(rows: Iterable[dict], stage):
    """按体积去重：续算重复行不应得到额外统计权重。"""
    selected = {}
    for row in rows:
        if row.get("stage") == stage:
            selected[float(row["volume_A3_atom"])] = row
    return [selected[key] for key in sorted(selected)]


def _numerical_pressure(rows):
    v = np.asarray([r["volume_A3_atom"] for r in rows], dtype=float)
    e = np.asarray([r["energy_eV_atom"] for r in rows], dtype=float)
    if len(v) < 3:
        return []
    # 5 点局部四次多项式的导数；它不使用 EOS 或应力信息。
    # 先平移能量和缩放体积，避免绝对能量与 Vandermonde 尺度问题。
    derivatives = []
    width = min(5, len(v))
    for i in range(len(v)):
        start = min(max(0, i - width // 2), len(v) - width)
        indexes = slice(start, start + width)
        local_v = v[indexes] - v[i]
        scale = float(np.max(np.abs(local_v)))
        coefficients = np.polynomial.polynomial.polyfit(
            local_v / scale, e[indexes] - e[i], deg=width - 1,
        )
        derivatives.append(coefficients[1] / scale)
    pressure = -np.asarray(derivatives) * EV_A3_TO_GPA
    output = []
    for i, (row, p) in enumerate(zip(rows, pressure)):
        stress_p = float(row["pressure_GPa"])
        output.append({
            "stage": row["stage"], "volume_A3_atom": float(v[i]),
            "pressure_stress_GPa": stress_p,
            "pressure_energy_derivative_GPa": float(p),
            "difference_GPa": stress_p - float(p),
            "is_boundary_stencil": bool(i < width // 2 or i >= len(v) - width // 2),
            "stencil_points": width,
        })
    return output


def analyze_rows(rows, natoms):
    """主拟合、数值敏感性及独立的应力—能量导数比较。"""
    coarse, fine = (_stage_rows(rows, s) for s in ("coarse", "fine"))
    primary_rows = fine if len(fine) >= 5 else coarse
    if len(primary_rows) < 5:
        raise ValueError("At least 5 points in one scan stage are needed for EOS analysis")
    fits = []
    errors = []

    def add_fit(label, selected, eos):
        if len(selected) < 5:
            return
        try:
            result = fit_eos(
                [r["volume_A3_atom"] for r in selected],
                [r["energy_eV_atom"] for r in selected], natoms, eos=eos,
            )
            result["label"] = label
            result["stage"] = selected[0]["stage"]
            fits.append(result)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            errors.append({"label": label, "error": str(exc)})

    primary_stage = primary_rows[0]["stage"]
    add_fit(f"{primary_stage}_full_BM3", primary_rows, "birchmurnaghan")
    if not fits:
        raise RuntimeError(f"Primary EOS fit failed: {errors}")
    primary = fits[0]
    add_fit(f"{primary_stage}_full_Vinet", primary_rows, "vinet")
    sparse = primary_rows[::2]
    if sparse[-1] is not primary_rows[-1]:
        sparse = sparse + primary_rows[-1:]
    add_fit(f"{primary_stage}_half_points_BM3", sparse, "birchmurnaghan")
    # 缩窄至主拟合 V0 附近，保证比较对象仍包含极小值两侧。
    v0 = primary["V0_A3_atom"]
    half_width = 0.65 * min(
        v0 - primary_rows[0]["volume_A3_atom"],
        primary_rows[-1]["volume_A3_atom"] - v0,
    )
    narrow = [r for r in primary_rows if abs(r["volume_A3_atom"] - v0) <= half_width]
    add_fit(f"{primary_stage}_narrow_BM3", narrow, "birchmurnaghan")
    if primary_stage == "fine":
        add_fit("coarse_full_BM3", coarse, "birchmurnaghan")
        add_fit("coarse_full_Vinet", coarse, "vinet")
    for fitted in fits:
        fitted["delta_a0_A"] = fitted["a0_A"] - primary["a0_A"]
        fitted["delta_B0_GPa"] = fitted["B0_GPa"] - primary["B0_GPa"]
    pressure_rows = _numerical_pressure(coarse) + _numerical_pressure(fine)
    stage_summaries = {}
    for stage in ("coarse", "fine"):
        interior = [r["difference_GPa"] for r in pressure_rows if r["stage"] == stage and not r["is_boundary_stencil"]]
        if interior:
            stage_summaries[stage] = {
                "n_interior_points": len(interior),
                "rmse_GPa": float(np.sqrt(np.mean(np.square(interior)))),
                "max_abs_difference_GPa": float(np.max(np.abs(interior))),
            }
    comparison = {
        "method": "Local 5-point degree-4 polynomial derivative (fewer points if unavailable); independent of EOS",
        "sign_convention": "ASE stress is positive in tension; P = -trace(stress)/3. Compression-positive P = -dE/dV.",
        "normalization": "E in eV/atom and V in A^3/atom; pressure multiplied by 160.2176634 to GPa",
        "boundary_note": "First/last two points use one-sided/asymmetric stencils and are excluded from reported interior errors; finite-difference truncation and energy noise remain.",
        "comparison_stage": primary_stage, "rows": pressure_rows,
        "stage_summaries": stage_summaries,
        **stage_summaries.get(primary_stage, {}),
    }
    return {
        "primary": primary, "fits": fits, "fit_errors": errors,
        "pressure_comparison": comparison,
        "sensitivity": {
            "a0_span_A": float(np.ptp([f["a0_A"] for f in fits])),
            "B0_span_GPa": float(np.ptp([f["B0_GPa"] for f in fits])),
            "interpretation": "Numerical sensitivity to range, point count and EOS; not MLFF physical uncertainty",
        },
    }
