"""由真实计算数据生成论文风格英文图，避免中文字体依赖。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii

from .eos import evaluate_eos, pressure_eos


def _save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return str(outdir / f"{name}.png")


def _footer(fig, model_label):
    caption = f"{model_label} | static ideal crystal; no thermal or zero-point corrections"
    fig.text(0.5, -0.005, textwrap.fill(caption, width=108),
             ha="center", va="top", fontsize=8, color="#555555")


def plot_material(rows, analysis, outdir, material, model_label):
    """保存 EV、局部放大、残差、PV 与敏感性图，返回图路径字典。"""
    primary = analysis["primary"]
    stages = {}
    for stage in ("coarse", "fine"):
        selected = {r["volume_A3_atom"]: r for r in rows if r.get("stage") == stage}
        if selected:
            stages[stage] = [selected[v] for v in sorted(selected)]
    all_rows = [r for selected in stages.values() for r in selected]
    if not all_rows:
        raise ValueError("No coarse/fine scan rows available for plotting")
    energy_ref = min(float(r["energy_eV_atom"]) for r in all_rows)
    paths = {}
    style = {"coarse": ("#1768ac", "o"), "fine": ("#d95f02", "s")}
    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.1, 4.7), layout="constrained")
        for stage, selected in stages.items():
            color, marker = style[stage]
            ax.scatter([r["volume_A3_atom"] for r in selected],
                       [(r["energy_eV_atom"] - energy_ref) * 1000 for r in selected],
                       c=color, marker=marker, s=25, label=f"{stage.capitalize()} scan", zorder=3)
        for fitted in analysis["fits"]:
            if fitted["label"] not in (primary["label"], "coarse_full_BM3"):
                continue
            v = np.linspace(fitted["volume_min_A3_atom"], fitted["volume_max_A3_atom"], 300)
            ax.plot(v, (evaluate_eos(v, fitted) - energy_ref) * 1000,
                    color=style[fitted["stage"]][0], lw=1.8,
                    label=f"BM3 ({fitted['stage']})")
        ax.axvline(primary["V0_A3_atom"], ls=":", color="black", lw=1, label="Fitted equilibrium")
        ax.set(xlabel=r"Volume ($\mathrm{\AA^3/atom}$)", ylabel=r"$E-E_{\rm lowest\ sampled}$ (meV/atom)",
               title=f"{material}: energy-volume equation of state")
        ax.legend(fontsize=8)
        _footer(fig, model_label)
        paths["energy_volume"] = _save(fig, outdir, "energy_volume")

        fig, ax = plt.subplots(figsize=(7.1, 4.7), layout="constrained")
        selected = stages.get("fine", stages.get("coarse"))
        v = np.asarray([r["volume_A3_atom"] for r in selected])
        e = np.asarray([r["energy_eV_atom"] for r in selected])
        ax.scatter(v, (e - energy_ref) * 1000, color="#1768ac", s=26, label="Calculated", zorder=3)
        smooth = np.linspace(v.min(), v.max(), 350)
        for fitted in analysis["fits"][:2]:
            ax.plot(smooth, (evaluate_eos(smooth, fitted) - energy_ref) * 1000,
                    label="BM3" if fitted["eos"] == "birchmurnaghan" else "Vinet", lw=1.7)
        ax.axvline(primary["V0_A3_atom"], ls=":", lw=1, color="black")
        ax.text(0.5, 0.97, f"a0 = {primary['a0_A']:.5f} A\nB0 = {primary['B0_GPa']:.3f} GPa",
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set(xlabel=r"Volume ($\mathrm{\AA^3/atom}$)", ylabel=r"$E-E_{\rm lowest\ sampled}$ (meV/atom)",
               title=f"{material}: near-equilibrium fit")
        # 在上方留出说明区，避免 Si 右端最高体积点被图例遮住。
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.16 * (high - low))
        ax.legend(fontsize=8, loc="upper right")
        _footer(fig, model_label)
        paths["equilibrium_zoom"] = _save(fig, outdir, "equilibrium_zoom")

        fig, ax = plt.subplots(figsize=(7.1, 4.2), layout="constrained")
        for fitted in analysis["fits"][:2]:
            ax.plot(v, (e - evaluate_eos(v, fitted)) * 1000, "o-", ms=4, lw=1,
                    label="BM3" if fitted["eos"] == "birchmurnaghan" else "Vinet")
        ax.axhline(0, c="black", lw=0.8)
        ax.set(xlabel=r"Volume ($\mathrm{\AA^3/atom}$)", ylabel="Calculated - fitted energy (meV/atom)",
               title=f"{material}: fine-scan fit residuals")
        ax.legend()
        _footer(fig, model_label)
        paths["fit_residuals"] = _save(fig, outdir, "fit_residuals")

        fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.7), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})
        for stage, selected in stages.items():
            color, marker = style[stage]
            axes[0].scatter([r["volume_A3_atom"] for r in selected], [r["pressure_GPa"] for r in selected],
                            c=color, marker=marker, s=23, label=f"Stress ({stage})")
            comparison = [r for r in analysis["pressure_comparison"]["rows"] if r["stage"] == stage]
            if comparison:
                cv = [r["volume_A3_atom"] for r in comparison]
                axes[0].plot(cv, [r["pressure_energy_derivative_GPa"] for r in comparison],
                             c=color, ls="--", lw=1.1, label=f"-dE/dV ({stage})")
                interior = [r for r in comparison if not r["is_boundary_stencil"]]
                boundary = [r for r in comparison if r["is_boundary_stencil"]]
                axes[1].plot([r["volume_A3_atom"] for r in interior], [r["difference_GPa"] for r in interior],
                             color=color, marker=marker, ms=4, lw=1)
                axes[1].scatter([r["volume_A3_atom"] for r in boundary], [r["difference_GPa"] for r in boundary],
                                facecolors="none", edgecolors=color, marker=marker, s=28)
        axes[0].plot(smooth, pressure_eos(smooth, primary), color="black", lw=1.2, label="BM3 (fine fit)")
        axes[0].axhline(0, color="black", lw=0.7)
        axes[1].axhline(0, color="black", lw=0.7)
        axes[0].set(ylabel="Pressure (GPa)", title=f"{material}: pressure from stress and energy derivative")
        axes[1].set(xlabel=r"Volume ($\mathrm{\AA^3/atom}$)", ylabel="Stress - derivative\n(GPa)")
        axes[1].text(0.98, 1.02, "Open markers: boundary stencils", transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=8)
        axes[0].legend(fontsize=8, ncols=2)
        _footer(fig, model_label)
        paths["pressure_volume"] = _save(fig, outdir, "pressure_volume")

        fits = analysis["fits"]
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True, layout="constrained")
        y = np.arange(len(fits))
        labels = [f"{f['label'].replace('_', ' ')} (n={f['npoints']})" for f in fits]
        for ax, field, xlabel, color in (
            (axes[0], "delta_a0_A", r"$a_0-a_{0,\rm primary}$ ($\mathrm{\AA}$)", "#1768ac"),
            (axes[1], "delta_B0_GPa", r"$B_0-B_{0,\rm primary}$ (GPa)", "#d95f02"),
        ):
            ax.barh(y, [f[field] for f in fits], color=color, alpha=0.85)
            ax.axvline(0, color="black", lw=0.7)
            ax.set(xlabel=xlabel, yticks=y, yticklabels=labels)
        axes[0].invert_yaxis()
        fig.suptitle(f"{material}: numerical sensitivity (not model physical uncertainty)", fontsize=11)
        _footer(fig, model_label)
        paths["fit_sensitivity"] = _save(fig, outdir, "fit_sensitivity")
    return paths


def plot_structure(atoms, outpath, material):
    """画出真实输入常规晶胞；显示周期边界等价角点，仅用于结构示意。"""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    cell = np.asarray(atoms.cell)
    fractions = atoms.get_scaled_positions(wrap=True)
    shown = {}
    # 补上分数坐标为 0 的周期等价 1 边界，视觉上完整显示晶胞。
    from itertools import product

    for fractional, number in zip(fractions, atoms.numbers):
        choices = [[x, 1.0] if abs(x) < 1e-8 else [x] for x in fractional]
        for entry in product(*choices):
            shown[tuple(np.round(entry, 9))] = int(number)
    position = np.asarray(list(shown)) @ cell
    numbers = list(shown.values())
    fig = plt.figure(figsize=(6.2, 5.6), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    corners = np.asarray(list(product((0, 1), repeat=3)))
    cartesian = corners @ cell
    for i, corner in enumerate(corners):
        for j in range(i + 1, len(corners)):
            if np.sum(np.abs(corner - corners[j])) == 1:
                ax.plot(*np.vstack((cartesian[i], cartesian[j])).T, color="#777777", lw=1)
    # 最近邻连线仅作示意，不参与能量或键级计算。
    distances = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=2)
    nearest = float(np.min(distances[distances > 1e-6]))
    for i in range(len(position)):
        for j in range(i + 1, len(position)):
            if distances[i, j] < nearest * 1.05:
                ax.plot(*position[[i, j]].T, color="#b9c3cc", lw=1.8, alpha=0.8)
    ax.scatter(*position.T, s=[140 * covalent_radii[z] ** 2 for z in numbers],
               c="#1768ac" if material == "Si" else "#da842f", edgecolor="white", linewidth=0.7,
               depthshade=True)
    ax.set(xlabel="x (A)", ylabel="y (A)", zlabel="z (A)",
           title=f"{material}: conventional cubic cell ({len(atoms)} atoms/cell)\nBoundary-equivalent atoms shown for clarity")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=30)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def plot_reference_comparison(runs, reference_csv, outpath):
    """每种材料相对于自己的静态 PBE/WIEN2k 参考值比较，不混比绝对能量。

    runs 为 {"Si": Path(运行目录), "Al": Path(运行目录)}。
    只读取真实运行 analysis.json，并把每个百分比的分子和分母保存为 JSON。
    百分差不是通用精度指标，也不是经过训练集排除的泛化测试。
    """
    with Path(reference_csv).open(encoding="utf-8-sig", newline="") as stream:
        references = {
            row["material"]: row for row in csv.DictReader(stream)
            if row["reference_id"] == "delta_wien2k_pbe"
        }
    records = []
    labels = []
    for material, run in runs.items():
        reference = references.get(material)
        if reference is None:
            raise ValueError(f"No verified delta_wien2k_pbe reference for {material}")
        analysis = json.loads((Path(run) / "analysis.json").read_text(encoding="utf-8"))
        fit = analysis["primary"]
        config_path = Path(run) / "config.json"
        model_path = Path(run) / "model.json"
        if config_path.exists() and model_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            model = json.loads(model_path.read_text(encoding="utf-8"))
            labels.append(f"{model.get('model_name', model.get('name', 'MACE'))}; {cfg.get('dtype', 'dtype not recorded')}")
        entry = {
            "material": material, "reference_id": reference["reference_id"],
            "reference_source_url": reference["source_url"],
            "reference_conditions": reference["functional_conditions"],
            "run_directory": str(Path(run)),
        }
        for name in ("a0_A", "B0_GPa"):
            ref = float(reference[name])
            actual = float(fit[name])
            entry[f"calculated_{name}"] = actual
            entry[f"reference_{name}"] = ref
            entry[f"relative_difference_{name}_percent"] = 100 * (actual / ref - 1)
        records.append(entry)
    if not records:
        raise ValueError("At least one completed material run is needed")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), layout="constrained")
        x = np.arange(len(records))
        for ax, field, title in (
            (axes[0], "a0_A", "Equilibrium lattice constant"),
            (axes[1], "B0_GPa", "Bulk modulus"),
        ):
            values = [r[f"relative_difference_{field}_percent"] for r in records]
            colors = ["#1768ac" if r["material"] == "Si" else "#da842f" for r in records]
            bars = ax.bar(x, values, color=colors, width=0.56)
            span = max(max(abs(v) for v in values), 0.1)
            ax.set_ylim(-1.45 * span, 1.45 * span)
            ax.axhline(0, color="black", lw=0.9)
            ax.grid(axis="y", alpha=0.22)
            ax.set_axisbelow(True)
            ax.set(xticks=x, xticklabels=[r["material"] for r in records],
                   ylabel="(MACE / own reference - 1) (%)", title=title)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + np.sign(value or 1) * span * 0.05,
                        f"{value:+.3f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        fig.suptitle("Si and Al vs each material's static PBE / WIEN2k reference", fontsize=12)
        unique_labels = list(dict.fromkeys(labels))
        footnote = (" | ".join(unique_labels) if unique_labels else "MACE pretrained model")
        footnote += " | BM3 fits; static ideal crystals; no thermal / zero-point corrections.\n"
        footnote += "Reference comparison does not establish general model accuracy or unseen-material generalization."
        fig.text(0.5, -0.012, footnote, ha="center", va="top", fontsize=8, color="#555555")
        fig.savefig(outpath, dpi=220, bbox_inches="tight")
        fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
    payload = {
        "figure": str(outpath), "comparisons": records,
        "interpretation": "Relative differences against each material's own static PBE/WIEN2k reference; not a universal model-accuracy or generalization claim",
    }
    outpath.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
