"""可恢复的 GPU 扫描及独立验证。每完成一个点即持久化。"""
import csv
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import write
from scipy.optimize import minimize

from .backend import ROOT, MODEL_NAME, calculator, environment, evaluate, fetch_model, now, save_json, sha256
from .structures import MATERIALS, make_crystal, scale_volume
from .units import EV_A3_TO_GPA


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row if k != "stress_eV_A3"))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class Experiment:
    def __init__(self, config, output, resume=False):
        self.config, self.out = config, Path(output)
        if self.out.exists() and any(self.out.iterdir()) and not resume:
            raise ValueError(f"Output is not empty: {self.out}. Use --resume with matching configuration, or a new directory.")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "structures").mkdir(exist_ok=True)
        self.env = environment()
        self.model_path, self.model_meta = fetch_model()
        inference_hashes = {name: sha256(Path(__file__).parent/name) for name in ["backend.py","structures.py","units.py"]}
        signature = {"config": config, "model_sha256": self.model_meta["sha256"], "versions": self.env["versions"], "inference_source_hashes":inference_hashes, "schema_version": 1}
        signature["fingerprint"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        identity = self.out / "identity.json"
        if (self.out / "raw_points.json").exists() and not identity.exists():
            raise ValueError("Resume rejected: raw data exist without an identity manifest")
        if identity.exists():
            saved = json.loads(identity.read_text(encoding="utf-8"))
            if signature != saved:
                raise ValueError("Resume rejected: configuration, model hash or dependency versions differ")
        save_json(identity, signature)
        save_json(self.out / "config.json", config)
        if not (self.out / "environment.json").exists():
            save_json(self.out / "environment.json", self.env)
            (self.out / "requirements-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        save_json(self.out / "model.json", self.model_meta)
        source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        save_json(self.out / "source_hashes_latest_attempt.json", source_hashes)
        attempt = self.out / "attempts" / (now().replace(":","-") + ".json")
        save_json(attempt,{"timestamp":now(),"environment":self.env,"source_hashes":source_hashes,"resume":resume})
        self.log = logging.getLogger(str(self.out.resolve()))
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self.log.handlers.clear()
        handler = logging.FileHandler(self.out / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.log.addHandler(handler)
        self.log.addHandler(logging.StreamHandler(sys.stdout))
        raw_path = self.out / "raw_points.json"
        self.rows = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
        self.calc, self.calc_info = calculator(self.model_path, config["dtype"], config["head"])
        warm = [evaluate(make_crystal(config["material"]), self.calc)["seconds"] for _ in range(3)]
        self.calc_info["warmup_seconds"] = warm
        self.calc_info["warmup_repeats"] = 3
        save_json(self.out / "calculator.json", self.calc_info)
        self.log.info("GPU ready: %s; %s; %s", self.env["gpu_name"], MODEL_NAME, config["dtype"])

    def point(self, a_A, stage, key, calc=None):
        prior = next((r for r in self.rows if r["key"] == key), None)
        if prior is not None:
            if abs(prior["a_A"] - a_A) > 1e-9:
                raise RuntimeError("Resume point geometry mismatch")
            return prior
        atoms = make_crystal(self.config["material"], a_A)
        row = evaluate(atoms, calc or self.calc)
        row.update(stage=stage, key=key, dtype=self.config["dtype"], timestamp=now(), structure=f"structures/{key}.extxyz")
        write(self.out / row["structure"], atoms, format="extxyz")
        self.rows.append(row)
        save_json(self.out / "raw_points.json", self.rows)
        write_csv(self.out / "raw_points.csv", self.rows)
        self.log.info("%s a=%.7f A E=%.9f eV/atom P=%+.6f GPa (%.3fs)", key, row["a_A"], row["energy_eV_atom"], row["pressure_GPa"], row["seconds"])
        return row

    def scan(self):
        cfg = self.config
        a_ref = MATERIALS[cfg["material"]]["a_ref"]
        half = cfg["coarse_volume_fraction"]
        for expansion in range(5):
            coarse = [self.point(a_ref * ratio ** (1 / 3), "coarse", f"coarse{expansion}_{i:03d}")
                      for i, ratio in enumerate(np.linspace(max(0.5, 1 - half), 1 + half, cfg["coarse_points"]))]
            idx = int(np.argmin([r["energy_eV_atom"] for r in coarse]))
            if 0 < idx < len(coarse) - 1:
                break
            if cfg["mode"] == "smoke":
                self.log.info("Smoke minimum is on boundary; production must bracket before fitting.")
                return
            half *= 1.7
        else:
            raise RuntimeError("Minimum not bracketed after 5 expansions; no formal fit accepted")
        if cfg["mode"] == "production":
            # 粗拟合只用于定位细扫描中心；最终参数取细扫描。
            from .eos import fit_eos
            fit = fit_eos([r["volume_A3_atom"] for r in coarse], [r["energy_eV_atom"] for r in coarse], MATERIALS[cfg["material"]]["n_atoms"])
            center = fit["a0_A"]
            for i, ratio in enumerate(np.linspace(1 - cfg["fine_volume_fraction"], 1 + cfg["fine_volume_fraction"], cfg["fine_points"])):
                self.point(center * ratio ** (1 / 3), "fine", f"fine_{i:03d}")
            fine = [r for r in self.rows if r["stage"] == "fine"]
            idx = int(np.argmin([r["energy_eV_atom"] for r in fine]))
            if idx in (0, len(fine) - 1):
                raise RuntimeError("Fine minimum on boundary; expand configuration and start a new run")

    def validations(self, analysis):
        from .eos import fit_eos
        cfg, fit = self.config, analysis["primary"]
        a0 = fit["a0_A"]
        n = MATERIALS[cfg["material"]]["n_atoms"]
        path = self.out / "validations.json"
        checks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        def persist(name, value):
            # NumPy 2.x 的标量布尔值需要显式转成 Python JSON 类型。
            checks[name] = json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
            save_json(path, checks)
        # 保存实际计算点与每个检查的判据；判据只衡量数值一致性。
        eq = self.point(a0, "equilibrium", "eos_equilibrium")
        persist("forces", {"max_scan_force_eV_A": max(r["max_force_eV_A"] for r in self.rows), "tolerance_eV_A": 1e-3,
                           "passed": max(r["max_force_eV_A"] for r in self.rows) < 1e-3,
                           "reason": "Ideal diamond and fcc sites are symmetry stationary under isotropic cubic strain; numerical forces verified."})
        if "pressure" in cfg["validations"] and "pressure" not in checks:
            derivatives = []
            for h in [0.002, 0.001]:
                minus = self.point(a0 * (1-h)**(1/3), "pressure_validation", f"pressure_minus_{h}")
                plus = self.point(a0 * (1+h)**(1/3), "pressure_validation", f"pressure_plus_{h}")
                pressure = -(plus["energy_eV_atom"]-minus["energy_eV_atom"])/(plus["volume_A3_atom"]-minus["volume_A3_atom"])*EV_A3_TO_GPA
                derivatives.append({"fractional_volume_step": h, "energy_derivative_pressure_GPa": pressure,
                                    "stress_pressure_GPa": eq["pressure_GPa"], "difference_GPa": pressure-eq["pressure_GPa"]})
            persist("pressure", {"at_eos_V0_pressure_GPa": eq["pressure_GPa"], "central_differences": derivatives,
                "equilibrium_tolerance_GPa": 0.1, "derivative_tolerance_GPa": 0.02,
                "passed": abs(eq["pressure_GPa"]) < .1 and abs(derivatives[-1]["difference_GPa"]) < .02,
                "convention": "ASE tensile stress positive, P=-trace(stress)/3; stress eV/A^3; no kinetic stress."})
        if "optimize" in cfg["validations"] and "optimize" not in checks:
            history = []
            def objective(x):
                a = float(np.exp(x[0]))
                atoms = make_crystal(cfg["material"], a)
                row = evaluate(atoms, self.calc)
                row.update(step=len(history), structure=f"structures/opt_{len(history):03d}.extxyz")
                write(self.out / row["structure"], atoms)
                history.append(row)
                write_csv(self.out / "optimization.csv", history)
                # 对数边长是唯一自由度：dE/dln(a)=-3*V*P；保持立方和分数坐标。
                return row["energy_eV_cell"], np.array([-3*row["volume_A3_cell"]*row["pressure_GPa"]/EV_A3_TO_GPA])
            result = minimize(objective, [np.log(MATERIALS[cfg["material"]]["a_ref"]*.98)], jac=True,
                              method="BFGS", options={"gtol": 1e-5, "maxiter": 60})
            optimized_a = float(np.exp(result.x[0]))
            final = evaluate(make_crystal(cfg["material"], optimized_a), self.calc)
            persist("optimize", {"method": "Independent stress-gradient BFGS on ln(a), ideal fractional coordinates fixed, external P=0",
                "a0_A": optimized_a, "delta_a_vs_EOS_A": optimized_a-a0, "pressure_GPa": final["pressure_GPa"],
                "optimizer_success": bool(result.success), "message": str(result.message), "evaluations": len(history),
                "a_tolerance_A": .002, "pressure_tolerance_GPa": .01,
                "passed": abs(optimized_a-a0) < .002 and abs(final["pressure_GPa"]) < .01 and bool(result.success)})
        if "periodicity" in cfg["validations"] and "periodicity" not in checks:
            values = []
            for label, atoms in [("primitive",make_crystal(cfg["material"],a0,cubic=False)),
                                  ("conventional",make_crystal(cfg["material"],a0)),
                                  ("2x2x2_conventional",make_crystal(cfg["material"],a0).repeat((2,2,2)))]:
                row = evaluate(atoms,self.calc)
                row["cell"] = label
                write(self.out / "structures" / f"periodicity_{label}.extxyz",atoms)
                values.append(row)
            dE = float(np.ptp([r["energy_eV_atom"] for r in values]))
            stress_difference = max(float(np.abs(np.array(r["stress_eV_A3"])-np.array(values[1]["stress_eV_A3"])).max()) for r in values)*EV_A3_TO_GPA
            persist("periodicity", {"cells":values,"energy_spread_eV_atom":dE,"max_stress_difference_GPa":stress_difference,
                     "energy_tolerance_eV_atom":1e-6,"stress_tolerance_GPa":1e-4,"passed":dE<1e-6 and stress_difference<1e-4})
        if "precision" in cfg["validations"] and "precision" not in checks:
            other_dtype = "float32" if cfg["dtype"] == "float64" else "float64"
            other, metadata = calculator(self.model_path,other_dtype,cfg["head"])
            metadata["warmup_seconds"] = [evaluate(make_crystal(cfg["material"],a0),other)["seconds"] for _ in range(3)]
            rows = []
            for original in [r for r in self.rows if r["stage"] == "fine"]:
                atoms = make_crystal(cfg["material"],original["a_A"])
                row = evaluate(atoms,other)
                row.update(dtype=other_dtype, stage="precision",delta_energy_eV_atom=row["energy_eV_atom"]-original["energy_eV_atom"],
                           delta_pressure_GPa=row["pressure_GPa"]-original["pressure_GPa"])
                write(self.out / "structures" / f"precision_{len(rows):03d}.extxyz",atoms)
                rows.append(row)
                write_csv(self.out / "precision.csv",rows)
                save_json(self.out / "precision_raw.json",rows)
            other_fit = fit_eos([r["volume_A3_atom"] for r in rows],[r["energy_eV_atom"] for r in rows],n)
            da, db = other_fit["a0_A"]-a0, other_fit["B0_GPa"]-fit["B0_GPa"]
            write_csv(self.out / "precision.csv",rows)
            persist("precision", {"other_dtype":other_dtype,"calculator":metadata,"fit":other_fit,
                "delta_a_A":da,"delta_B_GPa":db,"max_abs_delta_energy_eV_atom":max(abs(r["delta_energy_eV_atom"]) for r in rows),
                "max_abs_delta_pressure_GPa":max(abs(r["delta_pressure_GPa"]) for r in rows),
                "other_median_seconds":float(np.median([r["seconds"] for r in rows])),
                "primary_median_seconds":float(np.median([r["seconds"] for r in self.rows if r["stage"]=="fine"])),
                "a_tolerance_A":.001,"B_tolerance_GPa":.5,"passed":abs(da)<.001 and abs(db)<.5})
            del other
        if "timing" in cfg["validations"] and "timing" not in checks:
            samples = [evaluate(make_crystal(cfg["material"],a0),self.calc)["seconds"] for _ in range(5)]
            import torch
            persist("timing", {"gpu":self.env["gpu_name"],"model":MODEL_NAME,"dtype":cfg["dtype"],"n_atoms":n,
                "repeats":5,"seconds":samples,"median_seconds":float(np.median(samples)),
                "max_memory_allocated_bytes_process":torch.cuda.max_memory_allocated(),
                "scope":"Synchronized wall time of ASE energy+forces+stress; includes graph construction and host overhead; excludes download/load/warmup; calculator cache reset each repeat; no CPU speedup claimed"})
        for name in ["pressure","optimize","periodicity","precision","timing"]:
            if name not in cfg["validations"]:
                persist(name,{"status":"skipped by configuration","passed":None})
        return checks

    def run(self):
        try:
            self.scan()
            if self.config["mode"] == "smoke":
                save_json(self.out / "status.json",{"status":"completed","mode":"smoke","formal_results":False,"n_points":len(self.rows),"cuda_verified":True})
                return
            analysis = refit(self.out, plots=False)
            checks = self.validations(analysis)
            analysis = refit(self.out, plots=True)
            failed = [name for name,value in checks.items() if value.get("passed") is False]
            save_json(self.out / "status.json",{"status":"completed" if not failed else "completed_with_validation_issues", "mode":"production","formal_results":True,"failed_validations":failed,"timestamp":now()})
            self.log.info("Completed: a0=%.8f A B0=%.6f GPa; issues=%s",analysis["primary"]["a0_A"],analysis["primary"]["B0_GPa"],failed)
        except Exception as exc:
            self.log.exception("Run interrupted")
            save_json(self.out / "status.json",{"status":"interrupted","error":str(exc),"timestamp":now()})
            raise


def refit(directory, plots=True):
    from .eos import analyze_rows
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    if config["mode"] != "production":
        raise ValueError("Smoke data are not formal EOS results")
    rows = json.loads((directory/"raw_points.json").read_text(encoding="utf-8"))
    if len([r for r in rows if r.get("stage")=="fine"]) != config["fine_points"]:
        raise ValueError("Fine scan is incomplete; resume GPU calculation before creating formal fits")
    analysis = analyze_rows(rows,MATERIALS[config["material"]]["n_atoms"])
    save_json(directory/"analysis.json",analysis)
    write_csv(directory/"fits.csv",analysis["fits"])
    if plots:
        from .plotting import plot_material,plot_structure
        plot_material(rows,analysis,directory/"figures",config["material"],f"{MODEL_NAME}, {config['dtype']}, static lattice")
        plot_structure(make_crystal(config["material"],analysis["primary"]["a0_A"]),directory/"figures"/"structure.png",config["material"])
    return analysis
