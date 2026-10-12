"""统一命令行入口：check / run / refit。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .backend import ROOT, environment, save_json


def main():
    parser = argparse.ArgumentParser(description="MACE crystal equilibrium: CUDA-only inference")
    sub = parser.add_subparsers(dest="command",required=True)
    check = sub.add_parser("check",help="Check actual GPU and software environment")
    check.add_argument("--output",type=Path)
    run = sub.add_parser("run",help="Run Si first, then Al; smoke data are never formal results")
    run.add_argument("--material",choices=["Si","Al","all"],default="Si")
    run.add_argument("--mode",choices=["smoke","production"],default="production")
    run.add_argument("--config",type=Path)
    run.add_argument("--output",type=Path)
    run.add_argument("--resume",action="store_true")
    run.add_argument("--dtype",choices=["float64","float32"])
    run.add_argument("--validations",help="Comma-separated pressure,optimize,periodicity,precision,timing or 'none'")
    run.add_argument("--skip-validation",action="append",default=[],choices=["pressure","optimize","periodicity","precision","timing"])
    refit = sub.add_parser("refit",help="Refit/plot existing raw data without GPU inference")
    refit.add_argument("directory",type=Path)
    args = parser.parse_args()
    if args.command == "check":
        info = environment(require_cuda=False)
        print(json.dumps(info,ensure_ascii=False,indent=2))
        if args.output:
            save_json(args.output,info)
        if not info["cuda_available"]:
            raise SystemExit("CUDA unavailable: formal inference disabled. See README troubleshooting.")
    elif args.command == "refit":
        from .runner import refit as refit_run
        print(json.dumps(refit_run(args.directory)["primary"],indent=2))
    else:
        from .runner import Experiment
        config = json.loads((args.config or ROOT/"configs"/f"{args.mode}.json").read_text(encoding="utf-8"))
        if args.config and config["mode"] != args.mode:
            parser.error("--mode must match the supplied configuration")
        if args.dtype:
            config["dtype"] = args.dtype
        if args.validations is not None:
            config["validations"] = [] if args.validations == "none" else args.validations.split(",")
        allowed = {"pressure","optimize","periodicity","precision","timing"}
        if set(config["validations"]) - allowed:
            parser.error("Unknown validation name")
        config["validations"] = [x for x in config["validations"] if x not in args.skip_validation]
        if config["device"] != "cuda" or config["model"] != "MACE-MP-0b3-medium":
            parser.error("This audited workflow requires cuda and the pinned MACE-MP-0b3-medium checkpoint")
        if config["mode"] == "production" and (config["coarse_points"]<15 or config["fine_points"]<9):
            parser.error("Production requires >=15 coarse and >=9 fine points")
        if args.resume and args.output is None:
            parser.error("--resume requires explicit --output")
        output = args.output or ROOT/"results"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+args.mode)
        materials = ["Si","Al"] if args.material == "all" else [args.material]
        for material in materials:
            cfg = dict(config,material=material)
            target = output/material if args.material == "all" else output
            # 初始化错误也保留证据；输出目录冲突时不能覆盖旧 status。
            try:
                exp = Experiment(cfg,target,args.resume)
                exp.run()
            except Exception as exc:
                if target.exists():
                    from .backend import now
                    save_json(target/("last_error_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),
                              {"error":str(exc),"timestamp":now(),"requested_config":cfg})
                raise
            if args.material == "all":
                status = json.loads((target/"status.json").read_text(encoding="utf-8"))
                if status.get("failed_validations"):
                    raise RuntimeError(f"{material} validations failed; inspect before continuing to next material")


if __name__ == "__main__":
    main()
