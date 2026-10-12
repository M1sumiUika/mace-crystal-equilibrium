"""轻量交付检查：读取真实结果、检查引用文件和 Git 中的意外大文件。"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    for symbol in ["Si","Al"]:
        directory=ROOT/"results"/"production"/symbol
        rows=json.loads((directory/"raw_points.json").read_text(encoding="utf-8"))
        status=json.loads((directory/"status.json").read_text(encoding="utf-8"))
        assert status["status"]=="completed",status
        assert len([r for r in rows if r["stage"]=="coarse"])==21
        assert len([r for r in rows if r["stage"]=="fine"])==21
        for row in rows:
            assert (directory/row["structure"]).is_file(),row["structure"]
        identity=json.loads((directory/"identity.json").read_text(encoding="utf-8"))
        for name,expected in identity["inference_source_hashes"].items():
            assert hashlib.sha256((ROOT/"src"/"mace_crystal_equilibrium"/name).read_bytes()).hexdigest()==expected,name
        print(f"{symbol}: 42 production scan points, all saved structures, matching inference source hashes")
    missing=[]
    for path in [ROOT/"README.md",ROOT/"README_EN.md",*list((ROOT/"docs").glob("*.md"))]:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            target=target.split("#")[0]
            if not (path.parent/target).exists():
                missing.append([str(path.relative_to(ROOT)),target])
    assert not missing,missing
    files=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
    forbidden=[p for p in files if p and (p.startswith((".cache/",".venv/")) or p.endswith((".model",".pt",".pth")))]
    assert not forbidden,forbidden
    total=sum((ROOT/p).stat().st_size for p in files if p)
    print(f"Markdown links OK; tracked artifacts {total/1024**2:.2f} MiB; no model/cache/environment tracked")


if __name__=="__main__":
    main()
