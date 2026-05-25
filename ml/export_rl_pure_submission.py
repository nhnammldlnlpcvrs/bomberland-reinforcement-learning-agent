"""Export the isolated rl_pure submission zip."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "agent" / "hybrid_agent_rl_pure"


def export(checkpoint, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = ["agent.py", "model.py", "config.json"]
    missing = [name for name in files if not (AGENT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            zf.write(AGENT_DIR / name, arcname=name)
        ckpt = Path(checkpoint) if checkpoint else AGENT_DIR / "rl_pure_model.pth"
        if ckpt.exists():
            zf.write(ckpt, arcname="rl_pure_model.pth")
    with zipfile.ZipFile(out, "r") as zf:
        names = zf.namelist()
        agent_files = [name for name in names if Path(name).name == "agent.py"]
        if agent_files != ["agent.py"]:
            raise RuntimeError(f"Zip must contain exactly one root agent.py, got {agent_files}")
        if len(names) > 4:
            raise RuntimeError(f"Unexpected file count: {names}")
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt zip member: {bad}")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Exported {out} ({size_mb:.2f} MB) with files: {names}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="dist/rl_pure_submission.zip")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
