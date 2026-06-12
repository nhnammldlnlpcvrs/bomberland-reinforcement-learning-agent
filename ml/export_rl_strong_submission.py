from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = (
    "agent.py",
    "action_mask.py",
    "constants.py",
    "encoder.py",
    "frame_buffer.py",
    "metadata.json",
    "model.py",
    "policy.py",
    "utils.py",
    "policy.zip",
)


def export_submission(source: Path, output_dir: Path, zip_path: Path) -> None:
    if not (source / "agent.py").exists():
        raise FileNotFoundError(f"Missing required agent.py in {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()

    copied = []
    for filename in DEFAULT_FILES:
        src = source / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)

    if "policy.zip" not in copied:
        print("warning: policy.zip not found; exported agent will use safe fallback actions")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in copied:
            zf.write(output_dir / filename, arcname=filename)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"exported {zip_path} ({size_mb:.2f} MB, {len(copied)} files)")
    print(f"staging_dir={output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rl_strong as a flat legal submission zip.")
    parser.add_argument("--source", default="agent/rl_strong")
    parser.add_argument("--output_dir", default="submission_rl_strong")
    parser.add_argument("--zip_path", default="dist/rl_strong_submission.zip")
    args = parser.parse_args()
    export_submission(ROOT / args.source, ROOT / args.output_dir, ROOT / args.zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
