from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FILES = [
    "agent.py",
    "ml/__init__.py",
    "ml/features.py",
    "ml/models/__init__.py",
    "ml/models/simple_cnn_policy.py",
    "ml/checkpoints/action_ranker_bomb_fixed.pt",
]

MAX_ZIP_SIZE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_TOTAL_BYTES = 300 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 150 * 1024 * 1024
MAX_FILE_COUNT = 20
ALLOWED_EXTENSIONS = {
    ".py", ".txt", ".pt", ".pth", ".pkl", ".onnx", ".bin", ".json", ".yaml", ".yml", ".md",
    ".h5", ".pb", ".keras", ".tflite"
}


def _is_path_safe(path_str: str) -> bool:
    path = PurePosixPath(path_str)
    return not path.is_absolute() and not any(part == ".." for part in path.parts)


def _validate_zip_bytes(zip_data: bytes):
    if len(zip_data) > MAX_ZIP_SIZE_BYTES:
        return False, "zip_too_large", None
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")
    except Exception as exc:
        return False, f"invalid_zip:{exc}", None
    infos = zf.infolist()
    if len(infos) > MAX_FILE_COUNT:
        return False, "too_many_files", None

    extracted_total = 0
    agent_candidates = []
    manifest = {}
    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        if not _is_path_safe(name):
            return False, "unsafe_path", None
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            return False, f"disallowed_extension:{suffix}", None
        if info.file_size > MAX_SINGLE_FILE_BYTES:
            return False, "single_file_too_large", None
        extracted_total += info.file_size
        if extracted_total > MAX_EXTRACTED_TOTAL_BYTES:
            return False, "extracted_total_too_large", None
        manifest[name] = info.file_size
        if Path(name).name == "agent.py":
            agent_candidates.append(name)
    if len(agent_candidates) != 1:
        return False, "agent_py_missing_or_multiple", None
    if any(Path(name).name == "requirements.txt" for name in manifest):
        return False, "requirements_txt_forbidden", None
    try:
        compile(zf.read(agent_candidates[0]).decode("utf-8"), agent_candidates[0], "exec")
    except Exception as exc:
        return False, f"agent_py_syntax_error:{exc}", None
    return True, None, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export promoted hybrid model submission zip.")
    parser.add_argument("--source", default="submission")
    parser.add_argument("--output", default="dist/hybrid_model_optimized_submission.zip")
    args = parser.parse_args()

    source = ROOT / args.source
    output = ROOT / args.output
    missing = [name for name in FILES if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(source / name, arcname=name)

    zip_data = output.read_bytes()
    valid, reason, manifest = _validate_zip_bytes(zip_data)
    if not valid:
        raise RuntimeError(f"Submission zip failed validation: {reason}")

    with zipfile.ZipFile(output, "r") as zf:
        corrupt = zf.testzip()
    if corrupt:
        raise RuntimeError(f"Corrupt zip member: {corrupt}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
