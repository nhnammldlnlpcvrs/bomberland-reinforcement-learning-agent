from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _by_opponent(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("opponent")): row for row in results}


def _bad_runtime(row: dict[str, Any]) -> bool:
    return (
        int(row.get("invalid_action_count", 0)) > 0
        or int(row.get("crash_count", 0)) > 0
        or int(row.get("timeout_count", 0)) > 0
    )


def score_checkpoint(results: list[dict[str, Any]], baseline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    by_opp = _by_opponent(results)
    baseline_by_opp = _by_opponent(baseline or [])
    random_row = by_opp.get("random", {})
    simple_row = by_opp.get("simple", {})
    baseline_simple = baseline_by_opp.get("simple", {})
    random_win = float(random_row.get("win_rate", 0.0))
    simple_win = float(simple_row.get("win_rate", 0.0))
    simple_death = float(simple_row.get("death_rate", 1.0))
    baseline_simple_death = float(baseline_simple.get("death_rate", simple_death))
    baseline_simple_win = float(baseline_simple.get("win_rate", simple_win))
    bomb_rate = float(simple_row.get("place_bomb_frequency", 0.0))
    boxes_per_bomb = float(simple_row.get("average_boxes_destroyed_per_bomb", 0.0))
    bomb_suicide = float(simple_row.get("bomb_suicide_rate", 0.0))
    runtime_ok = all(not _bad_runtime(row) for row in results)

    reasons: list[str] = []
    if not runtime_ok:
        reasons.append("runtime_error_or_invalid")
    if random_win < 0.95:
        reasons.append("random_win_below_95")
    if simple_death > baseline_simple_death + 0.05:
        reasons.append("simple_death_regressed")
    if bomb_rate > 0 and boxes_per_bomb <= 0.5:
        reasons.append("bombs_without_box_value")
    if bomb_rate > 0 and bomb_suicide > 0.5:
        reasons.append("bomb_suicide_excessive")
    if bomb_rate == 0 and simple_win < baseline_simple_win + 0.02 and simple_death >= baseline_simple_death:
        reasons.append("no_bomb_without_survival_gain")

    score = (
        2.0 * simple_win
        - simple_death
        + 0.5 * random_win
        + min(1.0, boxes_per_bomb) * (0.2 if bomb_rate > 0 else 0.0)
        - max(0.0, bomb_suicide - 0.3)
    )
    return {
        "accepted": len(reasons) == 0,
        "score": float(score),
        "reasons": reasons,
        "random_win": random_win,
        "simple_win": simple_win,
        "simple_death": simple_death,
        "bomb_rate": bomb_rate,
        "boxes_per_bomb": boxes_per_bomb,
        "bomb_suicide": bomb_suicide,
    }


def update_best_files(candidate: str, eval_results: list[dict[str, Any]], save_dir: str, baseline_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    decision = score_checkpoint(eval_results, baseline_results)
    candidate_path = Path(candidate)
    if not candidate_path.exists():
        decision["accepted"] = False
        decision.setdefault("reasons", []).append("candidate_missing")
        return decision
    by_opp = _by_opponent(eval_results)
    simple = by_opp.get("simple", {})
    if decision["accepted"]:
        shutil.copy2(candidate_path, save_path / "latest_accepted.zip")
        history = save_path / "accepted_history"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, history / candidate_path.name)
    if float(simple.get("win_rate", 0.0)) >= _read_best_metric(save_path / "best_by_simple_win.json", "simple_win"):
        _save_named(candidate_path, save_path, "best_by_simple_win", decision)
    if -float(simple.get("death_rate", 1.0)) >= _read_best_metric(save_path / "best_by_death.json", "negative_simple_death"):
        best = dict(decision)
        best["negative_simple_death"] = -float(simple.get("death_rate", 1.0))
        _save_named(candidate_path, save_path, "best_by_death", best)
    if decision["score"] >= _read_best_metric(save_path / "best_by_score.json", "score"):
        _save_named(candidate_path, save_path, "best_by_score", decision)
    if decision["accepted"] and decision["score"] >= _read_best_metric(save_path / "best_overall.json", "score"):
        _save_named(candidate_path, save_path, "best_overall", decision)
    return decision


def _save_named(candidate_path: Path, save_path: Path, name: str, decision: dict[str, Any]) -> None:
    shutil.copy2(candidate_path, save_path / f"{name}.zip")
    (save_path / f"{name}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


def _read_best_metric(path: Path, key: str) -> float:
    if not path.exists():
        return -1e9
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get(key, -1e9))
    except Exception:
        return -1e9


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Bomberland strong checkpoint gates.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--baseline_json", default="")
    args = parser.parse_args()
    results = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8")) if args.baseline_json else None
    decision = update_best_files(args.candidate, results, args.save_dir, baseline)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
