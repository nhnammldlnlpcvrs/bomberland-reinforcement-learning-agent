#!/usr/bin/env python
"""Analyze current production conversion behavior from existing compact logs.

This script is intentionally analysis-only. It does not run matches, import the
agent, or modify production code. It prefers compact final-validation block
summaries and uses the existing enabled model-optimized benchmark only as a
secondary detailed sample when per-episode fields are present.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PRODUCTION_LABEL = "hybrid_model_optimized"
FINAL_BLOCK_PATTERN = "hybrid_model_final_validation_block*.json"
DETAILED_BENCHMARK = "hybrid_model_optimized_benchmark.json"


@dataclass
class WeightedAverage:
    total: float = 0.0
    weight: int = 0

    def add(self, value: Any, weight: int) -> None:
        if value is None or weight <= 0:
            return
        self.total += float(value) * weight
        self.weight += weight

    def value(self) -> Optional[float]:
        if self.weight <= 0:
            return None
        return self.total / self.weight


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(n: float, d: float) -> Optional[float]:
    if not d:
        return None
    return n / d


def sorted_block_paths(logs_dir: Path) -> List[Path]:
    def block_num(path: Path) -> int:
        stem = path.stem
        digits = "".join(ch for ch in stem.split("block")[-1] if ch.isdigit())
        return int(digits or 0)

    return sorted(logs_dir.glob(FINAL_BLOCK_PATTERN), key=block_num)


def aggregate_final_blocks(paths: Iterable[Path]) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    avg_rank = WeightedAverage()
    avg_survival = WeightedAverage()
    latency_avg = WeightedAverage()
    totals = Counter()
    seed_ranges: List[Any] = []
    valid_blocks = 0

    for path in paths:
        data = load_json(path)
        summary = (data.get("summary") or {}).get(PRODUCTION_LABEL)
        if not summary:
            continue

        matches = int(summary.get("matches") or 0)
        valid = bool(data.get("valid", False))
        valid_blocks += 1 if valid else 0
        seed_ranges.append(data.get("seed_range"))

        for key in (
            "wins",
            "draws",
            "losses",
            "self_bomb_deaths",
            "enemy_bomb_deaths",
            "timeouts",
            "errors",
            "invalid_actions",
            "act_calls",
        ):
            totals[key] += int(summary.get(key) or 0)

        avg_rank.add(summary.get("average_rank"), matches)
        avg_survival.add(summary.get("average_survival_step"), matches)
        latency_avg.add(summary.get("avg_latency_ms"), int(summary.get("act_calls") or 0))

        blocks.append(
            {
                "path": str(path),
                "valid": valid,
                "episode_count": data.get("episode_count"),
                "seed_range": data.get("seed_range"),
                "matches": matches,
                "wins": summary.get("wins"),
                "draws": summary.get("draws"),
                "losses": summary.get("losses"),
                "average_rank": summary.get("average_rank"),
                "average_survival_step": summary.get("average_survival_step"),
                "self_bomb_deaths": summary.get("self_bomb_deaths"),
                "enemy_bomb_deaths": summary.get("enemy_bomb_deaths"),
                "timeouts": summary.get("timeouts"),
                "errors": summary.get("errors"),
                "invalid_actions": summary.get("invalid_actions"),
            }
        )

    matches_total = int(totals["wins"] + totals["draws"] + totals["losses"])
    return {
        "source": "final_validation_blocks",
        "production_label": PRODUCTION_LABEL,
        "block_count": len(blocks),
        "valid_block_count": valid_blocks,
        "seed_ranges": seed_ranges,
        "matches": matches_total,
        "wins": int(totals["wins"]),
        "draws": int(totals["draws"]),
        "losses": int(totals["losses"]),
        "strict_rank1_frequency": pct(totals["wins"], matches_total),
        "draw_frequency": pct(totals["draws"], matches_total),
        "loss_frequency": pct(totals["losses"], matches_total),
        "average_rank_engine": avg_rank.value(),
        "average_survival_step": avg_survival.value(),
        "self_bomb_deaths": int(totals["self_bomb_deaths"]),
        "enemy_bomb_deaths": int(totals["enemy_bomb_deaths"]),
        "timeouts": int(totals["timeouts"]),
        "errors": int(totals["errors"]),
        "invalid_actions": int(totals["invalid_actions"]),
        "act_calls": int(totals["act_calls"]),
        "avg_latency_ms_weighted": latency_avg.value(),
        "blocks": blocks,
    }


def _get_section(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if isinstance(data.get("enabled"), dict):
        return "enabled", data["enabled"]
    if isinstance(data.get("results"), dict) and isinstance(data["results"].get("enabled"), dict):
        return "enabled", data["results"]["enabled"]
    return None, None


def analyze_detailed_benchmark(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "source": str(path),
            "available": False,
            "reason": "file_not_found",
        }

    data = load_json(path)
    section_name, section = _get_section(data)
    if not section:
        return {
            "source": str(path),
            "available": False,
            "reason": "enabled_section_not_found",
        }

    episodes = section.get("episodes") or []
    summary = (section.get("summary") or {}).get(PRODUCTION_LABEL) or {}
    rank_counts = Counter()
    alive_after_350_counts = Counter()
    production_appearances = 0
    production_matches = 0
    matches_gt_400 = 0
    matches_gt_450 = 0
    timeout_finishes = 0
    bomb_events = 0
    action_calls = 0
    action_bombs = 0
    self_bomb_deaths = 0
    late_game_appearances = 0
    late_game_bomb_events = 0

    for episode in episodes:
        roster = episode.get("roster") or []
        if PRODUCTION_LABEL not in roster:
            continue

        production_matches += 1
        steps = int(episode.get("steps") or 0)
        max_steps = int(episode.get("max_steps") or 500)
        if steps > 400:
            matches_gt_400 += 1
        if steps > 450:
            matches_gt_450 += 1
        if steps >= max_steps:
            timeout_finishes += 1

        survival_steps = episode.get("survival_steps") or []
        if survival_steps:
            alive_after_350_counts[str(sum(1 for s in survival_steps if int(s or 0) > 350))] += 1

        ranks = episode.get("ranks") or []
        per_episode = episode.get("per_episode") or {}
        runtime = episode.get("runtime") or []

        for idx, name in enumerate(roster):
            if name != PRODUCTION_LABEL:
                continue
            production_appearances += 1
            if idx < len(ranks):
                # Engine rank 0 is first/best; report as rank1..rank4.
                rank_counts[str(int(ranks[idx]) + 1)] += 1

            row = per_episode.get(name) or {}
            events = row.get("bomb_events") or []
            bomb_events += len(events)
            if row.get("self_bomb_death"):
                self_bomb_deaths += 1
            if steps > 350:
                late_game_appearances += 1
                late_game_bomb_events += len(
                    [e for e in events if int(e.get("placed_step", e.get("step", 0))) > 350]
                )

        for call in runtime:
            if call.get("name") != PRODUCTION_LABEL:
                continue
            action_calls += 1
            if int(call.get("action", -1)) == 5:
                action_bombs += 1

    matches = int(summary.get("matches") or production_appearances)
    wins = int(summary.get("wins") or 0)
    losses = int(summary.get("losses") or 0)
    draws = int(summary.get("draws") or max(0, matches - wins - losses))

    return {
        "source": str(path),
        "available": True,
        "section": section_name,
        "production_label": PRODUCTION_LABEL,
        "episode_count": len(episodes),
        "production_matches": production_matches,
        "production_appearances": production_appearances,
        "matches": matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "strict_rank1_frequency": pct(wins, matches),
        "loss_frequency": pct(losses, matches),
        "average_rank_engine": summary.get("average_rank"),
        "average_survival_step": summary.get("average_survival_step"),
        "rank_distribution_including_ties": dict(sorted(rank_counts.items(), key=lambda kv: int(kv[0]))),
        "rank_frequencies_including_ties": {
            str(rank): pct(rank_counts.get(str(rank), 0), production_appearances) for rank in range(1, 5)
        },
        "games_gt_400_steps": matches_gt_400,
        "games_gt_400_frequency": pct(matches_gt_400, production_matches),
        "games_gt_450_steps": matches_gt_450,
        "games_gt_450_frequency": pct(matches_gt_450, production_matches),
        "timeout_finishes": timeout_finishes,
        "timeout_finish_frequency": pct(timeout_finishes, production_matches),
        "alive_players_after_step_350_distribution": dict(sorted(alive_after_350_counts.items(), key=lambda kv: int(kv[0]))),
        "bomb_events": bomb_events,
        "bomb_events_per_appearance": pct(bomb_events, production_appearances),
        "action_calls": action_calls,
        "bomb_actions": action_bombs,
        "bomb_action_frequency": pct(action_bombs, action_calls),
        "late_game_appearances": late_game_appearances,
        "late_game_bomb_events": late_game_bomb_events,
        "late_game_bomb_events_per_late_appearance": pct(late_game_bomb_events, late_game_appearances),
        "self_bomb_deaths": self_bomb_deaths,
        "enemy_bomb_deaths": None,
        "enemy_kill_frequency": None,
        "safe_bomb_opportunities_declined": None,
        "late_game_pressure_opportunities_missed": None,
    }


def unavailable_metrics() -> Dict[str, str]:
    return {
        "enemy_kill_frequency": (
            "Unavailable from current production compact logs: kill ownership and victim events are not retained."
        ),
        "safe_bomb_opportunities_declined": (
            "Unavailable without frame-level board state and production-safe-bomb evaluator replay. Existing current-production logs are compact summaries."
        ),
        "late_game_pressure_opportunities_missed": (
            "Unavailable without frame-level enemy positions, blast maps, escape-area calculations, and the action actually declined at each late-game state."
        ),
        "exact_enemy_bomb_deaths_in_detailed_sample": (
            "Only aggregate final-validation block summaries retain enemy-bomb death counts for current production."
        ),
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(result: Dict[str, Any]) -> str:
    primary = result["primary_final_validation"]
    detail = result["secondary_detailed_sample"]
    unavailable = result["unavailable_metrics"]

    lines = [
        "# Production Conversion Analysis",
        "",
        "## Scope",
        "",
        "This is a Phase 1 analysis-only artifact. It does not run benchmarks, modify production behavior, or implement the conversion candidate.",
        "",
        "Current production label analyzed: hybrid_model_optimized (submission/agent.py).",
        "",
        "## Data Sources",
        "",
        f"- Primary compact validation blocks: {primary['block_count']} files, {primary['matches']} production appearances.",
        f"- Valid blocks: {primary['valid_block_count']} / {primary['block_count']}.",
        f"- Secondary detailed enabled sample: {detail.get('source')} ({'available' if detail.get('available') else 'unavailable'}).",
        "- Older replays from non-production agents were deliberately excluded to avoid mixing incompatible policies.",
        "",
        "## Primary Production Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Production appearances | {primary['matches']} |",
        f"| Strict rank 1 wins | {primary['wins']} ({fmt_pct(primary['strict_rank1_frequency'])}) |",
        f"| Draws | {primary['draws']} ({fmt_pct(primary['draw_frequency'])}) |",
        f"| Losses | {primary['losses']} ({fmt_pct(primary['loss_frequency'])}) |",
        f"| Avg engine rank | {fmt_num(primary['average_rank_engine'])} |",
        f"| Avg survival step | {fmt_num(primary['average_survival_step'])} |",
        f"| Self-bomb deaths | {primary['self_bomb_deaths']} |",
        f"| Enemy-bomb deaths | {primary['enemy_bomb_deaths']} |",
        f"| Timeout / error / invalid | {primary['timeouts']} / {primary['errors']} / {primary['invalid_actions']} |",
        "",
        "Interpretation: production is already a survival-heavy policy. The primary validation sample is dominated by draws and near-full-length survival, so the measurable bottleneck is conversion rather than basic survival.",
        "",
        "## Rank Distribution",
        "",
    ]

    if detail.get("available") and detail.get("production_appearances"):
        rank_dist = detail.get("rank_distribution_including_ties") or {}
        rank_freq = detail.get("rank_frequencies_including_ties") or {}
        lines.extend(
            [
                "Rank distribution requires per-episode ranks, which are available only in the secondary detailed enabled sample. Engine rank 1 below means best rank and includes ties/draw groups; strict unique wins are reported separately above.",
                "",
                "| User-facing rank | Count | Frequency |",
                "| --- | ---: | ---: |",
            ]
        )
        for rank in ("1", "2", "3", "4"):
            lines.append(f"| Rank {rank} | {rank_dist.get(rank, 0)} | {fmt_pct(rank_freq.get(rank))} |")
    else:
        lines.append("Per-rank distribution is unavailable because no current-production per-episode rank sample was found.")

    lines.extend(
        [
            "",
            "## Late-Game Shape",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    if detail.get("available"):
        lines.extend(
            [
                f"| Detailed production matches | {detail.get('production_matches')} |",
                f"| Games >400 steps | {detail.get('games_gt_400_steps')} ({fmt_pct(detail.get('games_gt_400_frequency'))}) |",
                f"| Games >450 steps | {detail.get('games_gt_450_steps')} ({fmt_pct(detail.get('games_gt_450_frequency'))}) |",
                f"| Timeout finishes | {detail.get('timeout_finishes')} ({fmt_pct(detail.get('timeout_finish_frequency'))}) |",
                f"| Alive players after step 350 distribution | {json.dumps(detail.get('alive_players_after_step_350_distribution', {}), sort_keys=True)} |",
                f"| Bomb events | {detail.get('bomb_events')} |",
                f"| Bomb events per production appearance | {fmt_num(detail.get('bomb_events_per_appearance'))} |",
                f"| PLACE_BOMB action frequency | {fmt_pct(detail.get('bomb_action_frequency'))} |",
                f"| Late-game bomb events per late appearance | {fmt_num(detail.get('late_game_bomb_events_per_late_appearance'))} |",
            ]
        )
    else:
        lines.append("| Detailed late-game metrics | unavailable |")

    lines.extend(
        [
            "",
            "## Opportunity Metrics",
            "",
            "| Metric | Status |",
            "| --- | --- |",
            f"| Enemy kill frequency | {unavailable['enemy_kill_frequency']} |",
            f"| Safe bomb opportunities declined | {unavailable['safe_bomb_opportunities_declined']} |",
            f"| Late-game pressure opportunities missed | {unavailable['late_game_pressure_opportunities_missed']} |",
            "",
            "These opportunity metrics need frame-level production replay data or a replayable state log with board, bombs, agents, chosen action, and candidate safe-action evaluation. The current compact logs are suitable for promotion gates but not for reconstructing missed tactical opportunities.",
            "",
            "## Findings",
            "",
            f"- Strict unique wins are low in the primary sample: {primary['wins']} / {primary['matches']} ({fmt_pct(primary['strict_rank1_frequency'])}).",
            f"- Loss rate is also low: {primary['losses']} / {primary['matches']} ({fmt_pct(primary['loss_frequency'])}), confirming survival is not the main weakness.",
            f"- Average survival is {fmt_num(primary['average_survival_step'])} steps, close to the 500-step cap used by validation.",
            "- Any conversion candidate should therefore be gated to low-risk states and measured primarily on rank-1 frequency and average rank without allowing self-bomb or loss regression.",
            "",
            "## Phase 1 Decision",
            "",
            "Proceed only to candidate design after review. This phase created analysis artifacts and an empty candidate skeleton only; no production behavior changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs", type=Path)
    parser.add_argument("--output", default=Path("logs/production_conversion_analysis.json"), type=Path)
    parser.add_argument("--report", default=Path("docs/PRODUCTION_CONVERSION_ANALYSIS.md"), type=Path)
    args = parser.parse_args()

    block_paths = sorted_block_paths(args.logs_dir)
    primary = aggregate_final_blocks(block_paths)
    secondary = analyze_detailed_benchmark(args.logs_dir / DETAILED_BENCHMARK)

    result = {
        "analysis_version": 1,
        "phase": "conversion-optimization-phase-1-analysis-only",
        "production_label": PRODUCTION_LABEL,
        "primary_final_validation": primary,
        "secondary_detailed_sample": secondary,
        "unavailable_metrics": unavailable_metrics(),
        "notes": [
            "No benchmarks were run by this script.",
            "No production files are imported or modified by this script.",
            "Rank distribution beyond strict wins is derived only from the secondary detailed sample when available.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    args.report.write_text(render_report(result), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
