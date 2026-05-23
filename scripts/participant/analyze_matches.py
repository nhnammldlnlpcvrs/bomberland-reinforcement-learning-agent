import argparse
import json
from collections import Counter
from pathlib import Path


WALL = 1
BOX = 2
STOP = 0
PLACE_BOMB = 5


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _get_nested(payload, *keys):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _agent_names(payload):
    candidates = [
        payload.get("team_ids"),
        _get_nested(payload, "meta", "agent_names"),
        payload.get("agents"),
        payload.get("agent_names"),
        payload.get("teams"),
        payload.get("participants"),
    ]
    for candidate in candidates:
        names = _as_list(candidate)
        if names:
            return [str(name) for name in names]
    return []


def _history(payload):
    for key in ("history", "steps", "frames", "trajectory", "replay"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _frame_step(frame, fallback):
    if not isinstance(frame, dict):
        return fallback
    for key in ("step", "_step", "t", "tick"):
        value = _safe_int(frame.get(key))
        if value is not None:
            return value
    return fallback


def _frame_players(frame):
    if not isinstance(frame, dict):
        return []
    return _as_list(frame.get("players") or frame.get("player_state") or frame.get("agents"))


def _frame_map(frame):
    if not isinstance(frame, dict):
        return []
    return _as_list(frame.get("map") or frame.get("grid") or frame.get("board"))


def _frame_bombs(frame):
    if not isinstance(frame, dict):
        return []
    return _as_list(frame.get("bombs"))


def _frame_actions(frame):
    if not isinstance(frame, dict):
        return []
    return _as_list(frame.get("actions") or frame.get("action"))


def _player_alive(player):
    if isinstance(player, dict):
        return bool(player.get("alive", player.get("is_alive", True)))
    values = _as_list(player)
    if len(values) >= 3:
        return bool(values[2])
    return True


def _player_pos(player):
    if isinstance(player, dict):
        row = player.get("row", player.get("x", player.get("r")))
        col = player.get("col", player.get("y", player.get("c")))
        row = _safe_int(row)
        col = _safe_int(col)
        if row is not None and col is not None:
            return row, col
        return None
    values = _as_list(player)
    if len(values) >= 2:
        row = _safe_int(values[0])
        col = _safe_int(values[1])
        if row is not None and col is not None:
            return row, col
    return None


def _bomb_tuple(bomb):
    if isinstance(bomb, dict):
        row = _safe_int(bomb.get("row", bomb.get("x", bomb.get("r"))))
        col = _safe_int(bomb.get("col", bomb.get("y", bomb.get("c"))))
        timer = _safe_int(bomb.get("timer", bomb.get("life", bomb.get("t"))), 0)
        owner = _safe_int(bomb.get("owner_id", bomb.get("owner", bomb.get("agent_id"))), -1)
    else:
        values = _as_list(bomb)
        row = _safe_int(values[0]) if len(values) > 0 else None
        col = _safe_int(values[1]) if len(values) > 1 else None
        timer = _safe_int(values[2], 0) if len(values) > 2 else 0
        owner = _safe_int(values[3], -1) if len(values) > 3 else -1
    if row is None or col is None:
        return None
    return row, col, timer, owner


def _in_bounds(game_map, row, col):
    return bool(game_map) and 0 <= row < len(game_map) and 0 <= col < len(game_map[row])


def _cell(game_map, row, col):
    if not _in_bounds(game_map, row, col):
        return WALL
    return _safe_int(game_map[row][col], WALL)


def _free_neighbors(pos, game_map, bomb_positions):
    if pos is None:
        return 0
    row, col = pos
    count = 0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = row + dr, col + dc
        if _cell(game_map, nr, nc) in (WALL, BOX):
            continue
        if (nr, nc) in bomb_positions:
            continue
        count += 1
    return count


def _zone_type(pos, game_map, bombs):
    bomb_positions = {(b[0], b[1]) for b in bombs if b is not None}
    free = _free_neighbors(pos, game_map, bomb_positions)
    if free <= 1:
        return "dead_end"
    if free == 2:
        return "corridor"
    return "open"


def _blast_tiles(game_map, bomb, players):
    row, col, _timer, owner = bomb
    radius = 1
    if 0 <= owner < len(players):
        player = players[owner]
        if isinstance(player, dict):
            radius = 1 + max(0, _safe_int(player.get("bomb_radius_bonus", player.get("radius_bonus")), 0))
        else:
            values = _as_list(player)
            if len(values) >= 5:
                radius = 1 + max(0, _safe_int(values[4], 0))

    tiles = {(row, col)}
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        for distance in range(1, radius + 1):
            nr, nc = row + dr * distance, col + dc * distance
            if not _in_bounds(game_map, nr, nc):
                break
            cell = _cell(game_map, nr, nc)
            if cell == WALL:
                break
            tiles.add((nr, nc))
            if cell == BOX:
                break
    return tiles


def _bombs_near(pos, bombs, max_dist=3):
    if pos is None:
        return False
    row, col = pos
    for bomb in bombs:
        if bomb is None:
            continue
        if abs(row - bomb[0]) + abs(col - bomb[1]) <= max_dist:
            return True
    return False


def _killer_owner(agent_idx, death_pos, prev_frame, death_frame):
    if death_pos is None:
        return None
    game_map = _frame_map(prev_frame) or _frame_map(death_frame)
    players = _frame_players(prev_frame) or _frame_players(death_frame)
    candidate_bombs = []
    for bomb in _frame_bombs(prev_frame) + _frame_bombs(death_frame):
        parsed = _bomb_tuple(bomb)
        if parsed is not None:
            candidate_bombs.append(parsed)

    owners = []
    for bomb in candidate_bombs:
        timer = bomb[2]
        if timer not in (0, 1):
            continue
        if death_pos in _blast_tiles(game_map, bomb, players):
            owners.append(bomb[3])
    if not owners:
        return None
    if agent_idx in owners:
        return agent_idx
    return owners[0]


def _find_agent_indices(names, team_name):
    wanted = team_name.lower()
    return [idx for idx, name in enumerate(names) if str(name).lower() == wanted]


def _fallback_agent_indices(payload, team_name):
    names = _agent_names(payload)
    indices = _find_agent_indices(names, team_name)
    if indices:
        return indices, names

    history = _history(payload)
    first_players = _frame_players(history[0]) if history else []
    if len(first_players) == 1 and team_name:
        return [0], [team_name]
    return [], names


def _outcome(agent_idx, payload, history):
    ranks = _as_list(payload.get("ranks") or payload.get("rank"))
    if ranks and agent_idx < len(ranks):
        rank = _safe_int(ranks[agent_idx])
        if rank == 0:
            winners = sum(1 for value in ranks if _safe_int(value) == 0)
            return ("draw" if winners > 1 else "win"), rank
        return "loss", rank

    final_players = _frame_players(history[-1]) if history else []
    alive = agent_idx < len(final_players) and _player_alive(final_players[agent_idx])
    alive_count = sum(1 for player in final_players if _player_alive(player))
    if alive and alive_count == 1:
        return "win", 0
    if alive and alive_count > 1:
        return "draw", 0
    return "loss", None


def _death_info(agent_idx, history):
    for index in range(1, len(history)):
        prev_players = _frame_players(history[index - 1])
        players = _frame_players(history[index])
        if agent_idx >= len(prev_players) or agent_idx >= len(players):
            continue
        if _player_alive(prev_players[agent_idx]) and not _player_alive(players[agent_idx]):
            prev_pos = _player_pos(prev_players[agent_idx])
            death_pos = _player_pos(players[agent_idx]) or prev_pos
            step = _frame_step(history[index], index)
            owner = _killer_owner(agent_idx, death_pos, history[index - 1], history[index])
            if owner == agent_idx:
                source = "own_bomb"
            elif owner is None or owner < 0:
                source = "unknown"
            else:
                source = "enemy_bomb"
            return {
                "step": step,
                "position": death_pos,
                "source": source,
                "frame_index": index,
            }
    return {"step": None, "position": None, "source": "unknown", "frame_index": None}


def _pre_death_context(agent_idx, history, death_index, window=5):
    if death_index is None:
        return {
            "corridor_or_dead_end": False,
            "zones": Counter(),
            "bomb_near": False,
        }
    zones = Counter()
    bomb_near = False
    start = max(0, death_index - window)
    for frame in history[start:death_index]:
        players = _frame_players(frame)
        if agent_idx >= len(players):
            continue
        pos = _player_pos(players[agent_idx])
        bombs = [_bomb_tuple(bomb) for bomb in _frame_bombs(frame)]
        bombs = [bomb for bomb in bombs if bomb is not None]
        zones[_zone_type(pos, _frame_map(frame), bombs)] += 1
        if _bombs_near(pos, bombs):
            bomb_near = True
    return {
        "corridor_or_dead_end": bool(zones["corridor"] or zones["dead_end"]),
        "zones": zones,
        "bomb_near": bomb_near,
    }


def _draw_context(agent_idx, history, total_steps):
    placed_bomb = False
    positions = []
    actions = []
    for frame in history:
        frame_actions = _frame_actions(frame)
        if agent_idx < len(frame_actions):
            action = _safe_int(frame_actions[agent_idx])
            actions.append(action)
            if action == PLACE_BOMB:
                placed_bomb = True
        players = _frame_players(frame)
        if agent_idx < len(players) and _player_alive(players[agent_idx]):
            pos = _player_pos(players[agent_idx])
            if pos is not None:
                positions.append(pos)

    recent = positions[-24:]
    unique_recent = len(set(recent))
    stop_recent = sum(1 for action in actions[-24:] if action == STOP)
    stuck_loop = bool(len(recent) >= 12 and unique_recent <= 4)
    if len(actions[-24:]) >= 12 and stop_recent >= 16:
        stuck_loop = True
    return {
        "near_500_steps": total_steps >= 480,
        "placed_bomb": placed_bomb,
        "stuck_loop": stuck_loop,
        "recent_unique_positions": unique_recent,
    }


def _match_seed(payload, path):
    for key in ("seed", "match_seed", "episode_seed"):
        if payload.get(key) is not None:
            return payload.get(key)
    return path.stem


def _total_steps(payload, history):
    for key in ("total_steps", "steps", "num_steps"):
        value = _safe_int(payload.get(key))
        if value is not None:
            return value
    if history:
        return _frame_step(history[-1], len(history) - 1)
    return 0


def analyze_file(path, team_name):
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON is not an object")

    history = _history(payload)
    agent_indices, names = _fallback_agent_indices(payload, team_name)
    if not agent_indices:
        return []

    total_steps = _total_steps(payload, history)
    results = []
    for agent_idx in agent_indices:
        outcome, rank = _outcome(agent_idx, payload, history)
        result = {
            "file": str(path),
            "seed": _match_seed(payload, path),
            "agent_idx": agent_idx,
            "agents": names,
            "total_steps": total_steps,
            "rank": rank,
            "outcome": outcome,
            "death": None,
            "pre_death": None,
            "draw": None,
            "suspicion": 0.0,
            "reasons": [],
        }

        if outcome == "loss":
            death = _death_info(agent_idx, history)
            context = _pre_death_context(agent_idx, history, death["frame_index"])
            result["death"] = death
            result["pre_death"] = context
            if death["source"] == "own_bomb":
                result["suspicion"] += 3
                result["reasons"].append("own bomb death")
            elif death["source"] == "unknown":
                result["suspicion"] += 1
                result["reasons"].append("unknown death source")
            if context["corridor_or_dead_end"]:
                result["suspicion"] += 2
                result["reasons"].append("pre-death corridor/dead-end")
            if context["bomb_near"]:
                result["suspicion"] += 1
                result["reasons"].append("bomb nearby before death")

        if outcome == "draw":
            draw = _draw_context(agent_idx, history, total_steps)
            result["draw"] = draw
            if draw["near_500_steps"]:
                result["suspicion"] += 2
                result["reasons"].append("near max steps")
            if not draw["placed_bomb"]:
                result["suspicion"] += 2
                result["reasons"].append("no bomb placed")
            if draw["stuck_loop"]:
                result["suspicion"] += 2
                result["reasons"].append("stuck loop")

        results.append(result)
    return results


def print_match_rows(results):
    if not results:
        return
    print("\n=== Matches ===")
    headers = ["outcome", "rank", "steps", "seed", "file"]
    rows = []
    for result in results:
        rows.append([
            result["outcome"],
            "" if result["rank"] is None else str(result["rank"]),
            str(result["total_steps"]),
            str(result["seed"]),
            Path(result["file"]).name,
        ])
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def print_summary(results, skipped, errors):
    total = len(results)
    outcomes = Counter(result["outcome"] for result in results)
    steps = [result["total_steps"] for result in results]
    deaths = Counter()
    deaths_corridor = 0
    draws_near_500 = 0
    draws_no_bomb = 0
    draws_stuck = 0

    for result in results:
        if result["outcome"] == "loss" and result["death"]:
            deaths[result["death"]["source"]] += 1
            if result["pre_death"] and result["pre_death"]["corridor_or_dead_end"]:
                deaths_corridor += 1
        if result["outcome"] == "draw" and result["draw"]:
            if result["draw"]["near_500_steps"]:
                draws_near_500 += 1
            if not result["draw"]["placed_bomb"]:
                draws_no_bomb += 1
            if result["draw"]["stuck_loop"]:
                draws_stuck += 1

    print("\n=== Summary ===")
    print(f"total matches: {total}")
    print(f"wins/draws/losses: {outcomes['win']}/{outcomes['draw']}/{outcomes['loss']}")
    print(f"avg steps: {_mean(steps):.1f}")
    print(
        "deaths by source: "
        f"own_bomb={deaths['own_bomb']}, "
        f"enemy_bomb={deaths['enemy_bomb']}, "
        f"unknown={deaths['unknown']}"
    )
    print(f"deaths in corridor/dead-end: {deaths_corridor}")
    print(f"draws near 500 steps: {draws_near_500}")
    print(f"draws with no bomb placed: {draws_no_bomb}")
    print(f"draws with stuck-loop signal: {draws_stuck}")
    if skipped:
        print(f"files skipped without target team: {skipped}")
    if errors:
        print(f"files with parse errors: {len(errors)}")
        for path, error in errors[:5]:
            print(f"  {path}: {error}")


def print_suspicious(results, limit=10):
    ranked = sorted(
        [result for result in results if result["suspicion"] > 0],
        key=lambda item: (item["suspicion"], item["total_steps"]),
        reverse=True,
    )
    print("\n=== Top Suspicious Matches ===")
    if not ranked:
        print("No suspicious loss/draw patterns detected.")
        return
    for result in ranked[:limit]:
        details = []
        if result["death"]:
            details.append(f"death_step={result['death']['step']}")
            details.append(f"death_pos={result['death']['position']}")
            details.append(f"death_source={result['death']['source']}")
        if result["draw"]:
            details.append(f"near_500={result['draw']['near_500_steps']}")
            details.append(f"placed_bomb={result['draw']['placed_bomb']}")
            details.append(f"stuck_loop={result['draw']['stuck_loop']}")
        reason = ", ".join(result["reasons"]) if result["reasons"] else "n/a"
        print(
            f"{Path(result['file']).name} | outcome={result['outcome']} | "
            f"steps={result['total_steps']} | score={result['suspicion']:.1f} | "
            f"reasons={reason} | {'; '.join(details)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze Bomberland JSON match logs for losses and draws.")
    parser.add_argument("--log_dir", required=True, help="Directory containing match JSON files.")
    parser.add_argument("--team_name", required=True, help="Team/agent name to analyze, e.g. HybridAgent.")
    parser.add_argument("--top", type=int, default=10, help="Number of suspicious matches to print.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"No logs found: directory does not exist: {log_dir}")
        return 0

    json_files = sorted(path for path in log_dir.rglob("*.json") if path.is_file())
    if not json_files:
        print(f"No JSON logs found in {log_dir}")
        return 0

    all_results = []
    skipped = 0
    errors = []
    for path in json_files:
        try:
            results = analyze_file(path, args.team_name)
        except Exception as exc:
            errors.append((path, exc))
            continue
        if results:
            all_results.extend(results)
        else:
            skipped += 1

    if not all_results:
        print(f"No matches containing team {args.team_name!r} found in {log_dir}")
        if skipped:
            print(f"JSON files scanned: {len(json_files)}")
        if errors:
            print(f"Files with parse errors: {len(errors)}")
            for path, error in errors[:5]:
                print(f"  {path}: {error}")
        return 0

    print(f"Analyzed team: {args.team_name}")
    print(f"JSON files scanned: {len(json_files)}")
    print_match_rows(all_results)
    print_summary(all_results, skipped, errors)
    print_suspicious(all_results, limit=max(0, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
