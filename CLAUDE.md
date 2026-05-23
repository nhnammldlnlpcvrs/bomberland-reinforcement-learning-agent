# CLAUDE.md — Bomberland AI Challenge Workspace

## Project Identity

**GDGoC AI Challenge 2026 — Bomberland**: Multi-agent RL competition on a 13×13 grid. 4 agents per match, turn-based, max 500 steps. Build an `Agent` class that survives, collects power-ups, and eliminates opponents.

## Critical Rules (Must Never Violate)

1. **`act()` must return in < 100 ms**. No I/O, no network, no subprocess, no disk writes inside `act()`.
2. **Imports restricted** to: `numpy`, standard library. For ML agents: `torch` (CPU inference only), `onnxruntime`. No `os`, `sys`, `socket`, `requests`, `threading`, etc.
3. **Agent class** must have `__init__(self, agent_id: int)` and `act(self, obs: dict) -> int`.
4. **Action space**: 0=STOP, 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN, 5=PLACE_BOMB.
5. **Board coordinates**: `players[i] = [row, col, alive, bombs_left, bomb_radius_bonus]`. LEFT/RIGHT change row, UP/DOWN change col.

## Quick Commands

```bash
# Test your agent against baselines (10 matches)
python -m scripts.participant.run_local_match \
  --agent_paths agent/hybrid_agent TacticalRuleAgent SmarterRuleAgent GeniusRuleAgent \
  --num_episodes 10

# Estimate TrueSkill rating
python -m scripts.participant.estimate_rankings \
  --agent_path agent/hybrid_agent --num_matches 100

# Benchmark latency
python -m scripts.participant.stability_benchmark \
  --agent_path agent/hybrid_agent --num_calls 1000

# Package for submission
zip -j submission.zip submission/agent.py submission/model.pth 2>/dev/null || \
  zip -j submission.zip submission/agent.py

# Verify package
unzip -l submission.zip
```

## Repository Map

```
project-root/
├── CLAUDE.md                  # You are here
├── .claude/                   # Claude CLI workspace config
│   ├── settings.json          # Model, permissions, ignore patterns
│   ├── agents/                # Custom subagent definitions
│   │   ├── reviewer.md        # AI Agent code reviewer
│   │   └── tester.md          # Test scenario engineer
│   └── skills/                # Domain knowledge for Claude
│       ├── commands/          # CLI commands reference
│       ├── react-patterns/    # Dashboard UI patterns
│       ├── api-conventions/   # WebSocket API spec
│       ├── hooks/             # Custom React hooks
│       └── plugins/           # MCP integration patterns
├── .mcp.json                  # MCP server configuration
├── engine/                    # Core game engine
│   ├── game.py                # BomberEnv class
│   ├── player.py              # Player class
│   ├── bomb.py                # Bomb class
│   └── map.py                 # Map generator (13×13 grid)
├── agent/                     # Agent implementations
│   ├── hybrid_agent/          # Main agent (submission candidate)
│   ├── tactical_rule_agent.py # Baseline: tactical rules
│   ├── genius_rule_agent.py   # Baseline: advanced rules
│   ├── smarter_rule_agent.py  # Baseline: smart rules
│   └── simple_rule_agent.py   # Baseline: simple rules
├── submission/                # Submission package
│   ├── agent.py               # Final agent for submission
│   └── model.pth              # (Optional) trained weights
├── scripts/                   # CLI tools
│   └── participant/           # Participant-facing scripts
├── competition/               # Competition infrastructure
├── logs/                      # Match logs & replays
└── docs/                      # Documentation
```

## Observation Schema

```python
obs = {
    "map": np.ndarray,       # (13, 13) int8 — 0=Grass, 1=Wall, 2=Box, 3=RadiusItem, 4=CapacityItem
    "players": np.ndarray,   # (4, 5) int8 — [row, col, alive, bombs_left, bomb_radius_bonus]
    "bombs": np.ndarray,     # (N, 4) int8 — [row, col, timer, owner_id]
}
```

### Coordinate Convention
- `row` = vertical axis (0=top, 12=bottom)
- `col` = horizontal axis (0=left, 12=right)
- Action 1 (LEFT): row -= 1
- Action 2 (RIGHT): row += 1
- Action 3 (UP): col -= 1
- Action 4 (DOWN): col += 1
- Walls occupy row 0, row 12, col 0, col 12 (outer border)

### Player Spawn Positions
- Agent 0: (1, 1)
- Agent 1: (11, 11)
- Agent 2: (1, 11)
- Agent 3: (11, 1)

### Bomb Mechanics
- Initial timer: 7 steps
- Blast radius: 1 + `bomb_radius_bonus` (max 5)
- Chain reactions: a bomb in another bomb's blast area detonates immediately
- Box destruction: 30% RadiusItem, 30% CapacityItem, 40% nothing
- Items are consumed on collection or destroyed by explosions

## Agent Development Guide

### Minimal Viable Agent

```python
import numpy as np
from collections import deque

class Agent:
    team_id = "MyAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: dict) -> int:
        p = obs["players"][self.agent_id]
        if not int(p[2]):
            return 0  # dead → STOP
        # ... your strategy ...
        return 4  # default: move DOWN
```

### Key Design Principles

1. **Safety first**: Before every bomb placement, verify an escape path exists. Before every move, check the destination is not about to explode.
2. **Danger map**: Compute a grid where `danger[r][c]` = steps until explosion. Account for chain reactions.
3. **BFS pathfinding**: Use `collections.deque` — no external libraries. Limit max depth to stay under 100ms.
4. **State reset**: Reset position history and internal caches at spawn detection.
5. **No global state**: The same Agent instance is reused across matches; `__init__` is called once, so reset per-match state inside `act()`.

### Performance Budget (100ms)

- 13×13 grid operations: nearly free (< 0.1ms)
- BFS up to depth 12: ~0.5ms
- Danger map computation: ~0.3ms
- 6 action evaluations: ~5ms total
- Headroom: ~94ms for remaining logic

If you use a neural network: load the model in `__init__()` (counts toward 20s startup), run single forward pass in `act()` (< 20ms on CPU for small networks).

## Submission Checklist

- [ ] `submission/agent.py` contains exactly one `Agent` class
- [ ] `Agent.__init__` accepts exactly one `agent_id: int` parameter
- [ ] `Agent.act(self, obs: dict) -> int` returns 0–5
- [ ] No banned imports in `agent.py`
- [ ] `act()` completes in < 100ms under all inputs
- [ ] Model file `submission/model.pth` (if applicable) is included
- [ ] `submission.zip` contains only `agent.py` and optionally `model.pth` (flat, no subdirectories)
- [ ] Verified: `unzip -l submission.zip` shows correct contents

## Using Claude Code in This Workspace

### Available Subagents
- **reviewer**: `Agent(".claude/agents/reviewer.md")` — reviews agent code for rules compliance
- **tester**: `Agent(".claude/agents/tester.md")` — generates adversarial test scenarios

### Available Skills
- **commands**: All CLI commands for testing, benchmarking, packaging
- **react-patterns**: Dashboard UI conventions and canvas rendering
- **api-conventions**: WebSocket protocol spec for dashboard-server communication
- **hooks**: Custom React hooks for match playback and metrics streaming
- **plugins**: MCP server integration patterns and custom simulator plugin spec

### Typical Workflow with Claude
1. "Review my agent for competition compliance" → invokes reviewer subagent
2. "Generate adversarial test cases for my agent" → invokes tester subagent
3. "Run 100 matches and report my TrueSkill" → uses commands skill
4. "Build a dashboard to watch training progress" → uses react-patterns + api-conventions + hooks skills

## Research and ML/DL/RL Workflow

- Treat `agent/hybrid_agent/` and `submission/` as production surfaces. Do not change them for research unless explicitly promoting a validated variant.
- Use `docs/RL_ROADMAP_FOR_BOMBERLAND.md` for the staged heuristic -> imitation -> DQN/PPO plan.
- Record important results in `docs/EXPERIMENT_LOG.md`.
- Follow `docs/SUBMISSION_CHECKLIST.md` before packaging any agent.
- Keep learned policies behind deterministic action masks and safety filters.
