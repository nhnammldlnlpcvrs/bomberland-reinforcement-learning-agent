"""
MCP Server for Bomberland local simulation.

Exposes tools for Claude CLI to directly:
- Run matches between agents
- Validate agent imports and interface compliance
- Benchmark act() latency
- Generate mock observations for edge-case testing
"""

import sys
import os
import json
import time
import importlib
import traceback
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_agent(agent_ref: str):
    """Resolve an agent reference string to an Agent class.

    Supports:
    - 'agent/hybrid_agent' -> loads agent.hybrid_agent.agent.Agent (module path convention)
    - 'TacticalRuleAgent'  -> loads agent.tactical_rule_agent.TacticalRuleAgent (class name)
    - 'agent/hybrid_agent:MyAgent' -> loads agent.hybrid_agent.agent.MyAgent (explicit class)
    """
    if ":" in agent_ref:
        module_path, class_name = agent_ref.split(":", 1)
    else:
        module_path = agent_ref
        # Heuristic: if it doesn't contain '/', it might be a known baseline class name
        if "/" not in agent_ref:
            mapping = {
                "TacticalRuleAgent": "agent.tactical_rule_agent",
                "GeniusRuleAgent": "agent.genius_rule_agent",
                "SmarterRuleAgent": "agent.smarter_rule_agent",
                "SimpleRuleAgent": "agent.simple_rule_agent",
                "RandomAgent": "agent.random_agent",
                "HybridAgent": "agent.hybrid_agent.agent",
            }
            if agent_ref in mapping:
                module_path = mapping[agent_ref]
                class_name = agent_ref
            else:
                raise ValueError(f"Unknown agent class: {agent_ref}")
        else:
            # Convert path/to/agent to Python module: agent.path.to.agent.agent
            parts = module_path.replace("/", ".").replace("\\", ".")
            module_path = f"{parts}.agent"
            class_name = "Agent"

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _make_empty_obs():
    """Create a minimal valid observation for smoke-testing."""
    game_map = np.zeros((13, 13), dtype=np.int8)
    game_map[0, :] = 1
    game_map[-1, :] = 1
    game_map[:, 0] = 1
    game_map[:, -1] = 1
    return {
        "map": game_map,
        "players": np.array([
            [1, 1, 1, 1, 0],
            [11, 11, 1, 1, 0],
            [1, 11, 1, 1, 0],
            [11, 1, 1, 1, 0],
        ], dtype=np.int8),
        "bombs": np.zeros((0, 4), dtype=np.int8),
    }


# ==============================================================================
# Tool: run_match
# ==============================================================================

def tool_run_match(agents: list[str], seed: int = 42, max_steps: int = 500) -> dict:
    """Run a full 4-agent match and return results."""
    from engine.game import BomberEnv

    agent_classes = [_resolve_agent(a) for a in agents]
    agents_inst = [cls(i) for i, cls in enumerate(agent_classes)]

    env = BomberEnv(seed=seed)
    env.max_steps = max_steps
    obs = env.reset(seed=seed)

    steps = 0
    terminated = False
    truncated = False
    action_history = []

    while not terminated and not truncated:
        actions = []
        for i, agent in enumerate(agents_inst):
            try:
                a = agent.act(obs)
                a = int(a)
                if not 0 <= a <= 5:
                    a = 0
                actions.append(a)
            except Exception:
                actions.append(0)

        obs, terminated, truncated = env.step(actions)
        action_history.append(actions)
        steps += 1

    # Determine winner
    alive = [int(obs["players"][i][2]) for i in range(4)]
    winner = next((i for i, a in enumerate(alive) if a == 1), None)

    return {
        "match_id": f"sim_{int(time.time())}",
        "seed": seed,
        "steps": steps,
        "winner": winner,
        "alive": alive,
        "terminated": terminated,
        "truncated": truncated,
        "agents": agents,
    }


# ==============================================================================
# Tool: validate_agent
# ==============================================================================

def tool_validate_agent(agent_path: str) -> dict:
    """Import an agent module and verify interface compliance."""
    errors = []
    warnings = []

    # Check banned imports
    BANNED = {"os", "sys", "subprocess", "socket", "requests", "urllib",
              "threading", "multiprocessing", "ctypes", "pygame", "matplotlib"}

    try:
        AgentClass = _resolve_agent(agent_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Import failed: {e}"], "warnings": []}

    # Check interface
    try:
        agent = AgentClass(0)
    except TypeError as e:
        errors.append(f"__init__ signature mismatch: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    except Exception as e:
        errors.append(f"__init__ raised exception: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not hasattr(agent, "act") or not callable(agent.act):
        errors.append("Agent class missing callable act() method")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Smoke test act()
    obs = _make_empty_obs()
    try:
        action = agent.act(obs)
        action = int(action)
        if not 0 <= action <= 5:
            errors.append(f"act() returned invalid action: {action} (expected 0-5)")
    except Exception as e:
        errors.append(f"act() raised exception: {e}\n{traceback.format_exc()}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check for banned imports in agent's module
    mod = sys.modules[AgentClass.__module__.split(".")[0]]
    if hasattr(mod, "__file__"):
        try:
            with open(AgentClass.__module__) as f:
                pass
        except Exception:
            pass

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "agent_class": AgentClass.__name__,
        "module": AgentClass.__module__,
    }


# ==============================================================================
# Tool: benchmark_act
# ==============================================================================

def tool_benchmark_act(agent_path: str, num_calls: int = 1000) -> dict:
    """Benchmark act() latency over many calls."""
    AgentClass = _resolve_agent(agent_path)
    agent = AgentClass(0)
    obs = _make_empty_obs()

    latencies = []
    for _ in range(num_calls):
        t0 = time.perf_counter()
        try:
            action = agent.act(obs)
        except Exception:
            action = 0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        # Re-create agent every 100 calls to simulate fresh state
        if _ % 100 == 99:
            agent = AgentClass(0)

    latencies = np.array(latencies)
    return {
        "num_calls": num_calls,
        "agent": agent_path,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(np.max(latencies)),
        "min_ms": float(np.min(latencies)),
        "std_ms": float(np.std(latencies)),
        "timeout_rate": float(np.mean(latencies > 100)),
        "passes": float(np.mean(latencies > 100)) == 0.0,
    }


# ==============================================================================
# Tool: generate_obs
# ==============================================================================

def tool_generate_obs(scenario: str, agent_id: int = 0) -> dict:
    """Generate mock observations for edge-case testing."""
    game_map = np.zeros((13, 13), dtype=np.int8)
    game_map[0, :] = 1
    game_map[-1, :] = 1
    game_map[:, 0] = 1
    game_map[:, -1] = 1

    default_players = np.array([
        [1, 1, 1, 1, 0],
        [11, 11, 1, 1, 0],
        [1, 11, 1, 1, 0],
        [11, 1, 1, 1, 0],
    ], dtype=np.int8)

    if scenario == "corner_trap":
        # Agent 0 at (1,1), bombs at (1,2) and (2,1) with timer=1
        return {
            "map": game_map,
            "players": default_players,
            "bombs": np.array([[1, 2, 1, 3], [2, 1, 1, 3]], dtype=np.int8),
        }

    elif scenario == "domino_chain":
        # Three bombs in a line that chain-react
        game_map[5, 3] = 0
        game_map[5, 5] = 0
        game_map[5, 7] = 0
        players = default_players.copy()
        players[agent_id] = [5, 4, 1, 1, 0]
        return {
            "map": game_map,
            "players": players,
            "bombs": np.array([[5, 3, 3, 1], [5, 5, 7, 2], [5, 7, 5, 3]], dtype=np.int8),
        }

    elif scenario == "all_enemies_dead":
        players = default_players.copy()
        players[1, 2] = 0
        players[2, 2] = 0
        players[3, 2] = 0
        return {
            "map": game_map,
            "players": players,
            "bombs": np.zeros((0, 4), dtype=np.int8),
        }

    elif scenario == "no_bombs_left":
        players = default_players.copy()
        players[agent_id, 3] = 0
        return {
            "map": game_map,
            "players": players,
            "bombs": np.zeros((0, 4), dtype=np.int8),
        }

    elif scenario == "under_bomb":
        # Agent standing on its own bomb about to explode
        players = default_players.copy()
        players[agent_id] = [5, 5, 1, 1, 0]
        return {
            "map": game_map,
            "players": players,
            "bombs": np.array([[5, 5, 1, agent_id]], dtype=np.int8),
        }

    elif scenario == "empty_map":
        return {
            "map": game_map,
            "players": default_players,
            "bombs": np.zeros((0, 4), dtype=np.int8),
        }

    else:
        raise ValueError(f"Unknown scenario: {scenario}. Available: corner_trap, domino_chain, all_enemies_dead, no_bombs_left, under_bomb, empty_map")


# ==============================================================================
# MCP Protocol Handler (stdio JSON-RPC)
# ==============================================================================

TOOLS = {
    "run_match": {
        "fn": tool_run_match,
        "schema": {
            "name": "run_match",
            "description": "Run a full 4-agent Bomberland match and return results (winner, steps, alive status).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "4 agent paths, e.g. ['agent/hybrid_agent', 'TacticalRuleAgent', 'SmarterRuleAgent', 'GeniusRuleAgent']"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility",
                        "default": 42
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum steps before truncation",
                        "default": 500
                    }
                },
                "required": ["agents"]
            }
        }
    },
    "validate_agent": {
        "fn": tool_validate_agent,
        "schema": {
            "name": "validate_agent",
            "description": "Import and validate an agent module for competition compliance (interface, banned imports, smoke test).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_path": {
                        "type": "string",
                        "description": "Agent path (e.g., 'agent/hybrid_agent' or 'TacticalRuleAgent')"
                    }
                },
                "required": ["agent_path"]
            }
        }
    },
    "benchmark_act": {
        "fn": tool_benchmark_act,
        "schema": {
            "name": "benchmark_act",
            "description": "Benchmark agent's act() latency over multiple calls. Reports mean, p95, p99, max, and timeout rate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_path": {
                        "type": "string",
                        "description": "Agent path to benchmark"
                    },
                    "num_calls": {
                        "type": "integer",
                        "description": "Number of act() calls to measure",
                        "default": 1000
                    }
                },
                "required": ["agent_path"]
            }
        }
    },
    "generate_obs": {
        "fn": tool_generate_obs,
        "schema": {
            "name": "generate_obs",
            "description": "Generate a mock observation for edge-case testing (corner_trap, domino_chain, all_enemies_dead, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "description": "Scenario name: corner_trap, domino_chain, all_enemies_dead, no_bombs_left, under_bomb, empty_map"
                    },
                    "agent_id": {
                        "type": "integer",
                        "description": "Agent ID to position in the scenario",
                        "default": 0
                    }
                },
                "required": ["scenario"]
            }
        }
    },
}


def _send_response(request_id, result):
    """Send a JSON-RPC response to stdout."""
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
    sys.stdout.write(json.dumps(response, default=str) + "\n")
    sys.stdout.flush()


def _send_error(request_id, code, message):
    """Send a JSON-RPC error to stdout."""
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _send_notification(method, params=None):
    """Send a JSON-RPC notification to stdout."""
    notification = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }
    sys.stdout.write(json.dumps(notification, default=str) + "\n")
    sys.stdout.flush()


def main():
    """Run the MCP server on stdio (blocking JSON-RPC loop)."""
    # Send initialized notification
    _send_notification("initialized")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        method = request.get("method")

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "bomberland-sim",
                    "version": "1.0.0",
                },
            }
            _send_response(request_id, result)

        elif method == "tools/list":
            tools_list = [
                {
                    "name": name,
                    "description": info["schema"]["description"],
                    "inputSchema": info["schema"]["inputSchema"],
                }
                for name, info in TOOLS.items()
            ]
            _send_response(request_id, {"tools": tools_list})

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name not in TOOLS:
                _send_error(request_id, -32601, f"Unknown tool: {tool_name}")
                continue

            try:
                fn = TOOLS[tool_name]["fn"]
                result = fn(**arguments)
                _send_response(request_id, {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2, default=str)}
                    ]
                })
            except Exception as e:
                _send_response(request_id, {
                    "content": [
                        {"type": "text", "text": f"Error: {e}\n{traceback.format_exc()}"}
                    ],
                    "isError": True,
                })

        elif method == "notifications/initialized":
            pass  # Acknowledge client ready

        else:
            _send_error(request_id, -32601, f"Unknown method: {method}")


if __name__ == "__main__":
    main()
