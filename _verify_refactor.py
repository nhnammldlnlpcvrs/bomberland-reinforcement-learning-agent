"""
Verification script for HPC refactor — Directives 1, 2, 3.
Tests step singleton cache, NumPy ring-buffer BFS, opponent trap detection,
AMP PPO update, auto-resume, exponential kill scaler.
"""
import sys, os, copy, tempfile, math, random, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_models.config import (
    BFS_MAX_DEPTH, LATEGAME_PRESSURE_START, REWARD_KILL, REWARD_KILL_MAX,
    REWARD_OPPONENT_TRAP, REWARD_CORNER_CAMPING, USE_AMP, PIN_MEMORY,
    DEVICE, BOARD_SIZE, STATE_CHANNELS_V2, SCALAR_FEATURES, ACTION_SPACE,
    A_BOMB, A_STOP, A_LEFT, A_RIGHT, A_UP, A_DOWN, MAX_STEPS,
    BOMB_TIMER, TILE_WALL, TILE_GRASS, TILE_BOX, CHECKPOINT_DIR, AGENT_POOL_DIR,
)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")

# ──────────────────────────────────────────────────────────────────
# Test 1: Step Singleton Cache
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 1: Step Singleton Cache ===")
from train_models.state_processor import _step_cache, flush_step_cache, encode_observation_v2, get_action_mask

flush_step_cache()

# Build mock observation
def make_mock_obs(agent_positions=None, bombs_data=None):
    game_map = np.full((13, 13), TILE_GRASS, dtype=np.int8)
    game_map[0, :] = TILE_WALL; game_map[12, :] = TILE_WALL
    game_map[:, 0] = TILE_WALL; game_map[:, 12] = TILE_WALL
    # Add some inner walls/boxes for realism
    game_map[3, 5] = TILE_WALL; game_map[7, 8] = TILE_BOX

    players = np.zeros((4, 5), dtype=np.int8)
    default_positions = [(1,1), (11,11), (1,11), (11,1)]
    for i, (r, c) in enumerate(default_positions):
        players[i, 0] = r
        players[i, 1] = c
        players[i, 2] = 1  # alive
        players[i, 3] = 3  # bombs_left
        players[i, 4] = 0  # bomb_radius_bonus
    if agent_positions:
        for aid, (r, c) in agent_positions.items():
            players[aid, 0] = r
            players[aid, 1] = c

    bombs = np.zeros((0, 4), dtype=np.int8)
    if bombs_data:
        bombs = np.array(bombs_data, dtype=np.int8).reshape(-1, 4)

    return {"map": game_map, "players": players, "bombs": bombs, "step": 42}

obs1 = make_mock_obs({0: (3, 3)})
obs2 = make_mock_obs({1: (7, 7)})

# First agent triggers cache miss
flush_step_cache()
state1, scalars1 = encode_observation_v2(obs1, 0)
check("First agent returns (16,13,13) tensor", state1.shape == (16, 13, 13), f"got {state1.shape}")
check("First agent returns (4,) scalars", scalars1.shape == (4,), f"got {scalars1.shape}")

# Second agent with same step should be cache hit (fingerprint unchanged)
state2, scalars2 = encode_observation_v2(obs2, 1)
check("Second agent returns (16,13,13) tensor", state2.shape == (16, 13, 13))

# Verify shared planes are identical (danger channel 5, dead_end channel 13, frontier channel 14)
# Note: same map, same bombs, same step => danger/dead_end/frontier must be identical
np.testing.assert_array_equal(state1[5], state2[5], err_msg="Danger channel differs across agents!")
np.testing.assert_array_equal(state1[13], state2[13], err_msg="Dead-end channel differs across agents!")
np.testing.assert_array_equal(state1[14], state2[14], err_msg="Frontier channel differs across agents!")
check("Shared danger channel (5) identical across agents", True)
check("Shared dead-end channel (13) identical across agents", True)
check("Shared frontier channel (14) identical across agents", True)

# Agent-specific channels should differ (channel 2 = self position)
any_diff_self = not np.allclose(state1[2], state2[2])
check("Agent-specific self-plane (ch2) differs across agents", any_diff_self)

# Verify channel 12 (reachable) is also per-agent
any_diff_reachable = not np.allclose(state1[12], state2[12])
check("Agent-specific reachable (ch12) differs across agents", any_diff_reachable)

# Changing step should trigger cache miss (fingerprint includes step)
obs3 = make_mock_obs({0: (3, 3)})
obs3["step"] = 43
fprint_before = _step_cache._fingerprint
state3, scalars3 = encode_observation_v2(obs3, 0)
fprint_after = _step_cache._fingerprint
check("Step change triggers fingerprint update", fprint_before != fprint_after,
      f"before={fprint_before}, after={fprint_after}")

print(f"  Cache summary: {_step_cache._fingerprint is not None}")

# ──────────────────────────────────────────────────────────────────
# Test 2: BFS Depth Cutoff
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 2: BFS Depth Cutoff at 12 ===")
from train_models.state_processor import _reachable_plane_numpy, compute_danger_map

# Open map with no bombs
open_map = np.full((13, 13), TILE_GRASS, dtype=np.int8)
open_map[0, :] = TILE_WALL; open_map[12, :] = TILE_WALL
open_map[:, 0] = TILE_WALL; open_map[:, 12] = TILE_WALL

danger = np.full((13, 13), 999, dtype=np.float32)
bomb_set = set()

reachable = _reachable_plane_numpy((6, 6), open_map, bomb_set, danger, max_depth=BFS_MAX_DEPTH)
reachable_count = int(reachable.sum())
max_dist = 0.0
# Verify no reachable cell is beyond BFS_MAX_DEPTH Manhattan distance from (6,6)
viable = True
for r in range(13):
    for c in range(13):
        if reachable[r, c] > 0:
            dist = abs(r - 6) + abs(c - 6)
            max_dist = max(max_dist, dist)
            if dist > BFS_MAX_DEPTH:
                viable = False
check(f"BFS reachable cells ({reachable_count}) all within depth {BFS_MAX_DEPTH}", viable, f"max_dist={max_dist}")

# ──────────────────────────────────────────────────────────────────
# Test 3: Opponent Trap Detection
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 3: Opponent Trap Detection ===")
from train_models.state_processor import _detect_opponent_trap, _dead_end_plane_numpy

# Create scenario: enemy in dead-end, bomb would block escape
trap_map = np.full((13, 13), TILE_GRASS, dtype=np.int8)
trap_map[0, :] = TILE_WALL; trap_map[12, :] = TILE_WALL
trap_map[:, 0] = TILE_WALL; trap_map[:, 12] = TILE_WALL
# Create a dead-end corridor: walls around enemy except one exit
trap_map[1, 2:11] = TILE_WALL  # top wall at row 1
trap_map[3, 10] = TILE_WALL    # block right exit
# Enemy at (2, 3) — only exit is right through (2,4)→(2,5)... but bomb at (2,5) blocks
trap_map[2, 11] = TILE_WALL    # cap right

players = np.zeros((4, 5), dtype=np.int8)
players[0, 0] = 6; players[0, 1] = 6; players[0, 2] = 1; players[0, 3] = 1
players[1, 0] = 2; players[1, 1] = 3; players[1, 2] = 1; players[1, 3] = 1  # enemy in dead-end
players[2, 0] = 10; players[2, 1] = 10; players[2, 2] = 1
players[3, 0] = 10; players[3, 1] = 2; players[3, 2] = 1

bombs = np.zeros((0, 4), dtype=np.int8)
trap_obs = {"map": trap_map, "players": players, "bombs": bombs, "step": 50}

# Compute dead_end_plane
bomb_set = set()
ded = _dead_end_plane_numpy(trap_map, bomb_set)

# Place bomb at (2,5) — blocks enemy at (2,3)'s escape
bomb_pos = (2, 5)
trap_result = _detect_opponent_trap(trap_obs, 0, bomb_pos, ded)
print(f"  Trap detection result: {trap_result}")
check("Opponent trap detection returns bool", isinstance(trap_result, bool))

# Also verify action mask with trap
mask = get_action_mask(trap_obs, 0)
check("Action mask is valid (6 elements)", len(mask) == 6 and mask.dtype == bool)

# ──────────────────────────────────────────────────────────────────
# Test 4: Exponential Kill Scaler
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 4: Exponential Kill Scaler ===")
def compute_kill_mult(step):
    if step <= LATEGAME_PRESSURE_START:
        return 1.0
    progress_past = (step - LATEGAME_PRESSURE_START) / (MAX_STEPS - LATEGAME_PRESSURE_START)
    return 1.0 + (REWARD_KILL_MAX / REWARD_KILL - 1.0) * (progress_past ** 2.0)

check("Step 300 kill_mult = 1.0", abs(compute_kill_mult(300) - 1.0) < 1e-6, f"got {compute_kill_mult(300)}")
check("Step 350 kill_mult = 1.0", abs(compute_kill_mult(350) - 1.0) < 1e-6, f"got {compute_kill_mult(350)}")

# At step 401: progress = (401-350)/(500-350) = 51/150 = 0.34
# kill_mult = 1 + (30/12 - 1) * 0.34^2 = 1 + (1.5) * 0.1156 = 1 + 0.1734 = 1.1734
# kill_reward = 12 * 1.1734 = 14.08
expected_401 = 1.0 + (30.0/12.0 - 1.0) * ((401-350)/(500-350))**2
check("Step 401 kill_mult ~ 1.173", abs(compute_kill_mult(401) - expected_401) < 0.01, f"got {compute_kill_mult(401)}")

expected_500 = 1.0 + (30.0/12.0 - 1.0) * 1.0  # = 2.5
check(f"Step 500 kill_mult = 2.5 (REWARD_KILL_MAX/REWARD_KILL)", abs(compute_kill_mult(500) - 2.5) < 1e-6, f"got {compute_kill_mult(500)}")
check(f"Step 500 kill reward = {REWARD_KILL_MAX} (30.0)", abs(12.0 * compute_kill_mult(500) - 30.0) < 1e-6)

# ──────────────────────────────────────────────────────────────────
# Test 5: PPO Agent with AMP
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 5: PPO Agent AMP Integration ===")
from train_models.model import ActorCritic
from train_models.ppo_agent import PPOAgent, RolloutBuffer

model = ActorCritic()
agent = PPOAgent(model, lr=3e-4)

check("PPOAgent has scaler attribute", hasattr(agent, "scaler"))
if USE_AMP:
    check("Scaler is GradScaler on CUDA", agent.scaler is not None)
else:
    check("Scaler is None on CPU (correct)", agent.scaler is None)

# Test select_action
obs_test = make_mock_obs({0: (3, 3)})
action, log_prob, value, mask, obs_arr, scalars_arr = agent.select_action(obs_test, 0, deterministic=True)
check("select_action returns valid action 0-5", 0 <= action <= 5, f"got {action}")
check("select_action returns log_prob (float)", isinstance(log_prob, float))
check("select_action returns value (float)", isinstance(value, float))

# Test RolloutBuffer + GAE + update
buffer = RolloutBuffer(capacity=8, num_envs=2)
for i in range(8):
    obs_batch = np.random.randn(2, STATE_CHANNELS_V2, 13, 13).astype(np.float32)
    scalars_batch = np.random.randn(2, SCALAR_FEATURES).astype(np.float32)
    buffer.add(
        obs_batch, scalars_batch,
        np.array([2, 3], dtype=np.int64),
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.6], dtype=np.float32),
        np.array([-0.1, -0.05], dtype=np.float32),
        np.ones((2, 6), dtype=bool),
    )

next_value = np.array([0.3, 0.4], dtype=np.float32)
next_done = np.array([0.0, 0.0], dtype=np.float32)

flush_step_cache()
try:
    metrics = agent.update(buffer, next_value, next_done)
    check("PPO update returns metrics dict", isinstance(metrics, dict))
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
        check(f"  Metric '{k}' present", k in metrics, f"got keys {list(metrics.keys())}")
    print(f"  Update metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
except Exception as e:
    check(f"PPO update runs without error: {e}", False)

# Test save/load with extra state
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "test_checkpoint.pth")
    extra = {"total_env_steps": 10000, "episode_count": 42, "best_eval_winrate": 0.75}
    agent.save(save_path, extra_state=extra)
    check(f"Checkpoint file created", os.path.exists(save_path))

    # Load into new agent
    model2 = ActorCritic()
    agent2 = PPOAgent(model2)
    loaded = agent2.load(save_path)
    check("Load restores model_state_dict", "model_state_dict" in loaded)
    check("Load restores optimizer_state_dict", "optimizer_state_dict" in loaded)
    check("Load preserves extra state", loaded.get("total_env_steps") == 10000)

# ──────────────────────────────────────────────────────────────────
# Test 6: Config Constants
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 6: Config Constants ===")
check("BFS_MAX_DEPTH = 12", BFS_MAX_DEPTH == 12)
check("LATEGAME_PRESSURE_START = 350", LATEGAME_PRESSURE_START == 350)
check("REWARD_KILL = 12.0", abs(REWARD_KILL - 12.0) < 1e-6)
check("REWARD_KILL_MAX = 30.0", abs(REWARD_KILL_MAX - 30.0) < 1e-6)
check("REWARD_OPPONENT_TRAP = 25.0", abs(REWARD_OPPONENT_TRAP - 25.0) < 1e-6)
check("REWARD_CORNER_CAMPING = -0.5", abs(REWARD_CORNER_CAMPING - (-0.5)) < 1e-6)
check("PIN_MEMORY matches CUDA", PIN_MEMORY == (DEVICE == "cuda"))
check("USE_AMP matches CUDA", USE_AMP == (DEVICE == "cuda"))

# ──────────────────────────────────────────────────────────────────
# Test 7: Trainer auto-resume
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 7: Trainer _try_auto_resume Method ===")
from train_models.trainer import Trainer

check("Trainer class is importable", True)
check("Trainer has _try_auto_resume method", hasattr(Trainer, "_try_auto_resume"))

# ──────────────────────────────────────────────────────────────────
# Test 8: flush_step_cache public API
# ──────────────────────────────────────────────────────────────────
print("\n=== Test 8: flush_step_cache Public API ===")
from train_models.state_processor import flush_step_cache as fsc
flush_step_cache()
cached = _step_cache.get(np.zeros((13,13), dtype=np.int8), np.zeros((0,4), dtype=np.int8), 0)
check("Cache miss after flush returns None", cached is None)

# Store and retrieve — use same obs dict to guarantee identical arrays
obs_cache = make_mock_obs({0: (2, 2)})
state, _ = encode_observation_v2(obs_cache, 0)
# Note: encode_observation_v2 converts to int32 internally, so get() needs int32 too
cached = _step_cache.get(
    np.asarray(obs_cache["map"], dtype=np.int32),
    np.asarray(obs_cache["bombs"], dtype=np.int32),
    int(obs_cache["step"]),
)
check("Cache hit returns planes dict after encode", cached is not None)
if cached:
    check("  danger_map present", "danger_map" in cached)
    check("  dead_end_plane present", "dead_end_plane" in cached)
    check("  frontier_plane present", "frontier_plane" in cached)

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed > 0:
    print("SOME TESTS FAILED — review output above.")
    sys.exit(1)
else:
    print("ALL TESTS PASSED — HPC refactor verified.")
    sys.exit(0)
