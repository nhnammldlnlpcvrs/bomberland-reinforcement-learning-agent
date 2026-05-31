"""
PPO Training Configuration for Bomberland Master Agent.

All tunable hyperparameters, paths, and environment constants.
"""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── PPO Hyperparameters ─────────────────────────────────────────────────────────
LR = 3e-4
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5

# ── Training loop ───────────────────────────────────────────────────────────────
ROLLOUT_STEPS = 2048          # steps collected per PPO update
UPDATE_EPOCHS = 4             # PPO epochs per update
BATCH_SIZE = 256              # minibatch size
TOTAL_TIMESTEPS = 10_000_000  # total env steps across all parallel envs
SAVE_INTERVAL = 50_000        # save checkpoint every N steps
EVAL_INTERVAL = 25_000        # run evaluation every N steps
SELF_PLAY_UPDATE_INTERVAL = 100_000  # snapshot into agent pool every N steps
LOG_INTERVAL = 1_000          # log to TensorBoard every N steps

# ── Environment ──────────────────────────────────────────────────────────────────
BOARD_SIZE = 13
MAX_STEPS = 500
NUM_AGENTS = 4
ACTION_SPACE = 6

# Action constants (matching engine/player.py)
A_STOP = 0
A_LEFT = 1
A_RIGHT = 2
A_UP = 3
A_DOWN = 4
A_BOMB = 5

MOVE_ACTIONS = [A_LEFT, A_RIGHT, A_UP, A_DOWN]
ALL_ACTIONS = [A_STOP, A_LEFT, A_RIGHT, A_UP, A_DOWN, A_BOMB]

# Direction deltas: (drow, dcol) matching engine convention
# LEFT=1 → row-1   RIGHT=2 → row+1   UP=3 → col-1   DOWN=4 → col+1
DIR_DELTA = {
    A_STOP: (0, 0),
    A_LEFT: (-1, 0),
    A_RIGHT: (1, 0),
    A_UP: (0, -1),
    A_DOWN: (0, 1),
}

BLAST_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# ── Map tile constants ───────────────────────────────────────────────────────────
TILE_GRASS = 0
TILE_WALL = 1
TILE_BOX = 2
TILE_RADIUS = 3
TILE_CAPACITY = 4

# ── Bomb constants ──────────────────────────────────────────────────────────────
BOMB_TIMER = 7
MAX_BOMB_RADIUS = 5
MAX_BOMB_CAPACITY = 5

# ── BFS / search limits ────────────────────────────────────────────────────────
BFS_MAX_DEPTH = 12             # hard cutoff for all BFS routines (13x13 grid)

# ── Network Architecture ────────────────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 64]
FC_HIDDEN = 128
STATE_CHANNELS = 7       # legacy 7-channel encoder
STATE_CHANNELS_V2 = 16   # production 16-channel encoder
SCALAR_FEATURES = 4
FRAME_STACK = 4          # temporal frame stacking (4×16=64 channels)

# ── Rewards ──────────────────────────────────────────────────────────────────────
# Terminal
REWARD_DEATH = -15.0
REWARD_OWN_BOMB_DEATH = -25.0
REWARD_WIN = 50.0
REWARD_KILL = 12.0

# Survival / progress
REWARD_LIVING = 0.03
REWARD_BOX_DESTROYED = 1.5          # per box, capped at 6.0
REWARD_ITEM_COLLECTED = 3.0
REWARD_BOMB_PLACED = 0.5            # base

# Action shaping
REWARD_STOP_PENALTY = -0.05
REWARD_BOMB_HOARDING = -0.02        # bomb available + safe to place + didn't place
REWARD_LOOP_PENALTY = -0.3
REWARD_REVISIT_PENALTY = -0.2

# Safety / escape
REWARD_DANGER_ZONE = -0.1
REWARD_ESCAPE_MARGIN_HIGH = 0.05    # >=3 safe neighbors
REWARD_ESCAPE_MARGIN_LOW = -0.5     # <=1 safe neighbor
REWARD_BOMB_ESCAPE_SUCCESS = 120.0
REWARD_TRAPPED_AFTER_BOMB = -140.0

# Exploration / territory
REWARD_ENTER_NEW_CELL = 5.0
REWARD_REACHABLE_INCREASE = 5.0
REWARD_CENTER_CONTROL = 0.01

# Late-game pressure
REWARD_LATEGAME_SURVIVAL = 0.02
REWARD_LATEGAME_PROXIMITY = 0.05    # within 5 cells of enemy after 60% progress
LATEGAME_PRESSURE_START = 350       # step when exponential kill scaling begins
REWARD_KILL_MAX = 30.0              # kill reward at step 500 (scales from 12.0)
REWARD_CORNER_CAMPING = -0.5        # penalty for staying in corners late game

# Opponent trapping
REWARD_OPPONENT_TRAP = 25.0         # bonus for trapping an enemy in a dead-end

# Bomb expected-value bonuses
REWARD_BOMB_BOX_HIT = 0.5           # per box in blast
REWARD_BOMB_ENEMY_THREAT = 1.5      # per enemy in blast line

# ── Paths ────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / "train_models" / "checkpoints"
AGENT_POOL_DIR = ROOT / "train_models" / "agent_pool"
LOG_DIR = ROOT / "train_models" / "logs"

# ── Inference (competition submission) ──────────────────────────────────────────
# Best checkpoint exported for submission is the one with highest eval win-rate.
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ── Warm-start ──────────────────────────────────────────────────────────────────
AUX_CHECKPOINT_PATH = ROOT / "ml" / "checkpoints" / "rl_agent_pure" / "aux_curriculum_model_v3.pt"

# ── Agent pool (self-play) ──────────────────────────────────────────────────────
POOL_MAX_SIZE = 20           # keep at most N historical checkpoints
POOL_INITIAL_AGENTS = [      # rule-based agents seeded into pool at start
    "TacticalRuleAgent",
    "SmarterRuleAgent",
    "GeniusRuleAgent",
]

# ── Device ───────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"     # automatic mixed precision on GPU only
PIN_MEMORY = DEVICE == "cuda"  # pinned memory for faster CPU→GPU transfers


def ensure_dirs():
    """Create all required directories."""
    for d in [CHECKPOINT_DIR, AGENT_POOL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
