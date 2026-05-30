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

# ── Network Architecture ────────────────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 64]
FC_HIDDEN = 128
STATE_CHANNELS = 7
SCALAR_FEATURES = 4

# ── Rewards ──────────────────────────────────────────────────────────────────────
REWARD_DEATH = -10.0
REWARD_WIN = 10.0
REWARD_LIVING = 0.01
REWARD_BOX_DESTROYED = 0.2
REWARD_ITEM_COLLECTED = 0.5
REWARD_DANGER_ZONE = -0.1
REWARD_BOMB_PLACED = 0.05
REWARD_KILL = 2.0

# ── Paths ────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / "train_models" / "checkpoints"
AGENT_POOL_DIR = ROOT / "train_models" / "agent_pool"
LOG_DIR = ROOT / "train_models" / "logs"

# ── Inference (competition submission) ──────────────────────────────────────────
# Best checkpoint exported for submission is the one with highest eval win-rate.
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

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


def ensure_dirs():
    """Create all required directories."""
    for d in [CHECKPOINT_DIR, AGENT_POOL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
