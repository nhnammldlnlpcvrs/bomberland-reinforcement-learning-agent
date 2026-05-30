BOARD_SIZE = 13
NUM_ACTIONS = 6
BOMB_TIMER = 7
MAX_STEPS = 500

TILE_GRASS = 0
TILE_WALL = 1
TILE_BOX = 2
TILE_ITEM_RADIUS = 3
TILE_ITEM_CAPACITY = 4

STOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
PLACE_BOMB = 5

MOVE_DELTAS = {
    STOP: (0, 0),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    UP: (0, -1),
    DOWN: (0, 1),
}
MOVE_ACTIONS = (LEFT, RIGHT, UP, DOWN)
BLAST_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))

CHANNELS = (
    "wall",
    "box",
    "grass",
    "item_radius",
    "item_capacity",
    "self_position",
    "enemy_positions",
    "alive_enemies",
    "bomb_positions",
    "bomb_timer_normalized",
    "bomb_owner_self",
    "bomb_owner_enemy",
    "danger_now",
    "danger_soon",
    "danger_future",
    "reachable_area",
    "legal_move_cells",
    "center_bias",
    "step_normalized",
)
N_CHANNELS = len(CHANNELS)

REWARD_WEIGHTS = {
    "win": 1000.0,
    "last_survivor_bonus": 300.0,
    "enemy_eliminated": 100.0,
    "destroy_box": 25.0,
    "collect_item": 35.0,
    "survival_step": 0.0,
    "enter_new_cell": 5.0,
    "increase_reachable_area": 5.0,
    "good_bomb_value": 120.0,
    "successful_bomb_escape": 120.0,
    "post_bomb_move_away": 12.0,
    "post_bomb_corridor_stay": -25.0,
    "post_bomb_early_death": -700.0,
    "bomb_destroy_box": 75.0,
    "trapped_after_bomb": -140.0,
    "death": -1000.0,
    "standing_in_danger": -150.0,
    "bomb_without_escape": -100.0,
    "useless_bomb": -40.0,
    "bomb_suicide": -500.0,
    "invalid_action": -10.0,
    "repeated_position": -15.0,
    "excessive_stop": -20.0,
}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
