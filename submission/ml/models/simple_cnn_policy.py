"""Tiny CNN policy for Bomberland imitation-learning experiments."""

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - depends on local env
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


INPUT_CHANNELS = 12
BOARD_SIZE = 13
NUM_ACTIONS = 6
TORCH_AVAILABLE = torch is not None


if TORCH_AVAILABLE:

    class SimpleCNNPolicy(nn.Module):
        """Small CPU-friendly CNN for 13x13 Bomberland feature tensors."""

        def __init__(self, input_channels=INPUT_CHANNELS, num_actions=NUM_ACTIONS):
            super().__init__()
            self.input_channels = int(input_channels)
            self.num_actions = int(num_actions)
            self.net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions),
            )

        def forward(self, x):
            return self.net(x)

else:

    class SimpleCNNPolicy:  # pragma: no cover - depends on local env
        """Placeholder that explains the optional PyTorch dependency."""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "PyTorch is not installed. Install torch to train or run the "
                "tiny imitation CNN baseline."
            ) from TORCH_IMPORT_ERROR


def build_model(input_channels=INPUT_CHANNELS, num_actions=NUM_ACTIONS):
    """Build the tiny policy model, or raise a clear error if torch is absent."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Skipping imitation model creation."
        ) from TORCH_IMPORT_ERROR
    return SimpleCNNPolicy(input_channels=input_channels, num_actions=num_actions)


def load_checkpoint(path, map_location="cpu"):
    """Load a saved imitation policy checkpoint."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Cannot load imitation checkpoints."
        ) from TORCH_IMPORT_ERROR

    checkpoint = torch.load(path, map_location=map_location)
    input_channels = int(checkpoint.get("input_channels", INPUT_CHANNELS))
    num_actions = int(checkpoint.get("num_actions", NUM_ACTIONS))
    model = build_model(input_channels=input_channels, num_actions=num_actions)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
