from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover
    BaseFeaturesExtractor = object

try:
    from .constants import BOARD_SIZE, N_CHANNELS
except ImportError:  # Loaded as a submission-local module.
    from constants import BOARD_SIZE, N_CHANNELS


if nn is not None:
    class BomberCNN(nn.Module):
        def __init__(self, in_channels=N_CHANNELS, features_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, features_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x.float())


    class BomberFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)
            channels = int(observation_space.shape[0])
            self.cnn = BomberCNN(channels, features_dim)

        def forward(self, observations):
            return self.cnn(observations)
else:
    BomberCNN = None
    BomberFeaturesExtractor = None
