try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - inference fallback handles this.
    torch = None
    nn = None

from features import NUM_CHANNELS


class HybridBCNet(nn.Module if nn is not None else object):
    def __init__(self, input_channels=NUM_CHANNELS, num_actions=6):
        if nn is None:
            raise ImportError("torch is required for HybridBCNet")
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)
