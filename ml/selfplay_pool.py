from __future__ import annotations

import random
from pathlib import Path


DEFAULT_BASELINES = ["random", "simple", "tactical", "agent/hybrid_agent_online_robust"]
ALIASES = {
    "online_robust": "agent/hybrid_agent_online_robust",
    "hybrid_agent_rl": "agent/hybrid_agent_rl",
    "rl_agent_pure": "agent/rl_agent_pure",
}

CURRICULUM = {
    "stage1": ["random", "simple"],
    "stage2": ["simple", "tactical", "agent/hybrid_agent_online_robust"],
    "stage3": ["simple", "tactical", "agent/hybrid_agent_online_robust", "checkpoints"],
    "stage4": ["tactical", "agent/hybrid_agent_online_robust", "agent/hybrid_agent_rl", "checkpoints"],
}


class SelfPlayPool:
    def __init__(self, checkpoint_dir="ml/checkpoints/rl_pure", stage="stage1", seed=None, opponents=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.stage = stage
        self.rng = random.Random(seed)
        self.override_opponents = list(opponents or [])

    def checkpoints(self):
        paths = []
        if self.checkpoint_dir.exists():
            paths.extend(self.checkpoint_dir.glob("*.zip"))
        archive_dir = Path("ml/checkpoints/rl_agent_pure")
        if archive_dir.exists() and archive_dir != self.checkpoint_dir:
            paths.extend(archive_dir.glob("*.zip"))
        return sorted({str(path) for path in paths})

    def opponents(self):
        entries = self.override_opponents or list(CURRICULUM.get(self.stage, DEFAULT_BASELINES))
        ckpts = self.checkpoints()
        expanded = []
        for entry in entries:
            if entry == "checkpoints":
                expanded.extend(ckpts)
            else:
                expanded.append(ALIASES.get(entry, entry))
        expanded = [entry for entry in expanded if self._is_available(entry)]
        return expanded or DEFAULT_BASELINES

    def sample(self, n=3):
        pool = self.opponents()
        return [self.rng.choice(pool) for _ in range(n)]

    def _is_available(self, entry):
        entry = ALIASES.get(entry, entry)
        if entry in {"random", "simple", "box_farmer", "smarter", "tactical"}:
            return True
        path = Path(entry)
        if path.suffix == ".zip":
            return path.exists()
        if path.is_dir():
            return (path / "agent.py").exists()
        if path.exists():
            return True
        return False
