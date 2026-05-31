from __future__ import annotations

import random
from pathlib import Path


class StrongSelfPlayPool:
    """Curated opponent pool for the strong Kaggle training track.

    This deliberately avoids blindly loading every old failed checkpoint. Only
    known baselines, the frame-stack survival checkpoint, and explicitly accepted
    checkpoints under the training save dir are included.
    """

    def __init__(self, save_dir: str, stage: str, seed: int = 0, extra_opponents: list[str] | None = None):
        self.save_dir = Path(save_dir)
        self.stage = str(stage)
        self.rng = random.Random(seed)
        self.extra_opponents = list(extra_opponents or [])

    def opponents(self) -> list[str]:
        pool: list[str] = ["random", "simple"]
        if self.stage in {"stage2", "stage3"}:
            pool.extend(["online_robust", "hybrid_agent_rl"])
        if self.stage == "stage3":
            pool.extend(self.accepted_checkpoints())
        for opponent in self.extra_opponents:
            if opponent not in pool:
                pool.append(opponent)
        return self._existing(pool)

    def accepted_checkpoints(self) -> list[str]:
        checkpoints: list[str] = []
        for name in ("best_overall.zip", "best_by_score.zip", "latest_accepted.zip"):
            path = self.save_dir / name
            if path.exists():
                checkpoints.append(str(path))
        history = self.save_dir / "accepted_history"
        if history.exists():
            checkpoints.extend(str(path) for path in sorted(history.glob("*.zip")))
        return checkpoints

    def sample(self) -> str:
        pool = self.opponents()
        return self.rng.choice(pool) if pool else "random"

    @staticmethod
    def _existing(pool: list[str]) -> list[str]:
        checked: list[str] = []
        for opponent in pool:
            if opponent in {"random", "simple", "online_robust"}:
                checked.append(opponent)
                continue
            if opponent == "hybrid_agent_rl" and not Path("agent/hybrid_agent_rl").exists():
                continue
            if opponent.endswith(".zip") and not Path(opponent).exists():
                continue
            checked.append(opponent)
        return checked
