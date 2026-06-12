"""Train the isolated rl_strong Bomberland PPO track.

This is a thin, stable entrypoint around the stronger frame-stacked PPO trainer.
It exists so experiments can target `agent/rl_strong` without touching the
production root `agent.py`.
"""

from __future__ import annotations

from ml.train_bomberland_strong.train import main


if __name__ == "__main__":
    main()
