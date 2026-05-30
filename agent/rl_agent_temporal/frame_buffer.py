from __future__ import annotations

from collections import deque

import numpy as np


class FrameBuffer:
    def __init__(self, frame_stack: int):
        self.frame_stack = max(1, int(frame_stack))
        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        frame = np.asarray(frame, dtype=np.float32)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self.stacked()

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if not self.frames:
            return self.reset(frame)
        self.frames.append(frame)
        return self.stacked()

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)
