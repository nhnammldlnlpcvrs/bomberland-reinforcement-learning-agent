from pathlib import Path

import numpy as np

from encoder import encode
from model import HybridBCNet

try:
    import torch
except Exception:  # pragma: no cover - rule fallback handles this.
    torch = None


class RLPolicy:
    def __init__(self, model_path=None):
        self.ok = False
        self.model = None
        if torch is None:
            return
        path = Path(model_path) if model_path is not None else Path(__file__).with_name("model.pth")
        if not path.exists():
            return
        try:
            model = HybridBCNet()
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.ok = True
        except Exception:
            self.ok = False
            self.model = None

    def logits(self, obs, agent_id):
        if not self.ok or self.model is None:
            return None
        try:
            x = torch.from_numpy(encode(obs, agent_id)).float().unsqueeze(0)
            with torch.no_grad():
                out = self.model(x).squeeze(0).cpu().numpy()
            return np.asarray(out, dtype=np.float32)
        except Exception:
            return None

    def predict(self, obs, agent_id):
        return self.logits(obs, agent_id)
