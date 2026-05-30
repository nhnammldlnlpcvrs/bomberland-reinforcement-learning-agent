from __future__ import annotations

import numpy as np

try:
    from .action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from .encoder import encode_observation
except ImportError:  # Loaded as a submission-local module.
    from action_mask import highest_prob_valid, legal_action_mask, sanitize_action
    from encoder import encode_observation


def predict_with_mask(model, obs, agent_id, deterministic=True):
    tensor_obs = encode_observation(obs, agent_id)
    action, _state = model.predict(tensor_obs, deterministic=deterministic)
    action, invalid = sanitize_action(action, obs, agent_id)
    if not invalid:
        return action

    try:
        import torch

        with torch.no_grad():
            obs_tensor = torch.as_tensor(tensor_obs[None], dtype=torch.float32, device=model.policy.device)
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
        return highest_prob_valid(probs, obs, agent_id)
    except Exception:
        valid = np.flatnonzero(legal_action_mask(obs, agent_id))
        return int(valid[0]) if valid.size else 0
