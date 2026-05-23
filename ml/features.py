"""Feature extraction placeholders for future Bomberland ML models."""


def encode_observation(obs):
    """
    Convert Bomberland obs into model-friendly features.

    Future idea:
    - multi-channel 13x13 tensor
    - player planes
    - bomb timer planes
    - danger map
    - item planes
    """
    raise NotImplementedError("encode_observation is a future ML placeholder")


def handcrafted_features(obs):
    """
    Extract tabular features:
    safe_cells, enemy_distance, bomb_pressure, territory_score, etc.
    """
    raise NotImplementedError("handcrafted_features is a future ML placeholder")

