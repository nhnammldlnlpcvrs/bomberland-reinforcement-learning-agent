# Bomberland Reward Design

Safe reward shaping should reflect long-term behavior:

- Win: large positive.
- Loss: large negative.
- Survival: small positive, capped to avoid camping.
- Kill: positive, but only if our survival remains safe.
- Territory: positive for safe reachable cells and open space.
- Bomb quality: positive for boxes, enemy pressure, and escape margin.
- Loop: negative for repeated local positions.

Bad rewards:

- Large per-step survival reward that causes draws.
- Large bomb reward that causes spam.
- Large chase reward that causes unsafe deaths.

