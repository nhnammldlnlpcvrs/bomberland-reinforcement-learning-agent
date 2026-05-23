# 06 - Self-play

Self-play trains against versions of the agent. It can improve trap handling and anti-camping behavior, but can also overfit to itself.

Practical approach:

1. Keep baseline opponents in the pool.
2. Add recent agent snapshots.
3. Add online-best style opponents.
4. Track draw rate and average steps.

Never accept self-play improvement without replay analysis.

