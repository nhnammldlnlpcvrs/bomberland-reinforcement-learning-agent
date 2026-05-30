from collections import deque

from constants import BOARD_SIZE


class AgentMemory:
    def __init__(self, agent_id, max_positions=16):
        self.agent_id = int(agent_id)
        self.positions = deque(maxlen=max_positions)
        self.actions = deque(maxlen=max_positions)
        self.bomb_positions = deque(maxlen=8)
        self.box_counts = deque(maxlen=20)
        self.item_counts = deque(maxlen=20)
        self.reachable_counts = deque(maxlen=20)

    def maybe_reset(self, pos, bombs):
        spawns = {
            0: (1, 1),
            1: (BOARD_SIZE - 2, BOARD_SIZE - 2),
            2: (1, BOARD_SIZE - 2),
            3: (BOARD_SIZE - 2, 1),
        }
        if pos == spawns.get(self.agent_id) and not bombs and len(self.positions) > 8:
            self.positions.clear()
            self.actions.clear()
            self.bomb_positions.clear()
            self.box_counts.clear()
            self.item_counts.clear()
            self.reachable_counts.clear()

    def observe_position(self, pos):
        self.positions.append(pos)

    def observe_action(self, action):
        self.actions.append(int(action))

    def observe_bomb(self, pos):
        self.bomb_positions.append(pos)

    def observe_progress(self, box_count, item_count, reachable_count):
        self.box_counts.append(int(box_count))
        self.item_counts.append(int(item_count))
        self.reachable_counts.append(int(reachable_count))

    def is_stuck(self):
        return (
            self.is_two_cell_loop()
            or self.is_three_cell_loop()
            or self.idle_steps() >= 3
            or self.no_progress_steps() >= 12
        )

    def is_two_cell_loop(self):
        recent = list(self.positions)[-6:]
        return len(recent) >= 6 and recent[-1] == recent[-3] == recent[-5] and recent[-2] == recent[-4] == recent[-6]

    def is_three_cell_loop(self):
        recent = list(self.positions)[-7:]
        return len(recent) >= 7 and recent[-1] == recent[-4] == recent[-7] and recent[-2] == recent[-5] and recent[-3] == recent[-6]

    def idle_steps(self):
        recent = list(self.positions)
        if not recent:
            return 0
        last = recent[-1]
        count = 0
        for pos in reversed(recent):
            if pos != last:
                break
            count += 1
        return count

    def no_progress_steps(self):
        if len(self.box_counts) < 8:
            return 0
        steps = min(len(self.box_counts), len(self.item_counts), len(self.reachable_counts))
        for window in range(min(steps, 16), 3, -1):
            boxes = list(self.box_counts)[-window:]
            items = list(self.item_counts)[-window:]
            reachable = list(self.reachable_counts)[-window:]
            box_progress = boxes[-1] < boxes[0]
            item_progress = items[-1] < items[0]
            reach_progress = reachable[-1] > reachable[0] + 2
            if box_progress or item_progress or reach_progress:
                return 0
        return min(steps, 16)

    def repeat_count(self, pos):
        return list(self.positions).count(pos)

    def recent_bomb_near(self, pos, radius=2):
        for br, bc in self.bomb_positions:
            if abs(pos[0] - br) + abs(pos[1] - bc) <= radius:
                return True
        return False
