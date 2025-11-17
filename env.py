
# env.py
"""
Simple Warehouse Grid Environment
State: (agent_row, agent_col, carrying_flag)
Actions: 0:UP,1:RIGHT,2:DOWN,3:LEFT,4:PICKUP/INTERACT,5:DROP
Reward shaping:
 - step: -0.1
 - collision/invalid move: -1
 - successful pickup: +1
 - successful drop at goal: +5
Episode ends after max_steps or successful delivery.
"""
import random
from typing import Tuple, List
import numpy as np

class WarehouseEnv:
    def __init__(self, rows=6, cols=8, shelves=None, pickup_loc=(1,1), drop_loc=(4,6), max_steps=200, seed=None):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.step_count = 0
        self.pickup_loc = pickup_loc
        self.drop_loc = drop_loc
        self.agent_pos = (0,0)
        self.carrying = False
        self.done = False
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # shelves/obstacles: list of (r,c)
        if shelves is None:
            # default simple shelf layout
            self.shelves = set([
                (2,2),(2,3),(2,4),
                (3,2),(3,3),(3,4),
            ])
        else:
            self.shelves = set(shelves)

        self.reset()

    def reset(self, random_start=False) -> Tuple[int,int,int]:
        self.step_count = 0
        self.carrying = False
        self.done = False
        if random_start:
            # choose random free cell
            free = [(r,c) for r in range(self.rows) for c in range(self.cols) if (r,c) not in self.shelves and (r,c)!=self.pickup_loc and (r,c)!=self.drop_loc]
            self.agent_pos = random.choice(free)
        else:
            # default start top-left free cell
            self.agent_pos = (0, 0)
            if self.agent_pos in self.shelves:
                self.agent_pos = (0, self.cols-1)
        return self._get_state()

    def _get_state(self):
        r,c = self.agent_pos
        return (r, c, int(self.carrying))

    def _in_bounds(self, pos):
        r,c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action:int):
        """
        action: 0:UP,1:RIGHT,2:DOWN,3:LEFT,4:PICKUP,5:DROP
        returns: state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        self.step_count += 1
        reward = -0.1  # small step penalty to encourage faster completion
        r,c = self.agent_pos

        if action == 0:  # UP
            nr, nc = r-1, c
        elif action == 1:  # RIGHT
            nr, nc = r, c+1
        elif action == 2:  # DOWN
            nr, nc = r+1, c
        elif action == 3:  # LEFT
            nr, nc = r, c-1
        elif action == 4:  # PICKUP/INTERACT
            nr, nc = r, c
            # pickup only if at pickup_loc and not carrying
            if not self.carrying and (r,c) == self.pickup_loc:
                self.carrying = True
                reward += 1.0
            else:
                reward -= 0.5
        elif action == 5:  # DROP
            nr, nc = r, c
            # drop only valid if carrying and at drop location
            if self.carrying and (r,c) == self.drop_loc:
                self.carrying = False
                reward += 5.0
                self.done = True
            else:
                reward -= 0.5
        else:
            nr, nc = r, c
            reward -= 0.5

        # movement handling
        if action in [0,1,2,3]:
            if not self._in_bounds((nr,nc)) or (nr,nc) in self.shelves:
                # invalid move: wall or shelf
                reward -= 1.0
                nr, nc = r, c  # stay
            else:
                self.agent_pos = (nr, nc)

        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), float(reward), self.done, {}

    def render(self) -> str:
        """Return a text representation of the grid"""
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        for (sr,sc) in self.shelves:
            grid[sr][sc] = '#'
        pr,pc = self.pickup_loc
        dr,dc = self.drop_loc
        grid[pr][pc] = 'P'
        grid[dr][dc] = 'D'
        ar,ac = self.agent_pos
        grid[ar][ac] = 'A' if not self.carrying else 'A*'
        lines = []
        for r in range(self.rows):
            row_str = ''
            for c in range(self.cols):
                row_str += f"{grid[r][c]:3}"
            lines.append(row_str)
        return "\n".join(lines)

    def observation_space_size(self):
        # rows * cols * 2 (carrying or not)
        return self.rows * self.cols * 2

    def encode_state(self, state):
        r,c,carry = state
        return (r * self.cols + c)*2 + int(carry)

    def decode_state(self, idx):
        carry = idx % 2
        cell = idx // 2
        r = cell // self.cols
        c = cell % self.cols
        return (r,c,carry)
