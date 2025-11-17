# env_gym.py
"""
Gymnasium-compatible wrapper for the Warehouse environment with an RGB renderer.
Observation: Discrete integer encoding (rows*cols*2)
Action space: Discrete(6)
Gym API: reset() -> obs, info ; step() -> obs, reward, terminated, truncated, info
Renderer: render(mode='rgb_array') returns a HxWx3 numpy uint8 array.
Also provides render_pil() which returns a PIL.Image for easy GIF creation.
"""
import numpy as np
import random
from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

CELL = 48  # pixels per grid cell for rendering

class WarehouseGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, rows=6, cols=8, shelves=None, pickup_loc=(1,1), drop_loc=(4,6), max_steps=200, seed=None):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.pickup_loc = tuple(pickup_loc)
        self.drop_loc = tuple(drop_loc)
        self.seed_val = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if shelves is None:
            self.shelves = set([(2,2),(2,3),(2,4),(3,2),(3,3),(3,4)])
        else:
            self.shelves = set(shelves)

        # Gym spaces
        self.observation_space = spaces.Discrete(self.rows * self.cols * 2)
        self.action_space = spaces.Discrete(6)

        # internal state
        self._reset_internal()

    def _reset_internal(self):
        self.step_count = 0
        self.carrying = False
        self.done = False
        self.agent_pos = (0,0)
        if self.agent_pos in self.shelves:
            self.agent_pos = (0, self.cols - 1)

    def seed(self, seed=None):
        self.seed_val = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_internal()
        # not random start by default; users can set attribute if they want random
        return self._encode_state((self.agent_pos[0], self.agent_pos[1], int(self.carrying))), {}

    def _in_bounds(self, pos):
        r,c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action: int):
        if self.done:
            obs = self._encode_state((self.agent_pos[0], self.agent_pos[1], int(self.carrying)))
            return obs, 0.0, True, False, {}

        self.step_count += 1
        reward = -0.1
        r,c = self.agent_pos

        if action == 0:  # UP
            nr, nc = r-1, c
        elif action == 1:  # RIGHT
            nr, nc = r, c+1
        elif action == 2:  # DOWN
            nr, nc = r+1, c
        elif action == 3:  # LEFT
            nr, nc = r, c-1
        elif action == 4:  # PICKUP
            nr, nc = r, c
            if (r,c) == self.pickup_loc and not self.carrying:
                self.carrying = True
                reward += 1.0
            else:
                reward -= 0.5
        elif action == 5:  # DROP
            nr, nc = r, c
            if (r,c) == self.drop_loc and self.carrying:
                self.carrying = False
                reward += 5.0
                self.done = True
            else:
                reward -= 0.5
        else:
            nr, nc = r, c
            reward -= 0.5

        # movement
        if action in (0,1,2,3):
            if not self._in_bounds((nr,nc)) or (nr,nc) in self.shelves:
                reward -= 1.0
                nr, nc = r, c
            else:
                self.agent_pos = (nr, nc)

        if self.step_count >= self.max_steps:
            self.done = True
            truncated = True
        else:
            truncated = False

        terminated = bool(self.done)
        obs = self._encode_state((self.agent_pos[0], self.agent_pos[1], int(self.carrying)))
        return obs, float(reward), terminated, truncated, {}

    def _encode_state(self, state: Tuple[int,int,int]) -> int:
        r,c,carry = state
        return (r * self.cols + c) * 2 + int(carry)

    def _decode_state(self, idx: int) -> Tuple[int,int,int]:
        carry = int(idx % 2)
        cell = idx // 2
        r = cell // self.cols
        c = cell % self.cols
        return (r, c, carry)

    def render(self, mode="rgb_array"):
        """Return HxWx3 np.uint8 image for Gym rendering."""
        img = self.render_pil()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr

    def render_pil(self):
        """Return a PIL Image representing the grid (useful for GIF creation)."""
        W = self.cols * CELL
        H = self.rows * CELL
        img = Image.new("RGBA", (W, H), (255,255,255,255))
        draw = ImageDraw.Draw(img)

        # draw grid and tiles
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * CELL
                y0 = r * CELL
                x1 = x0 + CELL - 1
                y1 = y0 + CELL - 1
                bbox = (x0, y0, x1, y1)

                if (r,c) in self.shelves:
                    draw.rectangle(bbox, fill=(50,50,50))  # dark shelf
                else:
                    draw.rectangle(bbox, outline=(180,180,180), fill=(245,245,245))

        # pickup (green) and drop (blue)
        pr,pc = self.pickup_loc
        dr,dc = self.drop_loc
        pr_bbox = (pc*CELL+6, pr*CELL+6, (pc+1)*CELL-6, (pr+1)*CELL-6)
        dr_bbox = (dc*CELL+6, dr*CELL+6, (dc+1)*CELL-6, (dr+1)*CELL-6)
        draw.ellipse(pr_bbox, fill=(80,200,120))
        draw.ellipse(dr_bbox, fill=(80,140,220))

        # agent (red). if carrying, draw star overlay
        ar,ac = self.agent_pos
        a_bbox = (ac*CELL+8, ar*CELL+8, (ac+1)*CELL-8, (ar+1)*CELL-8)
        draw.ellipse(a_bbox, fill=(220,60,60))
        if self.carrying:
            # small star / square on top-right of cell
            sx0 = (ac+1)*CELL - 16
            sy0 = ar*CELL + 4
            draw.rectangle((sx0, sy0, sx0+12, sy0+12), fill=(255,215,0))

        # grid lines
        for r in range(self.rows+1):
            draw.line((0, r*CELL, self.cols*CELL, r*CELL), fill=(200,200,200))
        for c in range(self.cols+1):
            draw.line((c*CELL, 0, c*CELL, self.rows*CELL), fill=(200,200,200))

        return img

    def close(self):
        return
