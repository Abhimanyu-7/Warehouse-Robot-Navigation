
# agent.py
import numpy as np
import random
import os
from typing import Tuple

class QLearningAgent:
    def __init__(self, state_size:int, action_size:int, alpha=0.5, gamma=0.99, epsilon=1.0, min_epsilon=0.05, eps_decay=0.995, seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay = eps_decay
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Q table
        self.Q = np.zeros((state_size, action_size), dtype=np.float32)

    def act(self, state_idx:int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return int(np.argmax(self.Q[state_idx]))

    def update(self, s_idx:int, a:int, r:float, s2_idx:int, done:bool):
        qsa = self.Q[s_idx, a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s2_idx])
        self.Q[s_idx, a] = qsa + self.alpha * (target - qsa)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
