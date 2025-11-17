
# evaluate.py
import argparse
from env import WarehouseEnv
from agent import QLearningAgent
import numpy as np
import time

def evaluate(args):
    env = WarehouseEnv(rows=args.rows, cols=args.cols,
                       pickup_loc=tuple(map(int, args.pickup.split(','))),
                       drop_loc=tuple(map(int, args.drop.split(','))),
                       max_steps=args.max_steps, seed=args.seed)
    state_size = env.observation_space_size()
    action_size = 6
    agent = QLearningAgent(state_size, action_size, seed=args.seed)
    agent.load(args.qtable)

    for ep in range(args.episodes):
        s = env.reset(random_start=False)
        s_idx = env.encode_state(s)
        total = 0.0
        done = False
        print(f"\n=== Episode {ep+1} ===")
        while not done:
            a = int(np.argmax(agent.Q[s_idx]))
            s2, r, done, _ = env.step(a)
            s_idx = env.encode_state(s2)
            total += r
            # text render
            print(env.render())
            print(f"Action: {a}, Reward: {r:.2f}, Total: {total:.2f}\n")
            time.sleep(args.render_delay)
        print(f"Episode finished. Total reward: {total:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default="models/q_table_best.npy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--pickup", type=str, default="1,1")
    parser.add_argument("--drop", type=str, default="4,6")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--render_delay", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args)

