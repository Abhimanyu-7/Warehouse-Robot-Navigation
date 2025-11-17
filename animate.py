# animate.py
"""
Run a trained Q-table policy for one episode, record frames and save a GIF.
Usage:
python animate.py --qtable /content/warehouse_robot/models/q_table_best.npy --out /content/warehouse_robot/episode.gif
"""
import argparse
import numpy as np
from env_gym import WarehouseGymEnv
from agent import QLearningAgent
from IPython.display import display
import os
from PIL import Image

def run_episode_and_save(env, agent, max_steps=500, save_path="episode.gif", fps=4):
    frames = []
    obs, _ = env.reset()
    s_idx = obs
    done = False
    step = 0
    while not done and step < max_steps:
        a = int(np.argmax(agent.Q[s_idx]))
        obs, r, terminated, truncated, _ = env.step(a)
        s_idx = obs
        # get PIL frame
        frame = env.render_pil().convert("P", palette=Image.ADAPTIVE)
        frames.append(frame)
        done = bool(terminated or truncated)
        step += 1

    if len(frames) == 0:
        raise RuntimeError("No frames captured")

    # save as GIF
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=False, duration=1000//fps, loop=0)
    print(f"Saved animation to {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default="models/q_table_best.npy")
    parser.add_argument("--out", type=str, default="episode.gif")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = WarehouseGymEnv(rows=args.rows, cols=args.cols, max_steps=args.max_steps, seed=args.seed)
    state_size = env.observation_space.n
    agent = QLearningAgent(state_size, env.action_space.n)
    agent.load(args.qtable)

    out_path = run_episode_and_save(env, agent, max_steps=args.max_steps, save_path=args.out, fps=args.fps)

    # If run in Colab, you can display it using IPython.display.Image
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=out_path))
    except Exception:
        pass
