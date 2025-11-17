# train.py
import os
import argparse
import numpy as np
from env import WarehouseEnv
from agent import QLearningAgent
import matplotlib.pyplot as plt

def train(args):
    env = WarehouseEnv(rows=args.rows, cols=args.cols,
                       shelves=None,
                       pickup_loc=tuple(map(int, args.pickup.split(','))),
                       drop_loc=tuple(map(int, args.drop.split(','))),
                       max_steps=args.max_steps, seed=args.seed)

    state_size = env.observation_space_size()
    action_size = 6
    agent = QLearningAgent(state_size, action_size,
                           alpha=args.alpha, gamma=args.gamma,
                           epsilon=args.epsilon, min_epsilon=args.min_epsilon,
                           eps_decay=args.eps_decay, seed=args.seed)

    rewards_history = []
    avg_history = []
    best_avg = -1e9

    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        state = env.reset(random_start=args.random_start)
        s_idx = env.encode_state(state)
        total_r = 0.0
        done = False

        while not done:
            a = agent.act(s_idx)
            s2, r, done, _ = env.step(a)
            s2_idx = env.encode_state(s2)
            agent.update(s_idx, a, r, s2_idx, done)
            s_idx = s2_idx
            total_r += r

        agent.decay_epsilon()

        rewards_history.append(total_r)
        avg = float(np.mean(rewards_history[-100:])) if len(rewards_history) > 0 else total_r
        avg_history.append(avg)

        if ep % args.log_every == 0 or ep == 1:
            print(f"Episode {ep}/{args.episodes} | Reward: {total_r:.2f} | Avg100: {avg:.2f} | Eps: {agent.epsilon:.3f}")

        # save checkpoints
        if ep % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"q_table_ep{ep}.npy")
            agent.save(ckpt_path)
            np.save(os.path.join(args.save_dir, "rewards.npy"), np.array(rewards_history))
            np.save(os.path.join(args.save_dir, "avg.npy"), np.array(avg_history))

        # save best
        if avg > best_avg:
            best_avg = avg
            agent.save(os.path.join(args.save_dir, "q_table_best.npy"))

    # save final
    agent.save(os.path.join(args.save_dir, "q_table_final.npy"))

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(rewards_history, label="Episode reward")
    plt.plot(avg_history, label="Avg100")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curve.png"))
    plt.close()

    print(f"Training completed. Models saved to: {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--pickup", type=str, default="1,1")
    parser.add_argument("--drop", type=str, default="4,6")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--min_epsilon", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # THIS LINE FIXES THE ERROR
    parser.add_argument("--checkpoint_every", type=int, default=200)

    args = parser.parse_args()

    train(args)
