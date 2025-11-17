# Warehouse Robot Navigation using Q-Learning (Gymnasium)

A Reinforcement Learning project where an autonomous agent learns to navigate a warehouse-like grid environment to reach a goal while avoiding obstacles.
The project uses Tabular Q-Learning with a custom Gymnasium-compatible environment.

## 🚀 Project Overview

This project implements an autonomous warehouse robot trained using Q-Learning, a model-free reinforcement learning algorithm.
The robot moves inside a grid, receives rewards based on its actions, and gradually learns the optimal navigation policy.

The project includes:

- Custom WarehouseEnv (Gymnasium environment)

- Q-learning implementation

- Training and evaluation pipeline

- Reward, exploration, and convergence visualizations

- Policy map, heatmaps, and rollout demonstrations

- Final PDF report + presentation-ready resources

## 🎯 Problem Statement

Warehouses rely heavily on mobile robots for transporting goods efficiently.
The objective is to design an RL agent that:

- Navigates a grid-based warehouse environment

- Avoids obstacles

- Learns shortest routes to the goal

- Maximizes cumulative rewards

- Operates without prior knowledge (pure RL learning)

## 🧠 Approach
### 1. Environment Design (WarehouseEnv)

- Grid world with
     - Start and goal states
     - Obstacles
     - Step penalty, goal reward, and obstacle penalty

- Observation: encoded as integer state

- Action space: {UP, DOWN, LEFT, RIGHT}

### 2. Q-Learning Implementation

- Tabular Q-Table: states × actions

- Update Rule:
```Q(s, a) ← Q(s, a) + α ( r + γ max(Q(s', ·)) − Q(s, a) )```

- Exploration strategy: Epsilon-Greedy

- Epsilon decay per episode

### 3. Training Loop

- Run N episodes

- Update Q-values

- Track reward, steps, epsilon

- Save plots + Q-table

### 4. Evaluation

- Generate rollout

- Compute success rate

- Produce policy arrow map

- Plot heatmaps and reward curves

## 📊 Results Summary

After training:

- Agent successfully discovers efficient path to goal

- Rewards per episode show clear learning curve

- Epsilon decreases, exploration → exploitation

- Q-values converge after sufficient episodes

- Policy map shows optimal movement from all states

- State visit heatmap confirms exploration

The agent demonstrates strong navigation behavior in a static grid world environment.

## 📁 Repository Structure
```
.
├── env/
│   └── warehouse_env.py        # Custom Gymnasium warehouse environment
│
├── notebooks/
│   └── RL_project.ipynb        # Main Jupyter notebook
│
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── make_gif.py             # Creates rollout GIF / MP4
│   └── plot_utils.py           # Helper for graphs
│
├── reports/
│   ├── Warehouse_RL_Report.pdf # Final project report
│   ├── figures/                # Saved plots
│   └── demo_rollout.mp4        # Navigation video
│
├── checkpoints/
│   └── q_table.npy             # Saved model
│
├── requirements.txt
└── README.md
```
## ⚙️ Installation

Clone the repository:
``
git clone <YOUR_REPOSITORY_URL>
cd warehouse-robot-rl
``

Install dependencies:
```
pip install -r requirements.txt
```

Example dependencies:
```
gymnasium
numpy
matplotlib
seaborn
pandas
imageio
opencv-python
reportlab
```

▶️ Run the Project
1. Train the Agent
`python scripts/train.py`

2. Evaluate the Agent
`python scripts/evaluate.py --q-table checkpoints/q_table.npy`

3. Generate Demo GIF / MP4
`python scripts/make_gif.py`
