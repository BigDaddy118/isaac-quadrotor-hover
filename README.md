# Quadcopter Hover — PPO with Isaac Gym

Reinforcement learning project that trains a quadcopter to hover at a target position using PPO (Proximal Policy Optimization) on NVIDIA Isaac Gym.

## Overview

The environment simulates 64 quadcopter agents in parallel. Each agent's goal is to reach and hold the target position `(0, 0, 2)` using 4 rotor thrusts as actions. The policy is trained end-to-end on GPU with Isaac Gym's PhysX backend.

The reward function combines:
- **Position error** — stay close to target
- **Attitude penalty** — keep the body level
- **Velocity / angular velocity penalty** — smooth, stable flight
- **Action penalty** — efficient rotor usage
- **Hover bonus** — reward for staying within 0.15 m of the target

## Files

| File | Purpose |
|------|---------|
| `quadcopter_hover.py` | Isaac Gym environment — sim, agents, physics, rewards |
| `ppo_agent.py` | Standalone PPO implementation (Actor-Critic + RolloutBuffer) |
| `train.py` | Training loop — rollout collection, PPO updates, logging, checkpoints |
| `evaluate.py` | Policy evaluation — hover accuracy, disturbance recovery, robustness |
| `test_gym.py` | Minimal smoke test to verify Isaac Gym installation |
| `requirements.txt` | Python dependencies |
| `cfg/task/QuadcopterHover.yaml` | Configuration for Isaac Gym task format |
| `tasks/quadcopter_hover.py` | Alternative VecTask-based environment implementation |

## Requirements

- **NVIDIA GPU** with CUDA support
- **Isaac Gym Preview 4** ([download](https://developer.nvidia.com/isaac-gym)) — must be installed manually
- Python 3.8+

```bash
pip install -r requirements.txt
```

> Isaac Gym is **not** installable via pip. Download it from NVIDIA's developer site and follow its installation instructions before running this project.

## Usage

### 1. Smoke test

Verify Isaac Gym works:

```bash
python test_gym.py
```

### 2. Train

```bash
python train.py --sim_device cuda:0 --graphics_device_id -1
```

Trains for 1,000 iterations. Checkpoints are saved to `checkpoints/` every 100 iterations (final: `checkpoints/ppo_final.pt`). Logs are written to `runs/train_log.csv`.

### 3. Evaluate

```bash
python evaluate.py --sim_device cuda:0 --graphics_device_id -1 --ckpt checkpoints/ppo_final.pt
```

Runs three test phases:
1. **Hover test** — no disturbances, 10 seconds
2. **Directional disturbance** — 20 N pushes from ±X, ±Y with recovery timing
3. **Random disturbance** — repeated random pushes every 2 seconds over 20 seconds

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_ENVS` | 64 | Parallel environments |
| `TARGET_POS` | (0, 0, 2) | Hover target (meters) |
| `MAX_THRUST` | 5.0 N | Max thrust per rotor |
| `DISTURB_FORCE` | 20.0 N | Disturbance force magnitude |
| `HOVER_THRESHOLD` | 0.15 m | Distance to count as "hovering" |
| `total_iterations` | 1000 | PPO training iterations |
| `lr` | 3e-4 | Learning rate |
| `num_steps` | 16 | Steps per rollout |
| `num_epochs` | 5 | PPO update epochs |

## Project Structure

```
├── quadcopter_hover.py     # Main environment
├── ppo_agent.py            # PPO algorithm
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── test_gym.py             # Gym smoke test
├── requirements.txt        # Python deps
├── cfg/
│   └── task/
│       └── QuadcopterHover.yaml
├── tasks/
│   └── quadcopter_hover.py # VecTask variant
└── assets/                 # Asset directory (placeholder)
```
