"""
Evaluation and disturbance testing for trained quadcopter policy.
Usage: python evaluate.py --sim_device cuda:0 --graphics_device_id -1 [--ckpt checkpoints/ppo_final.pt] [--headless]
"""

import os
import time
from isaacgym import gymutil

from quadcopter_hover import (
    QuadcopterEnv, NUM_ENVS, OBS_DIM, ACT_DIM, TARGET_POS,
    DISTURB_FORCE, DISTURB_DURATION, HOVER_THRESHOLD,
)
from ppo_agent import PPO
import torch
import numpy as np

SIM_DT = 1.0 / 60.0


def compute_metrics(positions, target):
    errors = np.linalg.norm(positions - target, axis=-1)
    return {
        "rmse_xyz": np.sqrt(np.mean(errors ** 2)),
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "hover_rate": np.mean(errors < HOVER_THRESHOLD),
    }


def evaluate(ckpt_path, headless=True):
    args = gymutil.parse_arguments(
        description="Quadcopter Evaluation",
        headless=headless,
    )
    if args.sim_device == "cpu":
        args.sim_device = "cuda:0"

    print("=" * 60)
    print("QUADCOPTER HOVER -- EVALUATION")
    print("=" * 60)

    env = QuadcopterEnv(args)

    device = args.sim_device if "cuda" in str(args.sim_device) else "cpu"
    ppo_config = {
        "lr": 3e-4, "gamma": 0.99, "lam": 0.95,
        "clip_param": 0.2, "value_coef": 0.5, "entropy_coef": 0.01,
        "max_grad_norm": 1.0, "num_epochs": 5, "batch_size": 256,
    }
    agent = PPO(OBS_DIM, ACT_DIM, ppo_config, device)
    agent.load(ckpt_path)

    target = TARGET_POS

    # =========================================
    # 1. Hover test (no disturbance)
    # =========================================
    print("\n[1/3] Hover test (no disturbance)...")
    obs = env.reset()
    all_positions = []
    eval_steps = 600  # 10 seconds

    for step in range(eval_steps):
        with torch.no_grad():
            actions, _, _ = agent.model.act(obs, deterministic=True)
        obs, rewards, dones, info = env.step(actions)
        pos = env.quad_pos[0].cpu().numpy()
        all_positions.append(pos)

    all_positions = np.array(all_positions)
    metrics = compute_metrics(all_positions, target)

    print("  RMSE:        {:.4f} m".format(metrics["rmse_xyz"]))
    print("  Mean Error:  {:.4f} m".format(metrics["mean_error"]))
    print("  Max Error:   {:.4f} m".format(metrics["max_error"]))
    print("  Hover Rate:  {:.1%}  (<{}m)".format(metrics["hover_rate"], HOVER_THRESHOLD))

    # =========================================
    # 2. Disturbance recovery test
    # =========================================
    print("\n[2/3] Disturbance recovery (force={}N, duration={}s)...".format(DISTURB_FORCE, DISTURB_DURATION))

    disturbance_steps = int(DISTURB_DURATION / SIM_DT)
    num_trials = 10
    recovery_times = []
    disturbance_dirs = ["+X", "-X", "+Y", "-Y"]

    for direction in disturbance_dirs:
        for trial in range(num_trials):
            env._reset_some([0])
            obs = env.reset()
            # 2-second warmup
            for s in range(120):
                with torch.no_grad():
                    actions, _, _ = agent.model.act(obs, deterministic=True)
                obs, _, _, _ = env.step(actions)

            if direction == "+X":
                force = np.array([DISTURB_FORCE, 0, 0])
            elif direction == "-X":
                force = np.array([-DISTURB_FORCE, 0, 0])
            elif direction == "+Y":
                force = np.array([0, DISTURB_FORCE, 0])
            else:
                force = np.array([0, -DISTURB_FORCE, 0])

            env.apply_disturbance(0, force)

            recovered = False
            for s in range(disturbance_steps):
                with torch.no_grad():
                    actions, _, _ = agent.model.act(obs, deterministic=True)
                obs, _, _, _ = env.step(actions)

                pos = env.quad_pos[0].cpu().numpy()
                err = np.linalg.norm(pos - target)
                if err < HOVER_THRESHOLD and not recovered:
                    recovery_time = s * SIM_DT
                    recovery_times.append(recovery_time)
                    recovered = True

            if not recovered:
                recovery_times.append(float("inf"))

    finite_times = [t for t in recovery_times if t != float("inf")]
    recovery_rate = len(finite_times) / len(recovery_times)
    avg_recovery = np.mean(finite_times) if finite_times else float("inf")

    print("  Recovery Rate:     {:.1%}".format(recovery_rate))
    print("  Avg Recovery Time: {:.3f} s".format(avg_recovery))
    if finite_times:
        print("  Min Recovery Time: {:.3f} s".format(np.min(finite_times)))

    # =========================================
    # 3. Random disturbance robustness
    # =========================================
    print("\n[3/3] Random disturbance robustness test...")

    obs = env.reset()
    total_steps = 1200  # 20 seconds
    disturb_interval = 120  # every 2 seconds
    hover_steps = 0
    total_errors = []
    crashed = False

    for step in range(total_steps):
        if step % disturb_interval == 0 and step > 0:
            disturbance = env.get_random_disturbance(magnitude=DISTURB_FORCE * 0.7)
            env.apply_disturbance(0, disturbance)

        with torch.no_grad():
            actions, _, _ = agent.model.act(obs, deterministic=True)
        obs, rewards, dones, info = env.step(actions)

        pos = env.quad_pos[0].cpu().numpy()
        err = np.linalg.norm(pos - target)
        total_errors.append(err)

        if dones[0]:
            crashed = True
            break
        if err < HOVER_THRESHOLD:
            hover_steps += 1

    total_errors = np.array(total_errors)
    print("  Survival:     {}".format("NO (crashed)" if crashed else "YES"))
    print("  Mean Error:   {:.4f} m".format(total_errors.mean()))
    print("  Hover Rate:   {:.1%}".format(hover_steps / len(total_errors)))
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/ppo_final.pt",
                        help="Path to checkpoint")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without graphics")
    args_extra, unknown = parser.parse_known_args()

    if not os.path.exists(args_extra.ckpt):
        print("Checkpoint not found: {}".format(args_extra.ckpt))
        print("Run train.py first, or specify with --ckpt <path>")
        exit(1)

    evaluate(args_extra.ckpt, headless=args_extra.headless)
