"""
PPO training script for quadcopter hover task.
Usage: python train.py --sim_device cuda:0 --graphics_device_id -1
"""

import os
import time
from datetime import datetime

from isaacgym import gymutil
from quadcopter_hover import QuadcopterEnv, NUM_ENVS, OBS_DIM, ACT_DIM
from ppo_agent import PPO, RolloutBuffer
import torch
import numpy as np

CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_param": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 1.0,
    "num_epochs": 5,
    "batch_size": 256,
    "num_steps": 16,
    "total_iterations": 1000,
    "save_interval": 100,
    "log_interval": 10,
    "num_envs": NUM_ENVS,
    "obs_dim": OBS_DIM,
    "act_dim": ACT_DIM,
    "log_dir": "runs",
    "ckpt_dir": "checkpoints",
}


def make_dirs(config):
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["ckpt_dir"], exist_ok=True)


def log_metrics(iteration, metrics, config):
    log_path = os.path.join(config["log_dir"], "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iteration,mean_reward,mean_ep_length,policy_loss,value_loss,entropy,approx_kl,fps\n")
    with open(log_path, "a") as f:
        f.write(
            "{},{:.4f},{:.2f},{:.6f},{:.6f},{:.6f},{:.6f},{:.1f}\n".format(
                iteration,
                metrics["mean_reward"],
                metrics["mean_ep_length"],
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["entropy"],
                metrics["approx_kl"],
                metrics["fps"],
            )
        )


def train():
    args = gymutil.parse_arguments(description="Quadcopter PPO Trainer")
    if args.sim_device == "cpu":
        args.sim_device = "cuda:0"

    config = CONFIG
    make_dirs(config)

    print("=" * 60)
    ts = datetime.now().strftime("%H:%M:%S")
    print("[{}] Creating environment on {}...".format(ts, args.sim_device))
    env = QuadcopterEnv(args)

    device = args.sim_device if "cuda" in str(args.sim_device) else "cpu"

    ts = datetime.now().strftime("%H:%M:%S")
    print("[{}] Creating PPO agent (obs={}, act={})...".format(ts, config["obs_dim"], config["act_dim"]))
    ppo = PPO(config["obs_dim"], config["act_dim"], config, device)

    buffer = RolloutBuffer(
        config["num_envs"], config["num_steps"],
        config["obs_dim"], config["act_dim"], device
    )

    obs = env.reset()
    episode_rewards = torch.zeros(config["num_envs"], device=device)
    episode_lengths = torch.zeros(config["num_envs"], device=device)
    completed_eps = []

    ts = datetime.now().strftime("%H:%M:%S")
    print("[{}] Training {} iterations...".format(ts, config["total_iterations"]))
    print("=" * 60)

    for it in range(config["total_iterations"]):
        t_start = time.time()
        buffer.step = 0

        # ---- Rollout ----
        for step in range(config["num_steps"]):
            with torch.no_grad():
                actions, log_probs, values = ppo.model.act(obs)

            next_obs, rewards, dones, info = env.step(actions)

            dones_bool = dones

            buffer.insert(obs, actions, log_probs, rewards, dones_bool, values)

            episode_rewards += rewards
            episode_lengths += 1

            if dones.any():
                done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_idx.dim() == 0:
                    done_idx = done_idx.unsqueeze(0)
                for idx in done_idx:
                    completed_eps.append({
                        "reward": episode_rewards[idx].item(),
                        "length": episode_lengths[idx].item(),
                    })
                episode_rewards[done_idx] = 0.0
                episode_lengths[done_idx] = 0.0

            obs = next_obs

        # ---- PPO Update ----
        with torch.no_grad():
            _, _, next_values = ppo.model.act(obs)

        losses = ppo.update(buffer, next_values)

        t_end = time.time()
        fps = (config["num_envs"] * config["num_steps"]) / (t_end - t_start)

        # ---- Logging ----
        recent = completed_eps[-100:] if completed_eps else []
        avg_ep_reward = np.mean([e["reward"] for e in recent]) if recent else 0.0
        avg_ep_len = np.mean([e["length"] for e in recent]) if recent else 0.0

        metrics = {
            "mean_reward": avg_ep_reward,
            "mean_ep_length": avg_ep_len,
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "approx_kl": losses["approx_kl"],
            "fps": fps,
        }

        if it % config["log_interval"] == 0:
            ts = datetime.now().strftime("%H:%M:%S")
            print("[{}] Iter {:4d} | Reward: {:7.2f} | EpLen: {:5.0f} | KL: {:.4f} | Entropy: {:.4f} | FPS: {:.0f}".format(
                ts, it, metrics["mean_reward"], metrics["mean_ep_length"],
                metrics["approx_kl"], metrics["entropy"], metrics["fps"]
            ))

        log_metrics(it, metrics, config)

        # ---- Checkpoint ----
        if it % config["save_interval"] == 0 and it > 0:
            ckpt_path = os.path.join(config["ckpt_dir"], "ppo_iter_{}.pt".format(it))
            ppo.save(ckpt_path)
            print("  -> Checkpoint: {}".format(ckpt_path))

    # ---- Final Save ----
    final_path = os.path.join(config["ckpt_dir"], "ppo_final.pt")
    ppo.save(final_path)
    print("\nTraining complete. Final model: {}".format(final_path))
    print("Logs: {}/train_log.csv".format(config["log_dir"]))


if __name__ == "__main__":
    train()
