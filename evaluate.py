"""
Evaluation and disturbance testing for trained quadcopter policy.
Usage: python evaluate.py --sim_device cuda:0 --graphics_device_id -1 [--ckpt checkpoints/ppo_final.pt] [--headless] [--record]
"""

import os
import time
from isaacgym import gymutil, gymapi

from quadcopter_hover_urdf import (
    QuadcopterEnv, NUM_ENVS, OBS_DIM, ACT_DIM, TARGET_POS,
    DISTURB_FORCE, DISTURB_DURATION, HOVER_THRESHOLD,
)
from ppo_agent import PPO
import torch
import numpy as np

SIM_DT = 1.0 / 60.0
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


def compute_metrics(positions, target):
    errors = np.linalg.norm(positions - target, axis=-1)
    return {
        "rmse_xyz": np.sqrt(np.mean(errors ** 2)),
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "hover_rate": np.mean(errors < HOVER_THRESHOLD),
    }


def record_hover(env, agent, output_path, duration=10.0):
    """Record hover demonstration video using Isaac Gym camera sensor."""
    cam_props = gymapi.CameraProperties()
    cam_props.width = VIDEO_WIDTH
    cam_props.height = VIDEO_HEIGHT
    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
    env.gym.set_camera_location(
        cam_handle, env.envs[0],
        gymapi.Vec3(3.0, 3.0, 3.0),
        gymapi.Vec3(0.0, 0.0, 2.0),
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    num_frames = int(duration * VIDEO_FPS)
    steps_per_frame = max(1, int((1.0 / VIDEO_FPS) / SIM_DT))

    import imageio
    frames = []
    obs = env.reset()

    print("Recording {} frames at {} fps ({:.1f}s)...".format(num_frames, VIDEO_FPS, duration))

    for frame_i in range(num_frames):
        for _ in range(steps_per_frame):
            with torch.no_grad():
                actions, _, _ = agent.model.act(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)

        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_stream(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
        env.gym.end_access_image_stream(env.sim)

        if img.size > 0:
            frame = img.reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 4)[:, :, :3]
            frames.append(frame)

        if (frame_i + 1) % 50 == 0:
            print("  {} / {} frames".format(frame_i + 1, num_frames))

    if frames:
        imageio.mimwrite(output_path, frames, fps=VIDEO_FPS, quality=8)
        print("Video saved: {}".format(output_path))
    else:
        print("Warning: no frames captured. Try running without headless mode.")


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
        "lr": 1e-4, "gamma": 0.99, "lam": 0.95,
        "clip_param": 0.2, "value_coef": 0.5, "entropy_coef": 0.02,
        "max_grad_norm": 1.0, "num_epochs": 5, "batch_size": 2048,
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
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record hover demonstration video")
    parser.add_argument("--output", type=str, default="videos/hover_demo.mp4",
                        help="Output video path")
    args_extra, unknown = parser.parse_known_args()

    if not os.path.exists(args_extra.ckpt):
        print("Checkpoint not found: {}".format(args_extra.ckpt))
        print("Run train.py first, or specify with --ckpt <path>")
        exit(1)

    if args_extra.record:
        args = gymutil.parse_arguments(description="Quadcopter Video Recording", headless=False)
        if args.sim_device == "cpu":
            args.sim_device = "cuda:0"
        env = QuadcopterEnv(args)
        device = args.sim_device if "cuda" in str(args.sim_device) else "cpu"
        ppo_config = {
            "lr": 1e-4, "gamma": 0.99, "lam": 0.95,
            "clip_param": 0.2, "value_coef": 0.5, "entropy_coef": 0.02,
            "max_grad_norm": 1.0, "num_epochs": 5, "batch_size": 2048,
        }
        agent = PPO(OBS_DIM, ACT_DIM, ppo_config, device)
        agent.load(args_extra.ckpt)
        os.makedirs(os.path.dirname(args_extra.output) or ".", exist_ok=True)
        record_hover(env, agent, args_extra.output, duration=10.0)
    else:
        evaluate(args_extra.ckpt, headless=args_extra.headless)
