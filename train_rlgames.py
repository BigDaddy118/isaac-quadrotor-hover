"""
Train quadcopter hover using rl_games PPO with VecTask environment.
Usage: python train_rlgames.py --sim_device cuda:0 --graphics_device_id -1
"""

import os
import sys
import yaml

from isaacgym import gymapi, gymutil
from rl_games.common import env_configurations, experiment, tr_helpers, vecenv
from rl_games.algos_torch import model_builder, network_builder
from rl_games.algos_torch import a2c_continuous
from rl_games.torch_runner import Runner


def create_rlgpu_env(**kwargs):
    from tasks.quadcopter_hover import QuadcopterHover
    return QuadcopterHover(**kwargs)


def main():
    args = gymutil.parse_arguments(description="rl_games PPO Trainer")
    if args.sim_device == "cpu":
        args.sim_device = "cuda:0"

    rl_device = args.sim_device if "cuda" in str(args.sim_device) else "cpu"

    # load configs
    task_cfg_path = os.path.join(os.path.dirname(__file__), "cfg", "task", "QuadcopterHover.yaml")
    train_cfg_path = os.path.join(os.path.dirname(__file__), "cfg", "train", "QuadcopterHoverPPO.yaml")

    with open(task_cfg_path, "r") as f:
        task_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    # register env
    env_configurations.register("rlgpu", {
        "vecenv_type": "RLGPU",
        "env_creator": lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    # merge task cfg into train cfg
    train_cfg["task"] = task_cfg

    # create runner and run
    runner = Runner(algo_observer=None)
    runner.load(train_cfg)
    runner.reset()
    runner.run({
        "train": True,
        "play": False,
        "checkpoint": "",
        "sigma": None,
    })


if __name__ == "__main__":
    main()
