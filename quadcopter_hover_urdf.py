import math
import os
import numpy as np
import isaacgym
import torch
from isaacgym import gymapi, gymtorch, gymutil

NUM_ENVS = 1024
ACT_DIM = 4
OBS_DIM = 13
TARGET_POS = np.array([0.0, 0.0, 2.0])
MAX_THRUST = 0.2

REWARD_POS_W = 2.0
REWARD_ATT_W = 0.5
REWARD_VEL_W = 0.3
REWARD_ANGVEL_W = 0.15
REWARD_ACT_W = 0.02
REWARD_HOVER_BONUS = 10.0
REWARD_SURVIVAL = 0.5
HOVER_THRESHOLD = 0.15

MAX_HEIGHT = 5.0
MIN_HEIGHT = 0.2
MAX_HORIZONTAL = 3.0

DISTURB_FORCE = 20.0
DISTURB_DURATION = 0.5

DR_ENABLE = True
DR_MASS_RANGE = 0.2
DR_THRUST_RANGE = 0.15


class QuadcopterEnv:
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 100.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 2

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params
        )
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "crazyflie.urdf")
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = False
        asset_opts.disable_gravity = False
        asset_opts.armature = 0.001
        asset_opts.collapse_fixed_joints = True
        asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self.quad_asset = self.gym.load_asset(self.sim, "", urdf_path, asset_opts)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self.quad_asset)
        self.num_dofs = self.gym.get_asset_dof_count(self.quad_asset)
        print(f"URDF loaded: {self.num_bodies} bodies, {self.num_dofs} DOFs")

        body_names = self.gym.get_asset_rigid_body_names(self.quad_asset)
        self.rotor_body_indices = [body_names.index(f"rotor_{i}") for i in range(4)]
        print(f"Rotor body indices: {self.rotor_body_indices}")

        num_per_row = int(math.sqrt(NUM_ENVS))
        lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        upper = gymapi.Vec3(2.0, 2.0, 5.0)

        self.envs = []
        self.actor_handles = []
        for i in range(NUM_ENVS):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 1.0)
            handle = self.gym.create_actor(env, self.quad_asset, pose, f"quad_{i}", i, 0)
            self.actor_handles.append(handle)

        self.gym.prepare_sim(self.sim)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_state).view(NUM_ENVS, 13)
        self.quad_pos = self.root_states[:, 0:3]
        self.quad_rot = self.root_states[:, 3:7]
        self.quad_vel = self.root_states[:, 7:10]
        self.quad_angvel = self.root_states[:, 10:13]

        self.target = torch.tensor(TARGET_POS, device=args.sim_device).float()

        # ---- domain randomization setup ----
        baseline_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
        self.baseline_masses = [p.mass for p in baseline_props]
        self.thrust_scales = torch.ones(NUM_ENVS, device=args.sim_device)

        self.total_bodies = NUM_ENVS * self.num_bodies  # for force tensors

    def reset(self):
        env_ids = list(range(NUM_ENVS))
        self._reset_some(env_ids)
        return self._compute_obs()

    def step(self, actions):
        self._apply_actions(actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        obs = self._compute_obs()
        rewards = self._compute_rewards(actions)

        z = self.quad_pos[:, 2]
        out_x = torch.abs(self.quad_pos[:, 0]) > MAX_HORIZONTAL
        out_y = torch.abs(self.quad_pos[:, 1]) > MAX_HORIZONTAL
        out_z = (z < MIN_HEIGHT) | (z > MAX_HEIGHT)
        dones = out_z | out_x | out_y

        if dones.any():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self._reset_some(reset_ids.tolist())
            self.gym.refresh_actor_root_state_tensor(self.sim)
            obs = self._compute_obs()

        info = {
            "mean_reward": rewards.mean().item(),
            "mean_height": z.mean().item(),
            "num_resets": dones.sum().item(),
        }
        return obs, rewards, dones, info

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _apply_actions(self, actions):
        actions = actions.clamp(0.0, 1.0)
        thrust = actions * MAX_THRUST * self.thrust_scales.unsqueeze(1)

        forces = torch.zeros(NUM_ENVS * self.num_bodies, 3, device=self.args.sim_device)
        for i, body_idx in enumerate(self.rotor_body_indices):
            base = torch.arange(NUM_ENVS, device=self.args.sim_device) * self.num_bodies + body_idx
            forces[base, 2] = thrust[:, i]
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.GLOBAL_SPACE
        )

    def _compute_obs(self):
        rel_pos = self.quad_pos - self.target
        return torch.cat([rel_pos, self.quad_vel, self.quad_rot, self.quad_angvel], dim=-1)

    def _compute_rewards(self, actions):
        pos_err = torch.norm(self.quad_pos - self.target, dim=-1)
        z = self.quad_pos[:, 2]

        qx = self.quad_rot[:, 0]
        qy = self.quad_rot[:, 1]
        body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = 1.0 - body_z_world_z.clamp(-1.0, 1.0)

        vel_mag = torch.norm(self.quad_vel, dim=-1)
        angvel_mag = torch.norm(self.quad_angvel, dim=-1)
        act_mag = torch.norm(actions, dim=-1)

        hover = (pos_err < HOVER_THRESHOLD).float() * REWARD_HOVER_BONUS
        alive = (z > MIN_HEIGHT) & (z < MAX_HEIGHT)

        rewards = (
            -REWARD_POS_W * pos_err
            - REWARD_ATT_W * tilt
            - REWARD_VEL_W * vel_mag
            - REWARD_ANGVEL_W * angvel_mag
            - REWARD_ACT_W * act_mag
            + hover
            + alive.float() * REWARD_SURVIVAL
        )
        return rewards

    def _reset_some(self, env_ids):
        n = len(env_ids)
        init_pos = self.target.cpu().numpy() + np.random.uniform(-0.5, 0.5, (n, 3))
        init_pos[:, 2] = np.maximum(0.5, init_pos[:, 2])

        self.root_states[env_ids, 0:3] = torch.tensor(init_pos, device=self.args.sim_device, dtype=torch.float)
        self.root_states[env_ids, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.args.sim_device)
        self.root_states[env_ids, 7:10] = 0.0
        self.root_states[env_ids, 10:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        if DR_ENABLE:
            env_ids_t = torch.tensor(env_ids, device=self.args.sim_device, dtype=torch.long)
            mass_scale = 1.0 + (torch.rand(n, device=self.args.sim_device) * 2 - 1) * DR_MASS_RANGE
            for idx, env_id in enumerate(env_ids):
                props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])
                scale = mass_scale[idx].item()
                for j, p in enumerate(props):
                    p.mass = self.baseline_masses[j] * scale
                self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id], props)

            thrust_scale = 1.0 + (torch.rand(n, device=self.args.sim_device) * 2 - 1) * DR_THRUST_RANGE
            self.thrust_scales[env_ids_t] = thrust_scale

        self.gym.sync_frame_time(self.sim)

    def apply_disturbance(self, env_id, force_vec):
        force_tensor = torch.zeros((NUM_ENVS * self.num_bodies, 3), device=self.args.sim_device)
        body_idx = env_id * self.num_bodies
        force_tensor[body_idx] = torch.tensor(force_vec, device=self.args.sim_device)
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.GLOBAL_SPACE
        )

    def get_random_disturbance(self, magnitude=20.0):
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0])


if __name__ == "__main__":
    args = gymutil.parse_arguments(description="Quadcopter Hover Environment (URDF force-control)")
    if args.sim_device == "cpu":
        args.sim_device = "cuda:0"
    env = QuadcopterEnv(args)
    obs = env.reset()
    for step_idx in range(1000):
        actions = torch.rand((NUM_ENVS, ACT_DIM), device=args.sim_device)
        obs, rewards, dones, info = env.step(actions)
        if step_idx % 100 == 0:
            msg = "Step {:3d} | Reward: {:.3f} | Height: {:.2f} | Resets: {}"
            print(msg.format(step_idx, info["mean_reward"], info["mean_height"], info["num_resets"]))
