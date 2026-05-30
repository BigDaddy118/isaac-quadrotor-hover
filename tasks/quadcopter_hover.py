import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask

class QuadcopterHover(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = cfg["env"]["maxEpisodeLength"]
        self.target_pos = torch.tensor([0.0, 0.0, 2.0], device=rl_device)

        self.max_rpm = 15000.0
        self.num_rotors = 4

        self.num_actions = self.num_rotors
        self.num_obs = 15
        self.num_states = 0

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.quad_pos = self.root_states[:, 0:3]
        self.quad_ori = self.root_states[:, 3:7]
        self.quad_vel = self.root_states[:, 7:10]
        self.quad_angvel = self.root_states[:, 10:13]

        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 100.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            gymapi.SIM_PHYSX, sim_params
        )

        # ===== 用代码构建四旋翼资产 =====
        quad_asset = self._build_quadcopter_asset()

        # 创建环境
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.5)
        upper = gymapi.Vec3(spacing, spacing, 3.5)
        num_envs_per_row = int(np.sqrt(self.num_envs))

        self.envs = []
        self.quad_handles = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_envs_per_row)
            quad_handle = self.gym.create_actor(env, quad_asset, gymapi.Transform(),
                                                f"quad{i}", i, 0)
            self.envs.append(env)
            self.quad_handles.append(quad_handle)

        # 获取旋翼关节索引（我们创建时命名为 rotor_0, rotor_1, rotor_2, rotor_3）
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.quad_handles[0])
        self.rotor_indices = [dof_names.index(f"rotor_{i}") for i in range(self.num_rotors)]
        self.num_dofs = len(dof_names)

    def _build_quadcopter_asset(self):
        """纯代码构建四旋翼：中心基座 + 四臂 + 四旋翼"""
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL

        asset = gymapi.Asset()

        # --- 中心基座 (盒体) ---
        body_dims = gymapi.Vec3(0.05, 0.05, 0.02)  # 5cm x 5cm x 2cm
        body_opts = gymapi.BodyProperties()
        body_opts.mass = 0.03  # 30g

        # --- 机臂 (圆柱体) ---
        arm_radius = 0.005
        arm_length = 0.08

        # --- 旋翼 (球体，模拟旋转质量) ---
        rotor_radius = 0.02

        # ========= 构建树形结构 =========
        # root: 中心基座
        root_link = gymapi.Link()
        root_link.name = "base"
        root_link.inertial.mass = 0.03
        root_link.inertial.ixx = 1e-5
        root_link.inertial.iyy = 1e-5
        root_link.inertial.izz = 2e-5

        # 盒体碰撞和视觉
        box_shape = gymapi.Shape()
        box_shape.geom = gymapi.Geom.Box
        box_shape.size = gymapi.Vec3(0.05, 0.05, 0.02)
        box_shape.collision = 0
        root_link.shapes.append(box_shape)

        asset.root_link = root_link

        # 四个旋翼臂 + 旋翼
        arm_directions = [
            (1.0, 0.0),   # 前
            (-1.0, 0.0),  # 后
            (0.0, 1.0),   # 左
            (0.0, -1.0),  # 右
        ]

        for i, (dx, dy) in enumerate(arm_directions):
            # 机臂
            arm_link = gymapi.Link()
            arm_link.name = f"arm_{i}"
            arm_link.inertial.mass = 0.005
            arm_link.inertial.ixx = 1e-6
            arm_link.inertial.iyy = 1e-6
            arm_link.inertial.izz = 1e-6

            # 圆柱形状
            cylinder_shape = gymapi.Shape()
            cylinder_shape.geom = gymapi.Geom.Cylinder
            cylinder_shape.size = gymapi.Vec3(arm_radius, arm_length)
            cylinder_shape.collision = 0
            arm_link.shapes.append(cylinder_shape)

            # 机臂关节（固定）
            arm_joint = gymapi.Joint()
            arm_joint.name = f"arm_joint_{i}"
            arm_joint.type = gymapi.JointType.Fixed
            arm_joint.parent = "base"
            arm_joint.child = f"arm_{i}"
            # 机臂沿水平方向延伸
            arm_joint.origin = gymapi.Transform()
            arm_joint.origin.p = gymapi.Vec3(dx * 0.06, dy * 0.06, 0.0)

            asset.links.append(arm_link)
            asset.joints.append(arm_joint)

            # 旋翼（球体 + 旋转关节）
            rotor_link = gymapi.Link()
            rotor_link.name = f"rotor_{i}"
            rotor_link.inertial.mass = 0.003
            rotor_link.inertial.i = gymapi.Vec3(1e-6, 1e-6, 2e-6)

            sphere_shape = gymapi.Shape()
            sphere_shape.geom = gymapi.Geom.Sphere
            sphere_shape.size = gymapi.Vec3(rotor_radius)
            sphere_shape.collision = 0
            rotor_link.shapes.append(sphere_shape)

            # 旋转关节（连续转动）
            rotor_joint = gymapi.Joint()
            rotor_joint.name = f"rotor_{i}"
            rotor_joint.type = gymapi.JointType.Revolute
            rotor_joint.parent = f"arm_{i}"
            rotor_joint.child = f"rotor_{i}"
            rotor_joint.origin = gymapi.Transform()
            rotor_joint.origin.p = gymapi.Vec3(dx * arm_length, dy * arm_length, 0.0)
            rotor_joint.axis = gymapi.Vec3(0.0, 0.0, 1.0)  # 绕 Z 轴旋转

            asset.links.append(rotor_link)
            asset.joints.append(rotor_joint)

        return asset

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

        rel_pos = self.quad_pos - self.target_pos.unsqueeze(0)
        vel = self.quad_vel
        roll, pitch, yaw = self._quat_to_euler(self.quad_ori)
        angvel = self.quad_angvel
        target_tile = self.target_pos.unsqueeze(0).repeat(self.num_envs, 1)

        self.obs_buf = torch.cat([
            rel_pos, vel,
            roll.unsqueeze(-1), pitch.unsqueeze(-1), yaw.unsqueeze(-1),
            angvel,
            target_tile
        ], dim=-1)

    def _quat_to_euler(self, quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def compute_reward(self):
        pos_error = torch.norm(self.quad_pos - self.target_pos.unsqueeze(0), dim=-1)
        reward_pos = -1.0 * pos_error

        roll, pitch, _ = self._quat_to_euler(self.quad_ori)
        attitude_error = torch.abs(roll) + torch.abs(pitch)
        reward_att = -0.1 * attitude_error

        vel_norm = torch.norm(self.quad_vel, dim=-1)
        reward_vel = -0.05 * vel_norm

        energy_penalty = -0.01 * torch.sum(self.actions ** 2, dim=-1)

        hover_bonus = 1.0 * (pos_error < 0.1).float()

        self.rew_buf[:] = reward_pos + reward_att + reward_vel + energy_penalty + hover_bonus

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        num_resets = len(env_ids)
        new_pos = self.target_pos.unsqueeze(0).repeat(num_resets, 1) + \
                  torch.randn(num_resets, 3, device=self.device) * 0.2
        new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0.5, 3.0)

        rand_axis = torch.randn(num_resets, 3, device=self.device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=-1, keepdim=True)
        rand_angle = torch.rand(num_resets, 1, device=self.device) * 0.2
        new_quat = self._axis_angle_to_quat(rand_axis, rand_angle)

        self.root_states[env_ids, 0:3] = new_pos
        self.root_states[env_ids, 3:7] = new_quat
        self.root_states[env_ids, 7:10] = 0.0
        self.root_states[env_ids, 10:13] = 0.0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(dof_states)
        dof_vel = dof_states[:, 1].view(self.num_envs, self.num_dofs)
        dof_vel[env_ids, :] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_states))

        self.progress_buf[env_ids] = 0

    def _axis_angle_to_quat(self, axis, angle):
        sin_half = torch.sin(angle / 2)
        cos_half = torch.cos(angle / 2)
        return torch.cat([cos_half, axis * sin_half], dim=-1)

    def pre_physics_step(self, actions):
        self.actions = actions.clone()
        rpm = (actions * 0.5 + 0.5) * self.max_rpm

        dof_velocities = torch.zeros(self.num_envs * self.num_dofs, device=self.device)
        dof_velocities = dof_velocities.view(self.num_envs, self.num_dofs)
        dof_velocities[:, self.rotor_indices] = rpm
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_velocities))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        pos = self.quad_pos
        out_of_bounds = (pos[:, 2] < 0.1) | (pos[:, 2] > 5.0) | \
                        (torch.abs(pos[:, 0]) > 3.0) | (torch.abs(pos[:, 1]) > 3.0)
        too_tilted = torch.abs(self._quat_to_euler(self.quad_ori)[0]) > 1.2
        timeout = self.progress_buf >= self.max_episode_length

        reset_ids = (out_of_bounds | too_tilted | timeout).nonzero(as_tuple=False).squeeze(-1)

        if len(reset_ids) > 0:
            self.reset_idx(reset_ids)

        self.compute_observations()
        self.compute_reward()