import math
import numpy as np
import isaacgym
import torch
from isaacgym import gymapi, gymtorch, gymutil

# 超参数
NUM_ENVS = 64
ACT_DIM = 4
TARGET_POS = np.array([0.0, 0.0, 2.0])
MAX_THRUST = 5.0

class QuadcopterEnv:
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()
        
        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1/60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_threads = 2
        
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id,
                                       gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")
        
        # 地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # 资产
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = False
        asset_opts.disable_gravity = False
        self.body_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.02, asset_opts)
        self.arm_asset = self.gym.create_capsule(self.sim, 0.01, 0.15, asset_opts)
        self.rotor_asset = self.gym.create_box(self.sim, 0.12, 0.01, 0.005, asset_opts)
        
        # 创建环境并记录旋翼的全局刚体索引
        self.envs = []
        self.num_bodies_per_env = 9  # 1机体 + 4机臂 + 4旋翼
        self.total_bodies = NUM_ENVS * self.num_bodies_per_env
        self.rotor_indices = torch.zeros((NUM_ENVS, 4), dtype=torch.int32, device=args.sim_device)
        num_per_row = int(math.sqrt(NUM_ENVS))
        lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        upper = gymapi.Vec3(2.0, 2.0, 5.0)
        
        for i in range(NUM_ENVS):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            # 机体
            body_pose = gymapi.Transform()
            body_pose.p = gymapi.Vec3(0, 0, 1.0)
            self.gym.create_actor(env, self.body_asset, body_pose, f"body_{i}", i, 0)
            # 机臂
            arm_offsets = [(0.15,0,0.02), (-0.15,0,0.02), (0,0.15,0.02), (0,-0.15,0.02)]
            for off in arm_offsets:
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*off)
                self.gym.create_actor(env, self.arm_asset, pose, f"arm_{i}", i, 0)
            # 旋翼
            rotor_offsets = [(0.2,0,0.02), (-0.2,0,0.02), (0,0.2,0.02), (0,-0.2,0.02)]
            for idx, off in enumerate(rotor_offsets):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*off)
                handle = self.gym.create_actor(env, self.rotor_asset, pose, f"rotor_{i}_{idx}", i, 0)
                # 存储全局刚体索引
                self.rotor_indices[i, idx] = self.gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM)
        
        self.gym.prepare_sim(self.sim)
        
        # 获取根状态张量（机体索引为0）
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_state).view(NUM_ENVS, 9, 13)
        self.quad_pos = self.root_states[:, 0, 0:3]
        self.quad_rot = self.root_states[:, 0, 3:7]
        self.quad_vel = self.root_states[:, 0, 7:10]
        
        self.target_pos = TARGET_POS
    
    def apply_actions(self, actions):
        """使用批量力张量接口：构造覆盖所有刚体的力张量，只在旋翼位置填力"""
        actions = actions.clamp(0.0, 1.0)
        thrusts = actions * MAX_THRUST          # [NUM_ENVS, 4]
        
        # 创建全局力张量 (total_bodies, 3) 全零
        forces_global = torch.zeros((self.total_bodies, 3), device=self.args.sim_device)
        # 旋翼索引展平 -> (NUM_ENVS*4,)
        indices = self.rotor_indices.view(-1).long()
        # 将推力填入对应位置的z分量
        forces_global[indices, 2] = thrusts.view(-1)
        
        # 调用批量施力，torqueTensor 置空，空间默认世界系
        self.gym.apply_rigid_body_force_tensors(self.sim,
                                                gymtorch.unwrap_tensor(forces_global),
                                                None,
                                                gymapi.CoordinateSpace.ENV_SPACE)
    
    def reset_some(self, env_ids):
        for env_id in env_ids:
            init_pos = self.target_pos.copy() + np.random.uniform(-0.5, 0.5, 3)
            init_pos[2] = max(0.5, init_pos[2])
            state = self.root_states[env_id, 0].clone()
            state[0:3] = torch.tensor(init_pos, device=self.args.sim_device)
            state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.args.sim_device)
            state[7:13] = 0.0
            self.root_states[env_id, 0] = state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states.reshape(-1, 13)))
        self.gym.sync_frame_time(self.sim)
    
    def step(self, actions):
        self.apply_actions(actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        pos_err = torch.norm(self.quad_pos - torch.tensor(self.target_pos, device=self.args.sim_device), dim=-1)
        reward = -pos_err
        
        z = self.quad_pos[:, 2]
        out = (z < 0.2) | (z > 5.0) | (torch.abs(self.quad_pos[:, 0]) > 3.0) | (torch.abs(self.quad_pos[:, 1]) > 3.0)
        env_ids = out.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_some(env_ids.tolist())
        
        return reward.mean().item(), self.quad_pos[:, 2].mean().item()

if __name__ == "__main__":
    args = gymutil.parse_arguments(description="Quadcopter Hover Final")
    if args.sim_device == 'cpu':
        args.sim_device = 'cuda:0'
    env = QuadcopterEnv(args)
    
    for step in range(1000):
        actions = torch.rand((NUM_ENVS, ACT_DIM), device=args.sim_device) * 0.5 + 0.5
        mean_reward, mean_height = env.step(actions)
        if step % 100 == 0:
            print(f"Step {step:3d} | Mean Reward: {mean_reward:.3f} | Mean Height: {mean_height:.2f}")