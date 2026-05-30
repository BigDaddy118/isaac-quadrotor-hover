import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1/60
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

# 第1个参数：计算设备ID(0=GPU0)，第2个参数：图形设备ID(-1=无图形)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 创建盒子
asset_options = gymapi.AssetOptions()
asset = gym.create_box(sim, 0.1, 0.1, 0.1, asset_options)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.0)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)
gym.create_actor(env, asset, pose, "box", 0, 1)

# 模拟 100 步，不调用任何图形相关函数
for i in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

print("Simulation completed successfully! 100 steps done.")