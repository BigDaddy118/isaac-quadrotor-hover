```markdown
# isaac-quadrotor-hover

基于 **NVIDIA Isaac Gym Preview 4** 的四旋翼悬停强化学习训练环境。  
✅ GPU 物理仿真环境已搭建，64 并行环境随机推力验证通过。  
⏭️ **下一步：接入 PPO 算法训练稳定悬停策略，并进行抗扰动测试。**

---

## 仓库结构

```
.
├── README.md               # 项目交接文档（本文件）
├── quadcopter_hover.py     # 四旋翼悬停环境（可直接运行）
├── test_gym.py             # Isaac Gym 安装验证脚本
├── requirements.txt        # Python 依赖清单
└── assets/                 # 预留：未来存放 URDF/MJCF 模型
```

---

## 项目目标
在 NVIDIA Isaac Gym 高性能仿真环境中，使用 PPO 算法训练一个能稳定悬停的四旋翼无人机控制策略，并通过外部扰动测试验证其鲁棒性。  
**最终交付**：训练好的策略网络 + 悬停视频/GIF + 实验报告。

---

## 当前进度（截至 2026-05-17）

### ✅ 已完成
- **服务器环境搭建**  
  - 平台：AutoDL，镜像 `PyTorch 2.0.0 + Python 3.8 + CUDA 11.8`，GPU：RTX 4090  
  - Isaac Gym Preview 4 手动安装（`isaacgym/python` 下 `pip install -e .`）  
  - 修复 NumPy 1.24+ 的 `np.float` 错误 → 降级至 `numpy==1.23.5`  
  - 启用 GPU PhysX（`physx.use_gpu = True`, `use_gpu_pipeline = True`）  
  - 成功运行 `1080_balls_of_solitude.py --sim_device cuda:0 --graphics_device_id -1`，输出 `Physics Device: cuda:0`

- **四旋翼仿真环境 `quadcopter_hover.py`**  
  - 用程序化几何体（盒子、胶囊、薄盒子）搭建简易四旋翼模型  
  - 64 个并行环境同时运行  
  - 动作空间：4 个旋翼推力比例（0~1），经 `MAX_THRUST=5N` 缩放后施加  
  - 观测空间（13 维）：相对目标位置(3) + 速度(3) + 姿态四元数(4) + 角速度(3)  
  - 施力方式：通过 `apply_rigid_body_force_tensors` 传入全局力张量（总刚体数×3）  
  - 奖励函数（简化版）：仅负位置误差 `-||pos - target||`  
  - 终止/重置：高度<0.2m 或 >5m，或水平距离>3m → 重置到目标点附近随机位置  
  - 随机动作测试通过：平均奖励约 -1.0~-0.7，平均高度约 1.0~1.8m

### ❌ 尚未完成（新对话的起点）
- **PPO 训练**  
  - 环境已具备 `step(actions)` 接口，但未接入任何 RL 算法  
  - 推荐使用服务器已有的 `rl_games` 库（位于 `/root/IsaacGymEnvs/`）或自写 PPO  
  - 需要编写训练脚本、配置超参数、定义网络结构  
- **奖励函数完善**  
  - 需加入姿态惩罚、速度惩罚、能耗惩罚、悬停成功奖励（权重已预留）  
- **模型保存与评估**、**外部扰动测试**

---

## 服务器环境速查（开机后直接可用）

**硬件/系统**：AutoDL 实例，Ubuntu 20.04，RTX 4090，CUDA 11.8，Python 3.8  
**所有文件位于** `/root/`  
**Isaac Gym 安装路径** `/root/isaacgym/`  
**`isaacgymenvs`（含 rl_games）**：`/root/IsaacGymEnvs/`  
**Conda 环境**：base，已装所有依赖，若需激活：
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

### 开机验证命令
```bash
# 1. 检查 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 2. 测试 Isaac Gym 基本功能
python /root/test_gym.py --sim_device cuda:0 --graphics_device_id -1

# 3. 运行四旋翼随机动作环境（确认一切正常）
python /root/quadcopter_hover.py --sim_device cuda:0 --graphics_device_id -1
```

### 依赖清单（`requirements.txt`）
```
numpy==1.23.5
torch==2.0.0+cu118
torchvision==0.15.1+cu118
scipy==1.10.1
pyyaml==6.0
pillow==9.4.0
imageio==2.35.1
ninja==1.13.0
```

---

## 核心代码说明（`quadcopter_hover.py`）

### 类结构
- **`QuadcopterEnv`**：环境主类
  - `__init__(args)`：创建仿真、加载资产、构建 64 个环境、获取根状态张量、存储旋翼刚体索引
  - `apply_actions(actions)`：将 `(64,4)` 的动作转为推力，构造全局力张量，调用 `apply_rigid_body_force_tensors`
  - `step(actions)`：施力 → `simulate` → `fetch_results` → `refresh` → 计算奖励、判断重置 → 返回平均奖励和平均高度
  - `reset_some(env_ids)`：将指定环境恢复到随机初始位置（目标点 ±0.5m），姿态重置为单位四元数

### 关键参数（硬编码，后续可改为 yaml 配置）
```python
NUM_ENVS = 64          # 并行环境数
ACT_DIM = 4            # 动作维度（4 个旋翼）
MAX_THRUST = 5.0       # 单个旋翼最大推力 (N)
TARGET_POS = [0, 0, 2] # 悬停目标位置
```
奖励权重（当前未全部启用，写在代码注释中）：
```python
REWARD_POS_W = 1.0      # 位置误差权重
REWARD_ATT_W = 0.5      # 姿态倾斜权重（建议添加）
REWARD_VEL_W = 0.1      # 速度惩罚
REWARD_ACT_W = 0.05     # 动作惩罚
REWARD_HOVER_BONUS = 10.0  # 悬停成功奖励
```

### 已踩坑与解决方案（代码中已修正）
| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ImportError: PyTorch was imported before isaacgym` | 导入顺序错误 | 必须在 `import torch` 之前 `import isaacgym` |
| `AttributeError: module 'numpy' has no attribute 'float'` | NumPy 1.24+ 移除了旧别名 | `pip install numpy==1.23.5` |
| `AttributeError: 'Gym' object has no attribute 'create_cylinder'` | Preview 4 无此 API | 改用 `create_box` 创建薄方片作为旋翼 |
| `apply_rigid_body_force_tensors` 参数错误 | 需要全局力张量，不传索引 | 构造 `(total_bodies, 3)` 的全零张量，在旋翼索引处填入力 |
| 物理设备始终为 CPU | 未启用 `physx.use_gpu` | 在 `SimParams` 中设置 `physx.use_gpu = True` 并配合 `use_gpu_pipeline` |

---

## ⏭️ 明天继续的标准流程（给新 AI 的指令）

1. **理解项目状态**：阅读本 README 和 `quadcopter_hover.py` 源码。
2. **恢复环境**：若服务器实例未释放，直接开机运行上述测试命令确认一切正常。
3. **任务一：完善奖励函数**  
   在 `step()` 方法中加入姿态惩罚、速度惩罚、动作惩罚和悬停奖励，可参考代码中预留的权重参数。
4. **任务二：接入 PPO 训练**  
   - 使用服务器已有的 `rl_games` 库（路径 `/root/IsaacGymEnvs/rl_games`）或自写 PPO。  
   - 将 `QuadcopterEnv` 封装成一个 `VecEnv` 风格的接口（需要 `reset` 返回全部环境观测，`step` 返回 obs, reward, done, info）。  
   - 配置参数：环境数 4096（充分利用 4090），horizon 16，mini-batch 256，学习率 3e-4，训练 500~1000 epochs。  
   - 启动训练并开启 TensorBoard 监控。
5. **任务三：保存与评估**  
   - 定期保存 checkpoint。  
   - 训练结束后测试悬停平均误差（RMSE）和抗扰动恢复能力。  
   - 录制视频或 GIF。
6. **任务四（可选）**：改用真实无人机 URDF 模型（如 Crazyflie），放入 `assets/` 目录。

---

## 📁 服务器上的关键路径速查
- 主代码：`/root/quadcopter_hover.py`  
- 测试脚本：`/root/test_gym.py`  
- Isaac Gym 安装：`/root/isaacgym/`  
- `isaacgymenvs`（含 rl_games）：`/root/IsaacGymEnvs/`  
- Conda 环境：`/root/miniconda3/`，base 环境已装好所有依赖  

---

## 🌱 未来改进方向
- 接入 `isaacgymenvs` 的 `VecTask` 基类，以支持官方 PPO 训练器  
- 引入域随机化（质量、惯量、推力系数）为 Sim2Real 做准备  
- 增加多任务（轨迹跟踪、避障）  

---

> **给新 AI 的提示**：此仓库是一个未完成的项目，请先阅读本 README 和 Python 源码，然后按照“下一步工作计划”继续开发。所有环境已就绪，核心代码可运行，当前主要任务是**完善奖励函数并接入 PPO 训练**。
```
