# Quadcopter Hover with Isaac Gym

基于 NVIDIA Isaac Gym 的四旋翼悬停强化学习环境。

## 环境要求
- Ubuntu 20.04, NVIDIA GPU (4090 测试通过)
- Isaac Gym Preview 4
- Python 3.8, PyTorch 2.0.0+cu118
- numpy==1.23.5

## 快速开始
1. 安装 Isaac Gym 并配置 GPU 物理
2. 安装依赖：`pip install -r requirements.txt`
3. 运行测试：`python test_gym.py --sim_device cuda:0 --graphics_device_id -1`
4. 运行四旋翼环境：`python quadcopter_hover.py --sim_device cuda:0 --graphics_device_id -1`

## 训练
（待补充 PPO 训练脚本）
