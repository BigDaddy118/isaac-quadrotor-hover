# 接续任务 — 四旋翼悬停 PPO 训练

## 目标
在 AutoDL 服务器上完成 Isaac Gym 四旋翼悬停 PPO 训练。

## 服务器信息
- JupyterLab: `https://a1004961-b812-7e6e03c3.nmb1.seetacloud.com:8443/jupyter/lab`
- Token: 在 AutoDL 终端执行 `jupyter server list` 获取
- SSH: `ssh -p 36961 root@connect.nmb1.seetacloud.com`
- 项目路径: `/root/isaac-quadrotor-hover/`

## 已完成的步骤
- [x] 项目文件已上传到服务器 `/root/isaac-quadrotor-hover/`
- [x] tar.gz 已解压，所有代码文件就位
- [ ] 环境验证（GPU、Isaac Gym、PyTorch）
- [ ] 冒烟测试 `python test_gym.py`
- [ ] 训练 `python train.py --sim_device cuda:0 --graphics_device_id -1`
- [ ] 评估

## 下一步（按顺序执行）
1. 验证 GPU 和 Isaac Gym 环境
2. 跑 test_gym.py 确认仿真正常
3. 开始 PPO 训练
4. 训练完成后跑 evaluate.py 评估

## 注意事项
- `import isaacgym` 必须在 `import torch` 之前
- Isaac Gym 路径: `/root/isaacgym/`（已安装？需验证）
- IsaacGymEnvs 路径: `/root/IsaacGymEnvs/`
- 服务器镜像: AutoDL, PyTorch 2.0.0, Python 3.8, CUDA 11.8

## 关键命令
```bash
cd /root/isaac-quadrotor-hover

# 验证环境
nvidia-smi
python -c "import isaacgym; print('OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 冒烟测试
python test_gym.py

# 训练
python train.py --sim_device cuda:0 --graphics_device_id -1
```
