# 智能体详述

本文档介绍代码包内智能体实现的关键模块，包括观测处理、特征处理、奖励处理、算法结构与评估配置。

## 观测处理

环境返回的 observation 包含智能体局部观测，可在 `PolicyObservationProcess` 中处理。  
代码包默认直接透传 `self.default_observation()` 返回的 301 维张量，选手可在此基础上进行归一化、聚合或拼接额外特征等特征工程。

```python
class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        # TODO (track 地形)：可按需从 env.goal_positions / env.goal_yaw
        # 或 env.scene.sensors["nav_scanner"] 构造特征并拼接到 obs。
        return obs
```

### 观测数据布局

| 区间 | 维度 | 含义 | 来源 |
| --- | --- | --- | --- |
| `[0:3]` | 3 | `base_ang_vel`，机体角速度，`scale=0.25` | Isaac Lab mdp |
| `[3:6]` | 3 | `projected_gravity`，重力方向投影 | Isaac Lab mdp |
| `[6:9]` | 3 | `velocity_commands`，速度命令 `(vx, vy, wz)` | command manager |
| `[9:21]` | 12 | `joint_pos_rel`，关节相对默认位置 | robot data |
| `[21:33]` | 12 | `joint_vel_rel`，关节速度，`scale=0.05` | robot data |
| `[33:45]` | 12 | `last_action`，上一帧动作 | action manager |
| `[45:301]` | 256 | `height_scan`，`16x16` 前方高度扫描 | height_scanner |

Track 地形扩展：环境额外提供 `env.goal_positions`（出口位置）、`env.goal_yaw`（出口朝向）和 `env.scene.sensors["nav_scanner"]`（前瞻遮挡扫描）。

## 特征处理

当前代码包中的 `PolicyObservationProcess` 直接使用 `default_observation()` 返回的原始观测，尚未做额外特征工程：

```python
def process(self):
    obs = self.default_observation()
    return obs
```

可以在此基础上扩展导航/地形相关特征。

## 奖励处理

代码包在 `reward_process.py` 提供了两个示例奖励函数：

| 奖励名称 | 作用 | 类型 |
| --- | --- | --- |
| `_reward_reach_goal` | 机器人与出口距离小于 `0.6m` 时给予 `1.0` 的稀疏奖励 | sparse |
| `_reward_forward_velocity` | 鼓励机器人沿本体 `x` 轴方向前进的 dense 奖励 | dense |

其余通用 locomotion 奖励继承自 `RewardProcessBase`，在 TOML 配置中激活 `[rewards.<name>]` 即可，无需重复实现。若需训练导航策略，请在 `reward_process.py` 中自行新增奖励函数。

## 算法介绍

代码包提供基于 PPO（Proximal Policy Optimization）的 Actor-Critic 实现。

| 算法 | 文件路径 | 说明 |
| --- | --- | --- |
| PPO + ActorCritic | `model/actor_critic.py` | 扁平 MLP Actor-Critic 网络，Gaussian 动作分布 |

### ActorCritic 网络结构

采用独立的 Actor 与 Critic 两个 MLP 网络：

| 组件 | 输入维度 | 隐藏层 | 输出维度 | 特殊设计 |
| --- | --- | --- | --- | --- |
| Actor | 301（policy obs） | `[512, 256, 128]` | 12（关节动作） | ELU 激活 |
| Critic | 316（critic obs） | `[512, 256, 128]` | 1（状态价值） | LayerNorm + ELU 激活 |

### 训练超参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `lr` | `3e-4` | 学习率 |
| `num_learning_epochs` | `5` | 每次更新的 epoch 数 |
| `num_mini_batches` | `4` | mini-batch 数量 |
| `num_steps_per_env` | `48` | 每个环境采集步数 |
| `model_save_interval` | `500` | 模型保存间隔（迭代次数） |

## 算法监控信息

算法上报了 reward 等指标，可通过腾讯开悟平台/客户端监控查看。

### 算法指标组

| 面板名称 | 指标 Key | 类型 | 说明 |
| --- | --- | --- | --- |
| 总损失 | `total_loss` | line | PPO 总 loss（`policy_loss + value_loss - entropy_loss`） |
| 价值损失 | `value_loss` | line | Critic 网络价值函数损失 |
| 策略损失 | `policy_loss` | line | Actor 网络策略梯度损失 |
| 熵损失 | `entropy_loss` | line | 策略熵，衡量探索程度 |

### 奖励指标组

| 面板名称 | 指标 Key | 类型 | 说明 |
| --- | --- | --- | --- |
| 线速度跟踪奖励 | `reward_track_lin_vel_xy` | line | XY 平面速度命令跟踪奖励 |

说明：以上为 `monitor_builder.py` 中定义的自定义指标。其余通用 reward 指标（如速度跟踪、姿态、步态等）由 `tools/conf/monitor_default.yaml` 和 `tools/conf/monitor_default_track.yaml` 展示。

## 模型保存限制策略

为避免保存频率过高，平台有安全限制。当前配置如下：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `model_save_interval` | `500` | 每 500 次训练迭代保存一次模型 |

注意：可在 `conf.py` 中调整 `model_save_interval`，但平台侧可能有额外最小间隔限制。

## 模型评估模式

可在腾讯开悟平台创建模型评估任务，并配置环境参数。示例如下：

```toml
[env]
# Number of parallel environments | 并行环境数量
num_envs = 16
# Episode length in seconds | 单回合时长(秒)
episode_length_s = 20.0
# Task type: standard / track | 任务类型
task = "standard"

[terrain]
# Terrain mode: standard / track | 地形模式
mode = "standard"
# Difficulty levels to evaluate (0=easiest, 9=hardest)
# standard 模式下: num_rows = max(level)+1 自动推导, num_cols = len(sub_terrains)
# track    模式下: num_parallel_tracks 默认 10; 仅当 max(level)+1 > 10 时才自动扩展
level = [0, 1, 3, 4]
# Force-disable curriculum for deterministic evaluation | 强制关闭课程学习
curriculum = false

# Standard mode: deterministic sub-terrain list (no proportion)
# Each terrain occupies one column; env_i → (level[(i//n_sub)%n_lv], sub[i%n_sub])
[terrain.standard]
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv"]

# Track mode: linear course config (used only when mode = "track")
[terrain.track]
# Number of sub-terrains chained along the track | 赛道串联的子地形数量
track_length = 5
# Sub-terrain sequence along the track (length must equal track_length, no duplicates)
# Allowed: pyramid_slope | pyramid_slope_inv | pyramid_stairs | pyramid_stairs_inv | open_entry_maze
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
# Parallel tracks along Y-axis | Y 轴并行赛道数
num_parallel_tracks = 10

# Velocity command limits | 速度指令上限
[commands]

[commands.limit]
# Linear velocity X range (m/s) | X 方向线速度范围
lin_vel_x = [0.0, 1.0]
# Linear velocity Y range (m/s) | Y 方向线速度范围
lin_vel_y = [-0.0, 0.0]
# Angular velocity Z range (rad/s) | Z 轴角速度范围
ang_vel_z = [0.0, 0.0]
```

### 注意事项

- 评估时部分训练配置（如域随机化、观测噪声）默认关闭，以保证一致性与可复现性。
- `task` 必须与 `mode` 保持一致；否则评估结果无意义（评分字段与地形语义错位）。
- `num_envs <= 16` 时会产出 `mp4` 可视化视频；`num_envs > 16` 时不产出视频。
