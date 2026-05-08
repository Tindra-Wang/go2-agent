# 数据协议

为了方便调用原始数据和特征数据，本文档整理了环境交互、观测、动作、奖励及终止信号的数据协议。

> 注意：基于 Isaac Lab 仿真环境，所有观测与动作数据均以 `torch.Tensor` 在 GPU 上传递，而非传统字典协议。

## 环境交互接口

### 环境重置（Reset）

```python
reset_data = env.reset(usr_conf)

# 成功时
obs, critic_obs = reset_data
# obs: torch.Tensor, shape=(num_envs, obs_dim)
# critic_obs: torch.Tensor, shape=(num_envs, critic_obs_dim)

# 失败时
reset_data = None
```

#### `reset()` 返回结构

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `obs` | `reset()[0]` | `torch.Tensor` | `(num_envs, obs_dim)` | policy 观测 |
| `critic_obs` | `reset()[1]` | `torch.Tensor` | `(num_envs, critic_obs_dim)` | critic 观测 |

### 环境步进（Step）

```python
data = env.step(actions)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data
dones = terminated | truncated
```

#### `step()` 返回结构

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `frame_no` | `step()[0]` | `int` | `>= 1` | 当前交互帧号 |
| `obs` | `step()[1]` | `torch.Tensor` | `(num_envs, obs_dim)` | 下一步 policy 观测 |
| `rewards` | `step()[2]` | `torch.Tensor` | `(num_envs,)` | 当前步总 reward |
| `terminated` | `step()[3]` | `torch.Tensor[bool]` | `(num_envs,)` | 真实终止 |
| `truncated` | `step()[4]` | `torch.Tensor[bool]` | `(num_envs,)` | 超时截断 |
| `infos` | `step()[5][0]` | `dict` | `-` | Isaac Lab extras |
| `privileged_obs` | `step()[5][1]` | `torch.Tensor` | `(num_envs, critic_obs_dim)` | critic 观测 |

## 观测数据协议（Observation）

### 策略观测（Policy Observation）

`obs = [proprio(45) | height_scan(256)]`，共 301 维。

#### `proprio`（45 维）字段布局

| 区间 | 维度 | 含义 | 来源 |
| --- | --- | --- | --- |
| `[0:3]` | 3 | `base_ang_vel`，机体角速度，`scale=0.25` | Isaac Lab mdp |
| `[3:6]` | 3 | `projected_gravity`，重力方向投影 | Isaac Lab mdp |
| `[6:9]` | 3 | `velocity_commands (vx, vy, wz)` | command manager |
| `[9:21]` | 12 | `joint_pos_rel`，关节相对默认位置 | robot data |
| `[21:33]` | 12 | `joint_vel_rel`，关节速度，`scale=0.05` | robot data |
| `[33:45]` | 12 | `last_action`，上一帧动作 | action manager |

#### `height_scan`（256 维）

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `height_scan` | `obs[:, 45:301]` | `torch.Tensor` | clip `[-5, 5]`，`scale=2.5` | `16x16` 前方高度扫描，256 条射线 |

### Track 地形附加原始张量（需自行消费）

代码包中的 `default_observation()` 返回 301 维 `obs`，不内置目标点或通行性特征。  
当 `terrain.mode = "track"` 时，环境额外暴露以下原始张量，需在 `policy_observation_process.py::process()` 中自行读取并拼接到 `obs`，同时同步修改 `critic_observation_process.py`、模型输入维度和 TOML 中的 `obs` 维度。

| 字段 | 类型 | Shape | 含义 |
| --- | --- | --- | --- |
| `env.goal_positions` | `torch.Tensor` | `(num_envs, 3)` | 目标点（迷宫出口）在世界坐标系下的 `(x, y, z)` |
| `env.goal_yaw` | `torch.Tensor` | `(num_envs,)` | 目标点在世界坐标系下的朝向（rad） |
| `env.scene.sensors["nav_scanner"]` | `RayCaster` | `-` | 前瞻遮挡扫描传感器，范围比 `height_scanner` 更大，适合避障/转向判断 |

这些都是原始值；如需构造新特征，请自行设计与实现。

### 评论家观测（Critic Observation）

`critic_obs = [critic_proprio(60) | height_scan(256)]`，共 316 维。

#### `critic_proprio`（60 维）字段布局

| 区间 | 维度 | 含义 |
| --- | --- | --- |
| `[0:3]` | 3 | `base_lin_vel`，机体线速度 |
| `[3:6]` | 3 | `base_ang_vel`，机体角速度 |
| `[6:9]` | 3 | `projected_gravity` |
| `[9:12]` | 3 | `velocity_commands` |
| `[12:24]` | 12 | `joint_pos_rel` |
| `[24:36]` | 12 | `joint_vel_rel` |
| `[36:48]` | 12 | `joint_effort`，关节力矩 |
| `[48:60]` | 12 | `last_action` |

补充：`critic_proprio`（60 维）相比 `proprio`（45 维）额外包含 `base_lin_vel` 和 `joint_effort` 等特权数据，仅供 Critic 训练使用，体现不对称 Actor-Critic 设计。

## 动作数据协议（Action）

```python
actions = torch.Tensor  # shape: (num_envs, 12)
```

| 字段名 | 类型 | Shape | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `actions` | `float32` | `(num_envs, 12)` | `[-1.0, 1.0]` | 12 个关节控制动作 |

### 关节动作维度

Go2 为 12 自由度四足机器人，4 条腿 × 3 个关节：

| 维度 | 关节组 | 说明 |
| --- | --- | --- |
| `0~2` | 前左腿 | hip / thigh / calf |
| `3~5` | 前右腿 | hip / thigh / calf |
| `6~8` | 后左腿 | hip / thigh / calf |
| `9~11` | 后右腿 | hip / thigh / calf |

注意：动作值为归一化偏移量，经 `action_scale`（默认 `0.25`）缩放后加到默认关节角度上，作为 PD 控制器目标角度。

## 奖励数据协议（Reward）

| 字段名 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `rewards` | `float32` | `(num_envs,)` | 当前步总 reward |

### 代码包默认激活的奖励项

以下 reward 已在 `train_env_conf_standard_locomotion.toml` 的 `[rewards.*]` 段激活：

| 奖励名称 | 作用 | 类型 |
| --- | --- | --- |
| `track_lin_vel_xy` | 奖励 XY 速度跟踪 | 正奖励 |
| `track_ang_vel_z` | 奖励 yaw 角速度跟踪 | 正奖励 |
| `lin_vel_z` | 惩罚 Z 方向跳动 | 负奖励 |
| `ang_vel_xy` | 惩罚 roll/pitch 角速度 | 负奖励 |
| `joint_acc` | 惩罚关节加速度 | 负奖励 |
| `joint_torques` | 惩罚关节扭矩 | 负奖励 |
| `dof_pos_limits` | 惩罚接近关节极限 | 负奖励 |
| `action_rate` | 惩罚动作帧间变化率 | 负奖励 |
| `undesired_contacts` | 惩罚非脚掌接触 | 负奖励 |
| `flat_orientation` | 惩罚非直立姿态 | 负奖励 |
| `reach_goal` | 到达目标点奖励（需 `env.goal_positions` 有效，默认 Track 地形生效） | 正奖励 |

其它框架内置 reward（如 `energy`、`feet_stumble`、`feet_height_body`、`termination` 等）未默认激活；如需启用，需在 `reward_process.py` 实现并在 TOML 中新增 `[rewards.<name>]` 段。

## 终止与超时协议

| 字段 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `terminated` | `bool` | `(num_envs,)` | 真实终止 |
| `truncated` | `bool` | `(num_envs,)` | 超时截断 |

| 条件 | `terminated` | `truncated` | 说明 |
| --- | --- | --- | --- |
| episode 最大时长 | False | True | 超时截断 |
| 姿态异常/摔倒 | True | False | 真实失败 |
| Track 到达出口 | True | False | 导航成功 |

算法侧应使用 `dones = terminated | truncated` 判断环境是否结束。

## 访问示例

### 解析 `step()` 返回

```python
def parse_env_step(data):
    if data is None:
        raise RuntimeError("env.step failed")
    frame_no, obs, rewards, terminated, truncated, extra = data
    infos, privileged_obs = extra
    critic_obs = privileged_obs if privileged_obs is not None else obs
    dones = torch.logical_or(terminated, truncated)
    return {
        "frame_no": frame_no,
        "obs": obs,
        "critic_obs": critic_obs,
        "rewards": rewards,
        "dones": dones,
        "terminated": terminated,
        "truncated": truncated,
        "infos": infos,
    }
```

### 解析 policy observation

```python
def parse_policy_obs(obs: torch.Tensor):
    return {
        "base_ang_vel": obs[:, 0:3],
        "projected_gravity": obs[:, 3:6],
        "velocity_commands": obs[:, 6:9],
        "joint_pos_rel": obs[:, 9:21],
        "joint_vel_rel": obs[:, 21:33],
        "last_action": obs[:, 33:45],
        "height_scan": obs[:, 45:301],
    }
```

## 常见问题

### Q1：Critic 观测和 Policy 观测的区别？

Critic 观测包含 `critic_proprio`（60 维），额外含 `base_lin_vel` 和 `joint_effort` 等特权信息，仅训练时可用，体现不对称 Actor-Critic 设计。

### Q2：返回格式与传统环境的区别？

`reset()` 返回 `(obs, critic_obs)` 而非字典。  
`step()` 返回 `(frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs))`，需用 `dones = terminated | truncated` 合并结束信号。