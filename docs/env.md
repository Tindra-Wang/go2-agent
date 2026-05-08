# 环境详述

本文档说明训练环境配置、交互数据、动作与监控指标。

## 环境配置

智能体与环境交互时，首先调用 `env.reset`，并传入 `usr_conf`（通常来自 `train_env_conf_standard_locomotion.toml`）进行定制化配置。

```python
# usr_conf 为用户传入的环境配置
reset_data = env.reset(usr_conf)
obs, critic_obs = reset_data
```

`train_env_conf_standard_locomotion.toml` 为标准模式训练示例配置。

### 主要配置项

| 配置项 | 类型 | 合法范围 | 说明 |
| --- | --- | --- | --- |
| `env.num_envs` | int | `[1, 4096]` | 并行环境数量 |
| `env.episode_length_s` | float | `> 0` | 最大 episode 时长（秒） |
| `terrain.mode` | string | `"standard"` \| `"track"` | 地形模式 |
| `terrain.num_rows` | int | `[1, 10]` | 难度级别数（沿 X 轴课程档位） |
| `terrain.num_cols` | int | `[1, 40]` | 同一难度下并行地块数（沿 Y 轴） |
| `terrain.difficulty_range` | list[float] | `[0.0, 1.0]` | 难度范围 |
| `terrain.curriculum` | bool | `true/false` | 是否启用地形课程学习 |
| `terrain.max_init_terrain_level` | int | `[0, 9]` | 机器人初始放置最大难度档 |
| `terrain.standard.*.proportion` | float | `[0, 1]` | 各子地形比例（总和须为 1.0） |
| `domain_rand.enable_domain_rand` | bool | `true/false` | 域随机化总开关 |
| `domain_rand.randomize_friction` | bool | `true/false` | 是否随机化地面摩擦系数 |
| `domain_rand.friction_range` | list[float] | `>= 0` | 摩擦系数采样范围 |
| `domain_rand.push_robots` | bool | `true/false` | 是否周期性施加外部推力 |
| `domain_rand.push_interval_s` | float | `> 0` | 推力间隔（秒） |
| `domain_rand.max_push_vel_xy` | float | `>= 0` | XY 平面最大推力速度（m/s） |
| `noise.add_noise` | bool | `true/false` | 是否在观测中加入噪声 |
| `init_state.pos` | list[float] | `z: [0.30, 0.60]` | 机器人初始位置 `[x, y, z]`（m） |
| `commands.resampling_time` | list[float] | `> 0` | 速度命令重采样区间 `[min, max]`（秒） |
| `commands.limit.lin_vel_x` | list[float] | `-` | X 方向线速度采样上限 |
| `commands.limit.lin_vel_y` | list[float] | `-` | Y 方向线速度采样上限 |
| `commands.limit.ang_vel_z` | list[float] | `-` | 偏航角速度采样上限 |
| `commands.ranges.lin_vel_x` | list[float] | `-` | X 方向线速度初始采样范围 |
| `commands.ranges.lin_vel_y` | list[float] | `-` | Y 方向线速度初始采样范围 |
| `commands.ranges.ang_vel_yaw` | list[float] | `-` | 偏航角速度初始采样范围 |
| `rewards.*.weight` | float | `-` | 各奖励项权重 |
| `rewards.*.params.*` | - | - | 各奖励项参数 |

### 补充说明

- `train_env_conf_standard_locomotion.toml` 仅训练时生效，请按上表描述配置。
- 配置错误会导致训练任务失败，可查看 `env` 模块错误日志排查。
- 评估任务配置需在腾讯开悟平台单独设置（见“智能体模型评估模式”）。

### 默认配置示例（节选）

```toml
[env]
num_envs = 2048
episode_length_s = 25

[env_conf]
seed = 0

[terrain]
mode = "standard"
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 5

[terrain.standard.pyramid_slope]
proportion = 0.15

[terrain.standard.pyramid_slope_inv]
proportion = 0.2

[terrain.standard.pyramid_stairs]
proportion = 0.25

[terrain.standard.pyramid_stairs_inv]
proportion = 0.3

[terrain.standard.maze]
proportion = 0.1

[domain_rand]
enable_domain_rand = true
randomize_friction = true
friction_range = [0.3, 1.5]
push_robots = true
push_interval_s = 15
max_push_vel_xy = 0.5

[noise]
add_noise = true

[init_state]
pos = [0.0, 0.0, 0.35]

[commands]
resampling_time = [10.0, 10.0]

[commands.limit]
lin_vel_x = [-2.0, 2.0]
lin_vel_y = [-1.5, 1.5]
ang_vel_z = [-1.5, 1.5]

[commands.ranges]
lin_vel_x = [0.0, 0.5]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-1.0, 1.0]
```

## 地形模式说明

### Standard 模式

合法子地形类型：

- `pyramid_slope`
- `pyramid_slope_inv`
- `pyramid_stairs`
- `pyramid_stairs_inv`
- `maze`

### 切换到 Track 模式

导航阶段训练时，需将 `terrain` 段替换为 Track 配置，并参考 `LocomotionConfig` 设计新的训练阶段参数：

```toml
[terrain.track]
track_length = 5
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
```

Track 模式合法子地形类型：

- `pyramid_slope`
- `pyramid_slope_inv`
- `pyramid_stairs`
- `pyramid_stairs_inv`
- `open_entry_maze`

注意：`open_entry_maze` 必须放在赛道最后，否则训练会报错。

## 环境信息

### `step` 返回字段

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `frame_no` | int | 当前交互帧号 |
| `obs` | torch.Tensor | 策略观测 `(num_envs, obs_dim)` |
| `rewards` | torch.Tensor | 当前步总 reward `(num_envs,)` |
| `terminated` | torch.Tensor[bool] | 真实终止（摔倒、目标达成） |
| `truncated` | torch.Tensor[bool] | 超时截断 |
| `infos` | dict | Isaac Lab / RSL-RL extras |
| `privileged_obs` | torch.Tensor \| None | critic 观测 `(num_envs, critic_obs_dim)` |

### 奖励信息（reward）

`reward` 是 Isaac Lab 根据 TOML 中 `[rewards.*]` 实时计算的每步奖励总和，`shape = (num_envs,)`。  
它是强化学习训练信号，不等于平台评分系统“总分”；总分由默认监控信息上报。

### 观测信息（observation）

策略观测 `obs` 传给 Actor 网络：

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

| 字段名 | 区间 | 类型 | 说明 |
| --- | --- | --- | --- |
| `height_scan` | `obs[:, 45:301]` | `torch.Tensor` | `16x16` 前方高度扫描，clip `[-5, 5]`，`scale=2.5` |

`privileged_obs`（316 维）用于 Critic 网络，在 `proprio` 基础上额外包含 `base_lin_vel`（机体线速度）与 `joint_effort`（关节力矩）等特权信息，仅训练时使用，体现不对称 Actor-Critic 设计。

补充：Track 地形下环境额外提供 `env.goal_positions`（目标点世界坐标）、`env.goal_yaw`（目标点朝向）和 `env.scene.sensors["nav_scanner"]`（前瞻遮挡扫描），可用于构造导航特征拼接到 `obs`。

### 额外信息（infos）

`infos` 是一个 `dict`，包含仿真环境附加信息。

## 动作空间

Go2 为 12 自由度四足机器人，动作空间是 12 维连续动作：

| 字段名 | 类型 | Shape | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `actions` | `float32` | `(num_envs, 12)` | `[-1.0, 1.0]` | 12 个关节控制动作 |

动作值为归一化偏移量，经 `action_scale`（默认 `0.25`）缩放后加到默认关节角度，作为 PD 控制目标角。关节维度对应：

| 维度 | 关节组 | 说明 |
| --- | --- | --- |
| `0~2` | 前左腿 | hip / thigh / calf |
| `3~5` | 前右腿 | hip / thigh / calf |
| `6~8` | 后左腿 | hip / thigh / calf |
| `9~11` | 后右腿 | hip / thigh / calf |

注意：具体关节顺序以 Isaac Lab Unitree Go2 资产配置为准。

## 时间信息

步（step）与帧（frame）一一对应。每一步中，智能体输出一个 12 维动作，环境更新状态后返回新观测、奖励及终止信号。`env.step()` 返回的 `frame_no` 即当前交互帧号。

## 环境监控信息

监控面板中 `env` 模块表示环境指标数据。系统每 1 分钟采集最新结束的 episode 数据并求均值展示。

### Standard 模式

#### 全局环境指标

| 面板名称 | 指标名称 | 说明 |
| --- | --- | --- |
| 已结束任务数 | `completed_count` | 正常完成 episode 数 |
| 已结束任务数 | `abnormal_count` | 异常终止 episode 数 |
| 已结束任务数 | `timeout_count` | 超时终止 episode 数 |
| 得分 | `total_score` | 单局总分均值 |
| 得分 | `distance_score` | 单局前进距离分均值 |
| 得分 | `time_score` | 单局时间分均值 |
| 得分 | `energy_score` | 单局能耗分均值 |
| 得分 | `pose_score` | 单局姿态分均值 |
| 步数 | `step` | 单局平均步数 |

#### 按地形类型分组指标（`[terrain_type]`）

| 面板名称 | 指标命名规律 | 说明 |
| --- | --- | --- |
| 地形-完成数 | `completed_count_[terrain_type]` | 该地形正常完成 episode 数 |
| 地形-失败数 | `abnormal_count_[terrain_type]` | 该地形异常终止 episode 数 |
| 地形-超时数 | `timeout_count_[terrain_type]` | 该地形超时终止 episode 数 |
| 地形-总分 | `total_score_[terrain_type]` | 该地形总分均值 |
| 地形-距离分数 | `distance_score_[terrain_type]` | 该地形前进距离分均值 |
| 地形-时间分数 | `time_score_[terrain_type]` | 该地形时间分均值 |
| 地形-能耗分数 | `energy_score_[terrain_type]` | 该地形能耗分均值 |
| 地形-姿态分数 | `pose_score_[terrain_type]` | 该地形姿态分均值 |
| 地形-步数 | `step_[terrain_type]` | 该地形平均步数 |

`[terrain_type]` 需替换为具体地形名（如 `pyramid_slope`、`maze`），不同地形对应不同 Tab。

### Track 模式

#### 全局环境指标

| 面板名称 | 指标名称 | 说明 |
| --- | --- | --- |
| 已结束任务数 | `completed_count` | 正常完成 episode 数 |
| 已结束任务数 | `abnormal_count` | 异常终止 episode 数 |
| 已结束任务数 | `timeout_count` | 超时终止 episode 数 |
| 得分 | `total_score` | 单局总分均值 |
| 得分 | `energy_score` | 单局能耗分均值 |
| 得分 | `pose_score` | 单局姿态分均值 |
| 得分 | `time_score` | 单局时间分均值（底层 key 为 `kaiwu_step_score`） |
| 步数 | `step_avg` | 单局平均步数 |
| Reward 均值 | `reward_mean` | 单局平均奖励 |
| Reward 均值 | `reward_std` | 单局奖励标准差 |

#### Track 赛道-难度档指标

| 面板名称 | 指标命名规律 | 说明 |
| --- | --- | --- |
| 赛道-完成数 | `completed_count_track_l{0~9}` | 各难度档正常完成 episode 数 |
| 赛道-失败数 | `abnormal_count_track_l{0~9}` | 各难度档异常终止 episode 数 |
| 赛道-超时数 | `timeout_count_track_l{0~9}` | 各难度档超时终止 episode 数 |
| 赛道-总分 | `total_score_track_l{0~9}` | 各难度档总分均值 |
| 赛道-能耗分数 | `energy_score_track_l{0~9}` | 各难度档能耗分均值 |
| 赛道-姿态分数 | `pose_score_track_l{0~9}` | 各难度档姿态分均值 |
| 赛道-时间分数 | `time_score_track_l{0~9}` | 各难度档时间分均值 |

### Reward 指标

代码包默认激活的 reward 对应监控面板如下：

| 面板名称 | 指标名称 | 说明 |
| --- | --- | --- |
| 线速度跟踪奖励 | `reward_track_lin_vel_xy` | XY 速度命令跟踪奖励均值 |
| 角速度跟踪奖励 | `reward_track_ang_vel_z` | yaw 角速度命令跟踪奖励均值 |
| 安全奖励 | `reward_undesired_contacts` | 非脚掌接触惩罚均值 |
| 安全奖励 | `reward_dof_pos_limits` | 关节位置极限惩罚均值 |
| 平坦姿态奖励 | `reward_flat_orientation` | 非直立姿态惩罚均值 |
| 到达目标 | `reward_reach_goal` | 到达目标点奖励均值（仅 Track 且 `env.goal_positions` 维护时生效） |

如在 `reward_process.py` 新增 reward，需要在 `agent_ppo/conf/monitor_builder.py` 参考 Group 2 示例添加对应监控面板。