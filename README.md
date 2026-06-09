# Go2-Agent 四足机器人自主导航运控

本仓库是腾讯开悟四足机器人自主导航运控赛题的 Go2 智能体代码包，面向 Unitree Go2 在 Isaac Lab 仿真环境中的复杂地形运动控制与赛道导航训练。

代码包含官方 PPO 基线与 `agent_diy` 自定义实现。当前 DIY 版本围绕稳定运动、Track 导航和安全约束做了扩展，包括 NP3O 风格多代价约束、历史观测编码、height scanner / nav scanner 编码、目标点观测与层级导航实验。

---

# 数据协议

## 环境交互接口

### reset

```python
reset_data = env.reset(usr_conf)
obs, critic_obs = reset_data
```

| 字段名 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `obs` | `torch.Tensor` | `(num_envs, obs_dim)` | Actor / policy 观测 |
| `critic_obs` | `torch.Tensor` | `(num_envs, critic_obs_dim)` | Critic 特权观测 |

### step

```python
data = env.step(actions)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data
dones = terminated | truncated
```

| 字段名 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `frame_no` | int | `>= 1` | 当前交互帧号 |
| `obs` | `torch.Tensor` | `(num_envs, obs_dim)` | 下一步 policy 观测 |
| `rewards` | `torch.Tensor` | `(num_envs,)` | 当前步总 reward |
| `terminated` | `torch.Tensor[bool]` | `(num_envs,)` | 真实终止，如摔倒或到达目标 |
| `truncated` | `torch.Tensor[bool]` | `(num_envs,)` | 超时截断 |
| `infos` | dict | - | Isaac Lab / RSL-RL 附加信息 |
| `privileged_obs` | `torch.Tensor` | `(num_envs, critic_obs_dim)` | Critic 观测 |

## 观测空间

默认 policy 观测为：

```text
obs = [proprio(45) | height_scan(256)]
```

共 301 维。DIY Track 导航阶段会按配置追加 terrain context、maze hint、goal observation 与 raw nav scanner。

| 区间 | 维度 | 含义 |
| --- | --- | --- |
| `[0:3]` | 3 | 机体角速度 `base_ang_vel` |
| `[3:6]` | 3 | 重力方向投影 `projected_gravity` |
| `[6:9]` | 3 | 速度命令 `(vx, vy, wz)` |
| `[9:21]` | 12 | 关节相对默认位置 |
| `[21:33]` | 12 | 关节速度 |
| `[33:45]` | 12 | 上一帧动作 |
| `[45:301]` | 256 | `16x16` 前方高度扫描 |

### Track 扩展观测

| 字段 | 维度 | 说明 |
| --- | --- | --- |
| `terrain_context` | 2 | `[terrain_type_norm, segment_progress]`，描述赛道段与段内进度 |
| `maze_nav_hint` | 2 | `[best_passage_angle_norm, best_passage_score]`，由 nav scanner 估计迷宫通道方向 |
| `goal_obs` | 4 | 目标点在机器人坐标系下的相对位置、距离与朝向 |
| `raw_nav_scan` | 32 | 前瞻遮挡扫描序列，供 1D CNN 编码 |

当前 `agent_diy` 的 Stage3 导航观测布局为：

```text
proprio(45) + height_scan(256) + terrain(2) + maze_hint(2) + goal(4) + nav_scan(32) = 341
```

若开启 HIM/GRU 历史编码，训练工作流会额外拼接：

```text
history_len * proprio_dim
```

默认 `history_len=10`、`proprio_dim=45`。

## Critic 观测

默认 Critic 观测为：

```text
critic_obs = [critic_proprio(60) | height_scan(256)]
```

共 316 维。其中 `critic_proprio` 相比 policy 观测额外包含机体线速度与关节力矩等训练期特权信息。

## 动作空间

Go2 为 12 自由度四足机器人，动作空间是 12 维连续关节动作。

| 字段名 | 类型 | Shape | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `actions` | `float32` | `(num_envs, 12)` | `[-1.0, 1.0]` | 12 个关节控制动作 |

动作值经 `action_scale` 缩放后加到默认关节角度，作为 PD 控制目标角。

| 维度 | 关节组 | 说明 |
| --- | --- | --- |
| `0~2` | 前左腿 | hip / thigh / calf |
| `3~5` | 前右腿 | hip / thigh / calf |
| `6~8` | 后左腿 | hip / thigh / calf |
| `9~11` | 后右腿 | hip / thigh / calf |

> 注意：Isaac Lab 内部实际 joint name 顺序以资产配置为准。DIY reward 中涉及关节索引的部分已尽量按 joint name 动态解析，避免硬编码顺序错误。

---

# 环境介绍

## 任务目标

训练 Go2 四足机器人在复杂地形上保持稳定运动，并在赛道模式中完成从起点到终点的自主导航。

任务包含两种模式：

| 模式 | 说明 |
| --- | --- |
| `standard` | 在坡面、楼梯、迷宫等地形上行走，评分关注前进距离、时间、能耗与姿态稳定性 |
| `track` | 多个子地形串联为单向赛道，机器人需要穿越赛道并到达迷宫出口 |

## 地形类型

| 地形类型 | 适用模式 | 说明 |
| --- | --- | --- |
| `pyramid_slope` | standard / track | 金字塔坡面，向上 |
| `pyramid_slope_inv` | standard / track | 金字塔坡面，向下 |
| `pyramid_stairs` | standard / track | 金字塔楼梯，向上 |
| `pyramid_stairs_inv` | standard / track | 金字塔楼梯，向下 |
| `maze` | standard | 随机生成障碍迷宫 |
| `open_entry_maze` | track | Track 终点迷宫，入口和出口开放 |

Track 模式下 `open_entry_maze` 必须放在赛道最后。

## 计分规则

### Standard 模式

```text
总分 = 0.4 * 前进距离分 + 0.2 * 时间分 + 0.2 * 能耗分 + 0.2 * 姿态分
```

| 子项 | 权重 | 含义 |
| --- | --- | --- |
| 前进距离分 | 0.4 | 按出生点到当前位置的 2D 欧氏距离归一化 |
| 时间分 | 0.2 | 走穿地形后，用时越短分越高 |
| 能耗分 | 0.2 | 平均关节机械功率越低分越高 |
| 姿态分 | 0.2 | 平均 roll / pitch 偏移越小分越高 |

### Track 模式

```text
总分 = 完成系数 * (0.4 * 时间分 + 0.4 * 姿态分 + 0.2 * 能耗分)
```

| 子项 | 权重 | 含义 |
| --- | --- | --- |
| 完成系数 | - | 批次内完成赛道的机器人比例 |
| 时间分 | 0.4 | 到达终点越快分越高 |
| 姿态分 | 0.4 | 姿态越稳定分越高 |
| 能耗分 | 0.2 | 关节机械功率越低分越高 |

## 终止条件

| 条件 | `terminated` | `truncated` | 说明 |
| --- | --- | --- | --- |
| 姿态异常 / 摔倒 | True | False | 主体或非期望部位接触地面 |
| Track 到达出口 | True | False | 导航成功 |
| 达到最大步数 | False | True | episode 超时 |

---

# 项目结构

| 路径 | 说明 |
| --- | --- |
| `agent_ppo/` | 官方 PPO 基线实现 |
| `agent_diy/` | 自定义智能体、算法、特征、奖励与模型 |
| `agent_diy/conf/` | 训练环境 TOML 与阶段配置 |
| `agent_diy/feature/` | policy / critic 观测处理与 reward 扩展 |
| `agent_diy/model/` | Actor-Critic、HIM/GRU、CNN 编码器与层级导航模型 |
| `agent_diy/workflow/` | 训练采样、PPO 更新、监控上报与层级导航 workflow |
| `docs/` | 赛题介绍、环境协议、数据协议与智能体说明 |
| `conf/` | 开悟平台应用与算法配置 |
| `tools/` | 通用工具与基础环境处理代码 |

---

# 智能体实现

## 基线 PPO

`agent_ppo` 提供官方基线 Actor-Critic：

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| Agent | `agent_ppo/agent.py` | 平台 Agent 入口 |
| PPO | `agent_ppo/algorithm/algorithm_ppo.py` | PPO 更新逻辑 |
| 模型 | `agent_ppo/model/actor_critic.py` | MLP Actor-Critic |
| 观测 | `agent_ppo/feature/policy_observation_process.py` | 默认 policy 观测处理 |
| 奖励 | `agent_ppo/feature/reward_process.py` | 默认 reward 扩展 |

## DIY PPO / NP3O 风格扩展

`agent_diy` 是主要开发目录，当前实现包含：

| 能力 | 说明 |
| --- | --- |
| 多代价约束 | 使用 `dof_pos_limits`、`torque_limit`、`dof_vel_limits` 三类 cost，并支持 penalty 调度 |
| HIM/GRU 历史编码 | 将历史 proprio 序列编码为 latent，改善部分可观测性 |
| HeightScan 2D CNN | 将 `16x16` height scan 编码后送入 Actor |
| NavScan 1D CNN | Track 阶段消费 raw nav scanner 序列 |
| Goal Observation | 将目标点相对位置、距离、朝向拼接到观测 |
| 地形上下文 | 提供赛道段编号与段内进度 |
| 迷宫导航提示 | 基于 nav scanner 估计更可能通向出口的开放方向 |
| Partial Load | 支持模型维度变化时部分加载 checkpoint |

## 层级导航实验

`agent_diy/agent_hier_nav.py` 与 `agent_diy/model/hier_nav_model.py` 提供层级式导航实验：

```text
nav_head(obs) -> 3D 速度命令 -> 冻结 locomotion actor -> 12D 关节动作 -> env.step()
```

该模式让 PPO 训练较高层的速度命令，底层运动 actor 负责把速度命令转换为 Go2 关节控制。

---

# 环境配置

训练配置通过 `agent_diy/conf/conf.py` 中的 `Config.CURRENT` 选择阶段。

| Stage | `task_type` | 配置文件 | 说明 |
| --- | --- | --- | --- |
| `LocomotionConfig` | `standard` | `agent_diy/conf/train_env_conf_standard_locomotion.toml` | 标准地形运动控制 |
| `NavConfig` | `track` | `agent_diy/conf/train_env_conf_track_nav.toml` | Track 端到端导航 |
| 层级导航 | `track` | `agent_diy/conf/train_env_conf_track_hier_nav.toml` | nav head + frozen loco 实验 |

常用配置项：

| 配置项 | 说明 |
| --- | --- |
| `env.num_envs` | 并行环境数量 |
| `env.episode_length_s` | episode 最大时长 |
| `terrain.mode` | `standard` 或 `track` |
| `terrain.curriculum` | 是否启用课程学习 |
| `terrain.standard.*.proportion` | Standard 子地形比例 |
| `terrain.track.track_length` | Track 赛道段数量 |
| `terrain.track.sub_terrains` | Track 子地形序列 |
| `domain_rand.*` | 摩擦、推力等域随机化 |
| `noise.add_noise` | 是否添加观测噪声 |
| `commands.*` | 速度命令采样范围 |
| `rewards.*` | reward 权重与参数 |

---

# 快速开始

## 1. 选择算法入口

本地训练测试入口为 `train_test.py`：

```python
algorithm_name = "ppo"  # 可选 "ppo" 或 "diy"
```

若要运行自定义实现，将其改为：

```python
algorithm_name = "diy"
```

## 2. 选择训练阶段

在 `agent_diy/conf/conf.py` 中修改：

```python
Config.CURRENT = NavConfig
```

常见选择：

| 目标 | 推荐设置 |
| --- | --- |
| 训练基础运动能力 | `Config.CURRENT = LocomotionConfig` |
| 训练 Track 端到端导航 | `Config.CURRENT = NavConfig` |
| 尝试层级导航 | 使用 `HierNavAgent` 与 `train_hier_nav_workflow.py` 配套入口 |

## 3. 调整 TOML

按目标模式编辑对应 TOML：

```text
agent_diy/conf/train_env_conf_standard_locomotion.toml
agent_diy/conf/train_env_conf_track_nav.toml
agent_diy/conf/train_env_conf_track_hier_nav.toml
```

## 4. 启动训练测试

```bash
python train_test.py
```

实际训练、评估与提交仍以腾讯开悟平台运行环境为准。

---

# 算法监控信息

## PPO 指标

| 指标 Key | 说明 |
| --- | --- |
| `total_loss` | PPO 总损失 |
| `policy_loss` | 策略损失 |
| `value_loss` | 价值函数损失 |
| `entropy_loss` | 策略熵 |
| `reward_mean` | rollout reward 均值 |
| `reward_std` | rollout reward 标准差 |

## 约束 / Cost 指标

| 指标 Key | 说明 |
| --- | --- |
| `cost_mean` | 多代价 cost 均值 |
| `violation_mean` | cost violation 均值 |
| `cost_source_id` | cost 来源标识 |

## 环境评分指标

Standard 模式关注：

| 指标 Key | 说明 |
| --- | --- |
| `total_score` | 单局总分 |
| `distance_score` | 前进距离分 |
| `time_score` | 时间分 |
| `energy_score` | 能耗分 |
| `pose_score` | 姿态分 |

Track 模式关注：

| 指标 Key | 说明 |
| --- | --- |
| `total_score` | 单局总分 |
| `time_score` | 到达目标耗时评分 |
| `energy_score` | 能耗分 |
| `pose_score` | 姿态分 |
| `completed_count_track_l{0~9}` | 各难度档完成数 |

---

# 开发指引

## 特征工程

| 方向 | 文件 | 说明 |
| --- | --- | --- |
| Policy 观测 | `agent_diy/feature/policy_observation_process.py` | 拼接目标点、地形上下文、导航扫描等特征 |
| Critic 观测 | `agent_diy/feature/critic_observation_process.py` | 同步补充训练期可用特权信息 |
| 历史信息 | `agent_diy/workflow/train_workflow.py` | 维护 history buffer 并拼接到 Actor 输入 |

修改观测维度时，需要同步检查：

| 位置 | 说明 |
| --- | --- |
| `agent_diy/conf/conf.py` | Stage 维度配置 |
| `agent_diy/model/model.py` | Actor / Critic 输入切分 |
| `agent_diy/feature/*observation_process.py` | 实际拼接逻辑 |
| TOML 配置 | 环境、传感器、reward 配置 |

## 奖励设计

| 方向 | 文件 | 说明 |
| --- | --- | --- |
| Locomotion reward | `agent_diy/feature/reward_process.py` | 稳定、能耗、步态、前进、姿态相关奖励 |
| Navigation reward | `agent_diy/feature/reward_process.py` | 目标接近、到达出口、通道选择、避障等 |
| Reward 权重 | `agent_diy/conf/*.toml` | 通过 `[rewards.<name>]` 激活并调权重 |
| 监控面板 | `agent_diy/conf/monitor_builder.py` | 新增 reward 后同步添加监控 |

## 模型结构

| 方向 | 文件 | 说明 |
| --- | --- | --- |
| Actor-Critic | `agent_diy/model/model.py` | 主模型结构 |
| HeightScan Encoder | `agent_diy/model/model.py` | 2D CNN 编码局部高度图 |
| NavScan Encoder | `agent_diy/model/model.py` | 1D CNN 编码前瞻扫描 |
| History Encoder | `agent_diy/model/model.py` | GRU / MLP 历史观测编码 |
| HierNav | `agent_diy/model/hier_nav_model.py` | 速度命令导航头 + 冻结运动策略 |

## 训练策略

| 阶段 | 建议 |
| --- | --- |
| 运动控制 | 先在 Standard 混合地形上训练稳定步态，重点观察摔倒率、姿态分与能耗 |
| Track 导航 | 在已有 locomotion checkpoint 上热启动，逐步加入 goal、nav scanner 与地形上下文 |
| 迷宫泛化 | 使用课程学习和难度档评估，重点观察 `completed_count_track_l{0~9}` |
| 安全约束 | 关注 cost 与 violation，不要只看 reward 上升 |

---

# 参考文档

| 文档 | 说明 |
| --- | --- |
| `docs/introduction.md` | 赛题目标、地形、评分与终止条件 |
| `docs/env.md` | 环境配置、观测、动作与监控指标 |
| `docs/protocol.md` | reset / step 数据协议与访问示例 |
| `docs/agent_description.md` | 智能体结构、算法、监控和评估配置 |

如文档与代码实现存在差异，以当前代码为准。
