# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — custom reward processor (lite baseline).
RewardProcess — 自定义奖励处理器（lite baseline）。

This file only ships two example rewards:
    1. _reward_reach_goal       — goal-reaching judgment (0.6 m)
    2. _reward_forward_velocity — forward velocity reward (dense, demonstrates reward writing style)
本文件仅预置两个示例 reward：
    1. _reward_reach_goal       — 赛题到达判定（0.6 m）
    2. _reward_forward_velocity — 前向速度奖励（dense，展示 reward 写法）

Other generic locomotion rewards (track_lin_vel_xy / joint_acc / action_rate, etc.)
are inherited from RewardProcessBase (see tools/base_env/base_reward.py).
Players only need to activate them in the TOML; no need to re-implement them here.
其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
选手在 TOML 中激活即可，无需在此重复实现。

If players need to train a navigation policy, please add more rewards in this file.
选手若需训练导航策略，请在本文件自行添加更多 reward。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the maze exit (returns 1.0 when distance < 0.6 m).
        到达迷宫出口奖励（distance < 0.6 m 时返回 1.0）。

        Note:
            The threshold must match the threshold of _goal_reached_termination
            in tools/unitree_rl_lab/.../velocity_env_cfg.py (currently 0.6 m),
            otherwise a "termination-reward dead zone" will appear.
            threshold 必须与 tools/unitree_rl_lab/.../velocity_env_cfg.py 中
            _goal_reached_termination 的 threshold 一致（当前 0.6 m），
            否则会产生"终止-奖励死区"。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame (the larger the better).
        前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）。

        This is an example reward that demonstrates how to read the robot state and
        build a dense signal.
        示例性 reward，展示如何读取机器人状态并构造 dense signal。
        """
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]
