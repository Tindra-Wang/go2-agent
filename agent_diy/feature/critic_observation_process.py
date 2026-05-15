# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → critic_obs = 316 dim
- Stage3:   num_goal_obs=4  → critic_obs = 320 dim
critic 观测布局：[critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → critic_obs = 316 维
- Stage3：  num_goal_obs=4  → critic_obs = 320 维
"""

import torch

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs.

    与 Isaac Lab CriticCfg 对齐的 critic 观测处理器，可选拼接 goal obs。
    """

    target_group = "critic"

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        Stage1/2: critic_obs = 316
        Stage3:   critic_obs = 316 + goal(4) = 320
        """
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self._goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs

    def _goal_position_in_robot_frame(self):
        """Compute goal position relative to robot in body frame (4 dim).

        计算目标点在机器人坐标系下的相对位置（4 维）。
        Returns: [rel_x, rel_y, rel_dist, rel_yaw] shape=(num_envs, 4)
        """
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "goal_positions") or env.goal_positions is None:
            return torch.zeros(num_envs, 4, device=device)

        robot = self._get_robot()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = env.goal_positions[:, :2]

        delta_world = goal_pos - robot_pos

        quat = robot.data.root_quat_w
        yaw = yaw_from_quat(quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        rel_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        rel_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        rel_dist = torch.norm(delta_world, dim=1)

        goal_yaw = env.goal_yaw if hasattr(env, "goal_yaw") and env.goal_yaw is not None else torch.zeros(num_envs, device=device)
        rel_yaw = wrap_to_pi(goal_yaw - yaw)

        return torch.stack([rel_x, rel_y, rel_dist, rel_yaw], dim=-1)
