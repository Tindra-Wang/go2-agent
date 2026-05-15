# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → obs = proprio + scan = 301 dim
- Stage3:   num_goal_obs=4  → obs = proprio + scan + goal = 305 dim
观测布局：[proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → obs = proprio + scan = 301 维
- Stage3：  num_goal_obs=4  → obs = proprio + scan + goal = 305 维
"""

import torch

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan and optional goal obs.

    带 height_scan 和可选 goal obs 的 policy 观测处理器。
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Stage1/2: proprio(45) + height_scan(256) = 301
        Stage3:   proprio(45) + height_scan(256) + goal(4) = 305
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
