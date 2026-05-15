# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → critic_obs = 316 dim
- Stage3:   num_goal_obs=11 → critic_obs = 60 + 256 + 4 + 7 = 327 dim
critic 观测布局：[critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → critic_obs = 316 维
- Stage3：  num_goal_obs=11 → critic_obs = 60 + 256 + 4 + 7 = 327 维
"""

import torch

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi
from agent_diy.feature.policy_observation_process import NUM_NAV_SECTORS, NUM_NAV_FEATURES


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs and nav_scanner sectors.

    与 Isaac Lab CriticCfg 对齐的 critic 观测处理器，可选拼接 goal obs + nav 扇区特征。
    """

    target_group = "critic"

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        Stage1/2: critic_obs = 316
        Stage3:   critic_obs = 316 + goal(4) + nav_sectors(7) = 327
        """
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self._goal_position_in_robot_frame()
            nav_obs = self._nav_scanner_sector_features()
            obs = self.concatenate_terms(obs, goal_obs, nav_obs)

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

    def _nav_scanner_sector_features(self):
        """Extract sector-based clearance features from nav_scanner (7 dim).

        从 nav_scanner 提取扇区通行距离特征（7 维）。
        """
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "scene") or "nav_scanner" not in env.scene.sensors:
            return torch.zeros(num_envs, NUM_NAV_FEATURES, device=device)

        sensor = env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]

        n_rays = scan.shape[1]
        sector_size = n_rays // NUM_NAV_SECTORS
        remainder = n_rays - sector_size * NUM_NAV_SECTORS

        sector_clearances = []
        start = 0
        for i in range(NUM_NAV_SECTORS):
            end = start + sector_size + (1 if i < remainder else 0)
            sector_data = scan[:, start:end]
            sector_clearances.append(sector_data.mean(dim=1))
            start = end

        sectors = torch.stack(sector_clearances, dim=1)
        sectors_norm = (sectors.clamp(-1.0, 5.0) + 1.0) / 6.0

        center_start = (sector_size + (1 if 0 < remainder else 0)) + (sector_size + (1 if 1 < remainder else 0))
        center_end = center_start + sector_size + (1 if 2 < remainder else 0)
        center_data = scan[:, center_start:center_end]
        min_ahead = center_data.min(dim=1).values
        min_ahead_norm = (min_ahead.clamp(-1.0, 5.0) + 1.0) / 6.0

        best_idx = sectors_norm.argmax(dim=1).float()
        best_direction = (best_idx / (NUM_NAV_SECTORS - 1)) * 2.0 - 1.0

        return torch.cat([
            sectors_norm,
            min_ahead_norm.unsqueeze(1),
            best_direction.unsqueeze(1),
        ], dim=1)
