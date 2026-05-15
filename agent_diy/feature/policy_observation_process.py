# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → obs = proprio + scan = 301 dim
- Stage3:   num_goal_obs=11 → obs = proprio + scan + goal(4) + nav_sectors(7) = 312 dim
观测布局：[proprio(45) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → obs = proprio + scan = 301 维
- Stage3：  num_goal_obs=11 → obs = proprio + scan + goal(4) + nav_sectors(7) = 312 维
"""

import torch

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi

NUM_NAV_SECTORS = 5
NUM_NAV_FEATURES = 7  # 5 sector clearances + min_ahead + best_direction


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan, goal obs, and nav_scanner sectors.

    带 height_scan、goal obs 和 nav_scanner 扇区特征的 policy 观测处理器。
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Stage1/2: proprio(45) + height_scan(256) = 301
        Stage3:   proprio(45) + height_scan(256) + goal(4) + nav_sectors(7) = 312
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
        nav_scanner 范围比 height_scanner 更大，提供远距离避障信息。

        Returns: [sector_0..4, min_ahead, best_direction] shape=(num_envs, 7)
          - sector_i: 第 i 扇区的平均通行距离（归一化到 ~[0,1]）
          - min_ahead: 中心扇区最近障碍距离（归一化）
          - best_direction: 最通畅扇区方向 [-1, 1]（左负右正）
        """
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "scene") or "nav_scanner" not in env.scene.sensors:
            return torch.zeros(num_envs, NUM_NAV_FEATURES, device=device)

        sensor = env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        # scan: (num_envs, N_rays) — positive = clear, negative = obstacle above

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

        sectors = torch.stack(sector_clearances, dim=1)  # (num_envs, 5)
        # Normalize: clip to [-1, 5] then scale to [0, 1]
        sectors_norm = (sectors.clamp(-1.0, 5.0) + 1.0) / 6.0

        # Min clearance in center sector (sector index 2)
        center_start = (sector_size + (1 if 0 < remainder else 0)) + (sector_size + (1 if 1 < remainder else 0))
        center_end = center_start + sector_size + (1 if 2 < remainder else 0)
        center_data = scan[:, center_start:center_end]
        min_ahead = center_data.min(dim=1).values
        min_ahead_norm = (min_ahead.clamp(-1.0, 5.0) + 1.0) / 6.0

        # Best direction: index of clearest sector mapped to [-1, 1]
        best_idx = sectors_norm.argmax(dim=1).float()
        best_direction = (best_idx / (NUM_NAV_SECTORS - 1)) * 2.0 - 1.0

        return torch.cat([
            sectors_norm,
            min_ahead_norm.unsqueeze(1),
            best_direction.unsqueeze(1),
        ], dim=1)
