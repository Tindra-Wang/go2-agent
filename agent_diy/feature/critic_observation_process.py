# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256) | terrain(2) | maze_hint(2) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2: num_goal_obs=0, num_nav_scan_obs=0, num_terrain_context_obs=0, num_maze_nav_hint_obs=0
            → critic_obs = 316 dim
- Stage3:   num_goal_obs=4, num_nav_scan_obs=32, num_terrain_context_obs=2, num_maze_nav_hint_obs=2
            → critic_obs = 316 + terrain(2) + maze_hint(2) + goal(4) + raw_nav_scan(32) = 356 dim
critic 观测布局：[critic_proprio(60) | height_scan(256) | terrain(2) | maze_hint(2) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2：num_goal_obs=0, num_nav_scan_obs=0, num_terrain_context_obs=0, num_maze_nav_hint_obs=0
            → critic_obs = 316 维
- Stage3：  num_goal_obs=4, num_nav_scan_obs=32, num_terrain_context_obs=2, num_maze_nav_hint_obs=2
            → critic_obs = 316 + terrain(2) + maze_hint(2) + goal(4) + raw_nav_scan(32) = 356 维
"""

import torch
import torch.nn.functional as F

from agent_diy.feature.policy_observation_process import PolicyObservationProcess


class CriticObservationProcess(PolicyObservationProcess):
    """Critic observation processor with goal obs plus raw nav scan for 1D CNN.

    critic 保留 goal 特征，并追加 raw nav_scanner 给 value/cost critic 的 1D CNN 使用。
    """

    target_group = "critic"
    goal_pos_scale = 10.0
    goal_pos_clip = 20.0

    def _get_num_nav_scan_obs(self) -> int:
        """Read raw nav_scanner obs dim from the active stage config."""
        cached = getattr(self, "_num_nav_scan_obs", None)
        if cached is not None:
            return cached

        from agent_diy.conf.conf import Config

        num_nav_scan_obs = int(getattr(Config.CURRENT, "num_nav_scan_obs", 0))
        self._num_nav_scan_obs = num_nav_scan_obs
        return num_nav_scan_obs

    def process(self):
        """Compute critic observation."""
        obs = self.default_observation()
        goal_obs = None
        nav_scan = None

        if self._get_num_terrain_context_obs() > 0:
            terrain_ctx = self._compute_terrain_context()
            obs = self.concatenate_terms(obs, terrain_ctx)

        if self._get_num_maze_nav_hint_obs() > 0:
            maze_hint = self._compute_maze_nav_hint()
            obs = self.concatenate_terms(obs, maze_hint)

        if self._get_num_goal_obs() > 0:
            goal_obs = self._goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        if self._get_num_nav_scan_obs() > 0:
            nav_scan = self._nav_scanner_raw_scan()
            obs = self.concatenate_terms(obs, nav_scan)

        return obs

    def _nav_scanner_raw_scan(self):
        """Return normalized raw nav_scanner rays for critic-side 1D CNNs."""
        env = self.env
        device = env.device
        num_envs = env.num_envs
        target_dim = self._get_num_nav_scan_obs()

        if target_dim <= 0:
            return torch.zeros(num_envs, 0, device=device)

        if not hasattr(env, "scene") or "nav_scanner" not in env.scene.sensors:
            return torch.zeros(num_envs, target_dim, device=device)

        sensor = env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        scan = scan.reshape(num_envs, -1)
        scan = torch.nan_to_num(scan, nan=5.0, posinf=5.0, neginf=-1.0)
        scan = (scan.clamp(-1.0, 5.0) + 1.0) / 6.0

        if scan.shape[1] != target_dim:
            scan = F.interpolate(scan.unsqueeze(1), size=target_dim, mode="linear", align_corners=False).squeeze(1)

        return torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=0.0)

    def _goal_position_in_robot_frame(self):
        """Compute final goal position relative to robot in body frame (4 dim)."""
        return super()._goal_position_in_robot_frame()
