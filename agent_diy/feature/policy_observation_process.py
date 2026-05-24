# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256) | terrain(2) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2: num_goal_obs=0, num_nav_scan_obs=0, num_terrain_context_obs=0
            → obs = proprio + scan = 301 dim
- Stage3:   num_goal_obs=4, num_nav_scan_obs=32, num_terrain_context_obs=2
            → obs = proprio + scan + terrain(2) + goal(4) + raw_nav_scan(32) = 339 dim
观测布局：[proprio(45) | height_scan(256) | terrain(2) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2：num_goal_obs=0, num_nav_scan_obs=0, num_terrain_context_obs=0
            → obs = proprio + scan = 301 维
- Stage3：  num_goal_obs=4, num_nav_scan_obs=32, num_terrain_context_obs=2
            → obs = proprio + scan + terrain(2) + goal(4) + raw_nav_scan(32) = 339 维
"""

import torch
import torch.nn.functional as F

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with goal obs plus raw nav scan for 1D CNN.

    policy 观测保留 goal 特征和给 1D CNN 使用的原始 nav_scanner 序列，不再拼接 nav_sector。
    """

    target_group = "policy"

    def _get_num_nav_scan_obs(self) -> int:
        """Read raw nav_scanner obs dim from the active stage config.

        本地兜底：即使运行环境中的 base ObservationProcess 没有该 helper，
        policy 侧也能正常读取 num_nav_scan_obs。
        """
        cached = getattr(self, "_num_nav_scan_obs", None)
        if cached is not None:
            return cached

        from agent_diy.conf.conf import Config

        num_nav_scan_obs = int(getattr(Config.CURRENT, "num_nav_scan_obs", 0))
        self._num_nav_scan_obs = num_nav_scan_obs
        return num_nav_scan_obs

    def _get_num_terrain_context_obs(self) -> int:
        cached = getattr(self, "_num_terrain_context_obs", None)
        if cached is not None:
            return cached

        from agent_diy.conf.conf import Config

        num_terrain_context_obs = int(getattr(Config.CURRENT, "num_terrain_context_obs", 0))
        self._num_terrain_context_obs = num_terrain_context_obs
        return num_terrain_context_obs

    def _get_num_maze_nav_hint_obs(self) -> int:
        cached = getattr(self, "_num_maze_nav_hint_obs", None)
        if cached is not None:
            return cached

        from agent_diy.conf.conf import Config

        num_maze_nav_hint_obs = int(getattr(Config.CURRENT, "num_maze_nav_hint_obs", 0))
        self._num_maze_nav_hint_obs = num_maze_nav_hint_obs
        return num_maze_nav_hint_obs

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Stage1/2: proprio(45) + height_scan(256) = 301
        Stage3:   proprio(45) + height_scan(256) + terrain(2) + maze_hint(2) + goal(4) + nav_scan(32) = 341
        """
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

    def _compute_terrain_context(self):
        """Return [terrain_type_norm, segment_progress] for track-mode terrain awareness.

        In track mode the policy sees which sub-terrain segment it is in and how far
        through, so it can learn per-terrain strategies (e.g. take the side on stairs).

        Returns zeros when not in track mode.
        """
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "goal_positions") or env.goal_positions is None:
            return torch.zeros(num_envs, 2, device=device)
        if not hasattr(env.scene, "env_origins") or env.scene.env_origins is None:
            return torch.zeros(num_envs, 2, device=device)

        robot = self._get_robot()
        robot_pos = robot.data.root_pos_w
        origins = env.scene.env_origins

        track_start_x = origins[:, 0]
        final_x = env.goal_positions[:, 0]

        from agent_diy.conf.conf import Config

        num_segments = int(getattr(Config.CURRENT, "track_num_segments", 5))

        total_dx = torch.clamp(final_x - track_start_x, min=1.0)
        segment_length = total_dx / num_segments
        progress = torch.clamp(robot_pos[:, 0] - track_start_x, min=0.0)
        segment_idx = torch.clamp((progress / segment_length).long(), 0, num_segments - 1)

        segment_start = segment_idx.float() * segment_length
        seg_progress = torch.clamp((progress - segment_start) / segment_length.clamp(min=0.01), 0.0, 1.0)
        terrain_type_norm = segment_idx.float() / max(num_segments - 1, 1)

        ctx = torch.stack([terrain_type_norm, seg_progress], dim=-1)
        return torch.nan_to_num(ctx)

    def _compute_maze_nav_hint(self):
        """Return [best_passage_angle_norm, best_passage_score] for maze navigation.

        Uses nav_scanner rays to find open passages and returns the direction of
        the passage that best aligns with the goal. Non-zero only in the maze segment.

        Returns zeros when not in maze or when nav_scanner is unavailable.
        """
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "goal_positions") or env.goal_positions is None:
            return torch.zeros(num_envs, 2, device=device)
        if not hasattr(env.scene, "env_origins") or env.scene.env_origins is None:
            return torch.zeros(num_envs, 2, device=device)
        if "nav_scanner" not in env.scene.sensors:
            return torch.zeros(num_envs, 2, device=device)

        robot = self._get_robot()
        robot_pos = robot.data.root_pos_w
        origins = env.scene.env_origins

        track_start_x = origins[:, 0]
        final_x = env.goal_positions[:, 0]

        from agent_diy.conf.conf import Config

        num_segments = int(getattr(Config.CURRENT, "track_num_segments", 5))

        total_dx = torch.clamp(final_x - track_start_x, min=1.0)
        segment_length = total_dx / num_segments
        progress = torch.clamp(robot_pos[:, 0] - track_start_x, min=0.0)
        segment_idx = torch.clamp((progress / segment_length).long(), 0, num_segments - 1)

        in_maze = (segment_idx == num_segments - 1).float()

        robot_pos_2d = robot_pos[:, :2]
        goal_pos = env.goal_positions[:, :2]
        delta_world = goal_pos - robot_pos_2d
        yaw = yaw_from_quat(robot.data.root_quat_w)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        goal_angle = torch.atan2(goal_body_y, goal_body_x)

        sensor = env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        n_rays = scan.shape[1]
        scan_flat = scan.reshape(num_envs, -1)

        clear = (scan_flat > -0.3).float()

        ray_angles = torch.linspace(-0.8, 0.8, n_rays, device=device)
        angle_diff = ray_angles.unsqueeze(0) - goal_angle.unsqueeze(1)
        goal_alignment = torch.cos(angle_diff).clamp(min=0.0)

        weights = clear * goal_alignment
        total_weight = weights.sum(dim=1).clamp(min=1e-6)
        weighted_angle = (weights * ray_angles.unsqueeze(0)).sum(dim=1) / total_weight
        best_score = total_weight / (n_rays * 0.5)

        best_angle_norm = torch.tanh(weighted_angle)

        result = torch.stack([best_angle_norm * in_maze, best_score * in_maze], dim=-1)
        return torch.nan_to_num(result)

    def _nav_scanner_raw_scan(self):
        """Return normalized raw nav_scanner rays for the actor-side 1D CNN.

        返回给 actor 端 1D CNN 使用的原始 nav_scanner 序列。
        Shape is fixed by Config.CURRENT.num_nav_scan_obs via 1D resampling.
        """
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
        env = self.env
        device = env.device
        num_envs = env.num_envs

        if not hasattr(env, "goal_positions") or env.goal_positions is None:
            return torch.zeros(num_envs, 4, device=device)

        robot = self._get_robot()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = env.goal_positions[:, :2]
        delta_world = goal_pos - robot_pos

        yaw = yaw_from_quat(robot.data.root_quat_w)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rel_x_m = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        rel_y_m = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        rel_dist = torch.norm(delta_world, dim=1)

        goal_yaw = env.goal_yaw if hasattr(env, "goal_yaw") and env.goal_yaw is not None else None
        rel_yaw = wrap_to_pi(goal_yaw - yaw) if goal_yaw is not None else torch.atan2(rel_y_m, rel_x_m)

        rel_x = torch.clamp(rel_x_m, -20.0, 20.0) / 10.0
        rel_y = torch.clamp(rel_y_m, -20.0, 20.0) / 10.0
        rel_dist = torch.tanh(rel_dist / 10.0)
        rel_yaw = rel_yaw / torch.pi
        return torch.nan_to_num(torch.stack([rel_x, rel_y, rel_dist, rel_yaw], dim=-1))
