# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2: num_goal_obs=0, num_nav_scan_obs=0 → obs = proprio + scan = 301 dim
- Stage3:   num_goal_obs=4, num_nav_scan_obs=32
            → obs = proprio + scan + goal(4) + raw_nav_scan(32) = 337 dim
观测布局：[proprio(45) | height_scan(256) | goal(4) | raw_nav_scan(num_nav_scan_obs)]
- Stage1/2：num_goal_obs=0, num_nav_scan_obs=0 → obs = proprio + scan = 301 维
- Stage3：  num_goal_obs=4, num_nav_scan_obs=32
            → obs = proprio + scan + goal(4) + raw_nav_scan(32) = 337 维
"""

import torch
import torch.nn.functional as F

from tools.base_env.observation_process import ObservationProcess, yaw_from_quat, wrap_to_pi


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with goal obs plus raw nav scan for 1D CNN.

    policy 观测保留 goal 特征和给 1D CNN 使用的原始 nav_scanner 序列，不再拼接 nav_sector。
    """

    target_group = "policy"
    goal_pos_scale = 10.0
    goal_pos_clip = 20.0

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

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Stage1/2: proprio(45) + height_scan(256) = 301
        Stage3:   proprio(45) + height_scan(256) + goal(4) + raw_nav_scan(num_nav_scan_obs)
        """
        obs = self.default_observation()
        goal_obs = None
        nav_scan = None

        if self._get_num_goal_obs() > 0:
            goal_obs = self._goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        if self._get_num_nav_scan_obs() > 0:
            nav_scan = self._nav_scanner_raw_scan()
            obs = self.concatenate_terms(obs, nav_scan)

        return obs

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
        scan = (scan.clamp(-1.0, 5.0) + 1.0) / 6.0

        if scan.shape[1] != target_dim:
            scan = F.interpolate(scan.unsqueeze(1), size=target_dim, mode="linear", align_corners=False).squeeze(1)

        return scan

    def _goal_position_in_robot_frame(self):
        """Compute goal position relative to robot in body frame (4 dim)."""
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

        rel_x_m = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        rel_y_m = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        rel_dist = torch.norm(delta_world, dim=1)

        goal_yaw = (
            env.goal_yaw
            if hasattr(env, "goal_yaw") and env.goal_yaw is not None
            else torch.zeros(num_envs, device=device)
        )
        rel_yaw = wrap_to_pi(goal_yaw - yaw)

        rel_x = rel_x_m.clamp(-self.goal_pos_clip, self.goal_pos_clip) / self.goal_pos_scale
        rel_y = rel_y_m.clamp(-self.goal_pos_clip, self.goal_pos_clip) / self.goal_pos_scale
        rel_dist = torch.tanh(rel_dist / self.goal_pos_scale)
        rel_yaw = rel_yaw / torch.pi

        return torch.stack([rel_x, rel_y, rel_dist, rel_yaw], dim=-1)
