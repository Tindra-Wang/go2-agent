# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch

from tools.base_env.base_reward import RewardProcessBase

try:
    from isaaclab.utils.math import quat_rotate_inverse as _isaac_quat_rotate_inverse
except Exception:
    _isaac_quat_rotate_inverse = None


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into body frame using inverse quaternion (wxyz).

    ``q``: (N, 4) or (N, K, 4); ``v``: (N, 3) or (N, K, 3). Matches Isaac Lab convention.
    """
    if _isaac_quat_rotate_inverse is not None:
        if v.ndim == 3:
            n_env, k, _ = v.shape
            q_e = q.unsqueeze(1).expand(n_env, k, 4).reshape(-1, 4)
            v_e = v.reshape(-1, 3)
            out = _isaac_quat_rotate_inverse(q_e, v_e)
            return out.view(n_env, k, 3)
        return _isaac_quat_rotate_inverse(q, v)

    # Fallback: MIT legged_gym-style (wxyz), vector shape (M, 3)
    if v.ndim == 3:
        n_env, k, _ = v.shape
        q_e = q.unsqueeze(1).expand(n_env, k, 4).reshape(-1, 4)
        v_e = v.reshape(-1, 3)
    else:
        q_e, v_e = q, v
    q_w = q_e[:, 0]
    q_vec = q_e[:, 1:4]
    a = v_e * (2.0 * q_w.unsqueeze(-1) ** 2 - 1.0)
    b = torch.cross(q_vec, v_e, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.sum(q_vec * v_e, dim=-1, keepdim=True) * 2.0
    out = a - b + c
    if v.ndim == 3:
        return out.view(n_env, k, 3)
    return out


class RewardProcess(RewardProcessBase):
    """
    Custom reward processor with user-defined reward terms
    自定义奖励处理器，包含用户自定义的奖励项
    """

    def _nav_scanner_blockage(self, obstacle_threshold: float = -0.3):
        """Return far-field blockage ratios from nav_scanner, or None if missing."""
        if "nav_scanner" not in self.env.scene.sensors:
            return None

        sensor = self.env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        scan = scan.view(self.env.num_envs, -1)
        blocked = (scan < obstacle_threshold).float()

        n_rays = blocked.shape[1]
        if n_rays < 2:
            ratio = blocked.mean(dim=1)
            return ratio, ratio, ratio

        mid = n_rays // 2
        left_blocked = blocked[:, :mid].mean(dim=1)
        right_blocked = blocked[:, mid:].mean(dim=1)
        blocked_ratio = blocked.mean(dim=1)
        return blocked_ratio, left_blocked, right_blocked

    def _reward_flat_orientation(self):
        """Penalize non-flat base orientation (deviation from upright).

        惩罚非平坦的基座朝向（偏离直立）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        """Penalize large joint velocities.

        惩罚大的关节速度。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).

        奖励长步幅（移动时脚部滞空时间超过阈值）。

        Args:
            command_name: Command term name. / 命令项名称。
            threshold: Minimum air time threshold. / 最小滞空时间阈值。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        # Compute reward
        # 计算奖励
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # No reward for zero commands
        # 当命令为零时不给奖励
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_air_time_variance_penalty(self):
        """Penalize variance in foot air/contact time (gait symmetry).

        惩罚脚部滞空/接触时间的方差（步态对称性）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        对齐 Isaac Lab feet_slide：用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        # Check which feet are in contact (3D force norm, max over history frames)
        # 检查哪些脚在接触地面（3D 力的范数，历史帧取最大值）
        contacts = (
                contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        # Get foot velocities (xy only)
        # 获取脚部速度（仅 xy 分量）
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        # Penalize velocity when in contact
        # 接触时惩罚速度
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        """Penalize joint position error from default pose.

        惩罚关节位置偏离默认姿态。

        Args:
            stand_still_scale: Scale factor when standing still. / 静止时的缩放因子。
            velocity_threshold: Velocity threshold to determine if moving. / 判断是否移动的速度阈值。
        """
        asset = self._get_robot_asset()
        cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
            reward,
            stand_still_scale * reward,
            )

    # def _reward_standing_posture(self):
    #     """Reward maintaining default standing posture (for standing task)."""
    #     asset = self._get_robot_asset()
    #     joint_deviation = torch.sum(
    #         torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    #     )
    #     return torch.exp(-joint_deviation * 10.0)

    # def _reward_feet_contact(self):
    #     """Reward four feet in contact (for standing task)."""
    #     contact = self._get_foot_contact()
    #     num_feet_in_contact = contact.sum(dim=1).float()
    #     return (num_feet_in_contact == 4).float()

    # def _reward_stand_velocity(self):
    #     """Penalize any movement (for standing task)."""
    #     asset = self._get_robot_asset()
    #     linear_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    #     angular_vel_penalty = torch.square(asset.data.root_ang_vel_b[:, 2])
    #     return linear_vel_penalty + angular_vel_penalty

    def _reward_obstacle_evasion(
            self,
            command_name: str = "base_velocity",
            obstacle_threshold: float = -0.3,
            near_x_end: int = 12,
            body_y_start: int = 3,
            body_y_end: int = 13,
            turn_std: float = 0.5,
    ):
        """Penalize forward-blocked path when robot is not actively turning.

        惩罚前方被障碍阻挡时未主动转向。

        Uses height_scan near-field window to detect tall obstacles (pillars/walls)
        directly ahead, and angular velocity to detect evasion turning.
        使用 height_scan 近场窗口检测正前方高障碍物，用角速度检测转向。

        Returns: blocked * not_evading * has_fwd_cmd
        返回：blocked * not_evading * has_fwd_cmd

        Grid layout (16x16, offset 0.75m fwd, res=0.1m):
          reshaped (N, 16, 16) -> dim0=y_idx, dim1=x_idx
          y: -0.75m(idx0) .. +0.75m(idx15)
          x: 0.0m(idx0) .. 1.5m(idx15)
        网格布局（16x16，前方偏移 0.75m，分辨率 0.1m）：
          reshape 为 (N, 16, 16) -> dim0=y_idx，dim1=x_idx
          y：-0.75m(idx0) .. +0.75m(idx15)
          x：0.0m(idx0) .. 1.5m(idx15)

        Default window:
          Y [3:13] = -0.45m ~ +0.55m (≈ passage width, catches side walls)
          X [:10]  = 0.0m ~ 0.9m (≈1s reaction at 0.5~1.0 m/s)
        默认窗口：
          Y [3:13] = -0.45m ~ +0.55m（≈通道宽度，可捕捉侧壁）
          X [:10]  = 0.0m ~ 0.9m（在 0.5~1.0 m/s 下约 1s 反应距离）
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]

        # raw height: base_z - hit_z (positive=ground below, negative=obstacle above)
        # 原始高度：base_z - hit_z（正值=下方为地面，负值=上方有障碍）
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        # near-field body-width window
        # 近场、身体宽度的窗口
        window = grid[:, body_y_start:body_y_end, :near_x_end]

        # column-projection: for each y-strip, any obstacle in forward range?
        # 列投影：每个 y 条带在前方范围内是否存在障碍物
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        near_blocked = col_blocked.mean(dim=-1)

        nav_stats = self._nav_scanner_blockage(obstacle_threshold)
        if nav_stats is not None:
            far_blocked, _, _ = nav_stats
            blocked = torch.maximum(near_blocked, 0.75 * far_blocked)
        else:
            blocked = near_blocked

        # evasion signal: turning hard -> low penalty
        # 规避信号：转弯幅度大 -> 惩罚低
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)

        # gate: only when forward command exists
        # 门控：仅在存在前进指令时生效
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()

        return blocked * not_evading * has_fwd_cmd

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).

        惩罚脚撞到垂直面。阈值 5× 对齐 legged_gym 原版。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    # -----------------------------------------------------------------------
    # Navigation rewards (Stage 3)
    # 导航奖励（第三阶段）
    # -----------------------------------------------------------------------

    def _reward_approach_goal(self):
        """Reward approaching the maze exit: -(current_dist - previous_dist).

        接近迷宫出口奖励：距离减少→正奖励，距离增加→负奖励。

        Requires env.goal_positions to be set by TerrainExitManager
        (auto-initialized via observation_process.goal_position_in_robot_frame).
        需要 TerrainExitManager 设置 env.goal_positions
        （通过 observation_process.goal_position_in_robot_frame 自动初始化）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
        goal_pos = self.env.goal_positions[:, :2]  # (N, 2)

        current_dist = torch.norm(goal_pos - robot_pos, dim=1)  # (N,)

        # 首次调用：初始化 previous_dist，返回零
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)

        # 距离变化（正=远离，负=接近）
        delta_dist = current_dist - self.env._previous_goal_dist

        # 重置的 env 不计算 delta（距离跳变）
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0

        # 更新 previous
        self.env._previous_goal_dist = current_dist.clone()

        # 返回负的距离变化 = 接近→正奖励
        return -delta_dist

    def _reward_reach_goal(self, threshold: float = 0.5):
        """Reward reaching the maze exit (distance < threshold).

        到达迷宫出口奖励（距离 < 阈值时返回 1.0）。

        Args:
            threshold: Distance threshold to consider goal reached (m).
                       判定到达目标的距离阈值（米）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]

        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_navigation_time(self):
        """Per-step penalty to encourage fast navigation.

        每步固定惩罚，鼓励快速到达出口。返回固定值 1.0，由 weight 控制大小。
        """
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_heuristic_navigation(self, obstacle_threshold: float = -0.3):
        """Main navigation signal: blend goal-tracking and clearance-following.

        主导航信号：前方通畅时奖励朝 goal 方向前进，前方阻挡时奖励朝空旷侧转向。

        Logic:
        - Use height_scanner 16x16 grid to detect forward blockage.
        - Clear ahead: reward = cos(heading_to_goal) * forward_speed (capped).
        - Blocked ahead: reward = alignment with clearance direction (turn towards open side).
        逻辑：
        - 用 height_scanner 16x16 grid 检测前方阻挡。
        - 前方通畅：reward = cos(朝向goal的角度) × 前向速度（截断）。
        - 前方阻挡：reward = 朝空旷侧转向的对齐程度。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()
        device = self.env.device

        # --- Forward blockage detection (height_scanner, 16x16) ---
        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        # Body-width forward window: Y[4:12], X[:8] (near-field ~0.8m)
        window = grid[:, 4:12, :8]
        near_blocked_ratio = (window < obstacle_threshold).float().mean(dim=(1, 2))

        nav_stats = self._nav_scanner_blockage(obstacle_threshold)
        if nav_stats is not None:
            far_blocked_ratio, nav_left_blocked, nav_right_blocked = nav_stats
            blocked_ratio = torch.maximum(near_blocked_ratio, 0.75 * far_blocked_ratio)
        else:
            nav_left_blocked = None
            nav_right_blocked = None
            blocked_ratio = near_blocked_ratio
        clear = (blocked_ratio < 0.15).float()
        blocked = 1.0 - clear

        # --- Goal direction in robot frame ---
        robot_pos = asset.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        delta_world = goal_pos - robot_pos
        # Robot heading from quaternion (wxyz)
        quat = asset.data.root_quat_w
        # yaw = atan2(2*(wz + xy), 1 - 2*(yy + zz))
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        # World → body frame rotation (2D)
        goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        goal_angle = torch.atan2(goal_body_y, goal_body_x)

        # --- Clear path: reward forward speed projected onto goal direction ---
        fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=1.0)
        cos_goal = torch.cos(goal_angle)
        clear_reward = torch.clamp(cos_goal, min=0.0) * fwd_speed

        # --- Blocked path: reward turning towards open side ---
        left_clear = (grid[:, 0:6, :8] < obstacle_threshold).float().mean(dim=(1, 2))
        right_clear = (grid[:, 10:16, :8] < obstacle_threshold).float().mean(dim=(1, 2))
        if nav_left_blocked is not None and nav_right_blocked is not None:
            left_clear = torch.maximum(left_clear, nav_left_blocked)
            right_clear = torch.maximum(right_clear, nav_right_blocked)
        # Desired turn: positive yaw_rate = turn left, negative = turn right.
        # Blend free-space preference with goal side so escape turns do not
        # continue in a direction that obviously moves away from the exit.
        clearance_delta = right_clear - left_clear
        clearance_sign = torch.sign(clearance_delta)
        goal_sign = torch.sign(goal_angle)
        desired_score = 0.65 * clearance_sign + 0.35 * goal_sign
        desired_sign = torch.sign(desired_score)
        desired_sign = torch.where(desired_sign == 0.0, goal_sign, desired_sign)
        yaw_rate = asset.data.root_ang_vel_b[:, 2]
        blocked_reward = torch.clamp(desired_sign * yaw_rate, min=0.0, max=1.0)

        return clear * clear_reward + blocked * blocked_reward

    def _reward_narrow_passage_alignment(
            self,
            obstacle_threshold: float = -0.3,
            side_threshold: float = 0.2,
            center_threshold: float = 0.12,
            lateral_std: float = 0.35,
            yaw_rate_std: float = 0.8,
    ):
        """Reward entering narrow passages while aligned with the goal direction."""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]

        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        forward_center = (grid[:, 6:10, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))
        left_side = (grid[:, 0:5, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))
        right_side = (grid[:, 11:16, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))

        side_walls = torch.minimum(left_side, right_side)
        narrow_gate = torch.clamp((side_walls - side_threshold) / (1.0 - side_threshold), 0.0, 1.0)
        center_clear_gate = torch.clamp((center_threshold - forward_center) / center_threshold, 0.0, 1.0)

        robot_pos = asset.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        delta_world = goal_pos - robot_pos
        quat = asset.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        goal_angle = torch.atan2(goal_body_y, goal_body_x)

        heading_alignment = torch.clamp(torch.cos(goal_angle), min=0.0)
        lateral_stability = torch.exp(-torch.abs(asset.data.root_lin_vel_b[:, 1]) / lateral_std)
        yaw_stability = torch.exp(-torch.abs(asset.data.root_ang_vel_b[:, 2]) / yaw_rate_std)

        return narrow_gate * center_clear_gate * heading_alignment * lateral_stability * yaw_stability

    def _reward_narrow_passage_instability(
            self,
            obstacle_threshold: float = -0.3,
            side_threshold: float = 0.2,
            center_threshold: float = 0.12,
            lateral_scale: float = 1.0,
            yaw_rate_scale: float = 0.25,
            forward_speed_threshold: float = 0.8,
    ):
        """Penalize side-slip, over-turning, and rushing inside narrow openings."""
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]

        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        forward_center = (grid[:, 6:10, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))
        left_side = (grid[:, 0:5, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))
        right_side = (grid[:, 11:16, 4:14] < obstacle_threshold).float().mean(dim=(1, 2))

        side_walls = torch.minimum(left_side, right_side)
        narrow_gate = torch.clamp((side_walls - side_threshold) / (1.0 - side_threshold), 0.0, 1.0)
        center_clear_gate = torch.clamp((center_threshold - forward_center) / center_threshold, 0.0, 1.0)

        lateral_vel = asset.data.root_lin_vel_b[:, 1]
        yaw_rate = asset.data.root_ang_vel_b[:, 2]
        fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0] - forward_speed_threshold, min=0.0)
        instability = lateral_scale * torch.square(lateral_vel) + yaw_rate_scale * torch.square(yaw_rate) + fwd_speed

        return narrow_gate * center_clear_gate * instability

    def _reward_goal_biased_corridor(
            self,
            obstacle_threshold: float = -0.3,
            max_angle: float = 1.5708,
            forward_speed_scale: float = 0.5,
    ):
        """Reward moving through open nav-scanner directions closest to the goal.

        Gated to maze segment only: on stairs/slopes the terrain context lets the
        policy learn terrain-adaptive strategies; in the maze the best strategy is
        to follow passages that point toward the exit.
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        if "nav_scanner" not in self.env.scene.sensors:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()

        # --- maze gate: only the last track segment ---
        if hasattr(self.env.scene, "env_origins") and self.env.scene.env_origins is not None:
            robot_pos = asset.data.root_pos_w
            origins = self.env.scene.env_origins
            track_start_x = origins[:, 0]
            final_x = self.env.goal_positions[:, 0]
            from agent_diy.conf.conf import Config

            num_segments = int(getattr(Config.CURRENT, "track_num_segments", 5))
            total_dx = torch.clamp(final_x - track_start_x, min=1.0)
            segment_length = total_dx / num_segments
            progress = torch.clamp(robot_pos[:, 0] - track_start_x, min=0.0)
            segment_idx = torch.clamp((progress / segment_length).long(), 0, num_segments - 1)
            in_maze = (segment_idx == num_segments - 1).float()
        else:
            in_maze = torch.ones(self.env.num_envs, device=self.env.device)

        sensor = self.env.scene.sensors["nav_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        scan = scan.view(self.env.num_envs, -1)

        n_rays = scan.shape[1]
        ray_angles = torch.linspace(-max_angle, max_angle, n_rays, device=self.env.device)
        clearance = (scan >= obstacle_threshold).float()

        robot_pos = asset.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        delta_world = goal_pos - robot_pos
        quat = asset.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        goal_angle = torch.atan2(goal_body_y, goal_body_x).clamp(-max_angle, max_angle)

        goal_alignment = torch.clamp(torch.cos(ray_angles.unsqueeze(0) - goal_angle.unsqueeze(1)), min=0.0)
        open_goal_score = clearance * goal_alignment
        best_score = open_goal_score.max(dim=1).values

        fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=forward_speed_scale) / forward_speed_scale
        return in_maze * best_score * fwd_speed

    def _reward_terrain_adaptive_pathing(self, obstacle_threshold: float = -0.3):
        """On pyramid terrains, reward seeking the lower side instead of climbing the centre.

        All four non-maze segments (slope / inv_slope / stairs / inv_stairs):
        the easiest path is always along the side of the track, not the centre.
        Uses height_scanner to find the clearer side and reward lateral + forward
        movement toward it.  Not active in maze.
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        if not hasattr(self.env.scene, "env_origins") or self.env.scene.env_origins is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        asset = self._get_robot_asset()
        robot_pos = asset.data.root_pos_w
        origins = self.env.scene.env_origins

        track_start_x = origins[:, 0]
        final_x = self.env.goal_positions[:, 0]

        from agent_diy.conf.conf import Config

        num_segments = int(getattr(Config.CURRENT, "track_num_segments", 5))
        total_dx = torch.clamp(final_x - track_start_x, min=1.0)
        segment_length = total_dx / num_segments
        progress = torch.clamp(robot_pos[:, 0] - track_start_x, min=0.0)
        segment_idx = torch.clamp((progress / segment_length).long(), 0, num_segments - 1)

        in_terrain = (segment_idx < num_segments - 1).float()

        sensor = self.env.scene.sensors["height_scanner"]
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        left_clear = (grid[:, 0:6, 2:12] >= obstacle_threshold).float().mean(dim=(1, 2))
        right_clear = (grid[:, 10:16, 2:12] >= obstacle_threshold).float().mean(dim=(1, 2))

        lateral_vel = asset.data.root_lin_vel_b[:, 1]
        fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=1.0)

        side_diff = right_clear - left_clear
        lateral_bonus = torch.tanh(1.5 * side_diff * (-lateral_vel))
        best_side_score = torch.maximum(left_clear, right_clear)

        return in_terrain * best_side_score * (fwd_speed + 0.4 * lateral_bonus)

    def _reward_deadend_escape(self, obstacle_threshold: float = -0.3, trapped_threshold: float = 0.3):
        """Reward turning when trapped in a dead-end (nav_scanner wider range).

        死胡同逃脱：用 nav_scanner（范围更大）检测大面积阻挡，奖励转向。

        When forward path is heavily blocked (ratio > trapped_threshold),
        reward angular velocity magnitude to encourage escape turning.
        当前方大面积阻挡（ratio > trapped_threshold）时，奖励角速度幅度以鼓励转向逃脱。
        """
        asset = self._get_robot_asset()

        # Use nav_scanner if available, fallback to height_scanner
        if "nav_scanner" in self.env.scene.sensors:
            sensor = self.env.scene.sensors["nav_scanner"]
        else:
            sensor = self.env.scene.sensors["height_scanner"]

        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        n_rays = scan.shape[1]
        # Treat as 1D forward scan: fraction of rays hitting obstacles
        blocked_ratio = (scan.squeeze(-1) < obstacle_threshold).float().mean(dim=1) if scan.ndim == 2 else (scan < obstacle_threshold).float().view(self.env.num_envs, -1).mean(dim=1)

        trapped = (blocked_ratio > trapped_threshold).float()

        # Reward turning speed when trapped
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        turn_reward = torch.clamp(yaw_rate, max=2.0)

        return trapped * turn_reward

    def _reward_wall_proximity_brake(self, obstacle_threshold: float = -0.3, far_scale: float = 0.35):
        """Penalize high forward speed when wall is detected nearby.

        近墙减速：前方近距离有墙时惩罚高前向速度。

        Uses height_scanner near-field (first 4 columns ≈ 0~0.4m) to detect
        imminent collision, then penalizes forward velocity proportionally.
        用 height_scanner 近场（前 4 列 ≈ 0~0.4m）检测即将碰撞，
        按比例惩罚前向速度。
        """
        asset = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]

        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)

        # Very near-field: body-width Y[4:12], X[:4] (0~0.4m ahead)
        near_window = grid[:, 4:12, :4]
        wall_proximity = (near_window < obstacle_threshold).float().mean(dim=(1, 2))

        nav_stats = self._nav_scanner_blockage(obstacle_threshold)
        if nav_stats is not None:
            far_blocked, _, _ = nav_stats
            wall_proximity = torch.maximum(wall_proximity, far_scale * far_blocked)

        # Penalize forward speed proportional to wall proximity
        fwd_speed = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0)
        return wall_proximity * fwd_speed

    # -----------------------------------------------------------------------
    # NP3O-style gait & uphill rewards
    # NP3O 风格的步态/上坡奖励
    # -----------------------------------------------------------------------
    #
    # 设计要点（参考 LocomotionWithNP3O/envs/legged_robot.py / configs/go2_constraint_him.py）:
    # 1. `_reward_upward` 用 `1 - projected_gravity_z` 给"机身仰角"正向回报，
    #    pyramid_slope_inv / pyramid_stairs_inv 等上坡地形会显著触发，直接修复
    #    "上坡/上楼梯训练不到位"的问题；
    # 2. `*_up` 系列（lin_vel_z_up / ang_vel_xy_up / orientation_up / collision_up
    #    / stumble_up / feet_slide_up）都用 `clamp(-projected_gravity_z, 0, 1)`
    #    作为门控，仅在大致直立时才惩罚，避免上坡时被动姿态被过度惩罚导致 agent
    #    干脆"挂机不动"；
    # 3. `_reward_feet_air_time` 已存在，配合 `_reward_hip_to_default` 抑制
    #    "小碎步 / 外八"步态。
    # -----------------------------------------------------------------------

    def _reward_upward(self):
        """Reward upright body during slope/stair climbing (1 - g_z).

        机身仰起时给正奖励，鼓励上坡/上楼梯主动抬身。
        平地时 ``projected_gravity_b[:, 2] ≈ -1``，奖励 ≈ 2；
        身体下倾（下坡或趴地）时奖励减小甚至为零。
        """
        asset = self._get_robot_asset()
        return 1.0 - asset.data.projected_gravity_b[:, 2]

    def _upright_gate(self):
        """Asymmetric posture gate: floor=0.6 when descending, floor=0 when climbing.

        非对称门控：
        - 上坡（机头朝上，``projected_gravity_b[:, 0] > 0``）→ floor=0，回归 NP3O 原版
          完全释放姿态惩罚的行为，避免上坡时被姿态约束打击导致 agent 干脆不上坡。
        - 下坡（机头朝下，``projected_gravity_b[:, 0] < 0``）→ floor=0.6，下楼/下坡
          保留适度姿态约束。从 0.7 回调到 0.6：0.7 过强导致下台阶时"尝试下行"的
          惩罚过大，策略选择"卡在边缘不动"作为局部最优。

        IsaacLab 中 ``projected_gravity_b`` = 世界重力 (0,0,-g) 旋到机体系。
        机头上仰 → ``g_x > 0``；机头下俯 → ``g_x < 0``。
        Reduced descent floor from 0.7 to 0.6: too strict a descent gate made
        "staying stuck on edge" cheaper than "attempting to step down".
        """
        asset = self._get_robot_asset()
        g_z = asset.data.projected_gravity_b[:, 2]
        g_x = asset.data.projected_gravity_b[:, 0]
        base = torch.clamp(-g_z, 0.0, 1.0)
        descending = (g_x < 0.0).float()
        floor = descending * 0.6
        return torch.maximum(base, floor)

    def _reward_lin_vel_z_up(self):
        """Penalize vertical linear velocity, gated by upright posture.

        惩罚 z 方向跳动，仅在直立时生效；上坡 pitch 大时不再过罚，避免挂机。
        """
        asset = self._get_robot_asset()
        return torch.square(asset.data.root_lin_vel_b[:, 2]) * self._upright_gate()

    def _reward_ang_vel_xy_up(self):
        """Penalize pitch/roll angular velocity, gated by upright posture.

        惩罚 pitch/roll 方向角速度，姿态严重倾倒时门控关闭。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * self._upright_gate()

    def _reward_orientation_up(self):
        """Penalize off-flat orientation, gated by upright posture.

        姿态偏离水平的惩罚（NP3O orientation_up 等价项）。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * self._upright_gate()

    def _reward_lateral_sway(
            self,
            command_name: str = "base_velocity",
            command_threshold: float = 0.05,
            forward_speed_threshold: float = 0.05,
            lateral_vel_scale: float = 1.0,
            roll_rate_scale: float = 0.25,
            roll_tilt_scale: float = 1.0,
    ):
        """Penalize side-to-side sway without suppressing stair-climbing pitch.

        楼梯上左右晃主要表现为 body-frame 侧向速度、roll 角速度和 roll 倾斜；
        这里仅约束这些横向分量，不惩罚 pitch，因此不会压制上台阶所需的机身仰角。
        """
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command(command_name)

        fwd_cmd = cmd[:, 0] > command_threshold
        fwd_speed = asset.data.root_lin_vel_b[:, 0] > forward_speed_threshold
        moving_forward = torch.logical_or(fwd_cmd, fwd_speed).float()

        lateral_vel = asset.data.root_lin_vel_b[:, 1]
        roll_rate = asset.data.root_ang_vel_b[:, 0]
        roll_tilt = asset.data.projected_gravity_b[:, 1]

        penalty = (
                lateral_vel_scale * torch.square(lateral_vel)
                + roll_rate_scale * torch.square(roll_rate)
                + roll_tilt_scale * torch.square(roll_tilt)
        )
        return penalty * moving_forward * self._upright_gate()

    # ----- Joint index helpers (IsaacLab Go2 joints are alphabetically grouped) -----
    # IsaacLab Go2 关节顺序按"先类型后腿名"字典序排列：
    #   [FL_hip, FR_hip, RL_hip, RR_hip,  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    #    FL_calf, FR_calf, RL_calf, RR_calf]
    # 这与 NP3O legged_gym 中"FR/FL/RR/RL × hip/thigh/calf"的索引完全不同；
    # 旧实现里硬编码的 `[0,3,6,9]` 实际上等于 [FL_hip, RR_hip, FL_thigh, RR_thigh]，
    # 既漏了一半 hip，又把 thigh 当成 hip 在罚——这是"挂机/外八/小碎步"训练
    # 不收敛的隐藏 bug。这里改为按 joint name 动态映射，避免一次性踩同一个坑。
    def _leg_joint_indices(self):
        """Return cached dict {leg: {hip,thigh,calf}} mapped from joint_names.

        按关节名解析每条腿的 hip/thigh/calf 索引，结果缓存在 ``self._leg_idx``。
        """
        if getattr(self, "_leg_idx", None) is not None:
            return self._leg_idx
        asset = self._get_robot_asset()
        names = list(asset.data.joint_names)
        idx = {leg: {} for leg in ("FL", "FR", "RL", "RR")}
        for i, n in enumerate(names):
            for leg in idx:
                if n.startswith(leg + "_"):
                    if "hip" in n:
                        idx[leg]["hip"] = i
                    elif "thigh" in n:
                        idx[leg]["thigh"] = i
                    elif "calf" in n:
                        idx[leg]["calf"] = i
        self._leg_idx = idx
        return idx

    def _reward_hip_to_default(self):
        """Penalize hip joints deviating from default angles (NP3O ``hip_pos``).

        惩罚四条腿的 hip 关节偏离 default，抑制"外八/碎步"步态。
        关节索引按名字解析，不再硬编码（避免 IsaacLab 与 NP3O 顺序差异踩坑）。
        """
        asset = self._get_robot_asset()
        idx = self._leg_joint_indices()
        hip_idx = [idx[leg]["hip"] for leg in ("FL", "FR", "RL", "RR")]
        delta = asset.data.joint_pos[:, hip_idx] - asset.data.default_joint_pos[:, hip_idx]
        return torch.sum(torch.square(delta), dim=1)

    def _reward_has_contact(self, command_threshold: float = 0.1):
        """NP3O ``has_contact``: reward all four feet on the ground when nearly idle.

        cmd≈0 时奖励"4 脚触地的比例"。直接对治评测中"右后腿长期悬空"的退化解：
        在低速指令下若一条腿没落地，本项就给不了满分，与 ``stand_nice`` 协同把
        agent 拉回标准四脚站姿。

        Args:
            command_threshold: cmd 范数低于该值视为静止 / "stand still" gate.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        contact = (forces.norm(dim=-1).max(dim=1)[0] > 1.0).float()
        cmd = self.env.command_manager.get_command("base_velocity")
        # idle 判定同时考察 ang_vel：原版只看线速度，"lin≈0 + |ang_z|=1" 时被错误
        # 视为 idle 并给四脚触地满分，让 agent 学到"原地转 + 站着拿满分"的退化解。
        # idle gate now also requires near-zero yaw-rate command, otherwise spinning
        # in place gets falsely rewarded as standing.
        lin_idle = torch.norm(cmd[:, :2], dim=1) < command_threshold
        ang_idle = torch.abs(cmd[:, 2]) < command_threshold * 2.0
        idle = (lin_idle & ang_idle).float()
        n_feet = max(1, contact.shape[1])
        return idle * contact.sum(dim=1) / n_feet

    def _reward_stand_nice(self, command_threshold: float = 0.1):
        """NP3O ``stand_nice``: pull joints back to default when idle & upright.

        在 cmd≈0 且姿态接近直立时，惩罚 |q - q_default|。直接处理"开局/中途挂机
        在奇怪扭曲姿态"的现象——只要还在挂机就一直被罚，必须复位到标准站姿。
        姿态门控用 ``1 - g_z``，倒地时不再继续累加惩罚。
        """
        asset = self._get_robot_asset()
        cmd = self.env.command_manager.get_command("base_velocity")
        # 同 has_contact：idle 必须同时 lin≈0 且 |ang_z|≈0，避免转向时被错判 idle。
        # Same fix as `has_contact`: require both linear and angular cmd near zero.
        lin_idle = torch.norm(cmd[:, :2], dim=1) < command_threshold
        ang_idle = torch.abs(cmd[:, 2]) < command_threshold * 2.0
        idle = (lin_idle & ang_idle).float()
        upright = 1.0 - asset.data.projected_gravity_b[:, 2]
        delta = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
        return delta * upright * idle

    def _reward_foot_mirror_up(self):
        """NP3O ``foot_mirror_up``: enforce diagonal-leg symmetry (FL≈RR, FR≈RL).

        对角腿对称：trot 步态下 FL 与 RR、FR 与 RL 应同相位、镜像姿态。
        若一条腿（如右后）长期悬空 / 与对角腿姿态严重错位，本项会持续给负反馈。
        姿态门控只在直立时生效，避免摔倒后继续累罚。
        """
        asset = self._get_robot_asset()
        idx = self._leg_joint_indices()
        q = asset.data.joint_pos
        fl = [idx["FL"]["hip"], idx["FL"]["thigh"], idx["FL"]["calf"]]
        fr = [idx["FR"]["hip"], idx["FR"]["thigh"], idx["FR"]["calf"]]
        rl = [idx["RL"]["hip"], idx["RL"]["thigh"], idx["RL"]["calf"]]
        rr = [idx["RR"]["hip"], idx["RR"]["thigh"], idx["RR"]["calf"]]
        diff1 = torch.sum(torch.square(q[:, fl] - q[:, rr]), dim=-1)
        diff2 = torch.sum(torch.square(q[:, fr] - q[:, rl]), dim=-1)
        return 0.5 * self._upright_gate() * (diff1 + diff2)

    def _reward_no_fly(self, command_threshold: float = 0.1):
        """Penalize "fewer than 2 feet on ground" when commanded to move (anti-air).

        移动时强制至少 2 足触地（trot 至少有一条对角触地），避免出现"三腿离地、
        一条腿勉强支撑"或"右后腿干脆挂着不参与支撑"的退化步态。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        contact = (forces.norm(dim=-1).max(dim=1)[0] > 1.0).float()
        n_contact = contact.sum(dim=1)
        cmd = self.env.command_manager.get_command("base_velocity")
        moving = (torch.norm(cmd[:, :2], dim=1) >= command_threshold).float()
        return ((n_contact < 2).float()) * moving

    def _reward_feet_height_body(
            self,
            command_name: str = "base_velocity",
            target_height: float = -0.30,
            tanh_mult: float = 2.0,
    ):
        """Penalize swing-foot height (body z) vs target; aligns Isaac Lab feet_height_body.

        摆动腿在机体坐标系 z 与 ``target_height`` 的偏差；配合 tanh 饱和，利于楼梯抬脚。
        仅在平面速度指令大于阈值时生效，与 ``feet_air_time`` 门控一致。
        """
        asset = self._get_robot_asset()
        asset_cfg = self._get_foot_asset_cfg()
        foot_ids = asset_cfg.body_ids
        foot_pos_w = asset.data.body_pos_w[:, foot_ids, :]
        root_pos = asset.data.root_pos_w[:, None, :]
        rel_w = foot_pos_w - root_pos
        q = asset.data.root_quat_w
        rel_b = _quat_rotate_inverse(q, rel_w)
        foot_z = rel_b[:, :, 2]
        err = foot_z - target_height
        penalty = torch.sum(torch.square(torch.tanh(err * tanh_mult)), dim=1)

        cmd = self.env.command_manager.get_command(command_name)
        moving = torch.norm(cmd[:, :2], dim=1) > 0.1
        return penalty * moving.float()

    def _reward_correct_base_height(self, target_height: float = 0.32):
        """Penalize vertical offset of root from desired height above env origin.

        基座高度相对环境原点 z 与 ``target_height`` 的平方误差；无射线时用简化近似。
        """
        asset = self._get_robot_asset()
        z = asset.data.root_pos_w[:, 2]
        if hasattr(self.env.scene, "env_origins") and self.env.scene.env_origins is not None:
            z = z - self.env.scene.env_origins[:, 2]
        return torch.square(z - target_height)

    def _reward_energy(self):
        """Mechanical power proxy: sum |tau * qdot| over joints (Isaac-style energy).

        优先 ``applied_torque`` / ``joint_effort`` × ``joint_vel``；缺失则退化为关节速度平方和。
        """
        asset = self._get_robot_asset()
        qd = asset.data.joint_vel
        if hasattr(asset.data, "applied_torque") and asset.data.applied_torque is not None:
            tau = asset.data.applied_torque
        elif hasattr(asset.data, "joint_effort") and asset.data.joint_effort is not None:
            tau = asset.data.joint_effort
        else:
            return torch.sum(torch.square(qd), dim=1)
        return torch.sum(torch.abs(tau * qd), dim=1)

    def _reward_action_smoothness(self):
        """Second-order penalty on actions: ||a_t - 2 a_{t-1} + a_{t-2}||^2.

        与 Isaac Lab / NP3O ``action_smoothness`` 一致；首步及 episode 首帧置零历史。
        """
        cur = self.env.action_manager.action
        n_env = cur.shape[0]
        if not hasattr(self, "_diy_smooth_prev") or self._diy_smooth_prev.shape != cur.shape:
            z = torch.zeros_like(cur)
            self._diy_smooth_prev = z.clone()
            self._diy_smooth_prev2 = z.clone()
            self._diy_smooth_ready = False

        if not self._diy_smooth_ready:
            self._diy_smooth_prev = cur.detach().clone()
            self._diy_smooth_prev2 = cur.detach().clone()
            self._diy_smooth_ready = True
            return torch.zeros(n_env, device=cur.device, dtype=cur.dtype)

        if hasattr(self.env, "episode_length_buf"):
            ep = self.env.episode_length_buf.squeeze(-1) if self.env.episode_length_buf.ndim > 1 else self.env.episode_length_buf
            first = ep <= 1
            self._diy_smooth_prev[first] = cur[first].detach()
            self._diy_smooth_prev2[first] = cur[first].detach()

        penalty = torch.sum(torch.square(cur - 2.0 * self._diy_smooth_prev + self._diy_smooth_prev2), dim=1)
        self._diy_smooth_prev2 = self._diy_smooth_prev.detach()
        self._diy_smooth_prev = cur.detach()
        return penalty

    def _reward_feet_contact_forces(self, max_contact_force: float = 100.0):
        """Penalize foot contact forces above ``max_contact_force`` (NP3O-style).

        对足部接触力范数超过阈值的超出部分求和，抑制跺脚与硬着陆。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        n = torch.norm(f, dim=-1)
        return torch.sum(torch.clip(n - max_contact_force, min=0.0), dim=1)

    # 注：collision_up 等价行为可由 `[rewards.undesired_contacts]`（基类提供）+
    # `[rewards.flat_orientation]` 共同覆盖；本文件不再额外定义 `_reward_collision_up`，
    # 避免依赖私有 helper，保持 reward_process 仅扩展 NP3O 中本仓库未提供的项。

    def _reward_forward_velocity(
            self,
            command_name: str = "base_velocity",
            max_speed: float = 1.0,
            cmd_threshold: float = 0.05,
            unconditional: bool = True,
    ):
        """Dense reward for actual forward speed in body frame.

        前向速度密集奖励：默认 ``unconditional=True``——**与 command 解耦**，无论评测端
        采到什么 command（cmd_x=0 / 负值 / 纯转向），策略都被持续激励向身体前向行进。
        这是适配"评测 command 不可控"的核心 shaping 信号：评分目标是前进距离，而本
        项是唯一保证"前进"成为内生偏好的密集奖励；`track_lin_vel_xy` 在 cmd=0 时给
        满分会把策略带回"站着"，必须用本项压住它。
        当 ``unconditional=False`` 时回退到旧行为（仅在 cmd_x > 阈值时发奖励）。
        """
        asset = self._get_robot_asset()
        fwd_vel = asset.data.root_lin_vel_b[:, 0]
        reward = torch.clamp(fwd_vel, min=0.0, max=max_speed)
        if unconditional:
            return reward
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > cmd_threshold).float()
        return reward * has_fwd_cmd

    def _reward_progress(self):
        """Reward gain in 2D distance from the spawn point (env origin) per step.

        距离 spawn 增量奖励：直接对治"在出生点附近转圈跑步"——本项只在机器人
        真正**远离 spawn** 时才给正奖励，原地画圈跑、来回往返都拿不到分。
        与 ``forward_velocity`` 的差异：后者是机体系前向速度，原地转圈跑也可以
        持续为正；本项基于世界系下"距离 spawn 的位移"，对应 standard 模式评分
        的 ``\\|pos_current - pos_spawn\\|`` 公式，是与评测目标最对齐的密集信号。

        实现：用上一步距离与当前距离的差。回合重置时清零，避免起步跳变。
        Mirrors the standard-mode scoring metric (||pos_current - pos_spawn||);
        directly penalizes "circling at spawn" because circular motion does not
        accumulate distance from the origin.
        """
        asset = self._get_robot_asset()
        pos = asset.data.root_pos_w[:, :2]
        if hasattr(self.env.scene, "env_origins") and self.env.scene.env_origins is not None:
            origin = self.env.scene.env_origins[:, :2]
        else:
            origin = torch.zeros_like(pos)
        current_dist = torch.norm(pos - origin, dim=1)

        prev = getattr(self, "_diy_progress_prev", None)
        if prev is None or prev.shape != current_dist.shape or prev.device != current_dist.device:
            self._diy_progress_prev = current_dist.clone()
            return torch.zeros_like(current_dist)

        delta = current_dist - self._diy_progress_prev

        # Reset on episode boundary so spawn-pos jump does not produce spurious reward.
        # 回合重置时清零，避免 spawn 切换造成距离跳变被误当作 progress。
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta = torch.where(reset_mask, torch.zeros_like(delta), delta)

        self._diy_progress_prev = current_dist.clone()

        # 仅奖励"远离 spawn"的位移；返回 m/step（典型 dt≈0.02s 时约 0.01 m/step）。
        # Reward only positive progress so back-and-forth motion does not accumulate.
        return torch.clamp(delta, min=0.0)

    def _reward_stall_penalty(self, vel_threshold: float = 0.15, progress_threshold: float = 0.002):
        """Penalize moving in body frame without making world-frame progress.

        停滞惩罚：当机体系有速度（在跑）但世界系位移增量接近零（没有远离 spawn）时
        给负奖励。直接对治两个失败模式：
        1. "出生点转圈"——机体前向速度为正但 progress≈0
        2. "台阶边卡住但腿在动"——有关节运动但不前进

        不惩罚真正静止的情况（vel < threshold），避免和起步/恢复阶段冲突。
        Penalizes "running but not advancing" — the circling and edge-stuck attractors.
        """
        asset = self._get_robot_asset()
        body_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        is_moving = (body_speed > vel_threshold).float()

        pos = asset.data.root_pos_w[:, :2]
        if hasattr(self.env.scene, "env_origins") and self.env.scene.env_origins is not None:
            origin = self.env.scene.env_origins[:, :2]
        else:
            origin = torch.zeros_like(pos)
        current_dist = torch.norm(pos - origin, dim=1)

        prev = getattr(self, "_diy_stall_prev_dist", None)
        if prev is None or prev.shape != current_dist.shape or prev.device != current_dist.device:
            self._diy_stall_prev_dist = current_dist.clone()
            return torch.zeros_like(current_dist)

        delta = current_dist - self._diy_stall_prev_dist

        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta = torch.where(reset_mask, torch.zeros_like(delta), delta)

        self._diy_stall_prev_dist = current_dist.clone()

        no_progress = (delta < progress_threshold).float()
        return is_moving * no_progress

    # --- Termination penalty / 终止惩罚 ---
    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out AND NOT goal-reached).

        惩罚真正的失败（被终止且非超时截断且非到达目标），对应经典 legged_gym 的
        `reset_buf * ~time_out_buf` 逻辑，同时排除导航成功终止。

        Returns:
            Float tensor (num_envs,): 1.0 for real failures, 0.0 otherwise.
            浮点张量 (num_envs,)：真实失败返回 1.0，其他情况返回 0.0。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs

        # 排除 goal_reached（导航成功不应被惩罚）
        if "goal_reached" in term_mgr.active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done

        return failure.float()
