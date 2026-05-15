#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import math
import os

import torch


def _get_algo_config():
    """Dynamically import Config class based on KAIWU_ALGORITHM env var.

    根据 KAIWU_ALGORITHM 环境变量动态加载对应算法的 Config 类，
    避免 tools/base_env 硬编码依赖 agent_ppo。支持 agent_ppo / agent_diy。

    Returns:
        Config class from agent_{algo}.conf.conf
    """
    algo = os.environ.get("KAIWU_ALGORITHM", "ppo")
    module_name = f"agent_{algo}.conf.conf"
    import importlib

    module = importlib.import_module(module_name)
    return module.Config


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    将向量从世界坐标系旋转到本体坐标系。

    Args:
        q: Quaternion (w, x, y, z) of shape (..., 4)
        v: Vector of shape (..., 3)

    Returns:
        Rotated vector of shape (..., 3)
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion.

    从四元数中提取偏航角（绕 Z 轴旋转角度）。

    Args:
        quat: Quaternion (w, x, y, z) of shape (..., 4)

    Returns:
        Yaw angle in radians of shape (...)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi].

    将角度归一化到 [-π, π] 范围内。

    Args:
        angles: Angles in radians

    Returns:
        Wrapped angles in [-pi, pi]
    """
    return (angles + math.pi) % (2 * math.pi) - math.pi


def extract_obs_without_command(obs):
    """
    Extract observation without command information
    提取不包含命令信息的观测数据

    Isaac Lab observation structure: [ang_vel(3), gravity(3), commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]
    Isaac Lab 观测结构：[ang_vel(3), gravity(3), commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]

    Returns: [ang_vel(3), gravity(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]
    返回：[ang_vel(3), gravity(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]

    Args:
        obs: observation tensor of shape (batch_size, 45) for Isaac Lab
        obs: 形状为 (batch_size, 45) 的观测张量（Isaac Lab）

    Returns:
        observation without command, shape (batch_size, 42)
        不包含命令的观测，形状 (batch_size, 42)
    """
    obs_dim = obs.shape[-1]

    if obs_dim == 45:
        # Isaac Lab format: [ang_vel(3), gravity(3), commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]
        return torch.concat(
            (
                obs[:, :6],  # ang_vel(3) + gravity(3)
                obs[:, 9:],  # joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)
            ),
            dim=1,
        )
    else:
        # Legacy IsaacGym format: [lin_vel(3), ang_vel(3), gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12)]
        return torch.concat(
            (
                obs[:, 3:9],  # ang_vel(3) + gravity(3)
                obs[:, 12:],  # dof_pos(12) + dof_vel(12) + actions(12)
            ),
            dim=1,
        )


def update_trajectory_history(trajectory_history, new_obs):
    """
    Update trajectory history with new observation (rolling window)
    使用新观测更新轨迹历史（滚动窗口）

    Args:
        trajectory_history: history tensor of shape (batch_size, history_length, obs_dim)
        trajectory_history: 形状为 (batch_size, history_length, obs_dim) 的历史张量
        new_obs: new observation of shape (batch_size, obs_dim)
        new_obs: 形状为 (batch_size, obs_dim) 的新观测

    Returns:
        updated trajectory history
        更新后的轨迹历史
    """
    return torch.concat(
        (trajectory_history[:, 1:], new_obs.unsqueeze(1)),
        dim=1,
    )


class ObservationProcess:
    """
    Observation processor
    观测处理器
    """

    target_group: str | None = None
    term_name: str = "custom_obs"

    def __init__(self, env=None):
        """
        Initialize observation processor
        初始化观测处理器

        Args:
            env: Isaac Lab 环境实例，可在 bridge wrapper 中延迟绑定。
        """
        self.env = env
        self._default_compute_observations = self._missing_default_compute_observations
        self._num_goal_obs = None

    def _missing_default_compute_observations(self):
        raise RuntimeError(
            "默认观测计算回调尚未注入。请通过 ObservationBridge 调用 `process()`，" "或在测试中手动设置 `_default_compute_observations`。"
        )

    def _get_robot(self):
        """获取 Isaac Lab 场景中的 robot 资产。"""
        if self.env is None or not hasattr(self.env, "scene"):
            raise RuntimeError("当前 observation process 尚未绑定有效 env.scene，无法读取自定义观测。")
        return self.env.scene["robot"]

    def _get_num_goal_obs(self) -> int:
        """Read num_goal_obs from Config.CURRENT (stage config). Cached."""
        cached = getattr(self, "_num_goal_obs", None)
        if cached is not None:
            return cached

        Config = _get_algo_config()

        num_goal_obs = Config.CURRENT.num_goal_obs

        self._num_goal_obs = num_goal_obs
        return num_goal_obs

    def default_observation(self):
        """返回被覆盖前的原始 Isaac Lab observation group。"""
        return self._default_compute_observations()

    def base_lin_vel(self):
        """类比 `mdp.observations.base_lin_vel()`：读取根坐标系线速度。"""
        return self._get_robot().data.root_lin_vel_b

    def base_speed_xy(self):
        """基于 `base_lin_vel` 派生水平面速度模长。"""
        return torch.linalg.vector_norm(self.base_lin_vel()[:, :2], dim=-1, keepdim=True)

    def height_scan(
            self, sensor_name: str = "height_scanner", offset: float = 0.5, clip_range: tuple | None = (-1.0, 5.0)
    ):
        """从指定 RayCaster 传感器获取高度扫描观测。

        类比 `mdp.observations.height_scan()`：读取射线投射传感器数据，
        计算传感器高度与地面命中点之间的高度差。

        Retrieve height scan observation from a named RayCaster sensor.

        Args:
            sensor_name: 场景中 RayCaster 传感器的注册名称，对应 SceneCfg 中的属性名。
                         默认 "height_scanner"（3m×3m 前方扫描），
                         也可传 "base_height_scanner"（1.6m×1.0m 基座扫描）。
            offset: 从高度差中减去的偏移量，默认 0.5m（与 Isaac Lab mdp.height_scan 一致）。
            clip_range: 裁剪范围 (min, max)，默认 (-1.0, 5.0)。传 None 不裁剪。

        Returns:
            torch.Tensor: shape (num_envs, num_rays) 的高度扫描数据。
                          每个值表示传感器位置相对于地面命中点的高度差减去偏移量。
        """
        if self.env is None or not hasattr(self.env, "scene"):
            raise RuntimeError("当前 observation process 尚未绑定有效 env.scene，无法读取传感器数据。")

        sensors = self.env.scene.sensors
        if sensor_name not in sensors:
            available = list(sensors.keys())
            raise KeyError(f"传感器 '{sensor_name}' 不存在于场景中。可用传感器: {available}")

        sensor = sensors[sensor_name]
        # height = sensor_z - hit_point_z - offset
        scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

        if clip_range is not None:
            scan = scan.clip(clip_range[0], clip_range[1])

        return scan

    def concatenate_terms(self, *terms):
        """按 Isaac observation group 的思路拼接多个 term 输出。"""
        return torch.cat(terms, dim=-1)

    def process(self):
        """
        [Optional] User-defined observation computation function
        【可选】用户自定义 observation 计算函数

        默认实现直接返回被覆盖前的原始 Isaac Lab observation group。
        用户可以在子类/自定义实现里基于 `self._default_compute_observations()`
        做裁剪、拼接、历史堆叠等后处理。
        """
        return self._default_compute_observations()

    def create_bridge(self, target_group: str | None = None, term_name: str | None = None):
        """创建 ObservationBridge，将 `process()` 包装成 Isaac Lab observation group。"""
        from tools.base_env.observation_bridge import ObservationBridge

        resolved_group = target_group or self.target_group
        if resolved_group is None:
            raise ValueError("ObservationProcess 未设置默认 target_group，请显式传入 target_group。")

        return ObservationBridge(
            self,
            target_group=resolved_group,
            term_name=term_name or self.term_name,
        )


# ----------------------------------------------------------------------------
# Module-level helpers: keep env.goal_positions maintained as terrain infrastructure,
# independent of whether the user's observation_process includes the goal term.
#
# 模块级工具：将 env.goal_positions 作为"地形基础设施"独立维护，
# 不再依赖用户 observation_process 是否包含 goal 相关 term 才触发更新。
#
# Usage: base_env.step() calls `ensure_goal_positions_ready(env)` + `update_goal_positions(env)`
# every step when task_type == "track", so scorer / termination / reward can
# rely on env.goal_positions being up-to-date regardless of the policy's obs shape.
# ----------------------------------------------------------------------------


def ensure_goal_positions_ready(env) -> bool:
    """Ensure env.goal_positions buffer and TerrainExitManager are initialized.

    确保 env.goal_positions 缓冲区与 TerrainExitManager 已初始化完毕。

    Idempotent: safe to call every step. Returns True when the buffer and
    manager are ready (or already were), False when initialization failed.

    幂等操作：可以每步调用。成功返回 True，失败返回 False。

    Args:
        env: Isaac Lab unwrapped environment instance.

    Returns:
        True if env.goal_positions and env._terrain_exit_manager are usable.
        True 表示缓冲区与 manager 都已就绪；False 表示初始化过程中出错。
    """
    if env is None:
        return False

    # --- Step 1: allocate buffers if missing ---
    if not hasattr(env, "goal_positions") or env.goal_positions is None:
        try:
            robot = env.scene["robot"]
            num_envs = robot.data.root_pos_w.shape[0]
            device = robot.data.root_pos_w.device
        except Exception:
            return False
        env.goal_positions = torch.zeros(num_envs, 3, device=device)
        env.goal_yaw = torch.zeros(num_envs, device=device)
        env._goal_manager_initialized = False

    # --- Step 2: init TerrainExitManager once ---
    if getattr(env, "_goal_manager_initialized", False):
        return True

    try:
        from unitree_rl_lab.terrains import TerrainExitManager
        from unitree_rl_lab.terrains.terrain_exit_manager import get_terrain_generator_runtime_state

        if not hasattr(env, "scene") or not hasattr(env.scene, "terrain"):
            env._goal_manager_initialized = True
            return False

        terrain = env.scene.terrain
        if terrain is None:
            env._goal_manager_initialized = True
            return False

        terrain_generator_state = get_terrain_generator_runtime_state(terrain)
        terrain_generator_cfg = getattr(terrain_generator_state, "cfg", terrain_generator_state)

        device = env.goal_positions.device
        manager = TerrainExitManager(device=device)

        if terrain.cfg.terrain_type == "plane" or terrain_generator_state is None:
            manager.initialize_for_plane(env.num_envs)
        else:
            terrain_size = terrain_generator_cfg.size
            manager.initialize_from_terrain_generator(terrain_generator_state, terrain_size)

        env._terrain_exit_manager = manager
        env._goal_manager_initialized = True
        return True

    except ImportError:
        env._goal_manager_initialized = True
        return False
    except Exception as e:
        import logging

        logging.warning(f"初始化 TerrainExitManager 失败: {e}")
        env._goal_manager_initialized = True
        return False


def update_goal_positions(env) -> bool:
    """Refresh env.goal_positions based on robot's current pose / terrain cell.

    基于机器人当前位姿和所在地形格刷新 env.goal_positions。

    Semantics (same as the original observation_process version):
      - Plane terrain: goal = 10 m in front of robot (in robot yaw direction).
      - Track mode: goal = exit of the finish row for each env's parallel track.
      - Standard grid mode: goal = exit of the env's current (level, type) block.

    语义与原 observation_process 版本完全一致：
      - plane：机器人前方 10 m
      - track：该赛道终点段出口
      - standard grid：(level, type) 对应块的出口

    Args:
        env: Isaac Lab unwrapped environment instance.

    Returns:
        True on successful update, False if manager is missing / not usable.
        成功更新返回 True；manager 缺失或无法查询返回 False。
    """
    if env is None:
        return False
    if not hasattr(env, "_terrain_exit_manager") or env._terrain_exit_manager is None:
        return False
    if not hasattr(env, "goal_positions") or env.goal_positions is None:
        return False

    manager = env._terrain_exit_manager
    terrain = env.scene.terrain

    from unitree_rl_lab.terrains.terrain_exit_manager import get_terrain_generator_runtime_state

    terrain_generator_state = get_terrain_generator_runtime_state(terrain)
    try:
        if terrain.cfg.terrain_type == "plane" or terrain_generator_state is None:
            # Plane mode: goal = 10 m in front of robot
            # Plane 模式：目标 = 机器人正前方 10 m
            robot = env.scene["robot"]
            robot_pos = robot.data.root_pos_w
            robot_quat = robot.data.root_quat_w
            robot_yaw = yaw_from_quat(robot_quat)

            distance = 10.0
            env.goal_positions[:, 0] = robot_pos[:, 0] + distance * torch.cos(robot_yaw)
            env.goal_positions[:, 1] = robot_pos[:, 1] + distance * torch.sin(robot_yaw)
            env.goal_positions[:, 2] = robot_pos[:, 2]
            env.goal_yaw.fill_(0.0)
        elif manager.is_track_mode:
            # Track mode: goal = finish-row exit of the parallel track (col)
            # Track 模式：目标 = 该并行赛道（col）终点段出口
            robot = env.scene["robot"]
            robot_pos_x = robot.data.root_pos_w[:, 0]
            terrain_types = terrain.terrain_types

            goal_pos, goal_yaw = manager.get_track_goal_positions(robot_pos_x, terrain_types)
            env.goal_positions.copy_(goal_pos)
            env.goal_yaw.copy_(goal_yaw)
        else:
            # Standard grid mode: query by (level, type)
            # Standard grid 模式：按 (level, type) 查询
            terrain_levels = terrain.terrain_levels
            terrain_types = terrain.terrain_types

            goal_pos, goal_yaw = manager.get_goal_positions(terrain_levels, terrain_types)
            env.goal_positions.copy_(goal_pos)
            env.goal_yaw.copy_(goal_yaw)
        return True
    except Exception as e:
        import logging

        logging.warning(f"update_goal_positions 失败: {e}")
        return False
