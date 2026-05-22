#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

import toml


# Valid task types (Isaac Lab native config format)
# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    Base class for training stage configuration.
    训练阶段配置基类。

    Subclass this and override fields to define a new training stage.
    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- Stage identity
    # 阶段标识 ---
    name = ""
    task_type = "standard"

    # --- Model architecture dimensions (Isaac Lab Unitree-Go2-Velocity constants)
    # These are fixed by the Isaac Lab task definition and the network structure;
    # users are not expected to change them. Do NOT move them into user TOML.
    # 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12  # Go2 joint action dim / Go2 关节动作维度
    num_proprio_obs = 45  # proprioceptive obs dim / 本体感知观测维度
    num_scan = 256  # 16x16 height-scan dim / 16x16 高度扫描维度
    num_goal_obs = 0  # optional goal observation dim / 可选 goal 观测维度
    num_nav_scan_obs = 0  # optional raw nav_scanner dim for 1D CNN / 可选 nav_scanner 原始序列维度
    num_critic_observations = 316  # proprio(45) + scan(256) + privileged(15)

    # --- Model architecture
    # 模型架构 ---
    model_class = "ActorCritic"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # --- Training hyperparameters
    # 训练超参数 ---
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    value_loss_coef = 1.0
    cost_value_loss_coef = 1.0
    cost_violation_loss_coef = 1.0
    entropy_coef = 0.01
    desired_kl = 0.01
    schedule = "adaptive"

    # --- Constrained PPO cost settings
    # 受约束 PPO cost 配置 ---
    # 默认对齐 NP3O Go2ConstraintHimRoughCfg：3 个具名 cost(dof_pos_limits/torque_limit/dof_vel_limits)。
    # Defaults align with NP3O 3 named costs.
    num_costs = 3
    cost_names = ["dof_pos_limits", "torque_limit", "dof_vel_limits"]
    cost_limit = 0.0  # Deprecated in algorithm path; kept only as scalar shorthand for cost_d_values.
    # cost_d_values 与 (1-gamma) * cost_return 同尺度；参考 NP3O 默认 0.0，
    # 每步 cost 期望已乘以 cost_scale，相当于 NP3O 中的 cost * dt。
    # cost_d_values shares scale with (1-gamma) * cost_return; 0.0 mirrors NP3O default
    # because per-step costs are pre-scaled by cost_scale (≈ control dt).
    cost_d_values = [0.0, 0.0, 0.0]
    # cost_scale ≈ NP3O `cost * dt`(0.02) 的等价系数,使派生 cost 与 d_values 同尺度。
    # cost_scale matches NP3O's `cost * dt` (0.02) so derived costs stay on the same scale as d_values.
    cost_scale = 0.02
    # NP3O 软极限阈值：超过此比例的关节位置/速度/扭矩才计入 cost。
    # NP3O soft-limit ratios: only joint pos/vel/torque beyond these ratios contribute to cost.
    soft_dof_pos_limit = 0.9
    soft_dof_vel_limit = 1.0
    soft_torque_limit = 1.0
    initial_penalty_weight = 0.1
    # penalty_mode: "scheduled" 为 NP3O 固定增长(默认), "adaptive" 为反馈式更新。
    # penalty_mode: "scheduled" matches NP3O fixed growth (default); "adaptive" keeps prior feedback law.
    penalty_mode = "scheduled"
    penalty_growth_rate = 1.0004
    penalty_lr = 0.05
    penalty_decay = 1.0
    penalty_max = 1.0
    # timeout_cost_bootstrap: "value" 用 cost_values 自举(类奖励路径,推荐),
    # "self" 复刻 NP3O 原版 costs += gamma * costs * timeout 行为。
    # timeout_cost_bootstrap: "value" bootstraps with cost_values (recommended, mirrors reward path),
    # "self" replicates the original NP3O `costs += gamma * costs * timeout` behavior.
    timeout_cost_bootstrap = "value"
    require_explicit_costs = False
    termination_as_cost = False  # 多代价路径下默认禁用单代价兜底；如需开启请在 stage override。
    # 启用从 env 状态直接计算 NP3O 三具名 cost。
    # Enable env-native computation of NP3O 3 named costs from robot state.
    use_native_costs = True

    # --- HIM-lite history encoder (NP3O actor history) ---
    # 历史编码器（NP3O actor 端历史观测，HIM-lite，无 contrastive loss）
    # 注：开启后 obs 维度增加 ``history_len * num_proprio_obs``，与 NP3O 行为对齐。
    # When enabled, policy obs is augmented by ``history_len * num_proprio_obs`` dims.
    use_history_encoder = True
    history_len = 10  # NP3O Go2ConstraintHimRoughCfg.env.history_len
    history_latent_dim = 16
    history_encoder_dims = [128, 64]
    nav_scan_latent_dim = 32
    nav_scan_cnn_channels = [16, 32]

    # --- Saving
    # 保存 ---
    model_save_interval = 100


class NavConfig(StageConfig):
    """
    Stage: nav — end-to-end navigation on track terrain.
    阶段：nav —— track 地形端到端导航训练。

    Hot-loads from a LocomotionConfig checkpoint; goal obs stay explicit while
    raw nav_scanner rays are consumed by the actor-side 1D CNN instead of
    hand-crafted nav sectors.
    从 LocomotionConfig checkpoint 热加载；goal 观测仍显式拼接，raw nav_scanner
    由 actor 端 1D CNN 消费，不再拼接手工 nav sector 特征。
    """

    name = "nav"
    task_type = "track"

    # goal obs (4): [rel_x, rel_y, rel_dist, rel_yaw]
    # Raw nav_scanner rays are appended to policy obs and consumed by the
    # actor-side 1D CNN. No hand-crafted nav_sector features are appended.
    num_goal_obs = 4
    num_nav_scan_obs = 32
    nav_scan_latent_dim = 32
    nav_scan_cnn_channels = [16, 32]
    # critic: 316 (base) + 4 (goal) + 32 (raw nav scan) = 352
    # agent.py adds goal/raw nav dims automatically.
    num_critic_observations = 316

    # Fine-tune learning rate (lower than locomotion stage)
    lr = 1e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48


class LocomotionConfig(StageConfig):
    """
    Stage: locomotion — learn stable walking on mixed terrain.
    阶段：locomotion —— 在混合地形上学习稳定行走。
    """

    name = "locomotion"
    task_type = "standard"


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。
    """

    # Switch stage by changing CURRENT
    # 通过修改 CURRENT 切换阶段
    CURRENT = NavConfig

    @staticmethod
    def resolve_cost_thresholds(stage):
        """Resolve algorithm-facing cost thresholds from stage config."""
        cost_d_values = getattr(stage, "cost_d_values", None)
        if cost_d_values is None or len(cost_d_values) == 0:
            cost_limit = float(getattr(stage, "cost_limit", 0.0))
            return [cost_limit] * int(stage.num_costs)

        resolved = [float(value) for value in cost_d_values]
        if len(resolved) == 1 and int(stage.num_costs) > 1:
            resolved = resolved * int(stage.num_costs)
        if len(resolved) != int(stage.num_costs):
            raise ValueError(
                f"Stage '{stage.name}' expected {stage.num_costs} cost_d_values, got {len(resolved)}"
            )
        return resolved

    @staticmethod
    def load_conf(logger):
        """
        Load user configuration file based on current stage.
        根据当前阶段加载用户配置文件。

        Args:
            logger: logger instance | 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. " f"Only {_VALID_TASKS} are supported."
            )

        # Determine if it's evaluation mode
        # 判断是否为评估模式
        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = f"tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_diy/conf/train_env_conf_{task_type}_{stage.name}.toml"

        usr_conf = _load_conf(usr_conf_file, logger)

        if usr_conf is None:
            error_msg = f"usr_conf is None, please check {usr_conf_file}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")

        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    递归将 override 字典合并到 base 字典中（override 优先）。

    Args:
        base: Base config dictionary | 基础配置字典
        override: Override config dictionary | 覆盖配置字典

    Returns:
        dict: Merged config dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    """
    Load config: first load base TOML, then deep-merge user TOML on top.
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base files provide model architecture dimensions (num_actions, num_proprio_obs, etc.)
    so user configs only need business-tunable parameters.
    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。

    Args:
        conf_file: Path to the user TOML config file | 用户配置文件路径
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # Determine base file by mode (eval or train)
    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # Load base config (optional — missing base is not fatal)
    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # Load user config
    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config
