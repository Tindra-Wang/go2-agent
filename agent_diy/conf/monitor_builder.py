#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    #
    # Note: this builder only keeps metrics that are unique to algorithm training
    # (loss-series metrics, episode_reward, track traversal progress).
    # Other reward_* metrics (velocity tracking, posture, gait, navigation rewards, etc.)
    # are rendered by the project-side tools/conf/monitor_default.yaml and
    # tools/conf/monitor_default_track.yaml, and are no longer redefined here,
    # to avoid duplicated panels with the same name in the final merged dashboard.
    #
    # 注意：本 builder 只保留算法训练独有的指标（loss 类、episode_reward、赛道穿越进度）。
    # 其余 reward_* 指标（速度跟踪、姿态、步态、导航奖励等）由项目侧
    # tools/conf/monitor_default.yaml 与 tools/conf/monitor_default_track.yaml 负责展示，
    # 这里不再重复定义，避免最终合并后的监控面板出现同名指标重复绘制。
    #
    # Extra DIY-only panels: constrained-PPO / NP3O-style losses and rollout stats from
    # ``agent_diy/algorithm/algorithm.py`` and ``agent_diy/workflow/train_workflow.py``.
    # 额外 DIY 面板：约束 PPO 损失与 workflow 滚动统计，便于观察收敛与安全约束。

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 1: Algorithm training loss metrics (unique to this builder, not covered by yaml)
        # Group 1: 算法训练损失指标（本 builder 独有，yaml 未覆盖）
        # ==============================================================
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="代价价值损失",
            name_en="cost_value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="cost_value_loss",
            expr="avg(cost_value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="约束违反损失",
            name_en="violation_loss",
            type="line",
        )
        .add_metric(
            metrics_name="violation_loss",
            expr="avg(violation_loss{})",
        )
        .end_panel()
        .add_panel(
            name="学习率",
            name_en="learning_rate",
            type="line",
        )
        .add_metric(
            metrics_name="learning_rate",
            expr="avg(learning_rate{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 2: Constrained PPO / cost signals (algorithm.learn)
        # Group 2: 约束与代价信号（算法更新）
        # ==============================================================
        .add_group(
            group_name="约束与代价",
            group_name_en="constraint_cost",
        )
        .add_panel(
            name="代价优势均值",
            name_en="mean_cost",
            type="line",
        )
        .add_metric(
            metrics_name="mean_cost",
            expr="avg(mean_cost{})",
        )
        .end_panel()
        .add_panel(
            name="代价回报均值",
            name_en="cost_return_mean",
            type="line",
        )
        .add_metric(
            metrics_name="cost_return_mean",
            expr="avg(cost_return_mean{})",
        )
        .end_panel()
        .add_panel(
            name="原始违反均值",
            name_en="raw_mean_violation",
            type="line",
        )
        .add_metric(
            metrics_name="raw_mean_violation",
            expr="avg(raw_mean_violation{})",
        )
        .end_panel()
        .add_panel(
            name="正违反均值",
            name_en="positive_mean_violation",
            type="line",
        )
        .add_metric(
            metrics_name="positive_mean_violation",
            expr="avg(positive_mean_violation{})",
        )
        .end_panel()
        .add_panel(
            name="惩罚系数均值",
            name_en="k_value_mean",
            type="line",
        )
        .add_metric(
            metrics_name="k_value_mean",
            expr="avg(k_value_mean{})",
        )
        .end_panel()
        .add_panel(
            name="惩罚系数最大值",
            name_en="k_value_max",
            type="line",
        )
        .add_metric(
            metrics_name="k_value_max",
            expr="avg(k_value_max{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 3: Rollout / episode aggregates (train_workflow, ~60s report)
        # Group 3: 滚动与回合汇总（workflow 约每分钟上报）
        # ==============================================================
        .add_group(
            group_name="滚动统计",
            group_name_en="rollout",
        )
        .add_panel(
            name="训练回合计数",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="avg(episode_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="单局奖励合计",
            name_en="episode_reward",
            type="line",
        )
        .add_metric(
            metrics_name="episode_reward",
            expr="avg(episode_reward{})",
        )
        .end_panel()
        .add_panel(
            name="步级奖励均值",
            name_en="reward_mean",
            type="line",
        )
        .add_metric(
            metrics_name="reward_mean",
            expr="avg(reward_mean{})",
        )
        .end_panel()
        .add_panel(
            name="步级奖励标准差",
            name_en="reward_std",
            type="line",
        )
        .add_metric(
            metrics_name="reward_std",
            expr="avg(reward_std{})",
        )
        .end_panel()
        .add_panel(
            name="步级代价均值",
            name_en="cost_mean",
            type="line",
        )
        .add_metric(
            metrics_name="cost_mean",
            expr="avg(cost_mean{})",
        )
        .end_panel()
        .add_panel(
            name="代价违反均值",
            name_en="violation_mean",
            type="line",
        )
        .add_metric(
            metrics_name="violation_mean",
            expr="avg(violation_mean{})",
        )
        .end_panel()
        .add_panel(
            name="代价来源编码",
            name_en="cost_source_id",
            type="line",
        )
        .add_metric(
            metrics_name="cost_source_id",
            expr="avg(cost_source_id{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 4: Reward metrics (examples, players can add more reward panels as needed)
        # Group 4: Reward 指标（示例，选手可按需补充更多 reward 面板）
        # ==============================================================
        .add_group(group_name="奖励指标", group_name_en="reward")
        .add_panel(name="线速度跟踪奖励", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy",
                        expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
