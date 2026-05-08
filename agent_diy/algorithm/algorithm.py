#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import RolloutStorage


class Algorithm:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        logger: Any = None,
        monitor: Any = None,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        cost_value_loss_coef: float = 1.0,
        cost_violation_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_value_loss: bool = True,
        num_mini_batches: int = 4,
        num_learning_epochs: int = 5,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        penalty_lr: float = 0.05,
        penalty_decay: float = 1.0,
        penalty_max: float = 1.0,
        penalty_mode: str = "scheduled",
        penalty_growth_rate: float = 1.0004,
        **kwargs,
    ):
        self.device = device
        self.actor_critic = model
        self.optimizer = optimizer
        self.logger = logger
        self.monitor = monitor

        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_violation_loss_coef = cost_violation_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_value_loss = normalize_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.penalty_lr = penalty_lr
        self.penalty_decay = penalty_decay
        self.penalty_max = penalty_max
        if penalty_mode not in {"scheduled", "adaptive"}:
            raise ValueError(f"Unknown penalty_mode: {penalty_mode}")
        self.penalty_mode = penalty_mode
        self.penalty_growth_rate = float(penalty_growth_rate)
        self.penalty_iter = 0

        stage = Config.CURRENT
        self.num_costs = stage.num_costs
        self.cost_d_values = Config.resolve_cost_thresholds(stage)
        self.min_std = torch.tensor(stage.min_normalized_std, device=device)
        self.k_value = torch.full((self.num_costs,), stage.initial_penalty_weight, device=device, dtype=torch.float32)

        self.train_step = 0
        self.last_report_monitor_time = 0
        self.storage = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        device = device or self.device
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            num_costs=self.num_costs,
            cost_d_values=self.cost_d_values,
            device=device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> tuple:
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            actions = self.actor_critic.act(obs)
            values = self.actor_critic.evaluate(critic_obs)
            cost_values = self.actor_critic.evaluate_cost(critic_obs)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            action_mean = self.actor_critic.action_mean.detach()
            action_std = self.actor_critic.action_std.detach()

        return actions, values, cost_values, log_probs, action_mean, action_std, obs.detach(), critic_obs.detach()

    def learn(self, list_sample_data=None):
        mean_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "cost_value_loss": 0.0,
            "entropy_loss": 0.0,
            "violation_loss": 0.0,
            "mean_cost": 0.0,
            "cost_return_mean": 0.0,
            "raw_mean_violation": 0.0,
            "positive_mean_violation": 0.0,
            "k_value_mean": 0.0,
            "k_value_max": 0.0,
            "total_loss": 0.0,
        }
        mean_penalty_signal = torch.zeros(self.num_costs, device=self.device, dtype=torch.float32)
        applied_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample_idx, sample in enumerate(generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                target_cost_values_batch,
                cost_advantages_batch,
                cost_returns_batch,
                cost_violation_batch,
                cost_d_values_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            entropy_batch = self.actor_critic.entropy
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std

            self._update_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

            surrogate_loss = self._compute_surrogate_loss(
                actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
            )
            value_loss = self._compute_value_loss(value_batch, returns_batch, target_values_batch)
            cost_value_loss = self._compute_value_loss(cost_value_batch, cost_returns_batch, target_cost_values_batch)
            violation_loss, violation_stats = self._compute_violation_loss(
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                cost_advantages_batch,
                cost_violation_batch,
            )

            entropy_loss = -self.entropy_coef * entropy_batch.mean()
            total_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                + self.cost_violation_loss_coef * violation_loss
                + entropy_loss
            )

            if not torch.isfinite(total_loss):
                if self.logger:
                    self.logger.warning(
                        f"[DIY] NaN/Inf loss detected at step {self.train_step}, mini-batch {sample_idx}. Skipping update."
                    )
                continue

            self.optimizer.zero_grad()
            total_loss.backward()

            grad_finite = True
            for p in self.actor_critic.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                if self.logger:
                    self.logger.warning(
                        f"[DIY] NaN/Inf gradient detected at step {self.train_step}, mini-batch {sample_idx}. Skipping update."
                    )
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_action_std()

            penalty_signal = violation_stats["penalty_signal"]
            mean_penalty_signal += penalty_signal.detach()
            applied_updates += 1

            batch_metrics = {
                "policy_loss": surrogate_loss.item(),
                "value_loss": value_loss.item(),
                "cost_value_loss": cost_value_loss.item(),
                "entropy_loss": (-entropy_loss).item(),
                "violation_loss": violation_loss.item(),
                "mean_cost": cost_advantages_batch.mean().item(),
                "cost_return_mean": cost_returns_batch.mean().item(),
                "raw_mean_violation": violation_stats["raw_mean_violation"].item(),
                "positive_mean_violation": violation_stats["positive_mean_violation"].item(),
                "total_loss": total_loss.item(),
            }
            for key, value in batch_metrics.items():
                mean_metrics[key] += 0.0 if value != value else value

        num_updates = max(applied_updates, 1)
        for key in mean_metrics:
            mean_metrics[key] /= num_updates

        if self.penalty_mode == "scheduled":
            self._update_penalty_scheduled()
        elif applied_updates > 0:
            self._update_penalty(mean_penalty_signal / applied_updates)
        mean_metrics["k_value_mean"] = self.k_value.mean().item()
        mean_metrics["k_value_max"] = self.k_value.max().item()

        self._report_training_metrics(mean_metrics)
        self.train_step += 1

        return (
            mean_metrics["policy_loss"],
            mean_metrics["value_loss"],
            mean_metrics["cost_value_loss"],
            mean_metrics["entropy_loss"],
            mean_metrics["violation_loss"],
        )

    def _clamp_action_std(self):
        if hasattr(self.actor_critic, "std") and self.min_std is not None:
            max_std_t = torch.full_like(self.actor_critic.std.data, 1.0e6)
            safe_std = torch.nan_to_num(
                self.actor_critic.std.data,
                nan=1.0,
                posinf=1.0e6,
                neginf=0.0,
            )
            self.actor_critic.std.data.copy_(torch.clamp(safe_std, min=self.min_std, max=max_std_t))

    def _compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_cost_surrogate_loss(self, ratio, cost_advantages_batch):
        surrogate = cost_advantages_batch * ratio.unsqueeze(-1)
        surrogate_clipped = cost_advantages_batch * torch.clamp(
            ratio.unsqueeze(-1), 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped)

    def _compute_value_loss(self, value_batch, returns_batch, target_values_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        if self.normalize_value_loss:
            value_loss = value_loss / (returns_batch.var() + 1e-8)
        return value_loss

    def _compute_violation_loss(
        self,
        actions_log_prob_batch,
        old_actions_log_prob_batch,
        cost_advantages_batch,
        cost_violation_batch,
    ):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        cost_surrogate = self._compute_cost_surrogate_loss(ratio, cost_advantages_batch)
        per_cost_surrogate = cost_surrogate.mean(dim=0)
        mean_violation = cost_violation_batch.mean(dim=0)
        raw_violation = per_cost_surrogate + mean_violation
        positive_violation = torch.relu(raw_violation)
        violation_loss = torch.sum(self.k_value * positive_violation)
        return violation_loss, {
            "raw_mean_violation": raw_violation.mean(),
            "positive_mean_violation": positive_violation.mean(),
            "penalty_signal": raw_violation,
        }

    def _update_penalty(self, mean_penalty_signal):
        bounded_signal = torch.nan_to_num(mean_penalty_signal, nan=0.0, posinf=self.penalty_max, neginf=-self.penalty_max)
        next_k_value = self.k_value * self.penalty_decay + self.penalty_lr * bounded_signal
        self.k_value = torch.clamp(next_k_value, min=0.0, max=self.penalty_max)

    def _update_penalty_scheduled(self):
        """NP3O-style penalty schedule: k *= growth_rate ** iter, capped at penalty_max.

        参考 ``LocomotionWithNP3O/algorithm/np3o.py::update_k_value``：每个外层迭代调用一次，
        以 ``growth_rate ** iter`` 放大 k 值，并通过 penalty_max 上限截断。
        """
        self.penalty_iter += 1
        scale = torch.tensor(self.penalty_growth_rate ** self.penalty_iter, device=self.device, dtype=self.k_value.dtype)
        max_t = torch.full_like(self.k_value, self.penalty_max)
        self.k_value = torch.minimum(max_t, self.k_value * scale)

    def _update_learning_rate(self, mu_batch, sigma_batch, old_mu_batch, old_sigma_batch):
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / (old_sigma_batch + 1.0e-5) + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _report_training_metrics(self, metrics):
        now = time.time()
        if self.monitor and now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "cost_value_loss": metrics["cost_value_loss"],
                "entropy_loss": metrics["entropy_loss"],
                "violation_loss": metrics["violation_loss"],
                "total_loss": metrics["total_loss"],
                "mean_cost": metrics["mean_cost"],
                "cost_return_mean": metrics["cost_return_mean"],
                "raw_mean_violation": metrics["raw_mean_violation"],
                "positive_mean_violation": metrics["positive_mean_violation"],
                "k_value_mean": metrics["k_value_mean"],
                "k_value_max": metrics["k_value_max"],
                "learning_rate": self.learning_rate,
            }
            self.monitor.put_data({"diy": monitor_data})
            self.last_report_monitor_time = now

        if self.logger:
            self.logger.info(
                "DIY update: "
                f"policy_loss={metrics['policy_loss']:.6f}, "
                f"value_loss={metrics['value_loss']:.6f}, "
                f"cost_value_loss={metrics['cost_value_loss']:.6f}, "
                f"violation_loss={metrics['violation_loss']:.6f}, "
                f"entropy_loss={metrics['entropy_loss']:.6f}, "
                f"mean_cost={metrics['mean_cost']:.6f}, "
                f"cost_return_mean={metrics['cost_return_mean']:.6f}, "
                f"raw_mean_violation={metrics['raw_mean_violation']:.6f}, "
                f"positive_mean_violation={metrics['positive_mean_violation']:.6f}, "
                f"k_value_mean={metrics['k_value_mean']:.6f}, "
                f"k_value_max={metrics['k_value_max']:.6f}, "
                f"total_loss={metrics['total_loss']:.6f}"
            )
