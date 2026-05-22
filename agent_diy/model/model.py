#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from agent_diy.conf.conf import Config


def resolve_nn_activation(activation: str) -> nn.Module:
    activation_map = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    return activation_map[activation]


class HistoryEncoder(nn.Module):
    """HIM-lite history encoder.

    历史观测编码器（NP3O HIM 简化版，无 contrastive loss）：
    将 ``history_len`` 步 proprio 拼接展平后，经 MLP 压缩为 ``latent_dim`` 维潜变量，
    再与当前 proprio + scan 拼接送入 actor，行为上等价于 NP3O 中
    ``actor_student_backbone`` 的历史分支去掉对比学习头的版本。
    """

    def __init__(
        self,
        history_len: int,
        proprio_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation_fn: nn.Module,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = history_len * proprio_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(type(activation_fn)())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.history_len = history_len
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NavScanEncoder(nn.Module):
    """1D CNN encoder for raw nav_scanner rays.

    原始 nav_scanner 序列编码器：raw rays 经 1D CNN 压缩成 latent 后送入 actor。
    """

    def __init__(
        self,
        scan_dim: int,
        channels: list[int],
        latent_dim: int,
        activation_fn: nn.Module,
    ) -> None:
        super().__init__()
        if scan_dim <= 0:
            raise ValueError(f"scan_dim must be positive, got {scan_dim}")
        if not channels:
            raise ValueError("channels must contain at least one Conv1d output channel")

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            layers.append(type(activation_fn)())
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, latent_dim))
        layers.append(type(activation_fn)())

        self.cnn = nn.Sequential(*layers)
        self.scan_dim = scan_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.cnn(x.unsqueeze(1))


class Model(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs: int | None = None,
        num_critic_obs: int | None = None,
        num_actions: int | None = None,
        actor_hidden_dims: tuple[int] | list[int] | None = None,
        critic_hidden_dims: tuple[int] | list[int] | None = None,
        activation: str | None = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_costs: int | None = None,
        history_len: int = 0,
        proprio_dim: int | None = None,
        history_latent_dim: int = 16,
        history_encoder_dims: list[int] | tuple[int, ...] | None = None,
        num_goal_obs: int = 0,
        num_nav_scan_obs: int = 0,
        nav_scan_latent_dim: int = 16,
        nav_scan_cnn_channels: list[int] | tuple[int, ...] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super(Model, self).__init__()

        stage = Config.CURRENT
        self.num_obs = num_obs if num_obs is not None else stage.num_proprio_obs + stage.num_scan
        self.num_critic_obs = num_critic_obs if num_critic_obs is not None else stage.num_critic_observations
        self.num_actions = num_actions if num_actions is not None else stage.num_actions
        self.num_costs = num_costs if num_costs is not None else stage.num_costs
        self.num_goal_obs = int(num_goal_obs)
        self.num_nav_scan_obs = int(num_nav_scan_obs)
        self.nav_scan_latent_dim = int(nav_scan_latent_dim)
        self.critic_base_obs_dim = self.num_critic_obs - self.num_goal_obs - self.num_nav_scan_obs
        if self.critic_base_obs_dim <= 0:
            raise ValueError(
                f"Invalid critic obs layout: num_critic_obs={self.num_critic_obs}, "
                f"num_goal_obs={self.num_goal_obs}, num_nav_scan_obs={self.num_nav_scan_obs}"
            )

        actor_hidden_dims = actor_hidden_dims or stage.actor_hidden_dims
        critic_hidden_dims = critic_hidden_dims or stage.critic_hidden_dims
        activation_fn = resolve_nn_activation(activation or stage.activation)

        # History encoder (HIM-lite): obs 末尾追加 history_len*proprio_dim 维历史，
        # 编码为 history_latent_dim 维潜变量后与当前观测拼接送入 actor。
        # critic 仍使用未增广的 critic_obs（NP3O 中 critic 端为特权观测，无需历史）。
        self.history_len = int(history_len) if history_len else 0
        self.proprio_dim = int(proprio_dim) if proprio_dim is not None else int(stage.num_proprio_obs)
        self.history_latent_dim = int(history_latent_dim)

        history_total = self.history_len * self.proprio_dim
        if history_total > 0 and self.num_obs <= history_total:
            raise ValueError(
                f"num_obs ({self.num_obs}) must exceed history_len*proprio_dim ({history_total})"
            )
        self.base_obs_dim = self.num_obs - history_total
        self.core_obs_dim = self.base_obs_dim - self.num_goal_obs - self.num_nav_scan_obs
        if self.core_obs_dim <= 0:
            raise ValueError(
                f"Invalid obs layout: base_obs_dim={self.base_obs_dim}, "
                f"num_goal_obs={self.num_goal_obs}, num_nav_scan_obs={self.num_nav_scan_obs}"
            )
        if self.history_len > 0:
            encoder_dims = list(history_encoder_dims or stage.history_encoder_dims)
            self.history_encoder = HistoryEncoder(
                history_len=self.history_len,
                proprio_dim=self.proprio_dim,
                hidden_dims=encoder_dims,
                latent_dim=self.history_latent_dim,
                activation_fn=activation_fn,
            )
        else:
            self.history_encoder = None

        if self.num_nav_scan_obs > 0:
            self.nav_scan_encoder = NavScanEncoder(
                scan_dim=self.num_nav_scan_obs,
                channels=list(nav_scan_cnn_channels or stage.nav_scan_cnn_channels),
                latent_dim=self.nav_scan_latent_dim,
                activation_fn=activation_fn,
            )
            self.critic_nav_scan_encoder = NavScanEncoder(
                scan_dim=self.num_nav_scan_obs,
                channels=list(nav_scan_cnn_channels or stage.nav_scan_cnn_channels),
                latent_dim=self.nav_scan_latent_dim,
                activation_fn=activation_fn,
            )
            self.cost_nav_scan_encoder = NavScanEncoder(
                scan_dim=self.num_nav_scan_obs,
                channels=list(nav_scan_cnn_channels or stage.nav_scan_cnn_channels),
                latent_dim=self.nav_scan_latent_dim,
                activation_fn=activation_fn,
            )
            nav_actor_dim = self.nav_scan_latent_dim
            nav_critic_dim = self.nav_scan_latent_dim
        else:
            self.nav_scan_encoder = None
            self.critic_nav_scan_encoder = None
            self.cost_nav_scan_encoder = None
            nav_actor_dim = 0
            nav_critic_dim = 0

        # Actor input: [proprio+height_scan | history_latent | optional_extra | nav_scan_latent]
        # 旧输入列保持在前面，方便从不带 raw nav_scanner 的 checkpoint 部分热加载。
        history_actor_dim = self.history_latent_dim if self.history_encoder is not None else 0
        actor_input_dim = self.core_obs_dim + history_actor_dim + self.num_goal_obs + nav_actor_dim
        critic_input_dim = self.critic_base_obs_dim + self.num_goal_obs + nav_critic_dim

        self.actor = self._build_mlp(actor_input_dim, actor_hidden_dims, self.num_actions, activation_fn)
        self.critic = self._build_critic(critic_input_dim, critic_hidden_dims, 1, activation_fn)
        self.cost_critic = self._build_critic(critic_input_dim, critic_hidden_dims, self.num_costs, activation_fn)

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int] | list[int], output_dim: int, activation_fn: nn.Module):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(type(activation_fn)())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_critic(input_dim: int, hidden_dims: tuple[int] | list[int], output_dim: int, activation_fn: nn.Module):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(type(activation_fn)())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        return None

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_input(self, obs: torch.Tensor) -> torch.Tensor:
        """Build actor input by encoding the trailing history slice when enabled.

        编码末尾的历史片段，并与当前 proprio+scan 拼接，作为 actor 输入。
        若未开启历史编码器，则原样返回。

        Layout: obs = [proprio+height_scan | optional_extra | raw_nav_scan | history_raw]
        Actor input = [proprio+height_scan | history_latent | optional_extra | nav_scan_latent]
        旧输入列保持在前面，新增 CNN latent 放到最后以便部分热加载。
        """
        if self.history_encoder is None and self.nav_scan_encoder is None:
            return obs

        # Split raw obs layout:
        # [proprio+height_scan | optional_extra | raw_nav_scan | history_raw]
        # Actor input layout:
        # [proprio+height_scan | history_latent | optional_extra | nav_scan_latent]
        core = obs[..., :self.core_obs_dim]
        offset = self.core_obs_dim

        if self.num_goal_obs > 0:
            goal = obs[..., offset:offset + self.num_goal_obs]
            offset += self.num_goal_obs
        else:
            goal = None

        if self.nav_scan_encoder is not None:
            nav_scan = obs[..., offset:offset + self.num_nav_scan_obs]
            offset += self.num_nav_scan_obs
            nav_latent = self.nav_scan_encoder(nav_scan)
        else:
            nav_latent = None

        history_latent = None
        if self.history_encoder is not None:
            history = obs[..., offset:]
            history_latent = self.history_encoder(history)

        terms = [core]
        if history_latent is not None:
            terms.append(history_latent)
        if goal is not None:
            terms.append(goal)
        if nav_latent is not None:
            terms.append(nav_latent)
        return torch.cat(terms, dim=-1)

    def _critic_input(self, critic_obs: torch.Tensor, nav_encoder: NavScanEncoder | None) -> torch.Tensor:
        """Build critic/cost-critic input by CNN-encoding trailing raw nav scan."""
        if nav_encoder is None:
            return critic_obs

        core = critic_obs[..., :self.critic_base_obs_dim]
        offset = self.critic_base_obs_dim

        if self.num_goal_obs > 0:
            goal = critic_obs[..., offset:offset + self.num_goal_obs]
            offset += self.num_goal_obs
        else:
            goal = None

        nav_scan = critic_obs[..., offset:offset + self.num_nav_scan_obs]
        nav_latent = nav_encoder(nav_scan)

        terms = [core]
        if goal is not None:
            terms.append(goal)
        terms.append(nav_latent)
        return torch.cat(terms, dim=-1)

    def update_distribution(self, obs: torch.Tensor):
        actor_in = self._actor_input(obs)
        mean = self.actor(actor_in)
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        actor_in = self._actor_input(obs)
        return self.actor(actor_in)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        critic_in = self._critic_input(critic_obs, self.critic_nav_scan_encoder)
        return self.critic(critic_in)

    def evaluate_cost(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        critic_in = self._critic_input(critic_obs, self.cost_nav_scan_encoder)
        return torch.nn.functional.softplus(self.cost_critic(critic_in))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
