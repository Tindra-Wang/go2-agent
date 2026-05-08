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
        **kwargs: dict[str, Any],
    ) -> None:
        super(Model, self).__init__()

        stage = Config.CURRENT
        self.num_obs = num_obs if num_obs is not None else stage.num_proprio_obs + stage.num_scan
        self.num_critic_obs = num_critic_obs if num_critic_obs is not None else stage.num_critic_observations
        self.num_actions = num_actions if num_actions is not None else stage.num_actions
        self.num_costs = num_costs if num_costs is not None else stage.num_costs

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
        if self.history_len > 0:
            encoder_dims = list(history_encoder_dims or stage.history_encoder_dims)
            self.history_encoder = HistoryEncoder(
                history_len=self.history_len,
                proprio_dim=self.proprio_dim,
                hidden_dims=encoder_dims,
                latent_dim=self.history_latent_dim,
                activation_fn=activation_fn,
            )
            actor_input_dim = self.base_obs_dim + self.history_latent_dim
        else:
            self.history_encoder = None
            actor_input_dim = self.num_obs

        self.actor = self._build_mlp(actor_input_dim, actor_hidden_dims, self.num_actions, activation_fn)
        self.critic = self._build_critic(self.num_critic_obs, critic_hidden_dims, 1, activation_fn)
        self.cost_critic = self._build_critic(self.num_critic_obs, critic_hidden_dims, self.num_costs, activation_fn)

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
        """
        if self.history_encoder is None:
            return obs
        base = obs[..., : self.base_obs_dim]
        history = obs[..., self.base_obs_dim :]
        latent = self.history_encoder(history)
        return torch.cat([base, latent], dim=-1)

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
        return self.critic(critic_obs)

    def evaluate_cost(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.softplus(self.cost_critic(critic_obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
