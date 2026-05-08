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

        self.actor = self._build_mlp(self.num_obs, actor_hidden_dims, self.num_actions, activation_fn)
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

    def update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs)
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
        return self.actor(obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_obs)

    def evaluate_cost(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.softplus(self.cost_critic(critic_obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
