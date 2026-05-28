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
    """HIM-lite history encoder (MLP variant, kept for backward compatibility).

    历史观测编码器（NP3O HIM 简化版，无 contrastive loss）：
    将 ``history_len`` 步 proprio 拼接展平后，经 MLP 压缩为 ``latent_dim`` 维潜变量。
    Legacy MLP path; prefer ``GRUHistoryEncoder`` for new training.
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


class GRUHistoryEncoder(nn.Module):
    """GRU-based history encoder (replaces MLP for sequential proprioception).

    用 GRU 替换 MLP 处理历史 proprio 序列：
    - 输入: [B, history_len * proprio_dim] → reshape → [B, history_len, proprio_dim]
    - GRU 逐帧编码，取末帧隐状态经 projection 得到 latent。
    相比 MLP，GRU 天然建模时序依赖，参数更少且对 history_len 变化更鲁棒。
    """

    def __init__(
            self,
            history_len: int,
            proprio_dim: int,
            hidden_dims: list[int],
            latent_dim: int,
            activation_fn: nn.Module,
            num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        # hidden_dims[0] as GRU hidden size; last element is projection hidden
        gru_hidden = hidden_dims[0] if hidden_dims else 128
        self.gru = nn.GRU(
            input_size=proprio_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        proj_layers: list[nn.Module] = []
        in_dim = gru_hidden
        for hidden in hidden_dims[1:]:
            proj_layers.append(nn.Linear(in_dim, hidden))
            proj_layers.append(type(activation_fn)())
            in_dim = hidden
        proj_layers.append(nn.Linear(in_dim, latent_dim))
        self.proj = nn.Sequential(*proj_layers) if proj_layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, history_len * proprio_dim]
        x = x.view(-1, self.history_len, self.proprio_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # [B, gru_hidden]
        return self.proj(last)


class HeightScanEncoder(nn.Module):
    """2D CNN encoder for 16x16 height_scanner grid.

    将 16x16 height_scanner 经 2D CNN 压缩为 latent 后送入 actor。
    输入: [B, 256] (flattened 16x16 grid)
    输出: [B, latent_dim]
    """

    def __init__(
            self,
            grid_size: int = 16,
            channels: list[int] | None = None,
            latent_dim: int = 256,
            activation_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]
        if activation_fn is None:
            activation_fn = nn.ELU()

        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(type(activation_fn)())
            in_ch = out_ch

        self.pool_size = min(4, grid_size)
        layers.append(nn.AdaptiveAvgPool2d(self.pool_size))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch * self.pool_size * self.pool_size, latent_dim))
        layers.append(type(activation_fn)())

        self.cnn = nn.Sequential(*layers)
        self.grid_size = grid_size
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256] → reshape → [B, 1, 16, 16]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.cnn(x.view(-1, 1, self.grid_size, self.grid_size))


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

        self.pool_bins = min(8, scan_dim)
        layers.append(nn.AdaptiveAvgPool1d(self.pool_bins))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * self.pool_bins, latent_dim))
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
            history_encoder_type: str = "gru",
            num_goal_obs: int = 0,
            num_nav_scan_obs: int = 0,
            height_scan_dim: int = 256,
            height_scan_latent_dim: int = 256,
            height_scan_cnn_channels: list[int] | tuple[int, ...] | None = None,
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

        # History encoder: GRU (default) or MLP (legacy).
        # 历史编码器：GRU（默认）或 MLP（兼容旧版）。
        # critic 仍使用未增广的 critic_obs（NP3O 中 critic 端为特权观测，无需历史）。
        self.history_len = int(history_len) if history_len else 0
        self.proprio_dim = int(proprio_dim) if proprio_dim is not None else int(stage.num_proprio_obs)
        self.history_latent_dim = int(history_latent_dim)
        encoder_type = history_encoder_type or getattr(stage, "history_encoder_type", "gru")

        history_total = self.history_len * self.proprio_dim
        if history_total > 0 and self.num_obs <= history_total:
            raise ValueError(
                f"num_obs ({self.num_obs}) must exceed history_len*proprio_dim ({history_total})"
            )
        self.base_obs_dim = self.num_obs - history_total
        # core_obs_dim = proprio + height_scan_raw (without goal/nav_scan/history)
        self.core_obs_dim = self.base_obs_dim - self.num_goal_obs - self.num_nav_scan_obs
        if self.core_obs_dim <= 0:
            raise ValueError(
                f"Invalid obs layout: base_obs_dim={self.base_obs_dim}, "
                f"num_goal_obs={self.num_goal_obs}, num_nav_scan_obs={self.num_nav_scan_obs}"
            )
        if self.history_len > 0:
            encoder_dims = list(history_encoder_dims or stage.history_encoder_dims)
            if encoder_type == "gru":
                self.history_encoder = GRUHistoryEncoder(
                    history_len=self.history_len,
                    proprio_dim=self.proprio_dim,
                    hidden_dims=encoder_dims,
                    latent_dim=self.history_latent_dim,
                    activation_fn=activation_fn,
                )
            else:
                self.history_encoder = HistoryEncoder(
                    history_len=self.history_len,
                    proprio_dim=self.proprio_dim,
                    hidden_dims=encoder_dims,
                    latent_dim=self.history_latent_dim,
                    activation_fn=activation_fn,
                )
        else:
            self.history_encoder = None

        # HeightScan 2D CNN encoder: 16x16 grid → latent (actor side only)
        # 将原始 height_scan(256) 替换为 2D CNN 编码后的 latent(256)，仅 actor 侧使用
        self.height_scan_dim = int(height_scan_dim)
        self.height_scan_latent_dim = int(height_scan_latent_dim)
        self.use_height_scan_encoder = bool(getattr(stage, "use_height_scan_encoder", True))
        if self.use_height_scan_encoder:
            hscan_channels = list(height_scan_cnn_channels or getattr(stage, "height_scan_cnn_channels", None) or [16, 32, 64])
            self.height_scan_encoder = HeightScanEncoder(
                grid_size=16,
                channels=hscan_channels,
                latent_dim=self.height_scan_latent_dim,
                activation_fn=activation_fn,
            )
        else:
            self.height_scan_encoder = None

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

        # Actor input: [proprio | height_scan_latent | extra | history_latent | goal | nav_scan_latent]
        # extra = terrain_context + maze_nav_hint (between height_scan and goal in core_obs)
        # 用 2D CNN latent(256) 替换原始 height_scan(256)，维度不变。
        history_actor_dim = self.history_latent_dim if self.history_encoder is not None else 0
        extra_actor_dim = max(0, self.core_obs_dim - self.proprio_dim - self.height_scan_dim)
        actor_input_dim = (self.proprio_dim + self.height_scan_latent_dim + extra_actor_dim
                           + history_actor_dim + self.num_goal_obs + nav_actor_dim)
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
        """Build actor input with 2D CNN height scan + GRU history + 1D CNN nav scan.

        Obs layout: [proprio | height_scan_raw | extra(terrain+maze) | goal | nav_scan_raw | history_raw]
        Actor input: [proprio | height_scan_latent | extra | history_latent | goal | nav_scan_latent]
        """
        proprio = obs[..., :self.proprio_dim]
        hscan_raw = obs[..., self.proprio_dim:self.proprio_dim + self.height_scan_dim]
        hscan_latent = self.height_scan_encoder(hscan_raw) if self.height_scan_encoder is not None else hscan_raw

        # Extra terms (terrain_context, maze_nav_hint) between height_scan and goal
        extra_start = self.proprio_dim + self.height_scan_dim
        extra_end = self.core_obs_dim
        extra = obs[..., extra_start:extra_end] if extra_end > extra_start else None

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

        terms = [proprio, hscan_latent]
        if extra is not None and extra.shape[-1] > 0:
            terms.append(extra)
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
