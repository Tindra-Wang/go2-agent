#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from common_python.utils.common_func import create_cls, Frame
import torch

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
)


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.costs = None
            self.dones = None
            self.values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        num_costs,
        cost_d_values=None,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_costs = num_costs

        cost_d_tensor = torch.as_tensor(
            cost_d_values if cost_d_values is not None else [0.0] * num_costs,
            dtype=torch.float32,
        )
        if cost_d_tensor.numel() != num_costs:
            raise ValueError(f"Expected {num_costs} cost thresholds, got {cost_d_tensor.numel()}")
        self.cost_d_values = cost_d_tensor.to(device).view(1, 1, num_costs)

        self._init_buffers(
            num_transitions_per_env,
            num_envs,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
        )

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    def _init_buffers(
        self,
        num_transitions_per_env,
        num_envs,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
    ):
        shape = (num_transitions_per_env, num_envs)

        self.observations = torch.zeros(*shape, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(*shape, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None

        self.actions = torch.zeros(*shape, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=self.device)
        self.mu = torch.zeros(*shape, *actions_shape, device=self.device)
        self.sigma = torch.zeros(*shape, *actions_shape, device=self.device)

        self.rewards = torch.zeros(*shape, 1, device=self.device)
        self.costs = torch.zeros(*shape, self.num_costs, device=self.device)
        self.values = torch.zeros(*shape, 1, device=self.device)
        self.cost_values = torch.zeros(*shape, self.num_costs, device=self.device)
        self.returns = torch.zeros(*shape, 1, device=self.device)
        self.cost_returns = torch.zeros(*shape, self.num_costs, device=self.device)
        self.advantages = torch.zeros(*shape, 1, device=self.device)
        self.cost_advantages = torch.zeros(*shape, self.num_costs, device=self.device)
        self.cost_violation = torch.zeros(*shape, self.num_costs, device=self.device)
        self.cost_d_storage = self.cost_d_values.expand(*shape, self.num_costs).clone()

        self.dones = torch.zeros(*shape, 1, device=self.device).byte()

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.costs[self.step].copy_(transition.costs.view(-1, self.num_costs))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.cost_values[self.step].copy_(transition.cost_values.view(-1, self.num_costs))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a, hid_c = self._normalize_hidden_states(hidden_states)
        if self.saved_hidden_states_a is None:
            self._init_hidden_state_storage(hid_a, hid_c)
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def _normalize_hidden_states(self, hidden_states):
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        return hid_a, hid_c

    def _init_hidden_state_storage(self, hid_a, hid_c):
        self.saved_hidden_states_a = [
            torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
        ]
        self.saved_hidden_states_c = [
            torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
        ]

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        last_values = torch.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.values.copy_(torch.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0))
        self.rewards.copy_(torch.nan_to_num(self.rewards, nan=0.0, posinf=0.0, neginf=0.0))

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.returns.copy_(torch.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0))
        self.advantages = self.returns - self.values
        adv_std = self.advantages.std()
        if not torch.isfinite(adv_std) or adv_std < 1e-6:
            adv_std = torch.ones_like(adv_std)
        self.advantages = (self.advantages - self.advantages.mean()) / (adv_std + 1e-8)
        self.advantages = torch.nan_to_num(self.advantages, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_cost_returns(self, last_cost_values, gamma, lam):
        last_cost_values = torch.nan_to_num(last_cost_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.cost_values.copy_(torch.nan_to_num(self.cost_values, nan=0.0, posinf=0.0, neginf=0.0))
        self.costs.copy_(torch.nan_to_num(self.costs, nan=0.0, posinf=0.0, neginf=0.0))

        advantage = torch.zeros_like(last_cost_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]

        self.cost_returns.copy_(torch.nan_to_num(self.cost_returns, nan=0.0, posinf=0.0, neginf=0.0))
        self.cost_advantages = self.cost_returns - self.cost_values
        adv_mean = self.cost_advantages.mean(dim=(0, 1), keepdim=True)
        adv_std = self.cost_advantages.std(dim=(0, 1), keepdim=True)
        adv_std = torch.where(torch.isfinite(adv_std) & (adv_std >= 1e-6), adv_std, torch.ones_like(adv_std))
        self.cost_advantages = (self.cost_advantages - adv_mean) / (adv_std + 1e-8)
        self.cost_advantages = torch.nan_to_num(self.cost_advantages, nan=0.0, posinf=0.0, neginf=0.0)

        violation = (1.0 - gamma) * (self.cost_returns - self.cost_d_storage) + adv_mean
        self.cost_violation = violation / (adv_std + 1e-8)
        self.cost_violation = torch.nan_to_num(self.cost_violation, nan=0.0, posinf=0.0, neginf=0.0)

    def get_statistics(self):
        done = self.dones.clone()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean(), self.costs.mean(dim=(0, 1))

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        flattened_data = self._flatten_buffers()

        for _ in range(num_epochs):
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield self._create_mini_batch(flattened_data, batch_idx)

    def _flatten_buffers(self):
        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else observations
        )
        return {
            "observations": observations,
            "critic_observations": critic_observations,
            "actions": self.actions.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "costs": self.costs.flatten(0, 1),
            "cost_values": self.cost_values.flatten(0, 1),
            "cost_returns": self.cost_returns.flatten(0, 1),
            "cost_advantages": self.cost_advantages.flatten(0, 1),
            "cost_violation": self.cost_violation.flatten(0, 1),
            "cost_d_values": self.cost_d_storage.flatten(0, 1),
            "old_actions_log_prob": self.actions_log_prob.flatten(0, 1),
            "old_mu": self.mu.flatten(0, 1),
            "old_sigma": self.sigma.flatten(0, 1),
        }

    def _create_mini_batch(self, flattened_data, batch_idx):
        return (
            flattened_data["observations"][batch_idx],
            flattened_data["critic_observations"][batch_idx],
            flattened_data["actions"][batch_idx],
            flattened_data["values"][batch_idx],
            flattened_data["advantages"][batch_idx],
            flattened_data["returns"][batch_idx],
            flattened_data["cost_values"][batch_idx],
            flattened_data["cost_advantages"][batch_idx],
            flattened_data["cost_returns"][batch_idx],
            flattened_data["cost_violation"][batch_idx],
            flattened_data["cost_d_values"][batch_idx],
            flattened_data["old_actions_log_prob"][batch_idx],
            flattened_data["old_mu"][batch_idx],
            flattened_data["old_sigma"][batch_idx],
            (None, None),
            None,
        )


def sample_process(collector):
    return collector.sample_process()


def build_frame(frame_no, obs, actions, dones, rewards):
    frame = Frame(
        frame_no=frame_no,
        obs=obs,
        actions=actions,
        done=dones,
        rewards=rewards,
    )
    return frame


def obs_normalizer(obs):
    return obs
