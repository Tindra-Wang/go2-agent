#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
from collections import defaultdict, deque

import torch

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import RolloutStorage
from tools.train_env_conf_validate import check_usr_conf
from tools.utils import load_reward_keys_from_monitor_config


DEFAULT_COST_INFO_KEYS = (
    "cost",
    "costs",
    "constraint_cost",
    "constraint_costs",
    "safety_cost",
    "safety_costs",
    "collision_cost",
)

DEFAULT_COST_EPISODE_KEYS = (
    "undesired_contacts",
    "collision",
    "collision_cost",
    "safety_cost",
)


def _initialize_training_state(env, agent, logger):
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)

    valid, message = check_usr_conf(usr_conf, is_eval=False, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    agent.algorithm.actor_critic.train()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    costbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_cost_sum = torch.zeros(agent.num_envs, stage.num_costs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    storage = agent.algorithm.storage

    data = env.reset(usr_conf)
    if data is None:
        error_message = "reset failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    obs, critic_obs = data
    if critic_obs is None:
        critic_obs = obs
    obs = torch.clone(obs)
    critic_obs = torch.clone(critic_obs)
    logger.info(f"obs.shape:{obs.shape}, critic_obs.shape:{critic_obs.shape}")

    reward_keys = load_reward_keys_from_monitor_config()
    logger.info(f"reward_keys list is {reward_keys}")

    return (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        costbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_cost_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    agent = agents[0]
    env = envs[0]

    (
        storage,
        obs,
        critic_obs,
        ep_infos,
        rewbuffer,
        costbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_cost_sum,
        cur_episode_length,
        reward_keys,
        usr_conf,
    ) = _initialize_training_state(env, agent, logger)

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0
    episode = 0

    while True:
        logger.info(f"Episode {episode} start, usr_conf is {usr_conf}")
        start_time = time.time()

        last_obs, last_critic_obs, storage_stats = run_episodes_(
            env,
            agent,
            storage,
            logger,
            last_obs,
            last_critic_obs,
            episode,
            ep_infos,
            cur_reward_sum,
            cur_cost_sum,
            cur_episode_length,
            rewbuffer,
            costbuffer,
            lenbuffer,
        )

        episode += 1

        agent.learn(list_sample_data=None)
        storage.clear()
        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time is {total_cost_time} s")

        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats)
            last_report_monitor_time = now

        ep_infos.clear()

        if episode % agent.save_interval == 0:
            agent.save_model()

    env.close()


def _extract_metric_value(ep_info, key, device):
    if key not in ep_info:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    metric = ep_info[key]
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric, device=device)
    return metric.float().mean()


def _aggregate_metrics(generic_metrics):
    aggregated = {}
    for metric_key, values in generic_metrics.items():
        aggregated[metric_key] = torch.stack(values).mean().item() if values else 0.0
    return aggregated


def _collect_episode_metrics(ep_infos, reward_keys, device):
    generic_metrics = defaultdict(list)
    for ep_info in ep_infos:
        for key in reward_keys:
            metric_value = _extract_metric_value(ep_info, key, device)
            generic_metrics[key].append(metric_value)
    return _aggregate_metrics(generic_metrics)


def report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats=None):
    if monitor is None:
        return

    monitor_data = {"episode_cnt": episode}

    if storage_stats:
        monitor_data["reward_mean"] = storage_stats.get("reward_mean", 0.0)
        monitor_data["reward_std"] = storage_stats.get("reward_std", 0.0)
        monitor_data["cost_mean"] = storage_stats.get("cost_mean", 0.0)
        monitor_data["violation_mean"] = storage_stats.get("violation_mean", 0.0)
        monitor_data["cost_source"] = storage_stats.get("cost_source", "unknown")

    if ep_infos:
        metrics = _collect_episode_metrics(ep_infos, reward_keys, agent.device)
        monitor_data.update(metrics)
        monitor_data["episode_reward"] = sum(monitor_data.get(key, 0) for key in reward_keys)

    monitor.put_data({os.getpid(): monitor_data})


def _process_env_step_result(data, episode, logger):
    if data is None:
        error_message = "step failed, please check"
        logger.error(error_message)
        raise Exception(error_message)

    frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data

    critic_obs = torch.clone(privileged_obs) if privileged_obs is not None else torch.clone(obs)
    obs = torch.clone(obs)

    if obs is None:
        logger.error(f"episode {episode}, obs is None after processing!")
        raise Exception(f"episode {episode}, obs is None after processing!")

    dones = torch.logical_or(terminated, truncated)
    return frame_no, obs, critic_obs, rewards, dones, infos


def _extract_explicit_cost_tensor(raw_cost, num_envs, num_costs, device):
    if isinstance(raw_cost, torch.Tensor):
        cost_tensor = raw_cost.to(device=device, dtype=torch.float32)
    else:
        cost_tensor = torch.as_tensor(raw_cost, device=device, dtype=torch.float32)

    if cost_tensor.ndim == 0:
        return None
    if cost_tensor.ndim == 1:
        if cost_tensor.shape[0] != num_envs:
            return None
        return cost_tensor.unsqueeze(-1)
    if cost_tensor.ndim == 2:
        if cost_tensor.shape[0] != num_envs:
            return None
        if cost_tensor.shape[1] == num_costs:
            return cost_tensor
        if cost_tensor.shape[1] == 1 and num_costs == 1:
            return cost_tensor
        return None
    return None


def _select_costs_from_infos(infos, num_envs, num_costs, device):
    if not isinstance(infos, dict):
        return None, None

    for key in ("costs", "cost"):
        if key not in infos:
            continue
        cost_tensor = _extract_explicit_cost_tensor(infos[key], num_envs, num_costs, device)
        if cost_tensor is not None:
            return cost_tensor, f"infos[{key}]"
    return None, None


def _derive_costs(infos, rewards, dones, agent):
    num_envs = rewards.shape[0]
    num_costs = agent.stage.num_costs
    costs = torch.zeros(num_envs, num_costs, dtype=torch.float32, device=agent.device)
    cost_source = "zeros"

    if infos is None:
        if getattr(agent.stage, "require_explicit_costs", False):
            raise RuntimeError("[DIY] Explicit per-env cost tensors are required, but env infos is None.")
        return costs, cost_source

    explicit_costs, explicit_source = _select_costs_from_infos(infos, num_envs, num_costs, agent.device)
    if explicit_costs is not None:
        return explicit_costs, explicit_source

    if getattr(agent.stage, "require_explicit_costs", False):
        available_keys = sorted(infos.keys()) if isinstance(infos, dict) else []
        raise RuntimeError(
            "[DIY] Explicit per-env cost tensors are required, but neither infos['costs'] nor infos['cost'] "
            f"contained a valid tensor for {num_envs} envs and {num_costs} costs. Available info keys: {available_keys}"
        )

    episode_info = infos.get("episode") if isinstance(infos, dict) else None
    if isinstance(episode_info, dict):
        for key in DEFAULT_COST_EPISODE_KEYS:
            if key in episode_info:
                metric = episode_info[key]
                if not isinstance(metric, torch.Tensor):
                    metric = torch.tensor(metric, device=agent.device, dtype=torch.float32)
                metric = metric.to(agent.device, dtype=torch.float32)
                agent.logger.warning(
                    f"[DIY] Falling back to episode summary cost infos['episode']['{key}']; broadcasting scalar summary to all envs."
                )
                costs[:, 0] = torch.clamp(metric.view(-1)[0], min=0.0)
                return costs, f"episode[{key}]"

    if agent.stage.termination_as_cost:
        agent.logger.warning("[DIY] Falling back to termination_as_cost because explicit per-env cost tensors were not found.")
        costs[:, 0] = dones.float().view(-1)
        return costs, "termination_as_cost"

    return costs, cost_source


def _move_tensors_to_device(obs, critic_obs, rewards, dones, costs, device):
    return (
        obs.to(device),
        critic_obs.to(device),
        rewards.to(device),
        dones.to(device),
        costs.to(device),
    )


def _update_transition_data(
    transition,
    actions,
    values,
    cost_values,
    actions_log_prob,
    action_mean,
    action_sigma,
    obs,
    critic_obs,
    rewards,
    costs,
    dones,
    infos,
    agent,
):
    transition.actions = actions
    transition.values = values
    transition.cost_values = cost_values
    transition.actions_log_prob = actions_log_prob
    transition.action_mean = action_mean
    transition.action_sigma = action_sigma
    transition.observations = obs
    transition.critic_observations = critic_obs
    transition.rewards = rewards.clone()
    transition.costs = costs.clone()
    transition.dones = dones

    if "time_outs" in infos:
        timeout_mask = infos["time_outs"].unsqueeze(1).to(agent.device)
        transition.rewards += agent.algorithm.gamma * torch.squeeze(transition.values * timeout_mask, 1)
        transition.costs += agent.algorithm.gamma * transition.costs * timeout_mask


def _update_episode_statistics(
    dones,
    rewards,
    costs,
    infos,
    cur_reward_sum,
    cur_cost_sum,
    cur_episode_length,
    rewbuffer,
    costbuffer,
    lenbuffer,
    ep_infos,
):
    if "episode" in infos:
        ep_infos.append(infos["episode"])

    cur_reward_sum += rewards
    cur_cost_sum += costs
    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
    if new_ids.numel() > 0:
        costbuffer.extend(cur_cost_sum[new_ids[:, 0]].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_cost_sum[new_ids[:, 0]] = 0
    cur_episode_length[new_ids] = 0


def _compute_advantages_and_returns(storage, agent, critic_obs, logger, cost_source):
    last_critic_obs = torch.clone(critic_obs)
    last_values = agent.algorithm.actor_critic.evaluate(last_critic_obs.detach()).detach()
    last_cost_values = agent.algorithm.actor_critic.evaluate_cost(last_critic_obs.detach()).detach()
    storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)
    storage.compute_cost_returns(last_cost_values, agent.algorithm.gamma, agent.algorithm.lam)

    storage_stats = {
        "reward_mean": storage.rewards.mean().item(),
        "reward_std": storage.rewards.std().item(),
        "cost_mean": storage.costs.mean().item(),
        "violation_mean": storage.cost_violation.mean().item(),
        "cost_source": cost_source,
    }

    return storage_stats


def run_episodes_(
    env,
    agent,
    storage,
    logger,
    last_obs,
    last_critic_obs,
    episode,
    ep_infos,
    cur_reward_sum,
    cur_cost_sum,
    cur_episode_length,
    rewbuffer,
    costbuffer,
    lenbuffer,
):
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs
    last_cost_source = "uninitialized"

    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            predict_result = agent.predict((obs, critic_obs))
            (
                actions,
                values,
                cost_values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
            ) = predict_result
            joint_actions = actions
            command_actions = torch.clip(joint_actions, -6.0, 6.0).to(agent.device)
            if i == 0:
                logger.info(f"clipped_action:{command_actions}")

            data = env.step(command_actions)
            frame_no, obs, critic_obs, rewards, dones, infos = _process_env_step_result(data, episode, logger)
            costs, cost_source = _derive_costs(infos, rewards, dones, agent)
            last_cost_source = cost_source

            obs, critic_obs, rewards, dones, costs = _move_tensors_to_device(
                obs, critic_obs, rewards, dones, costs, agent.device
            )

            _update_episode_statistics(
                dones,
                rewards,
                costs,
                infos,
                cur_reward_sum,
                cur_cost_sum,
                cur_episode_length,
                rewbuffer,
                costbuffer,
                lenbuffer,
                ep_infos,
            )

            _update_transition_data(
                transition,
                actions,
                values,
                cost_values,
                actions_log_prob,
                action_mean,
                action_sigma,
                detach_obs,
                detach_critic_obs,
                rewards,
                costs,
                dones,
                infos,
                agent,
            )
            storage.add_transitions(transition)
            transition.clear()

            if i == 0:
                info_keys = sorted(infos.keys()) if isinstance(infos, dict) else []
                info_shapes = {}
                if isinstance(infos, dict):
                    for key, value in infos.items():
                        if isinstance(value, torch.Tensor):
                            info_shapes[key] = list(value.shape)
                        else:
                            info_shapes[key] = type(value).__name__
                logger.info(
                    f"rollout shapes: obs={detach_obs.shape}, critic_obs={detach_critic_obs.shape}, costs={costs.shape}, cost_source={cost_source}"
                )
                logger.info(f"rollout infos keys: {info_keys}")
                logger.info(f"rollout infos shapes: {info_shapes}")

        storage_stats = _compute_advantages_and_returns(storage, agent, critic_obs, logger, last_cost_source)
        last_obs = torch.clone(obs)

    return last_obs, critic_obs, storage_stats
