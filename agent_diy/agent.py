#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch
import torch.optim as optim

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.feature.definition import ActData
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.stage = stage
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]
        self.num_actions = stage.num_actions
        self.num_goal_obs = int(getattr(stage, "num_goal_obs", 0))
        self.num_critic_obs = stage.num_critic_observations + self.num_goal_obs
        self.proprio_dim = int(stage.num_proprio_obs)
        self.base_num_obs = self.proprio_dim + int(stage.num_scan) + self.num_goal_obs
        self.history_len = int(getattr(stage, "history_len", 0)) if getattr(stage, "use_history_encoder", False) else 0
        self.num_obs = self.base_num_obs + self.history_len * self.proprio_dim
        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval
        self._eval_history = None

        self.model = Model(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            num_costs=stage.num_costs,
            history_len=self.history_len,
            proprio_dim=self.proprio_dim,
            history_latent_dim=int(getattr(stage, "history_latent_dim", 16)),
            history_encoder_dims=list(getattr(stage, "history_encoder_dims", [128, 64])),
            num_goal_obs=self.num_goal_obs,
        ).to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        self.logger.info(f"DIY Actor MLP: {self.model.actor}")
        self.logger.info(f"DIY Critic MLP: {self.model.critic}")
        self.logger.info(f"DIY Cost Critic MLP: {self.model.cost_critic}")

        params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(params, lr=stage.lr)
        self.algorithm = Algorithm(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage.lr,
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
            value_loss_coef=stage.value_loss_coef,
            cost_value_loss_coef=stage.cost_value_loss_coef,
            cost_violation_loss_coef=stage.cost_violation_loss_coef,
            entropy_coef=stage.entropy_coef,
            desired_kl=stage.desired_kl,
            schedule=stage.schedule,
            penalty_lr=stage.penalty_lr,
            penalty_decay=stage.penalty_decay,
            penalty_max=stage.penalty_max,
            penalty_mode=getattr(stage, "penalty_mode", "scheduled"),
            penalty_growth_rate=getattr(stage, "penalty_growth_rate", 1.0004),
        )
        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=(self.num_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.num_actions,),
            device=self.device,
        )

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        (obs, critic_obs) = list_obs_data
        with torch.no_grad():
            (
                actions,
                values,
                cost_values,
                actions_log_prob,
                action_mean,
                action_sigma,
                observations,
                critic_observations,
            ) = self.algorithm.act(obs, critic_obs)

        return (
            actions,
            values,
            cost_values,
            actions_log_prob,
            action_mean,
            action_sigma,
            observations,
            critic_observations,
        )

    def _augment_obs_for_eval(self, obs):
        """Concatenate the eval-side history buffer onto the raw obs (HIM-lite eval path).

        在评估路径下也维护历史观测，使评估时的 actor 输入与训练时一致；
        若 ``history_len <= 0`` 则原样返回。
        """
        if self.history_len <= 0:
            return obs
        if (
            self._eval_history is None
            or self._eval_history.shape[0] != obs.shape[0]
            or self._eval_history.device != obs.device
        ):
            self._eval_history = torch.zeros(
                obs.shape[0], self.history_len, self.proprio_dim, device=obs.device, dtype=obs.dtype
            )
        return torch.cat([obs, self._eval_history.flatten(1)], dim=-1)

    def _update_eval_history(self, raw_obs):
        if self.history_len <= 0 or self._eval_history is None:
            return
        proprio = raw_obs[:, : self.proprio_dim].detach()
        self._eval_history = torch.cat([self._eval_history[:, 1:], proprio.unsqueeze(1)], dim=1)

    def exploit(self, list_obs_data):
        obs = list_obs_data[0] if isinstance(list_obs_data, (list, tuple)) else list_obs_data
        aug_obs = self._augment_obs_for_eval(obs)
        with torch.no_grad():
            actions = self.algorithm.actor_critic.act_inference(aug_obs)
        self._update_eval_history(obs)
        return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        return self.algorithm.learn(list_sample_data)

    def predict_local(self, obs, critic_obs):
        return self.algorithm.act(obs, critic_obs)

    def action_process(self, act_data):
        if getattr(act_data, "action", None) is not None and act_data.action.ndim == 1:
            act_data.action = act_data.action.unsqueeze(0)
        return act_data

    def observation_process(self, obs_q):
        return obs_q

    def reset(self):
        # 评估端历史缓冲在 episode 切换时由调用方触发清理。
        # Evaluation-side history is cleared by the caller across episode boundaries.
        self._eval_history = None
        return None

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        has_mismatch = False
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self.cur_model_name = model_file_path

    def _load_model_partial(self, model, pretrained, model_file_path):
        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for key in current_state:
            if key not in pretrained:
                skipped_keys.append(key)
                continue

            old_param = pretrained[key]
            new_param = current_state[key]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(key)
            else:
                with torch.no_grad():
                    new_param.zero_()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}->{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(
            f"Partial load model {model_file_path}: "
            f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
        )
        for info in partial_keys:
            self.logger.info(f"  Partial: {info}")
