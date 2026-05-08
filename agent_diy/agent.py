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
        self.num_critic_obs = stage.num_critic_observations
        self.num_obs = stage.num_proprio_obs + stage.num_scan
        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        self.model = Model(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            num_costs=stage.num_costs,
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

    def exploit(self, list_obs_data):
        obs = list_obs_data[0] if isinstance(list_obs_data, (list, tuple)) else list_obs_data
        with torch.no_grad():
            actions = self.algorithm.actor_critic.act_inference(obs)
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
        self.model.load_state_dict(pretrained)
        self.cur_model_name = model_file_path
        self.logger.info(f"load model {model_file_path} successfully")
