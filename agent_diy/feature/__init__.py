from importlib import import_module

from agent_diy.feature.critic_observation_process import CriticObservationProcess
from agent_diy.feature.policy_observation_process import PolicyObservationProcess

__all__ = [
    "CriticObservationProcess",
    "PolicyObservationProcess",
    "RewardProcess",
]


def __getattr__(name):
    if name == "RewardProcess":
        return import_module("agent_diy.feature.reward_process").RewardProcess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
