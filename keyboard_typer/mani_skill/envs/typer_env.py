from dataclasses import dataclass, field

from typing import Union

import gymnasium as gym
import torch
from mani_skill.utils.registration import register_env

from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from keyboard_typer.curriculum.stages import StageZeroHandler, StageOneHandler
from keyboard_typer.mani_skill.envs.typer_base_env import (
    MAX_EPISODE_STEPS,
    TyperBaseEnv,
    TyperEnvBaseConfig,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TyperEnvConfig(TyperEnvBaseConfig):
    tcp_threshold: float = 0.01
    keyboard_initial_pose: list = field(
        default_factory=lambda: [
            0.3,
            0.1,
            0.5,
        ]
    )  # Relative values, Z will scale with kb_height


@register_env("TyperEnv-v0", max_episode_steps=MAX_EPISODE_STEPS)
class TyperEnv(TyperBaseEnv):
    def __init__(self, *args, config: TyperEnvConfig, robot_uids: str = "", **kwargs):
        self.stage = 1  # Current stage
        self.handlers = {0: StageZeroHandler(self), 1: StageOneHandler(self)}
        self.target_key_pos = None
        self.target_finger = None
        super().__init__(*args, config=config, robot_uids="", **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.stage in self.handlers:
            self.target_key_pos, self.target_finger = self.handlers[self.stage].initialize(
                self.keyboard_pos,
                self.num_envs,
                self.device,
                self.agent,
                self.tcp_viz,
                self.goal_viz,
                self.kb_manager,
            )
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def _get_obs_extra(self, info: dict):
        if self.stage in self.handlers:
            finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
            return self.handlers[self.stage].get_obs(
                finger_tip_pos, self.target_finger, self.target_key_pos
            )
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def compute_normalized_dense_reward(self, obs, action, info):
        if self.stage in self.handlers:
            finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
            qvel = self.agent.robot.qvel
            reward, self.reward_dict = self.handlers[self.stage].compute_reward(
                finger_tip_pos,
                self.target_finger,
                self.target_key_pos,
                qvel,
                self.tcp_viz,
            )
            return reward
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def get_reward_details(self):
        return self.reward_dict


if __name__ == "__main__":
    config = TyperEnvConfig()

    n_envs = 1

    env = gym.make(
        "TyperEnv-v0",
        config=config,
        render_mode="human",
        num_envs=n_envs,
        obs_mode="state_dict",
        parallel_in_single_scene=False,
    )
    env.reset()

    action = env.action_space.sample()
    while True:
        env.unwrapped.render_human()

        env.step(action)
