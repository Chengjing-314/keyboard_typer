import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

from keyboard_typer.constants import ASSETS_ROOT
from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from typing import Union
from dataclasses import dataclass
import gymnasium as gym
from keyboard_typer.utils.scene_builder.table import TableSceneBuilder
import torch


@dataclass
class TyperEnvConfig:
    robot_uids: Union[str, tuple] = "xarm7_right"
    control_freq: int = 50
    bimanual: bool = False
    initial_agent_poses: Union[sapien.Pose, list[sapien.Pose]] = None


@register_env("TyperEnv-v0", max_episode_steps=100)
class TyperEnv(BaseEnv):
    def __init__(self, *args, config=TyperEnvConfig(), robot_uids="xarm7_right", **kwargs):
        SUPPORTED_ROBOTS = ["xarm7_right", "xarm7_left"]
        agent: Union[XArm7AbilityLeft, XArm7AbilityRight]
        self.config = config
        # NOTE: robot uids are overridden by the config
        super().__init__(*args, robot_uids=self.config.robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, initial_agent_poses=self.config.initial_agent_poses)
        if self.config.bimanual:
            for agent in self.agent.agents:
                agent.reset()
        else:
            self.agent.reset()

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)


def main():
    config = TyperEnvConfig()

    config.robot_uids = ("xarm7_right", "xarm7_left")
    config.bimanual = True
    config.initial_agent_poses = [None, None]

    env = gym.make("TyperEnv-v0", config=config, render_mode="human")
    env.reset()

    while True:
        env.render_human()


if __name__ == "__main__":
    main()
