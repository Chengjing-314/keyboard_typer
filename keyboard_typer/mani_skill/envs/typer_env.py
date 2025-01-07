# python3 keyboard_typer/mani_skill/envs/typer_env.py

import json
from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from keyboard_typer.constants import ASSETS_ROOT
from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from keyboard_typer.utils.scene_builder.table import TableSceneBuilder
from keyboard_typer.utils.keyboard.keyboard import BlackKeyboard


@dataclass
class TyperEnvConfig:
    robot_uids: Union[str, tuple] = "xarm7_right"
    control_freq: int = 50
    bimanual: bool = False
    initial_agent_poses: Union[sapien.Pose, list[sapien.Pose]] = None
    fix_keyboard: bool = False


@register_env("TyperEnv-v0", max_episode_steps=100)
class TyperEnv(BaseEnv):
    def __init__(self, *args, config=TyperEnvConfig(), robot_uids="xarm7_right", **kwargs):
        SUPPORTED_ROBOTS = ["xarm7_right", "xarm7_left"]
        agent: Union[XArm7AbilityLeft, XArm7AbilityRight]
        self.config = config
        self.kb_manager = BlackKeyboard()
        # NOTE: robot uids are overridden by the config
        super().__init__(*args, robot_uids=self.config.robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, initial_agent_poses=self.config.initial_agent_poses)
        # if self.config.bimanual:
        #     for agent in self.agent.agents:
        #         agent.reset()
        # else:
        #     self.agent.reswt()

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # TODO: refactor for more keyboard
        urdf_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/mobility.urdf"
        bb_box_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/bounding_box.json"

        builder = self.scene.create_urdf_loader()
        bbox_max, bbox_min = (
            json.load(open(bb_box_path))["max"],
            json.load(open(bb_box_path))["min"],
        )
        length, height, width = np.array(bbox_max) - np.array(bbox_min)
        self.scale = 0.45 / max(width, length, height)
        builder.scale = self.scale
        articulation_builders = builder.parse(str(urdf_path))["articulation_builders"]
        builder = articulation_builders[0]
        self.keyboard_height = self.scale * height * 0.5
        builder.initial_pose = sapien.Pose(p=[0.45, 0.0, self.scale * height * 0.5])
        self.keyboard = builder.build(name="keyboard")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        num_envs = len(env_idx)
        keyboard_pos = torch.zeros((num_envs, 3))
        keyboard_pos[..., :2] = torch.rand((num_envs, 2)) * 0.05 - 0.025
        keyboard_pos[..., 2] = self.keyboard_height

        if self.config.bimanual:
            for agent in self.agent.agents:
                agent.reset()
        else:
            self.agent.reset()

        keyboard_pose = Pose.create_from_pq(p=keyboard_pos)
        self.keyboard.set_pose(keyboard_pose)
        self.target_key_press, self.target_key_indices = self.kb_manager.generate_key_press(
            num_envs
        )

        keyboard_active_keys = self.keyboard.get_active_joints()
        query_keys = [keyboard_active_keys[i] for i in self.target_key_indices]
        target_key_pos = []
        for env, key in enumerate(query_keys):
            target_key_pos.append(key.get_global_pose().raw_pose[env, :3])

        self.target_key_pos = torch.stack(target_key_pos, dim=0)

    def _get_obs_extra(self, info):
        # key position
        # finger tip position
        # keyboard position

        if not self.config.bimanual:
            finger_tip_pos = self.agent.get_finger_tip_pos()
        else:
            raise NotImplementedError

        obs = dict(target_key_pos=self.target_key_pos, finger_tip_pos=finger_tip_pos)

        return obs


def main():
    config = TyperEnvConfig()

    # config.robot_uids = ("xarm7_right", "xarm7_left")
    # config.bimanual = True
    # config.initial_agent_poses = [None, None]

    env = gym.make("TyperEnv-v0", config=config, render_mode="human", num_envs=1, obs_mode="state")
    env.reset()

    while True:
        env.unwrapped.render_human()


if __name__ == "__main__":
    main()
