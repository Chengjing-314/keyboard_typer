from dataclasses import dataclass, field

from typing import Union

import gymnasium as gym
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from keyboard_typer.curriculum.stages import StageHandler
from keyboard_typer.mani_skill.envs.typer_base_env import (
    MAX_EPISODE_STEPS,
    TyperBaseEnv,
    TyperEnvBaseConfig,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TyperEnvConfig(TyperEnvBaseConfig):
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
        self.handlers = {1: StageHandler(self)}
        self.target_key_pos = None
        self.target_finger = None
        self.key_default_qpos = None
        self.num_chars = 2
        super().__init__(*args, config=config, robot_uids="", **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.stage in self.handlers:
            if self.target_key_pos is None:
                self.target_key_pos, self.target_finger, self.target_key_indices = self.handlers[
                    self.stage
                ].initialize(
                    env_idx,
                    self.num_chars,
                    self.keyboard_pos,
                    self.num_envs,
                    self.device,
                    self.agent,
                    self.tcp_viz,
                    self.goal_viz,
                    self.kb_manager,
                )
                self.key_default_qpos = torch.zeros(
                    (len(env_idx)), device=self.device
                )  # FIXME: this is not working
                self.key_press_progress = torch.zeros((len(env_idx)), device=self.device).long()
            else:
                (
                    self.target_key_pos[env_idx, :],
                    self.target_finger[env_idx],
                    self.target_key_indices[env_idx],
                ) = self.handlers[self.stage].initialize(
                    env_idx,
                    self.num_chars,
                    self.keyboard_pos,
                    self.num_envs,
                    self.device,
                    self.agent,
                    self.tcp_viz,
                    self.goal_viz,
                    self.kb_manager,
                )
                self.key_press_progress[env_idx] = 0

        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def _get_obs_extra(self, info: dict):
        if self.stage in self.handlers:
            finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
            keyboard_qpos = self.keyboard.qpos
            return self.handlers[self.stage].get_obs(
                self.key_press_progress,
                self.num_chars,
                finger_tip_pos,
                self.target_finger,
                self.target_key_pos,
                keyboard_qpos,
            )
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def compute_normalized_dense_reward(self, obs, action, info):
        if self.stage in self.handlers:
            finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
            qvel = self.agent.robot.qvel
            keyboard_qpos = self.keyboard.qpos
            reward, self.reward_dict = self.handlers[self.stage].compute_reward(
                info,
                self.key_press_progress,
                finger_tip_pos,
                self.target_finger,
                self.target_key_pos,
                self.target_key_indices,
                qvel,
                self.tcp_viz,
                self.goal_viz,
                keyboard_qpos,
                self.key_default_qpos,
            )
            return reward
        else:
            raise NotImplementedError(f"Stage {self.stage} is not implemented")

    def evaluate(self):
        target_finger_pos = self.agent.get_finger_tip_pos(flatten=False)[
            torch.arange(self.num_envs), self.target_finger
        ]
        reached = (
            torch.norm(
                self.target_key_pos[torch.arange(self.num_envs), self.key_press_progress]
                - target_finger_pos,
                dim=-1,
            )
            <= 0.01
        )

        current_target_key_indices = self.target_key_indices[
            torch.arange(self.num_envs), self.key_press_progress
        ]

        target_key_qpos = self.keyboard.qpos[
            torch.arange(self.num_envs), current_target_key_indices
        ]

        pressed = target_key_qpos > 0.002

        self.key_press_progress[pressed] = torch.clamp(
            self.key_press_progress[pressed] + 1, 0, self.num_chars - 1
        )

        finished = (self.key_press_progress == (self.num_chars - 1)) & pressed

        return {"success": reached & finished}

    def get_reward_details(self):
        return self.reward_dict

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 1, 1, torch.pi / 2, 0.01, 100)]

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=250,
            control_freq=50,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 36,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 12,
                found_lost_pairs_capacity=2**28,
            ),
        )


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

    import numpy as np

    action = np.array(
        [
            0.158,
            0.413,
            -0.791,
            0.245,
            3.1415927,
            1.324,
            -0.00349066,
            0.0,
            0.795,
            0.584,
            1.081,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    while True:
        env.unwrapped.render_human()

        # env.step(None)

        env.step(action)
