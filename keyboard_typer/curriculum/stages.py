from abc import ABC, abstractmethod
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.agents.base_agent import BaseAgent

from keyboard_typer.utils.keyboard.keyboard import Keyboard

import numpy as np


class StageHandler(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_obs(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        pass


class StageZeroHandler(StageHandler):
    def initialize(
        self,
        keyboard_pos: torch.Tensor,
        num_envs: int,
        device: torch.device,
        agent: BaseAgent,
        tcp_viz: Actor,
        goal_viz: Actor,
        kb_manager: Keyboard = None,
    ):
        target_key_pos = keyboard_pos + torch.tensor([0, 0, 0.05], device=device).unsqueeze(
            0
        ).repeat(num_envs, 1)
        target_finger = torch.randint(0, 5, (num_envs,), device=device)

        finger_tip_pos = agent.get_finger_tip_pos(flatten=False)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))
        goal_viz.set_pose(Pose.create_from_pq(target_key_pos))

        return target_key_pos, target_finger

    def get_obs(self, finger_tip_pos, target_finger, target_key_pos):
        num_envs = len(finger_tip_pos)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]
        finger_tip_pos_flat = finger_tip_pos.flatten(start_dim=1)

        tcp_distance = torch.norm(target_finger_pos - target_key_pos, dim=-1)

        obs = dict(
            target_finger_idx=target_finger.unsqueeze(-1),
            finger_tip_pos=finger_tip_pos_flat,
            target_key_pos=target_key_pos,
            tcp_distance=tcp_distance,
        )

        return obs

    def compute_reward(self, finger_tip_pos, target_finger, target_key_pos, qvel, tcp_viz):
        num_envs = len(finger_tip_pos)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]
        tcp_distance = torch.norm(target_finger_pos - target_key_pos, dim=-1)

        tcp_distance_reward = torch.exp(-5 * tcp_distance)
        qvel_penalty = torch.exp(-0.1 * torch.norm(qvel, dim=-1))

        reward = tcp_distance_reward + qvel_penalty

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))

        reward_dict = dict(
            tcp_distance_reward=tcp_distance_reward.mean(dim=-1).item(),
            over_all_distance_reward=0,  # Placeholder
            rotation_distance_reward=0,  # Placeholder
            key_actuation_reward=0,  # Placeholder
            velocity_penalty=qvel_penalty.mean(dim=-1).item(),  # Placeholder
            wrong_key_penalty=0,  # Placeholder
        )

        return reward, reward_dict


class StageOneHandler(StageHandler):
    def initialize(
        self,
        keyboard_pos: torch.Tensor,
        num_envs: int,
        device: torch.device,
        agent: BaseAgent,
        tcp_viz: Actor,
        goal_viz: Actor,
        kb_manager: Keyboard = None,
    ):
        key_pos = kb_manager.get_default_key_pos()
        key_pos = torch.tensor(key_pos, device=device).unsqueeze(0).repeat(
            num_envs, 1, 1
        ) + keyboard_pos.repeat(num_envs, 1, 1)

        target_finger = torch.randint(0, 5, (num_envs,), device=device)

        self.target_key_press, self.target_key_indices = kb_manager.simulate_key_presses_over_time(
            1, num_envs
        )  # only one key press
        self.target_key_indices = torch.tensor(np.array(self.target_key_indices).T, device=device)
        target_key_pos = key_pos[
            torch.arange(num_envs), self.target_key_indices[:, 0]
        ] + torch.tensor([0, 0, 0.05], device=device)

        self.target_key_pos = target_key_pos

        finger_tip_pos = agent.get_finger_tip_pos(flatten=False)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))
        goal_viz.set_pose(Pose.create_from_pq(self.target_key_pos))

        return target_key_pos, target_finger

    def get_obs(self, finger_tip_pos, target_finger, target_key_pos):
        num_envs = len(finger_tip_pos)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]
        finger_tip_pos_flat = finger_tip_pos.flatten(start_dim=1)

        tcp_distance = torch.norm(target_finger_pos - target_key_pos, dim=-1)

        obs = dict(
            target_finger_idx=target_finger.unsqueeze(-1),
            finger_tip_pos=finger_tip_pos_flat,
            target_key_pos=target_key_pos,
            tcp_distance=tcp_distance,
        )

        return obs

    def compute_reward(self, finger_tip_pos, target_finger, target_key_pos, qvel, tcp_viz):
        num_envs = len(finger_tip_pos)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger]
        tcp_distance = torch.norm(target_finger_pos - target_key_pos, dim=-1)

        tcp_distance_reward = torch.exp(-5 * tcp_distance)
        distance_scale = torch.sigmoid(10 * (tcp_distance - 0.1))
        qvel_penalty = torch.exp(-0.1 * torch.norm(qvel, dim=-1)) * distance_scale

        reward = tcp_distance_reward + qvel_penalty

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))

        reward_dict = dict(
            tcp_distance_reward=tcp_distance_reward.mean(dim=-1).item(),
            over_all_distance_reward=0,  # Placeholder
            rotation_distance_reward=0,  # Placeholder
            key_actuation_reward=0,  # Placeholder
            velocity_penalty=qvel_penalty.mean(dim=-1).item(),  # Placeholder
            wrong_key_penalty=0,  # Placeholder
        )

        return reward, reward_dict
