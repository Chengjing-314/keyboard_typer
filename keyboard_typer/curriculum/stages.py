from abc import ABC, abstractmethod
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.agents.base_agent import BaseAgent

from keyboard_typer.utils.keyboard.keyboard import Keyboard

import numpy as np




class Stage(ABC):
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


class StageHandler(Stage):
    def initialize(
        self,
        env_idx: torch.Tensor,
        num_chars: int,
        keyboard_pos: torch.Tensor,
        num_envs: int,
        device: torch.device,
        agent: BaseAgent,
        tcp_viz: Actor,
        goal_viz: Actor,
        kb_manager: Keyboard,
    ):
        num_envs = len(env_idx)
        key_pos = kb_manager.get_default_key_pos()
        key_pos = torch.tensor(key_pos, device=device).unsqueeze(0).repeat(
            num_envs, 1, 1
        ) + keyboard_pos.repeat(num_envs, 1, 1)

        # target_finger = torch.randint(1, 2, (num_envs,), device=device)

        target_finger = torch.randint(
            1, 2, (num_envs, num_chars), device=device
        )  # FIXME: only training for 1 finger

        self.target_key_press, self.target_key_indices = kb_manager.simulate_key_presses_over_time(
            num_chars, num_envs
        )

        self.target_key_indices = torch.tensor(np.array(self.target_key_indices).T, device=device)

        batch_indices = (
            torch.arange(num_envs, device=key_pos.device)
            .unsqueeze(1)
            .expand(-1, self.target_key_indices.shape[1])
        )

        target_key_pos = key_pos[batch_indices, self.target_key_indices]

        self.target_key_pos = target_key_pos[:, 0]

        finger_tip_pos = agent.get_finger_tip_pos(flatten=False)
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), target_finger[:, 0]]

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))
        goal_viz.set_pose(Pose.create_from_pq(self.target_key_pos))

        return target_key_pos, target_finger, self.target_key_indices

    def get_obs(
        self,
        key_press_progress: torch.Tensor,
        num_chars: int,
        finger_tip_pos: torch.Tensor,
        target_finger: torch.Tensor,
        target_key_pos: torch.Tensor,
        keyboard_qpos: torch.Tensor,
        hand_qpos: torch.Tensor,    
        desired_hand_qpos: torch.Tensor,
    ):
        num_envs = len(finger_tip_pos)

        current_target_finger = target_finger[torch.arange(num_envs), key_press_progress]
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), current_target_finger]

        current_target_key_pos = target_key_pos[
            torch.arange(num_envs),
            key_press_progress,
        ]

        tcp_distance = torch.norm(target_finger_pos - current_target_key_pos, dim=-1)

        mask = key_press_progress < num_chars - 1
        next_key_press_progress = key_press_progress.clone()

        next_key_press_progress[mask] += 1

        batch_idx = torch.arange(num_envs, device=target_key_pos.device)
        one_step_look_ahead_key_pos = torch.zeros_like(current_target_key_pos)
        one_step_look_ahead_finger_idx = torch.zeros_like(
            current_target_finger, device=current_target_finger.device
        )  # This might need to change if we add thumb as it is index 0

        one_step_look_ahead_key_pos[mask] = target_key_pos[
            batch_idx[mask],
            next_key_press_progress[mask],
        ]

        one_step_look_ahead_finger_idx[mask] = target_finger[
            batch_idx[mask],
            next_key_press_progress[mask],
        ]
        
        qpos_distance = torch.norm(hand_qpos - desired_hand_qpos, dim=-1) 
        
        obs = dict(
            target_finger_idx=current_target_finger,
            one_step_look_ahead_finger_idx=one_step_look_ahead_finger_idx,
            target_finger_pos=target_finger_pos,
            current_target_key_pos=current_target_key_pos,
            one_step_look_ahead_key_pos=one_step_look_ahead_key_pos,
            keyboard_qpos=keyboard_qpos,
            tcp_distance=tcp_distance,
            qpos_distance=qpos_distance
        )

        return obs

    def compute_reward(
        self,
        info,
        action,
        key_press_progress,
        finger_tip_pos,
        target_finger,
        target_key_pos,
        target_key_indices,
        wrist_rot,
        qpos,
        qvel,
        tcp_viz,
        goal_viz,
        keyboard_qpos,
        key_default_qpos,
        desired_hand_qpos,
        desired_wrist_rot,
    ):
        num_envs = len(finger_tip_pos)
        current_target_finger = target_finger[
            torch.arange(num_envs),
            key_press_progress,
        ]

        target_finger_pos = finger_tip_pos[torch.arange(num_envs), current_target_finger]
        current_target_key_pos = target_key_pos[
            torch.arange(num_envs),
            key_press_progress,
        ]

        tcp_distance = torch.norm(target_finger_pos - current_target_key_pos, dim=-1)
        tcp_distance_reward =  - torch.tanh(5 * tcp_distance) # this encourage the finger to move closer to the key as a 0.2 distance will be -1

        hand_qpos = qpos[:, -10:]
        hand_qpos_reward = -torch.tanh(
            0.5 * torch.norm(hand_qpos - desired_hand_qpos.unsqueeze(0), dim=-1)
        )
        
        qvel_arm_only = qvel[:, :7]
        qvel_penalty = -torch.tanh(0.03 * torch.norm(qvel_arm_only, dim=-1))

        current_target_key_indices = target_key_indices[
            torch.arange(num_envs),
            key_press_progress,
        ]
        target_key_qpos = keyboard_qpos[torch.arange(num_envs), current_target_key_indices]
        press_ratio = (target_key_qpos - key_default_qpos) / ((0.002 - key_default_qpos) + 1e-6)
        key_actuation_reward = 2 * torch.tanh(press_ratio)

        num_keys = keyboard_qpos.shape[1]
        all_key_indices = torch.arange(num_keys, device=target_key_indices.device).repeat(
            num_envs, 1
        )
        mask = all_key_indices != current_target_key_indices.unsqueeze(-1)
        non_target_key_indices = all_key_indices[mask].view(num_envs, -1)
        non_target_key_qpos = keyboard_qpos[
            torch.arange(num_envs).unsqueeze(-1), non_target_key_indices
        ]
        non_target_key_activation_count = (non_target_key_qpos > 0.002).sum(dim=-1)
        exist_wrong_key = non_target_key_activation_count > 0 
        non_target_activation_penality = exist_wrong_key * -0.1
        time_penality = -0.05 # This is needed to prevent the agent from not pressing the key
        
        
        arm_action = action[:, :7] 
        action_regularization = -0.2 * torch.norm(arm_action, dim=-1)
        

        
        distance_reward_weight = 1.0
        velocity_penalty_weight = 8.0
        actuation_reward_weight = 3.0 # 2.0 as base, 3.0 to encourage pressing 
        hand_pose_weight = 5.0
        action_regularization_weight = 8.0 

        reward = (
            distance_reward_weight * tcp_distance_reward
            + velocity_penalty_weight * qvel_penalty
            + actuation_reward_weight * key_actuation_reward
            + hand_pose_weight * hand_qpos_reward
            + action_regularization_weight * action_regularization
            + non_target_activation_penality
            + time_penality
        )

        reward[info["success"]] += 3
        
        reward /= 16.0

        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))
        goal_viz.set_pose(Pose.create_from_pq(current_target_key_pos))

        rot_dot_product = torch.abs(torch.sum(wrist_rot * desired_wrist_rot, dim=-1))
        rot_dot_product = torch.clamp(rot_dot_product, min=0.0, max=1.0)
        
        theta = torch.acos(rot_dot_product)
        theta_penalty = -2 *torch.tanh(0.3 * theta)

        reward_dict = dict(
            tcp_distance_reward=tcp_distance_reward.mean(dim=-1).item(),
            over_all_distance_reward=0,  # Placeholder
            rotation_distance_reward=theta_penalty.mean(dim=-1).item(),  # Placeholder
            key_actuation_reward=key_actuation_reward.mean(dim=-1).item(),
            velocity_penalty=qvel_penalty.mean(dim=-1).item(),  # Placeholder
            # velocity_penalty=0,
            wrong_key_penalty=non_target_activation_penality.mean(dim=-1).item(),
            # wrong_key_penalty=0,
            hand_qpos_reward=hand_qpos_reward.mean(dim=-1).item(),
            action_regularization=action_regularization.mean(dim=-1).item(),
        )

        return reward, reward_dict
