from abc import ABC, abstractmethod
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.agents.base_agent import BaseAgent

from keyboard_typer.utils.keyboard.keyboard import Keyboard
from keyboard_typer.utils.reward.reward_util import tolerance_torch
from dataclasses import dataclass


import numpy as np


@dataclass
class StageConfig:
    distance_reward_weight: float = 1.0
    actuation_reward_weight: float = 3.0
    hand_qpos_reward_weight: float = 5.0
    action_regularization_weight: float = 8.0
    wrong_key_penality_weight: float = 0.1
    penetration_penalty_weight: float = 0.1
    penetration_zone: float = 1e-3

class Stage(ABC):
    def __init__(self, env, config: StageConfig):
        self.env = env
        self.config = config
        self.distance_reward_weight = config.distance_reward_weight
        self.actuation_reward_weight = config.actuation_reward_weight
        self.hand_qpos_reward_weight = config.hand_qpos_reward_weight
        self.penetration_penalty_weight = config.penetration_penalty_weight
        self.action_regularization_weight = config.action_regularization_weight
        self.wrong_key_penality_weight = config.wrong_key_penality_weight
        self.penetration_zone = config.penetration_zone

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
            qpos_distance=qpos_distance,
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
        # Select the current target finger for each environment.
        current_target_finger = target_finger[
            torch.arange(num_envs),
            key_press_progress,
        ]
        target_finger_pos = finger_tip_pos[torch.arange(num_envs), current_target_finger]
        current_target_key_pos = target_key_pos[
            torch.arange(num_envs),
            key_press_progress,
        ]

        # ----------------------------
        # TCP Distance Reward
        # ----------------------------
        # Compute distance between target finger position and key position.
        tcp_distance = torch.norm(target_finger_pos - current_target_key_pos, dim=-1)
        tcp_distance_reward = torch.exp(-tcp_distance / 0.08)
        
        # ----------------------------
        # Non-target Finger Position Penalty
        # ----------------------------
        # Get Z-coordinates of all finger tips
        finger_tip_z = finger_tip_pos[:, :, 2]  # Extract Z-axis

        # Get Z-coordinates of non-target fingers
        finger_mask = torch.ones_like(finger_tip_z, dtype=torch.bool)
        finger_mask[torch.arange(num_envs), current_target_finger] = False
        non_target_finger_z = finger_tip_z[finger_mask].view(num_envs, -1)

        # Define keyboard height threshold (assuming key_default_qpos is the resting height)
        keyboard_height_threshold = target_key_pos[:,0,2] + self.penetration_zone 
        min_non_target_finger_z = torch.min(non_target_finger_z, dim=-1)[0]

        # Compute penetration depth (how far below the keyboard they go)
        penetration_depth = torch.clamp(keyboard_height_threshold - min_non_target_finger_z, min=0, max  = 0.02)

        # Apply exponential penalty (similar to rewards)
        penetration_penalty = -torch.exp(penetration_depth / 0.02)  + 1

        # ----------------------------
        # Key Actuation Reward
        # ----------------------------
        current_target_key_indices = target_key_indices[
                torch.arange(num_envs),
                key_press_progress,
            ]
        target_key_qpos = keyboard_qpos[torch.arange(num_envs), current_target_key_indices]

        # Define actuation threshold
        actuation_threshold = 0.0025
        press_progress = torch.clamp(target_key_qpos / actuation_threshold, 0, 1)
        key_actuation_reward = torch.exp(-(1 - press_progress) / 0.5) # 0.5 for 1st exp
        
        
        # ----------------------------
        # Hand Qpos Reward
        # ----------------------------
        hand_qpos = qpos[:, -10:]
        hand_qpos_weight = torch.tensor([1.0,0.3, 1.0, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0], device=hand_qpos.device).unsqueeze(0)
        weighted_hand_qpos_distance = (hand_qpos - desired_hand_qpos) * hand_qpos_weight
        qpos_distance = torch.norm(weighted_hand_qpos_distance, dim=-1)
        hand_qpos_reward = torch.exp(-qpos_distance / 5.0) # was 2.5
        
        # ----------------------------
        # Non-target Key Penalty
        # ----------------------------
        num_keys = keyboard_qpos.shape[1]
        all_key_indices = torch.arange(num_keys, device=target_key_indices.device).repeat(
            num_envs, 1
        )
        mask = all_key_indices != current_target_key_indices.unsqueeze(-1)
        non_target_key_indices = all_key_indices[mask].view(num_envs, -1)
        non_target_key_qpos = keyboard_qpos[
            torch.arange(num_envs).unsqueeze(-1), non_target_key_indices
        ]
        non_target_key_activation_count = (non_target_key_qpos > 0.0025).sum(dim=-1)
        exist_wrong_key = non_target_key_activation_count > 0
        # non_target_activation_penalty = (exist_wrong_key - 1) * 0.5   # if there is wrong key, we dont give additional reward

        # ----------------------------
        # Action Regularization
        # ----------------------------
        arm_action = action[:, :7]
        action_norm = torch.norm(arm_action, dim=-1)
        # Here we assume the maximum expected norm is 1.0; adjust if needed.
        normalized_action_norm = action_norm  # If max norm is 1, then norm is already normalized.
        action_regularization = -self.action_regularization_weight * normalized_action_norm

        # ----------------------------
        # Combine Rewards
        # ----------------------------
        reward = (
            self.distance_reward_weight * tcp_distance_reward
            + self.actuation_reward_weight * key_actuation_reward
            + self.hand_qpos_reward_weight * hand_qpos_reward
            # + self.penetration_penalty_weight * penetration_penalty
        )

        # reward /= (  self.distance_reward_weight + self.actuation_reward_weight + self.penetration_penalty_weight) #! no hand qpos reward for testing 
        reward /= (  self.distance_reward_weight + self.actuation_reward_weight + self.hand_qpos_reward_weight) #! no hand qpos reward for testing 

        # Apply a bonus for success.
        # reward[info["pressed"]] += 5
        reward[info["success"]] += 10 
        # reward /= (
        #     5
        #     + self.actuation_reward_weight
        #     + self.distance_reward_weight
        #     + self.action_regularization_weight
        # )

        # ----------------------------
        # Visualization Updates
        # ----------------------------
        tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))
        goal_viz.set_pose(Pose.create_from_pq(current_target_key_pos))

        reward_dict = dict(
            tcp_distance_reward=tcp_distance_reward.mean().item() * self.distance_reward_weight,
            key_actuation_reward=key_actuation_reward.mean().item() * self.actuation_reward_weight,
            # key_actuation_reward=0,
            wrong_key_penalty=0,
            # action_regularization=action_regularization.mean().item()
            # * self.action_regularization_weight,
            action_regularization=0,
            hand_qpos_reward=hand_qpos_reward.mean().item() * self.hand_qpos_reward_weight,
            penetration_penalty=penetration_penalty.mean().item() * self.penetration_penalty_weight,
        )

        return reward, reward_dict
