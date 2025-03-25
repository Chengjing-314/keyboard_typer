import sys
import glob
import os
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import h5py
import yaml
import numpy as np
import transforms3d
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import tyro  # Argument parsing
from fastdev.sim_webui.webui import SimWebUI

current_path = Path(__file__).parent
sys.path.append(str(current_path.parent))

from robot_agent.utils import action_in_ee_frame, update_single_arm_qpos
from robot_agent.real_env import RealEnv
from neural_teleop.models.hand_traj_state import HandTrajState
from dataclasses import dataclass

# Constants for functionality selection
USE_REAL_HAND = True
USE_REAL_ARM = True
ENABLE_LEFT_ARM = False
ENABLE_RIGHT_ARM = True
ENABLE_BIMANUAL = ENABLE_LEFT_ARM and ENABLE_RIGHT_ARM


@dataclass
class TrajReplayConfig:
    traj_path: str = "/home/jiyue/Downloads/object_success_dict.pkl"
    object_name: str = "mustard_bottle"
    replay_idx: int = 3


def initialize_real_env(config_path: str, init_qpos: tuple) -> RealEnv:
    """Initialize the Real Environment based on configuration."""
    with open(config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    right_config = yaml_config["right"]

    real_env = RealEnv(
        use_arm=USE_REAL_ARM,
        use_hand=USE_REAL_HAND,
        control_config=right_config,
        init_qpos=init_qpos,
        is_right=True,
        is_explore=True,
        use_servo_control=False,
        enable_finger_tactile=False,
        enable_palm_tactile=False,
    )
    return real_env


def initialize_starting_pose(real_env: RealEnv, target_ee_pose, target_hand_qpos, steps=30) -> None:
    target_qpos = real_env.robot_step(target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos)

    for _ in range(steps):
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()


def trajectory_replay(real_env: RealEnv, config: TrajReplayConfig):
    target_ee_pose = np.zeros(7)
    target_hand_qpos = np.zeros(6)

    trajectories = np.load(config.traj_path, allow_pickle=True)
    trajectory = trajectories[config.object_name]["traj"][config.replay_idx]

    target_ee_poses = np.zeros((trajectory.shape[0], 7))
    target_hand_qposes = trajectory[:, 6:][:, [2, 3, 4, 5, 1, 0]]

    for i in range(trajectory.shape[0]):
        target_ee_poses[i, :3] = trajectory[i, :3]
        target_ee_poses[i, 3:] = R.from_euler("XYZ", trajectory[i, 3:6]).as_quat(scalar_first=True)

    # Adjust position constraints
    min_height = max(min(target_ee_poses[:, 2]), 0.1)
    target_ee_poses[:, :3] -= target_ee_poses[0, :3]
    target_ee_poses[:, 0] += 0.15
    target_ee_poses[:, 2] += min_height

    initialize_starting_pose(real_env, target_ee_poses[0], target_hand_qposes[0])

    for i in range(260):
        target_ee_pose = target_ee_poses[i] if i < len(target_ee_poses) else target_ee_poses[-1]
        target_hand_qpos = target_hand_qposes[i] if i < len(target_hand_qposes) else target_hand_qposes[-1]

        if i > 200:
            target_hand_qpos = apply_grasping_lift(target_hand_qpos, i)

        target_qpos = real_env.robot_step(target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos)
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()


def apply_grasping_lift(qpos, i, pre_grasping_step=200, grasping_step=50):
    if pre_grasping_step < i < pre_grasping_step + grasping_step:
        scaling_factor = 1 + min((i - pre_grasping_step) / 30, 1)
        qpos[0] *= scaling_factor
        qpos[1] *= scaling_factor
        qpos[-2] *= scaling_factor
    return qpos


def main(config: TrajReplayConfig):
    left_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    right_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    init_qpos = (left_init_qpos, right_init_qpos)

    real_env = initialize_real_env(
        "/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability/real_control/configs/kinematics_config/pinocchio_bimanual_xarm7_ability.yml",
        init_qpos,
    )  # Adjust with your actual config path
    trajectory_replay(real_env, config)


if __name__ == "__main__":
    # tyro.cli(main)
    traj_path: str = "/home/jiyue/Downloads/object_success_dict.pkl"
    with open(traj_path, "rb") as f:
        trajectories = np.load(f, allow_pickle=True)

    mustard_bottle_trajs = np.stack(trajectories["mustard_bottle"]["traj"], axis=0)
    mustard_bottle_trajs = torch.from_numpy(mustard_bottle_trajs).float()

    hand_state = HandTrajState.from_variables(
        "ability_hand_manual_spheres_right", variables=mustard_bottle_trajs, rotation_mode="rxyz"
    )
    replay_idx = 7
    replay_traj = hand_state.variables[replay_idx].numpy()

    webui = SimWebUI()
    hand_state.visualize_via_webui(webui)

    # --------- init --------
    left_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    right_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    init_qpos = (left_init_qpos, right_init_qpos)
    real_env = initialize_real_env(
        "/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability/real_control/configs/kinematics_config/pinocchio_bimanual_xarm7_ability.yml",
        init_qpos,
    )

    # --------- test first frame --------
    import IPython; IPython.embed() # noqa
    tl = replay_traj[0][:3].copy()
    offset_for_7th_traj = np.array([0.8, -0.3, 0.05])
    tl += offset_for_7th_traj
    # array([0.37480436, 0.00101135, 0.03246577])

    rot = transforms3d.euler.euler2quat(*replay_traj[0][3:6], axes="rxyz")
    ee_pose = np.concatenate([tl, rot])
    hand_qpos = replay_traj[0][6:][[2, 3, 4, 5, 1, 0]]

    target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
    real_env.control_qpos(target_qpos)
    real_env.wait_until_next_control_signal()

    import IPython; IPython.embed() # noqa
    # --------- test grasp frames --------
    grasp_frame = 200
    for frame_idx in range(grasp_frame):
        tl = replay_traj[frame_idx][:3].copy()
        tl += offset_for_7th_traj
        rot = transforms3d.euler.euler2quat(*replay_traj[frame_idx][3:6], axes="rxyz")
        ee_pose = np.concatenate([tl, rot])
        hand_qpos = replay_traj[frame_idx][6:][[2, 3, 4, 5, 1, 0]]

        target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()

    for _ in range(50):
        target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()

    import IPython; IPython.embed() # noqa

    grasp_qpos_offset = np.array([0.3, 0.3, 0.3, 0.3, 0.3, -0.3])
    post_grasp_qpos = hand_qpos.copy() + grasp_qpos_offset
    for _ in range(30):
        target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=post_grasp_qpos)
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()


    # --------- test arm min pose --------
    # arm_min_qpos_in_degree = np.array([-14.4, 40.7, 10.7, 32.4, 173.9, 96.8, 0])
    # arm_min_qpos = arm_min_qpos_in_degree / 180 * np.pi

    # min_pose = real_env.robot.compute_ee_pose(arm_min_qpos)
    # min_quat = transforms3d.quaternions.mat2quat(min_pose.rotation)
    # min_ee_pose_7d = np.concatenate([min_pose.translation, min_quat])
    # min_qpos = real_env.robot.hand.get_hand_state()['raw_pos']
    # min_z = min_pose.translation[2]

