import sys
import glob
import os
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict

import argparse
import h5py
import yaml
import numpy as np
import transforms3d
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# from sim_web_visualizer.parser.yourdfpy import URDF

import sys 
sys.path.append("/home/jiyue/Documents/repo/real_xarm_ability")

from real_control.xarm7_ability import XArm7Ability
from control.motion_control_config import MotionControlConfig

from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat, quat2euler

from robot_agent.utils import parse_args, action_in_ee_frame, update_single_arm_qpos
from robot_agent.real_env import RealEnv

TEST_FUNC = 0

USE_REAL_HAND = True
USE_REAL_ARM = True

ENABLE_LEFT_ARM = False
ENABLE_RIGHT_ARM = True
ENABLE_BIMANUAL = ENABLE_LEFT_ARM and ENABLE_RIGHT_ARM
    
def main():
    args = parse_args()
    kinematics_config = args.kinematics_cfg
    kinematics_path = Path(kinematics_config)
    if not Path(kinematics_config).is_absolute():
        kinematics_path = kinematics_path.absolute()
    with kinematics_path.open("r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        left_config = yaml_config["left"]
        right_config = yaml_config["right"]
    
    sys.argv = [sys.argv[0]]

    # # # Default
    left_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    right_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])

    init_qpos = (left_init_qpos, right_init_qpos)
    
    robot_base_pose=(
        np.array([0, 0.4, 0, 1, 0, 0, 0]),
        np.array([0, -0.4, 0, 1, 0, 0, 0]),
    )
    
    # init qpos
    # left_joint_names = motion_controls[0].get_joint_names()
    # right_joint_names = motion_controls[0].get_joint_names()
    
    print("========init real env========")

    real_env = RealEnv(
        use_arm=USE_REAL_ARM,
        use_hand=USE_REAL_HAND,
        control_config=right_config,
        init_qpos=init_qpos,
        is_right=True,
        is_explore=True,
        use_servo_control=False,
        enable_finger_tactile=False,
        enable_palm_tactile=False
    )
    
    print("test funtions: ", TEST_FUNC)

    try:
        if TEST_FUNC == 0:
            test_basic(real_env)

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down.\n")
        # right_robot.stop()
        real_env.stop()
        # right_robot.stop()

def initialize_starting_pose(robot, motion_control):
    pass
        

def test_basic(real_env: RealEnv):
    target_ee_pose = np.zeros(7)
    target_hand_qpos = np.zeros(6)
    
    initialize_starting_pose(real_env, real_env.motion_control)
    
    # trajectory = np.load("/home/jiyue/Downloads/hand_actions.npy")
    
    trajectories = np.load("/home/jiyue/Downloads/object_success_dict.pkl", allow_pickle=True)
    
    print(trajectories.keys())
    
    replay_object_name = "mustard_bottle"
    
    idx = 3
    
    trajectory = trajectories[replay_object_name]['traj'][idx]
    
    print(trajectories[replay_object_name]['idx'][idx])
    
    # import IPython; IPython.embed()
    
    
    from scipy.spatial.transform import Rotation as R
    
    # preprocess
    
    assert trajectory.shape[1] == 12
    
    target_ee_poses = np.zeros((trajectory.shape[0], 7))
    target_hand_qposes = trajectory[:, 6:]
    target_hand_qposes = target_hand_qposes[:, [2, 3, 4, 5, 1, 0]]
 
    
    for i in range(trajectory.shape[0]):
        target_ee_poses[i, :3] = trajectory[i, :3]
        target_ee_poses[i, 3:] = R.from_euler("XYZ", trajectory[i, 3:6]).as_quat(scalar_first=True)
        
    
    min_height = min(target_ee_poses[:, 2])
    
    if min_height < 0.1:
        min_height += 0.1
    else:
        min_height += 0.055
    
    target_ee_poses[:, :3] -= target_ee_poses[0, :3]
    
    print("min_height: ", min_height)
        
    
    target_ee_poses[:,0] += 0.15
    
    max_reach_foward = max(target_ee_poses[:,0])
    max_side_way = max(target_ee_poses[:,1])
    
    if max_reach_foward > 0.4:
        target_ee_poses[:,0] -= (max_reach_foward - 0.45)
        
    if max_side_way > 0.05:
        target_ee_poses[:,1] -= (max_side_way - 0.1)
    
    
    target_ee_poses[:,2] += min_height 
    
    
    # target_ee_poses[:, :3] = np.zeros_like(target_ee_poses[:, :3])
    # target_ee_poses[:,0] = 0.3
    # target_ee_poses[:,2] = 0.15
    # target_ee_poses[:, 3] = 1.0
    # target_ee_poses[:, 4] = 0
    # target_ee_poses[:, 5] = 0
    # target_ee_poses[:, 6] = 0
    
    
    
    
    # print("target_ee_poses: ", target_ee_poses[:10])
    
    # print("x_max, x_min: ", np.max(target_ee_poses[:,0]), np.min(target_ee_poses[:,0]))
    # print("y_max, y_min: ", np.max(target_ee_poses[:,1]), np.min(target_ee_poses[:,1]))
    # print("z_max, z_min: ", np.max(target_ee_poses[:,2]), np.min(target_ee_poses[:,2]))
    
    # exit(0)
        
    # pre-step 30 steps
    
    for i in range(30):
        target_qpos = real_env.robot_step(target_ee_pose=target_ee_poses[0], hand_joint_angle=target_hand_qposes[0])
        real_env.wait_until_next_control_signal()
        # time.sleep(0.1)
 
    
    # for target_ee_pose, target_hand_qpos in zip(target_ee_poses, target_hand_qposes):
    #     target_qpos = real_env.robot_step(target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos)
    #     real_env.wait_until_next_control_signal()
    #     for _ in range(10):
    #         real_env.control_qpos(target_qpos)
            # time.sleep(0.1)
            
            
    pre_grasping_pause_frame = 30
    pre_grasping_step = 200
    grasping_step = 50
            
    for i in range(260 + pre_grasping_pause_frame):
        target_ee_pose = target_ee_poses[i]
        print(target_ee_pose[:3])
        target_hand_qpos = target_hand_qposes[i]
        
        
        if i > pre_grasping_step and i < pre_grasping_step + pre_grasping_pause_frame:
            
            target_hand_qpos = target_hand_qposes[199] 
            
            print("pausing")
            
        if i > pre_grasping_step + pre_grasping_pause_frame:
            
            target_hand_qpos[0] *= (1 + 1 * min((i - pre_grasping_step - pre_grasping_pause_frame) / 30, 1))
            target_hand_qpos[1] *= (1 + 1 * min((i - pre_grasping_step - pre_grasping_pause_frame) / 30, 1))
            target_hand_qpos[-2] *= (1 + 1 * min((i - pre_grasping_step - pre_grasping_pause_frame) / 30, 1))
        
            if i > pre_grasping_pause_frame + pre_grasping_step:
                print("grabbing")
            
            if i > pre_grasping_pause_frame + pre_grasping_step + grasping_step:
                print("lifting")
           
        
        target_qpos = real_env.robot_step(target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos) 
        real_env.wait_until_next_control_signal()
        for _ in range(15):
            real_env.control_qpos(target_qpos)
            # time.sleep(0.1)
    
    # IPython.embed()
    
    
    # for i in range(50):
    #     target_ee_pose = target_ee_poses[100 + i]
    #     target_hand_qpos = target_hand_qposes[100 + i]
        
    #     target_qpos = real_env.robot_step(target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos)
    #     real_env.wait_until_next_control_signal()
    #     for _ in range(10):
    #         real_env.control_qpos(target_qpos)
            # time.sleep(0.1)


if __name__ == "__main__":
    main()
    # draw_fsr_pcd()
