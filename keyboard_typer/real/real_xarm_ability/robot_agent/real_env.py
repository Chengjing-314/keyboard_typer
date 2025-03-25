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
# import open3d as o3d
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat, mat2euler, euler2mat, quat2euler
# from sim_web_visualizer.parser.yourdfpy import URDF

from keyboard_typer.real.real_xarm_ability.real_control.xarm7_ability import XArm7Ability
from keyboard_typer.real.real_xarm_ability.control.motion_control_config import MotionControlConfig

from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat, quat2euler

from keyboard_typer.real.real_xarm_ability.robot_agent.utils import (
    parse_args,
    action_in_ee_frame,
    update_single_arm_qpos,
    pose_to_matrix,
    update_rotation,
)


class RealEnv:
    def __init__(
        self,
        use_arm,
        use_hand,
        control_config,
        init_qpos,
        is_right=True,
        is_explore=True,
        use_servo_control=False,
        enable_finger_tactile=True,
        enable_palm_tactile=False,
        enable_joint_teaching=False,
    ):
        motion_control_config = MotionControlConfig.from_dict(control_config["control"])
        self.motion_control = (
            motion_control_config.build(device="cuda:0") if control_config is not None else None
        )

        self.is_right = is_right

        if self.is_right:
            self.motion_control.set_current_qpos(init_qpos[1])
        else:
            self.motion_control.set_current_qpos(init_qpos[0])

        self.robot = XArm7Ability(
            use_arm=use_arm,
            use_hand=use_hand,
            # arm_ip="192.168.1.242",
            arm_ip="192.168.1.209",
            # hand_tty_index="usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0",
            hand_tty_index="usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
            is_right=self.is_right,
            use_servo_control=use_servo_control,
            enable_finger_tactile=enable_finger_tactile,
            enable_palm_tactile=enable_palm_tactile,
            finger_port='/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44231313430351F0D1B1-if00',
            # finger_port="/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_4423131343035140A011-if00",
            palm_port="/dev/ttyACM1",
            enable_joint_teaching=enable_joint_teaching,
        )

        self.robot.reset()

        # For safety, we set the velocity to a lower value during initialization
        if not use_servo_control:
            original_velocity_limit = self.robot.max_arm_velocity
            new_velocity_limit = original_velocity_limit / 3
            self.robot.max_arm_velocity = new_velocity_limit
        else:
            original_max_velocity = self.robot.arm_velocity_limit
            new_velocity_limit = original_max_velocity / 3
            self.robot.arm_velocity_limit = new_velocity_limit

        if not enable_joint_teaching:
            if self.is_right:
                self.robot.control_arm_qpos(init_qpos[1][:7])
            else:
                self.robot.control_arm_qpos(init_qpos[0][:7])

        self.robot.start()

        # Set back the robot velocity to original
        if not use_servo_control:
            # left_robot.max_arm_velocity = original_velocity_limit
            self.robot.max_arm_velocity = original_velocity_limit
        else:
            # left_robot.arm_velocity_limit = original_max_velocity
            self.robot.arm_velocity_limit = original_max_velocity

        self.robot.hand.start_thread()

        ee_pose = self.motion_control.compute_ee_pose(self.motion_control.get_current_qpos())
        self.init_rot = R.from_quat((ee_pose[4], ee_pose[5], ee_pose[6], ee_pose[3])).as_matrix()

    # call it before control
    def wait_until_next_control_signal(self):
        self.robot.wait_until_next_control_signal()

    # in the main, call the funtion in a loop at the final step of control
    def control_qpos(self, target_qpos):
        self.robot.control_arm_qpos(target_qpos[:7])
        self.robot.control_hand_qpos(target_qpos[7:])

    def robot_step(self, target_ee_pose, hand_joint_angle):
        hand_rot = target_ee_pose[3:]
        # hand_rot_world = R.from_matrix(self.init_rot).as_quat(scalar_first=True)
        hand_rot_world = hand_rot
        hand_rot = hand_rot_world
        # target_ee_pose[3:] = np.array([1, 0, 0, 0])
        target_ee_pose[3:] = hand_rot
        # target_ee_pose[:3] = np.dot(hand_rot_world.as_matrix(), target_ee_pose[:3])

        # import pdb; pdb.set_trace()

        new_qpos = update_single_arm_qpos(target_ee_pose, self.motion_control, repeat_times=1)
        self.motion_control.set_current_qpos_with_hand(
            new_qpos, self.robot.hand.joint_remap_6_qpos_to_10(hand_joint_angle)
        )

        return np.concatenate([new_qpos[:7], hand_joint_angle])

    def update_motion_control_qpos(self):
        hand_state = self.robot.hand.get_hand_state()
        finger_qpos = hand_state["raw_pos"]
        robot_qpos = self.motion_control.get_current_qpos()
        self.motion_control.set_current_qpos_with_hand(
            robot_qpos, self.robot.hand.joint_remap_6_qpos_to_10(finger_qpos)
        )

    def get_real_obs(self):
        hand_state = self.robot.hand.get_hand_state()

        obs = []

        return obs

    def get_touch_state(self):
        hand_state = self.robot.hand.get_hand_state()
        touch = hand_state["touch"]
        touch = touch[[13, 11, 12, 9, 7, 8, 10, 5, 6, 4, 0, 2, 3, 1]]
        touch = np.insert(touch, [0, 3, 5, 5, 7, 10], 0)

        return touch

    def agent_ee_step(self, obs):
        action = int(self.agent.step(obs))
        # import pdb; pdb.set_trace()
        print("========action========")
        print(action)
        if action in self.action_to_direction.keys():
            local_ee_action = self.action_to_direction[action]
        elif action == 12:
            local_ee_action = np.zeros(6)
        else:
            raise ValueError("Invalid action")

        hand_state = self.robot.hand.get_hand_state()
        hand_joint_angle = hand_state["raw_pos"]
        return local_ee_action, hand_joint_angle
