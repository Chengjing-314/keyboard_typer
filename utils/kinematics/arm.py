import os
from timeit import default_timer as timer

import numpy as np
import torch

import pytorch_kinematics as pk
import pytorch_seed

import sapien
from scipy.spatial.transform import Rotation as R
import pytransform3d
from scipy.interpolate import interp1d
from pytransform3d.rotations import (
    quaternion_slerp,
    quaternion_from_matrix,
    matrix_from_quaternion,
    active_matrix_from_intrinsic_euler_xyz,
    intrinsic_euler_xyz_from_active_matrix,
)
from .ability_hand import batch_quaternion_to_euler_intrinsic_xyz


def interpolate_rpy(init_quaternion, goal_quaternions, steps):
    rpy_interpolated_steps = []
    for i in range(steps):

        quaternions_interpolated = np.array(
            [
                quaternion_slerp(init_quaternion[0], goal_quaternions[j], i / (steps - 1))
                for j in range(goal_quaternions.shape[0])
            ]
        )

        interpolated_rotations = np.array(
            [matrix_from_quaternion(quat) for quat in quaternions_interpolated]
        )

        rpy_interpolated = np.array(
            [intrinsic_euler_xyz_from_active_matrix(matrix) for matrix in interpolated_rotations]
        )
        rpy_interpolated_steps.append(rpy_interpolated)

    rpy_interpolated = torch.tensor(rpy_interpolated_steps, dtype=torch.float64)
    return rpy_interpolated


def compute_arm_fk(chain, qpos):
    ret = chain.forward_kinematics(qpos, end_only=False)
    # look up the transform for a specific link
    tg = ret["ee_link"]
    # get transform matrix (1,4,4), then convert to separate position and unit quaternion
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot


def compute_arm_ik(offset, arm, urdf_path, robot_keyframe, goal_in_world_frame, device):
    steps = 5
    robot_root_pose = robot_keyframe.pose
    init_qpos = robot_keyframe.qpos[:7]
    chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), "ee_link").to(
        dtype=torch.float64, device=device
    )
    robot_pos = torch.tensor(robot_root_pose.p).to(device)
    robot_rot = pytransform3d.rotations.euler_from_quaternion(
        torch.tensor(robot_root_pose.q), 0, 1, 2, False
    )
    robot_rot = torch.tensor(robot_rot, dtype=torch.float64).to(device)
    robot_root = pk.Transform3d(dtype=torch.float64, pos=robot_pos, rot=robot_rot, device=device)

    # goal in world frame
    goal_pos = goal_in_world_frame[:, :3] + offset
    goal_rot = goal_in_world_frame[:, 3:]

    # init in root frame
    init_pos, init_quat = compute_arm_fk(chain, init_qpos)
    init_sapirn_pos = arm.wrist_link.pose.p

    goal_in_world_frame = pk.Transform3d(
        dtype=torch.float64, pos=goal_pos, rot=goal_rot, device=device
    )
    goal_in_robot_frame = robot_root.inverse().compose(goal_in_world_frame)

    # goal in root frame
    goal_pos = goal_in_robot_frame.get_matrix()[:, :3, 3]
    goal_quat = pk.matrix_to_quaternion(goal_in_robot_frame.get_matrix()[:, :3, :3])

    # iterpolation
    interpolated_xyz = torch.tensor(
        np.linspace(init_pos.cpu().numpy(), goal_pos.cpu().numpy(), steps)
    ).to(device)
    interpolated_rpy = interpolate_rpy(init_quat.cpu().numpy(), goal_quat.cpu().numpy(), steps).to(
        device
    )

    limit = torch.tensor(chain.get_joint_limits(), dtype=torch.float64, device=device)
    cur_qpos = arm.robot.get_qpos()[:, :7].to(dtype=torch.float64)

    for i in range(steps):
        interpolated_goal_in_robot_frame = pk.Transform3d(
            dtype=torch.float64, pos=interpolated_xyz[i], rot=interpolated_rpy[i], device=device
        )

        ik = pk.PseudoInverseIK(
            chain,
            retry_configs=cur_qpos,
            max_iterations=30,
            num_retries=1,
            joint_limits=limit.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2,
        )
        # solve IK
        sol = ik.solve(interpolated_goal_in_robot_frame)
        solutions = sol.solutions
        batch_indices = torch.arange(solutions.shape[0], device=device)
        if sol.converged.all() == True:
            selected_solutions = solutions[:, 0, :].view(-1, 7)
        # num goals x num retries x DOF tensor of joint angles; if not converged, best solution found so far
        else:
            raise ValueError("ik not converge")

        cur_qpos = selected_solutions

    return selected_solutions


if __name__ == "__main__":
    urdf_path = "/data/keyi/code/Assistive_Teleop_Maniskill3/assets/robot/heti/xarm7_ability/xarm7_ability_left_hand_glb_nmm_cs.urdf"
    device = torch.device("cuda:0")
    robot_root_pose = sapien.Pose([0, 0.4, 0], [1, 0, 0, 0])
    goal_in_world_frame = torch.tensor(
        [[0.33982402, 0.20034271, 0.07351755, 3.1322026, 1.0512958, 1.5811945]]
    )
    compute_arm_ik(urdf_path, robot_root_pose, goal_in_world_frame, device)
