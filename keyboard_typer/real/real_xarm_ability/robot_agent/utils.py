from typing import List
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler


def parse_args():
    description = """\
        Read the camera config and launch the camera driver and/or hand detector.
        --------------------------------
        Example: python3 script/visual_module.launch.py --cfg example/example_camera_config.yaml
        """
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "--kinematics-cfg",
        "-k",
        required=True,
        help="File path to the kinematics config.",
    )
    args = parser.parse_args()
    return args

def action_in_ee_frame(delta, motion_control):
    ee_pose = motion_control.compute_ee_pose(motion_control.get_current_qpos())
    w_trans, w_rot = ee_pose[:3], ee_pose[3:]
    w2l = R.from_quat(w_rot, scalar_first=True)
    new_trans = w2l.apply(delta[:3]) + np.array(w_trans)
    new_rot = (w2l * R.from_rotvec(delta[3:])).as_quat(scalar_first=True)
    return np.array([*new_trans, *new_rot])

def pose_to_matrix(pose):
    tx, ty, tz = pose[:3]
    rotation = R.from_quat(pose[3:], scalar_first=True)
    rotation_matrix = rotation.as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = np.array([tx, ty, tz])
    return matrix

def update_single_arm_qpos(target_ee_pose, motion_control, repeat_times=1):
    new_qpos = np.zeros(7)
    motion_control.step(target_ee_pose[:3], target_ee_pose[3:], repeat_times)
    new_qpos = motion_control.get_current_qpos()
    
    return new_qpos

def update_bimanual_arm_hand_qpos(target_ee_pose: List[np.ndarray], 
                                target_hand_qpos: List[np.ndarray], 
                                motion_controls: List, 
                                repeat_times: int = 1) -> List[np.ndarray]:
    """update bimanual arm and hand qpos, in base frame

    Args:
        target_ee_pose (List[np.ndarray]): [left, right] target end effector pose, each in shape of (7,)
        target_hand_qpos (List[np.ndarray]): [left, right] target hand qpos, each in shape of (6,)
        motion_controls (List): [left, right] pinocchio motion control models
        repeat_times (int, optional): repeat time for ik iterations: 100 * repeat_times. Defaults to 1.

    Returns:
        List[np.ndarray]: [left, right] target qpos
    """    
    qpos_list = []
    
    left_qpos = update_single_arm_qpos(target_ee_pose[0], motion_controls[0], repeat_times)
    left_qpos = np.concatenate([left_qpos, target_hand_qpos[0]])
    qpos_list.append(left_qpos)
    
    right_qpos = update_single_arm_qpos(target_ee_pose[1], motion_controls[1], repeat_times)
    right_qpos = np.concatenate([right_qpos, target_hand_qpos[1]])
    qpos_list.append(right_qpos)
    
    return qpos_list

def update_rotation(local_rot, rpy_step):
    cur_rot_matrix = euler2mat(*local_rot, axes='rxyz')
    step_rot_matrix = euler2mat(*rpy_step, axes='rxyz')

    new_rot_matrix = np.dot(cur_rot_matrix, step_rot_matrix)

    new_local_rot = mat2euler(new_rot_matrix, axes='rxyz')

    return new_local_rot