import sys
from pathlib import Path

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro  # Argument parsing

current_path = Path(__file__).parent
sys.path.append(str(current_path.parent))
sys.path.append("/home/jiyue/Documents/chengjing/neural-teleop/thirdparty/ugg/LION")

from robot_agent.real_env import RealEnv
from dataclasses import dataclass, field

sys.path.append("/home/jiyue/Documents/chengjing/neural-teleop/")
from scripts.baselines.lifting_il import DexYCBILModelNoPrivlige
import sys
from safetensors.torch import load_file

import pyrealsense2 as rs
import torch

from einops import rearrange
from thirdparty.ugg.LION.third_party.pvcnn.functional.sampling import furthest_point_sample
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from fastdev.xform import matrix_to_rotation_6d, rotation_6d_to_matrix
from neural_teleop.models.arm_traj_state import ArmTrajState, ArmConfig
from neural_teleop.models.hand_traj_state import HandTrajState, HAND_CONFIGS    
from fastdev.sim_webui import SimWebUI

import IPython
import time

def get_pointcloud():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    # Start streaming/

    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)
    threshold_filter = rs.threshold_filter(min_dist=0.15, max_dist=1.5)
    try:
        # Skip initial frames for auto-exposure adjustment
        for _ in range(5):
            pipeline.wait_for_frames()
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = threshold_filter.process(depth_frame)

        # Check if frames are valid
        if not depth_frame or not color_frame:
            print("Failed to get frames.")
            return None, None
        # Create point cloud object and map it to the color frame
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        # Convert to numpy arrays
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3)
        colors = colors[:, ::-1]  # BGR to RGB
        colors = colors / 255.0
        return vertices, colors
    finally:
        pipeline.stop()


# Constants for functionality selection
USE_REAL_HAND = True
USE_REAL_ARM = True
ENABLE_LEFT_ARM = False
ENABLE_RIGHT_ARM = True
ENABLE_BIMANUAL = ENABLE_LEFT_ARM and ENABLE_RIGHT_ARM


@dataclass
class DeployConfig:
    ckpt_path: str = "/home/jiyue/Documents/chengjing/neural-teleop/models/baseline_10k.safetensors"
    camera_extrinsic: list = field(
        default_factory=lambda: [
            [
                [
                    -0.9983663144152767,
                    0.0015526814368588444,
                    -0.057116472415692086,
                    0.5587072272707383,
                ],
                [0.05237326930563374, 0.4244868089813931, -0.9039181321685245, 0.7771719761582357],
                [
                    0.022841692211752702,
                    -0.9054327905378455, -0.4238746500036934, 0.4434275742248567, ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    num_points: int = 2048
    x_center: float = 0.456
    y_center: float = 0.174
    z_bounding: tuple = (-0.05, 0.5)
    x_bounding: float = 0.3
    y_bounding: float = 0.25
    num_pred_frames: int = 50
    real_robot_control_config_path: str = "/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability/real_control/configs/kinematics_config/pinocchio_bimanual_xarm7_ability.yml"
    control_per_step: int = 30
    grasp_phase: int = 50
    lift_phase: int = 30
    
    hand_pre_grasp_translation: tuple = (0.245, -0.02, 0.352) # read this from the ufactory sdk 
    hand_pre_grasp_rotation: tuple = (0, np.pi/2, 0) # hands facing downward
    robot_pre_grasp_qpos: list = field(
        default_factory=lambda: np.concatenate(
            [np.array([-11.8, -53.0, 14.8, 15.5, 171.9, 23.2, 18.8]) / 180 * np.pi, np.zeros(10)]
        ).tolist()
    )
    
    finger_grasp_weight: tuple = (1, 0.8, 0.6, 0.4, 0.4, 0.1)
    


def initialize_real_env(config_path: str, config:DeployConfig) -> RealEnv:
    """Initialize the Real Environment based on configuration."""
    with open(config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    right_config = yaml_config.get("right", {})

    left_init_qpos = np.concatenate(
        [np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)]
    )
    
    right_init_qpos = np.concatenate(
        [np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)]
    ) 
    
    right_pre_grasp_qpos = np.array(config.robot_pre_grasp_qpos)

    init_qpos = (left_init_qpos, right_init_qpos)

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
        right_pre_grasp_qpos=right_pre_grasp_qpos,
    )
    return real_env



def step_to_qpose(real_env, qpos, step):
    
    real_env.wait_until_next_control_signal()
    for _ in range(step):
        real_env.control_qpos(qpos)


def deploy(real_env: RealEnv, config: DeployConfig):

    camera_extrinsic = np.array(config.camera_extrinsic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # TODO: move the arm to the side to take picture

    photo_pose = np.concatenate(
        [np.array([90.0, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)]
    )

    step_to_qpose(real_env, photo_pose, 30)


    IPython.embed(header="photo_pose")

    # manipulation_starting_pose = np.concatenate(
    #     [np.array([1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)]
    # )
    
    manipulation_starting_pose = np.array(config.robot_pre_grasp_qpos)

    pcd, _ = get_pointcloud()

    pcd_homo = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
    pcd = (camera_extrinsic @ pcd_homo.T).T[:, :3]
    pcd = np.squeeze(pcd, -1)
    mask = (
        (
            (pcd[:, 0] > config.x_center - config.x_bounding)
            & (pcd[:, 0] < config.x_center + config.x_bounding)
        )
        & (
            (pcd[:, 1] > config.y_center - config.y_bounding)
            & (pcd[:, 1] < config.y_center + config.y_bounding)
        )
        & (pcd[:, 2] > config.z_bounding[0])
        & (pcd[:, 2] < config.z_bounding[1])
    )

    pcd = pcd[mask]

    pcd = furthest_point_sample(
        torch.tensor(pcd.T.astype(np.float32), device="cuda").unsqueeze(0),
        num_samples=config.num_points,
    )

    pcd = pcd.cpu().numpy().squeeze(0).T



    webui = SimWebUI()

    pc_id = webui.add_point_cloud_asset(points=pcd)
    webui.set_point_cloud_state(asset_id=pc_id, scene_index=0, color="blue", point_size=0.001)

    IPython.embed(header="pc viz")


    model = DexYCBILModelNoPrivlige(config)
    model.load_state_dict(load_file(config.ckpt_path))
    model = model.to(device)
    model.eval()

    arm = ArmTrajState(arm_config=ArmConfig(robot_pre_garsp_qpos=config.robot_pre_grasp_qpos))

    rx, ry, rz = config.hand_pre_grasp_rotation

    default_rotation_matrix = R.from_euler(
        "XYZ",
        [rx, ry, rz]
    ).as_matrix()
    default_rotation_matrix = (
        matrix_to_rotation_6d(torch.tensor(default_rotation_matrix).unsqueeze(0))
        .squeeze(0)
    )

    cur_hand_variables = torch.zeros(15, device="cuda")
    cur_hand_variables[0] = config.hand_pre_grasp_translation[0]
    cur_hand_variables[1] = config.hand_pre_grasp_translation[1] 
    cur_hand_variables[2] = config.hand_pre_grasp_translation[2]
    cur_hand_variables[3:9] = default_rotation_matrix
    cur_hand_variables = cur_hand_variables.unsqueeze(0)

    results = [cur_hand_variables.clone()]

    history_pred_hand_variables = []

    step_to_qpose(real_env, manipulation_starting_pose, 30)


    IPython.embed()

    pcd = torch.tensor(pcd, device=device).unsqueeze(0)
    for _ in range(99):
        pred_hand_variables = model(pcd, cur_hand_variables)
        pred_hand_variables = rearrange(
            pred_hand_variables,
            "b (n_frames d) -> b n_frames d",
            n_frames=config.num_pred_frames,
        )
        history_pred_hand_variables.insert(0, pred_hand_variables.clone())
        # ref: https://github.com/Shaka-Labs/ACT/blob/c5a3cc8fe8fc49bdd2ea10b16ba5ee650c206a31/evaluate.py#L108C1-L110C66
        actions_for_curr_step = []
        for history_idx in range(min(len(history_pred_hand_variables), config.num_pred_frames)):
            actions_for_curr_step.append(history_pred_hand_variables[history_idx][:, history_idx])
        actions_for_curr_step = torch.stack(actions_for_curr_step, dim=1)
        k = 0.01
        exp_weights = torch.exp(-k * torch.arange(actions_for_curr_step.shape[1], device=device))
        exp_weights = exp_weights / exp_weights.sum(dim=0, keepdim=True)
        cur_hand_variables = (actions_for_curr_step * exp_weights.unsqueeze(-1).unsqueeze(0)).sum(
            dim=1
        )
        results.append(cur_hand_variables.clone())

    results = torch.stack(results, dim=1)
    
    # hand = HandTrajState(HAND_CONFIGS["ability_hand_manual_spheres_right"], joint_values= torch.zeros((1,3,6)))
    
    
    # hand.variables = results.clone()
    
    # hand.visualize_via_webui_with_pc(webui, pcd.cpu().numpy().squeeze(0))
    
    # IPython.embed(header="check hand viz")
    

    results = results.detach().cpu().numpy().squeeze(0)

    ee_pose = np.zeros((results.shape[0], 7))
    ee_pose[:, :3] = results[:, :3]
    rot = rotation_6d_to_matrix(torch.tensor(results[:, 3:9], device="cpu")).numpy()
    ee_pose[:, 3:] = R.from_matrix(rot).as_quat(scalar_first=True)
    
    # our order: thumb_q1, thumb_q2, index_q1, middle_q1, ring_q1, pinky_q1

    qpos_ = results[:, 9:]
    qpos = qpos_[:, [2, 3, 4, 5, 1, 0]]
    
    arm.visualize_sequence(ee_pose, qpos_, pcd.cpu().numpy().squeeze(0), webui)
    
    # fmt:off
    IPython.embed(header="arm ik visualziation") # noqa
    # fmt:on

    initialize_starting_pose(real_env, ee_pose[0], qpos[0])

    IPython.embed(header="after this stepping begin")

    for i in range(len(ee_pose) * 2):
        curr_idx = i // 2
        target_ee_pose = ee_pose[curr_idx] if curr_idx < len(ee_pose) else ee_pose[-1]
        target_hand_qpos = qpos[curr_idx] if curr_idx < len(qpos) else qpos[-1]

        move_qpos = real_env.robot_step(
            target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos
        )
        real_env.wait_until_next_control_signal()

        for _ in range(config.control_per_step):
            real_env.control_qpos(move_qpos)

    for i in range(config.grasp_phase):
        hand_qpos = qpos[-1]
        hand_qpos[0] *= 1 + 1 * min((i / 30, 1))
        hand_qpos[1] *= 1 + 1 * min((i / 30, 1))
        hand_qpos[-2] *= 1 + 1 * min((i / 30, 1))
        move_qpos = real_env.robot_step(target_ee_pose=ee_pose[-1], hand_joint_angle=hand_qpos)
        real_env.wait_until_next_control_signal()

        for _ in range(config.control_per_step):
            real_env.control_qpos(move_qpos)
            
            
    for i in range(config.lift_phase):
        hand_qpos = qpos[-1]
        hand_qpos[0] *= 2
        hand_qpos[1] *= 2
        hand_qpos[-2] *= 2
        wrist_pos = ee_pose[-1]
        wrist_pos[2] = 0.005 * i
        move_qpos = real_env.robot_step(target_ee_pose=wrist_pos, hand_joint_angle=hand_qpos)
        real_env.wait_until_next_control_signal()

        for _ in range(config.control_per_step):
            real_env.control_qpos(move_qpos)



def fake_deploy(real_env: RealEnv, config: DeployConfig):
    
    pcd = np.load("/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability/example/real_pcd.npy")


    webui = SimWebUI()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DexYCBILModelNoPrivlige(config)
    model.load_state_dict(load_file(config.ckpt_path))
    model = model.to(device)
    model.eval()

    arm = ArmTrajState(arm_config=ArmConfig(robot_pre_garsp_qpos=config.robot_pre_grasp_qpos))

    rx, ry, rz = config.hand_pre_grasp_rotation

    default_rotation_matrix = R.from_euler(
        "XYZ",
        [rx, ry, rz]
    ).as_matrix()
    default_rotation_matrix = (
        matrix_to_rotation_6d(torch.tensor(default_rotation_matrix).unsqueeze(0))
        .squeeze(0)
    )
    
    manipulation_starting_pose = np.array(config.robot_pre_grasp_qpos)

    cur_hand_variables = torch.zeros(15, device="cuda")
    cur_hand_variables[0] = config.hand_pre_grasp_translation[0]
    cur_hand_variables[1] = config.hand_pre_grasp_translation[1] 
    cur_hand_variables[2] = config.hand_pre_grasp_translation[2]
    cur_hand_variables[3:9] = default_rotation_matrix
    cur_hand_variables = cur_hand_variables.unsqueeze(0)

    results = [cur_hand_variables.clone()]

    history_pred_hand_variables = []
    
    
    IPython.embed(header = "start")

    
    #! we update the ik pose here
    step_to_qpose(real_env, manipulation_starting_pose, 30)
    real_env.motion_control.set_current_qpos(manipulation_starting_pose) # must have as we did not use robot step


    pcd = torch.tensor(pcd, device=device).unsqueeze(0)
    for _ in range(99):
        pred_hand_variables = model(pcd, cur_hand_variables)
        pred_hand_variables = rearrange(
            pred_hand_variables,
            "b (n_frames d) -> b n_frames d",
            n_frames=config.num_pred_frames,
        )
        history_pred_hand_variables.insert(0, pred_hand_variables.clone())
        # ref: https://github.com/Shaka-Labs/ACT/blob/c5a3cc8fe8fc49bdd2ea10b16ba5ee650c206a31/evaluate.py#L108C1-L110C66
        actions_for_curr_step = []
        for history_idx in range(min(len(history_pred_hand_variables), config.num_pred_frames)):
            actions_for_curr_step.append(history_pred_hand_variables[history_idx][:, history_idx])
        actions_for_curr_step = torch.stack(actions_for_curr_step, dim=1)
        k = 0.01
        exp_weights = torch.exp(-k * torch.arange(actions_for_curr_step.shape[1], device=device))
        exp_weights = exp_weights / exp_weights.sum(dim=0, keepdim=True)
        cur_hand_variables = (actions_for_curr_step * exp_weights.unsqueeze(-1).unsqueeze(0)).sum(
            dim=1
        )
        results.append(cur_hand_variables.clone())

    results = torch.stack(results, dim=1)
    

    results = results.detach().cpu().numpy().squeeze(0)

    ee_pose = np.zeros((results.shape[0], 7))
    ee_pose[:, :3] = results[:, :3]
    rot = rotation_6d_to_matrix(torch.tensor(results[:, 3:9], device="cpu")).numpy()
    ee_pose[:, 3:] = R.from_matrix(rot).as_quat(scalar_first=True)
    
    # our order: thumb_q1, thumb_q2, index_q1, middle_q1, ring_q1, pinky_q1
    # real order: index_q1, index_q2, middle_q1, middle_q2, thumb_q2, thumb_q1

    qpos_ = results[:, 9:]
    qpos = qpos_[:, [2, 3, 4, 5, 1, 0]]
    
    arm.visualize_sequence(ee_pose, qpos_, pcd.cpu().numpy().squeeze(0), webui)
    
    IPython.embed(header="arm ik visualziation, exit the robot will start moving") 
    


    for i in range(len(ee_pose) * 2):
        curr_idx = i // 2
        target_ee_pose = ee_pose[curr_idx] if curr_idx < len(ee_pose) else ee_pose[-1]
        target_hand_qpos = qpos[curr_idx] if curr_idx < len(qpos) else qpos[-1]

        move_qpos = real_env.robot_step(
            target_ee_pose=target_ee_pose, hand_joint_angle=target_hand_qpos
        )

        for _ in range(config.control_per_step):
            real_env.control_qpos(move_qpos)
        
        real_env.wait_until_next_control_signal()
    
    print("*****************Start grasping******************")
    
    IPython.embed(header="start grasping")
    
    finger_grasp_weight = np.array(config.finger_grasp_weight)
            
    for i in range(config.grasp_phase):
        hand_qpos = qpos[-1].copy()
        
        hand_qpos*= 1 + finger_grasp_weight * min((i / (config.grasp_phase  * 0.8) , 1))

        move_qpos_grasp = move_qpos.copy()
        move_qpos_grasp[7:] = hand_qpos

        for _ in range(config.control_per_step):
            real_env.control_qpos(move_qpos_grasp)
        real_env.wait_until_next_control_signal()
    
    print("-------------------------Start lifting--------------------")
   

        
    fake_qpos = np.zeros(17)
    fake_qpos[:7] = move_qpos[:7] 
    # lifting_sequnce_wirst = np.repeat(ee_pose[-1].reshape(1, -1), config.lift_phase, axis=0)
    # lifting_sequnce_wirst[:, 2] = np.linspace(ee_pose[-1, 2], ee_pose[-1, 2] + 0.005 * config.lift_phase, config.lift_phase)    
    # lifting_sequence_hand = np.zeros((config.lift_phase, 6), dtype=np.float32)
    # arm.visualize_sequence(lifting_sequnce_wirst, lifting_sequence_hand, pcd.cpu().numpy().squeeze(0), webui, starting_qpos=fake_qpos)
    
    IPython.embed(header="start lifting") # lifting sequence
    
    real_env.motion_control.set_current_qpos(fake_qpos)
    lifting_wrist_pos = ee_pose[-1].copy()
    lifting_wrist_pos[2] += 0.005 * config.lift_phase
    move_qpos = real_env.robot_step(target_ee_pose=lifting_wrist_pos, hand_joint_angle=hand_qpos)

    step_to_qpose(real_env, move_qpos, config.lift_phase) # lifting
    

def unwarp_wrist_rotation(wrist_rotations) -> np.ndarray:
    angles_unwrapped = np.unwrap(wrist_rotations, axis=0)

    for i in range(angles_unwrapped.shape[1]):
        if np.any(angles_unwrapped[:, i] > 2 * np.pi):            
            angles_unwrapped[:, i] = angles_unwrapped[:, i] - 2 * np.pi
        elif np.any(angles_unwrapped[:, i] < -2 * np.pi):
            angles_unwrapped[:, i] = angles_unwrapped[:, i] + 2 * np.pi
    return angles_unwrapped


def main(config: DeployConfig):
    real_env = initialize_real_env(
        config.real_robot_control_config_path, config
    )  # Adjust with your actual config path

    fake_deploy(real_env, config)


if __name__ == "__main__":
    tyro.cli(main)
