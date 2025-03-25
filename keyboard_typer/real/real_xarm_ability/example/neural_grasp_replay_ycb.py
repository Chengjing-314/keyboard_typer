import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transforms3d
import yaml
from fastdev.sim_webui.webui import SimWebUI
from fastdev.xform import matrix_to_rotation_6d
from fastdev.utils.tensor_utils import to_numpy
from scipy.spatial.transform import Rotation as R

from neural_teleop.models.hand_traj_state import HandTrajState
from neural_teleop.constants import DATA_ROOT
from neural_teleop.models.scene_state import TableTopSceneState
from einops import repeat, rearrange

current_path = Path(__file__).parent
sys.path.append(str(current_path.parent))

from robot_agent.real_env import RealEnv
from robot_agent.utils import action_in_ee_frame, update_single_arm_qpos

# Constants for functionality selection
USE_REAL_HAND = True
USE_REAL_ARM = True
ENABLE_LEFT_ARM = False
ENABLE_RIGHT_ARM = True
ENABLE_BIMANUAL = ENABLE_LEFT_ARM and ENABLE_RIGHT_ARM

GLOBAl_OFFSET = np.array([0.45, 0.2, 0.01])



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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tyro.cli(main)
    data_dir: str = "/home/jiyue/Documents/chengjing/neural-teleop/data/real_dexycb_grasps/samples_dexycb_v2"
    files = [x for x in os.listdir(data_dir) if x.endswith(".npz")]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    print("files: ", files)

    all_scene_metas = []
    all_root_poses, all_joint_values = [], []
    for file in files:
        with open(os.path.join(data_dir, file), "rb") as f:
            data = np.load(f, allow_pickle=True)
            all_scene_metas.extend(data["scene_metas"].tolist())
            all_root_poses.append(data["root_poses"])
            all_joint_values.append(data["joint_values"])
            joint_names = data["joint_names"].tolist()
    all_root_poses = np.concatenate(all_root_poses, axis=0)
    all_joint_values = np.concatenate(all_joint_values, axis=0)
    result_path = os.path.join(data_dir, "results.json")
    with open(result_path, "r") as f:
        results = np.asarray(json.load(f)["success"])
    assert len(all_scene_metas) == len(results)

    obj_indices = {}
    for i, scene_meta in enumerate(all_scene_metas):
        obj_code = scene_meta["object_codes"][0]
        obj_pose = np.asarray(scene_meta["object_poses"][0])
        obj_pose_hash = hash(obj_pose.tobytes())
        if (obj_code, obj_pose_hash) not in obj_indices:
            obj_indices[(obj_code, obj_pose_hash)] = []
        if results[i]:
            obj_indices[(obj_code, obj_pose_hash)].append(i)

    POSE_INDEX = 1

    selected_object_indices = {k: v for k, v in obj_indices.items() if k[0] == "009_gelatin_box"}
    selected_posed_object_indices = selected_object_indices[list(selected_object_indices.keys())[POSE_INDEX]]

    scene_metas = [all_scene_metas[i] for i in selected_posed_object_indices]
    root_poses = all_root_poses[selected_posed_object_indices]
    joint_values = all_joint_values[selected_posed_object_indices]

    scene_state = TableTopSceneState.from_metas(
        scene_metas, object_data_dir=str(DATA_ROOT / "dexycb/models/"), object_data_source="dexycb", device=device
    )
    root_tl = torch.from_numpy(root_poses[..., :3, 3]).unsqueeze(1)
    root_rot = torch.from_numpy(root_poses[..., :3, :3]).unsqueeze(1)
    hand_state = HandTrajState(
        hand_config="ability_hand_manual_spheres_right",
        joint_values=torch.from_numpy(joint_values).unsqueeze(1),
        root_rotations=matrix_to_rotation_6d(root_rot),
        root_translations=root_tl,
        rotation_mode="6d",
        device=device,
    )
    hand_state.rotation_mode = "rxyz"

    # array([0.31904743, 0.0004699 , 0.14165016])

    # --------- init --------
    left_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    right_init_qpos = np.concatenate([np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)])
    init_qpos = (left_init_qpos, right_init_qpos)
    real_env = initialize_real_env(
        "/home/jiyue/Documents/chengjing/neural-teleop/real_xarm_ability/real_control/configs/kinematics_config/pinocchio_bimanual_xarm7_ability.yml",
        init_qpos,
    )

    # --------- test coordinate --------
    # cur_pose = real_env.robot.compute_ee_pose(np.asarray(real_env.robot.get_arm_qpos()))
    # cur_quat = transforms3d.quaternions.mat2quat(cur_pose.rotation)
    # cur_tl = cur_pose.translation
    # ee_pose = np.concatenate([cur_tl, cur_quat])
    # cur_hand_qpos = real_env.robot.hand.get_hand_state()['raw_pos']

    # target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=cur_hand_qpos)
    # real_env.control_qpos(target_qpos)
    # real_env.wait_until_next_control_signal()

    # --------- test init frame --------
    init_pose_in_obj_coord = np.array(
        [[0.0, 0.0, 1.0, -0.13], [0.0, 1.0, 0.0, -0.2], [-1.0, 0.0, 0.0, 0.142], [0.0, 0.0, 0.0, 1.0]]
    )
    init_qpos = np.array([0.27181052, 0.27189041, 0.27260949, 0.27013267, -0.00455414, -0.03315737])
    init_tl = init_pose_in_obj_coord[:3, 3].copy()
    init_rxyz = transforms3d.euler.mat2euler(init_pose_in_obj_coord[:3, :3], axes="rxyz")
    init_var = np.concatenate([init_tl, init_rxyz, init_qpos])
    init_var = torch.from_numpy(init_var).float().unsqueeze(0).to(device)
    init_vars = repeat(init_var, "t d -> b t d", b=hand_state.num_scenes)

    tl = init_pose_in_obj_coord[:3, 3].copy()
    tl += GLOBAl_OFFSET
    rot = transforms3d.quaternions.mat2quat(init_pose_in_obj_coord[:3, :3])
    ee_pose = np.concatenate([tl, rot])
    target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=init_qpos)
    real_env.control_qpos(target_qpos)
    real_env.wait_until_next_control_signal()

    # --------- motion planning --------
    w_smooth_tl: float = 50.0
    w_smooth_rot: float = 1.0
    w_smooth_joint: float = 0.2

    smooth_w = torch.tensor(
        [w_smooth_tl] * 3 + [w_smooth_rot] * 3 + [w_smooth_joint] * hand_state.num_dofs,
        device=device,
        dtype=torch.float,
    )

    grasp_var = hand_state.variables.to(device)
    assert hand_state.rotation_mode == "rxyz"

    steps = torch.linspace(0, 1, 40, device=device).unsqueeze(-1)
    opt_var = init_vars + steps[1:-1] * (grasp_var - init_vars)
    traj_var = torch.cat([init_vars, opt_var, grasp_var], dim=-2)
    hand_state.variables = traj_var
    init_var = traj_var.clone()

    hand_state.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [hand_state.root_translations, hand_state.root_rotations, hand_state.joint_values], lr=0.01
    )

    for step in range(200):
        hand_state.apply_forward_kinematics()
        spheres = rearrange(
            hand_state.collision_spheres, "n_scene n_frame n_sphere dim -> n_scene (n_frame n_sphere) dim"
        )
        center_sdf, _, _ = scene_state.query_scene_signed_distances_v2(spheres[..., :3])  # outside is positive
        sphere_sdf = center_sdf - spheres[..., 3]
        sphere_sdf = -sphere_sdf  # flip sign, now inside is positive
        sphere_sdf += 0.01  # add a small offset to avoid penetration
        sphere_sdf[sphere_sdf <= 0] = 0  # only consider penetration
        E_pen = sphere_sdf.sum()

        dis_diff = torch.diff(hand_state.variables, dim=1)
        dis_diff_std = dis_diff.std(dim=1)
        vel_diff = torch.diff(dis_diff, dim=1)
        vel_diff_std = vel_diff.std(dim=1)
        E_smooth = dis_diff_std + vel_diff_std
        E_smooth = E_smooth * smooth_w
        E_smooth = E_smooth.sum()

        cost = E_pen + 0.1 * E_smooth

        # print(f"Step {step}/{cfg.num_iters}, E_pen: {E_pen:.4f}, E_smooth: {E_smooth:.4f}")
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # set back to the beginning/ending state
        with torch.no_grad():
            hand_state.root_translations[:, 0] = init_var[:, 0, :3]
            hand_state.root_translations[:, -1] = init_var[:, -1, :3]
            hand_state.root_rotations[:, 0] = init_var[:, 0, 3:6]
            hand_state.root_rotations[:, -1] = init_var[:, -1, 3:6]
            hand_state.joint_values[:, 0] = init_var[:, 0, 6:]
            hand_state.joint_values[:, -1] = init_var[:, -1, 6:]

    webui = SimWebUI()
    scene_state.visualize_via_webui(webui, frame_range=hand_state.num_frames)
    hand_state.visualize_via_webui(webui)


    import IPython; IPython.embed(header="after mp, before deploy")  # noqa

    # --------- test contact frame --------
    # selected_idx = 6

    # tl = hand_state.root_translations[selected_idx, 0].numpy().copy()
    # offset = np.array([0.45, 0.2, 0.0])
    # tl += offset

    # rot = transforms3d.quaternions.mat2quat(
    #     hand_state.convert_root_rotation(target_mode="mat")[selected_idx, 0].numpy()
    # )
    # ee_pose = np.concatenate([tl, rot])
    # hand_qpos = hand_state.joint_values[selected_idx, 0].numpy()[[2, 3, 4, 5, 1, 0]]

    # target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
    # real_env.control_qpos(target_qpos)
    # real_env.wait_until_next_control_signal()

    # succ indices:
    # cleanser 0, 7
    # cleanser 1, 3

    # --------- test replay frames --------
    selected_idx = 4
    for frame_idx in range(hand_state.num_frames):
        tl = to_numpy(hand_state.root_translations[selected_idx, frame_idx]).copy()
        tl += GLOBAl_OFFSET

        rot = transforms3d.quaternions.mat2quat(
            to_numpy(hand_state.convert_root_rotation(target_mode="mat")[selected_idx, frame_idx])
        )
        ee_pose = np.concatenate([tl, rot])
        hand_qpos = to_numpy(hand_state.joint_values[selected_idx, frame_idx])[[2, 3, 4, 5, 1, 0]]

        target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
        for _ in range(20):
            real_env.control_qpos(target_qpos)
            real_env.wait_until_next_control_signal()

    import IPython; IPython.embed() # noqa

    # for _ in range(50):
    #     target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=hand_qpos)
    #     real_env.control_qpos(target_qpos)
    #     real_env.wait_until_next_control_signal()

    grasp_offset = 0.5
    # grasp_qpos_offset = np.array([0.3, 0.3, 0.3, 0.3, 0.3, -0.3])
    grasp_qpos_offset = np.array([grasp_offset, grasp_offset, grasp_offset, grasp_offset, grasp_offset, -grasp_offset])
    for post_grasp_frame_idx in range(30):
        post_grasp_qpos = hand_qpos.copy() + grasp_qpos_offset / 30.0 * post_grasp_frame_idx
        target_qpos = real_env.robot_step(target_ee_pose=ee_pose, hand_joint_angle=post_grasp_qpos)
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()

    for post_grasp_frame_idx in range(20):
        real_env.control_qpos(target_qpos)
        real_env.wait_until_next_control_signal()

