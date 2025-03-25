# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from threading import Lock
from typing import List, Optional

import curobo.util_file
import numpy as np
import torch
import transforms3d.quaternions
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import create_reacher_mpc

from teleop.control.base import BaseMotionControl


def get_content_path():
    from pathlib import Path

    root_path = Path(__file__).parent.parent.parent
    content_path = root_path / "assets/curobo"
    return str(content_path.absolute())


class CuroboMPCMotionControl(BaseMotionControl):
    def __init__(
        self,
        robot_name: str,
        robot_config_path: str,
        torch_device: torch.device,
    ):
        curobo.util_file.get_content_path = get_content_path
        self.robot_name = robot_name
        self._qpos_lock = Lock()

        # MPC configs
        self.tensor_args = TensorDeviceType(device=torch_device)

        # Create arm reach MPC
        world_file = None
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_path))[
            "robot_cfg"
        ]
        self.ee_name = robot_cfg["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self.mpc = create_reacher_mpc(
            robot_cfg, world_file, tensor_args=self.tensor_args, use_cuda_graph=True
        )

        # Print joint information
        print(f"Number of C-space coordinates: {self.get_dof()}")
        print("Joint names: ")
        joint_names = self.get_joint_names()
        for i in range(self.get_dof()):
            print(f"[{i}] {joint_names[i]}")

        # Data cache
        self.dt = self.mpc.rollout_fn.dt
        zero_position = torch.zeros((1, self.get_dof()), device=torch_device)
        current_robot_state = JointState.from_position(zero_position)
        self.current_robot_state = current_robot_state.to(self.tensor_args)

    def step(self, pos: Optional[np.ndarray], quat: Optional[np.ndarray], repeat=1):
        rot = transforms3d.quaternions.quat2mat(quat)
        goal_pose = Pose(
            position=torch.as_tensor(pos, **vars(self.tensor_args)).unsqueeze(0),
            rotation=torch.as_tensor(rot[:3, :3], **vars(self.tensor_args)).unsqueeze(
                0
            ),
        )

        mpc_result = self.mpc.solve(
            Goal(current_state=self.current_robot_state, goal_pose=goal_pose),
            shift_steps=1,
        )
        self.current_robot_state = mpc_result.action

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        batch_qpos = qpos[None, :].copy()
        joint_state = JointState.from_position(
            torch.from_numpy(batch_qpos).to(
                self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        )
        state = self.mpc.rollout_fn.compute_kinematics(joint_state)
        pos = state.ee_pos_seq[0].cpu().numpy()
        quat = state.ee_quat_seq[0].cpu().numpy()

        return np.concatenate([pos, quat])

    def get_current_qpos(self) -> np.ndarray:
        with self._qpos_lock:
            qpos = self.current_robot_state.position.cpu().numpy()[0]
        return qpos

    def set_current_qpos(self, qpos: np.ndarray):
        with self._qpos_lock:
            qpos_tensor = self.tensor_args.to_device(qpos[None])
            self.current_robot_state.position[:] = qpos_tensor[:]

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return self.mpc.rollout_fn.d_action

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:
        return self.mpc.rollout_fn.dynamics_model.robot_model.joint_names

    def is_use_gpu(self) -> bool:
        return True
