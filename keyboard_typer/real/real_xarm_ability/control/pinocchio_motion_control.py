from pathlib import Path
from threading import Lock
from typing import List, Optional, Dict

import numpy as np
import pinocchio as pin
import yaml

from control.base import BaseMotionControl


class PinocchioMotionControl(BaseMotionControl):
    def __init__(
        self,
        robot_name: str,
        robot_config_path: str,
    ):
        self.robot_name = robot_name
        self._qpos_lock = Lock()

        # Config
        robot_config_path = Path(robot_config_path)
        if not robot_config_path.is_absolute():
            base_path = (
                Path(__file__).parent.parent / "assets/curobo/configs/robot"
            )
            robot_config_path = (base_path / robot_config_path).resolve().absolute()

        with Path(robot_config_path).open("r") as f:
            cfg = yaml.safe_load(f)["robot_cfg"]
        ik_damping = cfg["kinematics"]["ik_damping"]
        ee_name = cfg["kinematics"]["ee_link"]
        self.ik_damping = float(ik_damping) * np.eye(6)
        self.ik_eps = float(cfg["kinematics"]["eps"])
        self.dt = float(cfg["dt"])
        self.ee_name = ee_name
        self.fsr_names = cfg["kinematics"]["fsr"]
        # control_params = cfg["dynamics"]

        urdf_path = Path(cfg["urdf_path"])
        if not urdf_path.is_absolute():
            base_path = Path(__file__).parent.parent / "assets/curobo/assets"
            urdf_path = (base_path / urdf_path).resolve().absolute()

        # import ipdb; ipdb.set_trace()
        self.model: pin.Model = pin.buildModelFromUrdf(str(urdf_path))
        self.data: pin.Data = self.model.createData()
        frame_mapping: Dict[str, int] = {}

        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        if self.ee_name not in frame_mapping:
            raise ValueError(
                f"End effector name {ee_name} not find in robot with path: {urdf_path}."
            )
        
        for fsr_name in self.fsr_names:
            if fsr_name not in frame_mapping:
                raise ValueError(
                    f"FSR name {fsr_name} not find in robot with path: {urdf_path}."
                )
        
        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[ee_name]
        self.fsr_frame_ids = []
        for fsr in self.fsr_names:
            self.fsr_frame_ids.append(frame_mapping[fsr])
        
        
        print("========frame mapping========")
        print(frame_mapping)

        print("========fsr frame id========")
        print(self.fsr_frame_ids)
        
        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        
        self.fsr_poses = []
        for fsr_id in self.fsr_frame_ids:
            self.fsr_poses.append(pin.updateFramePlacement(self.model, self.data, fsr_id))
        
        
        for joint_id, joint in enumerate(self.model.joints):
            name = self.model.names[joint_id]
            print(f"joint {joint_id}: {name}, DOF: {joint.nq}")

    def step(self, pos: Optional[np.ndarray], quat: Optional[np.ndarray], repeat=1):
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        pose_vec = np.concatenate([pos, xyzw])
        oMdes = pin.XYZQUATToSE3(pose_vec)
        with self._qpos_lock:
            qpos = self.qpos.copy()

        for k in range(100 * repeat):
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < self.ik_eps:
                break

            v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
            qpos = pin.integrate(self.model, qpos, v * self.dt)
            
            if k == (100 * repeat - 1) and np.linalg.norm(err) > self.ik_eps:
                print("IK not converge")

        self.set_current_qpos(qpos)

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        xyzw_pose = pin.SE3ToXYZQUAT(oMf)

        return np.concatenate(
            [
                xyzw_pose[:3],
                np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
            ]
        )
    
    def compute_fsr_poses(self, qpos: np.ndarray) -> List[np.ndarray]:
        pin.forwardKinematics(self.model, self.data, qpos)
        fsr_poses = []
        for fsr_id in self.fsr_frame_ids:
            oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, fsr_id)
            xyzw_pose = pin.SE3ToXYZQUAT(oMf)
            fsr_poses.append(np.concatenate(
                [
                    xyzw_pose[:3],
                    np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
                ]
            ))
        return np.array(fsr_poses)

    def get_current_qpos(self) -> np.ndarray:
        with self._qpos_lock:
            return self.qpos

    def set_current_qpos(self, qpos: np.ndarray):
        with self._qpos_lock:
            self.qpos = qpos
            pin.forwardKinematics(self.model, self.data, self.qpos)
            self.ee_pose = pin.updateFramePlacement(
                self.model, self.data, self.ee_frame_id
            )
            fsr_poses = []
            for fsr_id in self.fsr_frame_ids:
                fsr_poses.append(pin.updateFramePlacement(self.model, self.data, fsr_id))
            self.fsr_poses = fsr_poses
    
    def set_current_qpos_with_hand(self, qpos: np.ndarray, hand_qpos: np.ndarray):
        with self._qpos_lock:
            self.qpos = np.concatenate((qpos[:7], hand_qpos[2:], hand_qpos[:2]))
            pin.forwardKinematics(self.model, self.data, self.qpos)
            self.ee_pose = pin.updateFramePlacement(
                self.model, self.data, self.ee_frame_id
            )
            fsr_poses = []
            for fsr_id in self.fsr_frame_ids:
                fsr_poses.append(pin.updateFramePlacement(self.model, self.data, fsr_id))
            self.fsr_poses = fsr_poses

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:
        # Pinocchio by default add a dummy joint name called "universe"
        names = list(self.model.names)
        return names[1:]

    def is_use_gpu(self) -> bool:
        return False
