# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import yaml
from keyboard_typer.real.real_xarm_ability.control.base import BaseMotionControl
import sys
from keyboard_typer.constants import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT / "keyboard_typer/real/real_xarm_ability"))


@dataclass
class MotionControlConfig:
    type: str

    robot_name: str
    robot_config_path: str

    # Low pass filter
    low_pass_alpha: float = 0.1

    # Orientation control
    disable_orientation_control: bool = False

    # Motion scaling factor, factor smaller than 1 means that robot hand will move smaller distance than human hand
    motion_scaling_factor: float = 1

    _TYPE = ["curobo", "pinocchio"]

    def __post_init__(self):
        # Motion control type check
        if self.type not in self._TYPE:
            raise ValueError(f"Motion control type must be one of {self._TYPE}")

        # Robot configs file path check
        robot_config_path = Path(self.robot_config_path)
        asset_path = (
            Path(__file__).absolute().parent.parent
            / "assets/curobo/configs/robot"
        )
        if not robot_config_path.is_absolute():
            robot_config_path = asset_path / robot_config_path
            robot_config_path = robot_config_path.absolute()
        if not robot_config_path.exists():
            raise ValueError(f"Config path {robot_config_path} does not exist")
        self.urdf_path = str(robot_config_path)

    @classmethod
    def from_file(cls, config_path):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["control"]
            if cfg is None:
                return None
            else:
                config = MotionControlConfig(**cfg)
                return config

    @classmethod
    def from_dict(cls, cfg: Dict):
        if cfg is None:
            return None
        else:
            config = MotionControlConfig(**cfg)
            return config

    def build(self, device="cuda:0") -> BaseMotionControl:

        if self.type == "curobo":
            from control.curobo_motion_control import CuroboMPCMotionControl
            import torch

            motion_control = CuroboMPCMotionControl(
                robot_name=self.robot_name,
                robot_config_path=self.robot_config_path,
                torch_device=torch.device(device),
            )
        elif self.type == "pinocchio":
            from control.pinocchio_motion_control import PinocchioMotionControl

            motion_control = PinocchioMotionControl(
                robot_name=self.robot_name,
                robot_config_path=self.robot_config_path,
            )
        else:
            raise ValueError(f"Motion control type must be one of {self._TYPE}")

        return motion_control


def get_motion_control_config(config_path) -> Optional[MotionControlConfig]:
    config = MotionControlConfig.from_file(config_path)
    return config


if __name__ == "__main__":
    # Path below is relative to this file
    import numpy as np

    test_config = get_motion_control_config(
        "../../assets/config/kinematics_config/kuka_allegro.yml"
    )
    print(test_config)
    control = test_config.build()
    print(control.get_joint_names(), control.get_ee_name())
    print(control.compute_ee_pose(np.zeros(7)))
