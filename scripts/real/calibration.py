"""
python scripts/real/calibration.py
"""

import sys
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import IPython
import numpy as np
import yaml

from keyboard_typer.constants import PROJECT_ROOT
from keyboard_typer.real.real_xarm_ability.robot_agent.real_env import RealEnv

sys.path.append(str(PROJECT_ROOT / "real/real_xarm_ability"))


def get_real_env(is_right: bool = False, enable_joint_teaching: bool = False) -> RealEnv:
    # args = parse_args()
    kinematics_config = str(
        PROJECT_ROOT
        / "keyboard_typer/real/real_xarm_ability/real_control/configs/kinematics_config/pinocchio_bimanual_xarm7_ability.yml"
    )
    kinematics_path = Path(kinematics_config)
    if not Path(kinematics_config).is_absolute():
        kinematics_path = kinematics_path.absolute()
    with kinematics_path.open("r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        left_config = yaml_config["left"]
        right_config = yaml_config["right"]

    control_config = left_config if not is_right else right_config

    hand_init_qpos = np.array(
        [
            0.27141102949727186,
            0.27021257043266805,
            0.2754857903169248,
            0.27125123495532466,
            -0.017577399614188933,
            -0.025407332169600363,
        ]
    )

    left_init_qpos = np.concatenate(
        [np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, np.zeros(10)]
    )
    right_init_qpos = np.concatenate(
        [np.array([-1.8, 7.7, 1.8, 13.7, 180, 83.9, -0.2]) / 180 * np.pi, hand_init_qpos]
    )

    init_qpos = (left_init_qpos, right_init_qpos)

    USE_REAL_HAND = True
    USE_REAL_ARM = True

    real_env = RealEnv(
        use_arm=USE_REAL_ARM,
        use_hand=USE_REAL_HAND,
        control_config=control_config,
        init_qpos=init_qpos,
        is_right=is_right,
        is_explore=True,
        use_servo_control=False,
        enable_finger_tactile=False,
        enable_palm_tactile=False,
        enable_joint_teaching=enable_joint_teaching,
    )
    return real_env


if __name__ == "__main__":
    real_env = get_real_env(enable_joint_teaching=True)
    index_poke_qpos = np.array(
        [0.0000, 0.3940, 2.0940, 2.0940, 2.0940, 2.0940, 1.0890, 2.6590, 2.6590, 2.6590]
    )
    real_env.control_hand_qpos(index_poke_qpos)

    IPython.embed(header="before init")

    # cur_arm_qpos = np.asarray(real_env.robot.get_arm_qpos())

    arm_qpos = np.array(
        [-0.03049554, 0.13442656, 0.33543366, 0.26725397, 3.00873256, 1.46274078, -0.0035109]
    )
    target_qpos = np.concatenate([arm_qpos, np.zeros(10)])
    real_env.control_qpos(target_qpos)
    real_env.wait_until_next_control_signal()

    IPython.embed()
