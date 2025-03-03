# python3 base_arm_agent.py -r 'xarm7_right'
# python3 base_arm_agent.py -r 'xarm7_left'

import sapien
import numpy as np
import torch


import copy

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPoseControllerConfig
from mani_skill.utils import sapien_utils

from keyboard_typer.constants import ASSETS_ROOT


from rich import print
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConstants:
    ARM_STIFFNESS: float = 500
    ARM_DAMPING: float = 100
    ARM_FORCE_LIMIT: float = 50

    HAND_STIFFNESS: float = 500
    HAND_DAMPING: float = 100
    HAND_FORCE_LIMIT: float = 50

    ARM_POSE_PD_EE_POS_LOWER: float = -0.01
    ARM_POSE_PD_EE_POS_UPPER: float = 0.01
    ARM_POSE_PD_EE_ROT_LOWER: float = -0.05
    ARM_POSE_PD_EE_ROT_UPPER: float = 0.05

    ARM_DOF: int = 7
    HAND_DOF: int = 6  # we use linear mimic joint for hand

    NORMALIZE_PDEE: bool = True
    NORMALIZE_PD_ARMJOINT: bool = False
    NORMALIZE_PD_HANDJOINT: bool = False


@register_agent()
class XArm7AbilityBase(BaseAgent):
    uid = "xarm7_ability_base"
    urdf_path = ""
    fix_root_link = True

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose([0, 0.4, 0], [1, 0, 0, 0]),
            qpos=np.array(
                [
                    -0.03141593,
                    0.13439035,
                    0.03141593,
                    0.23911011,
                    3.14159265,
                    1.46433124,
                    -0.00349066,
                ]
                + [0] * 10
            ),
        )
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    hand_joint_names = [
        "thumb_q1",
        "index_q1",
        "middle_q1",
        "ring_q1",
        "pinky_q1",
        "thumb_q2",
        "index_q2",
        "middle_q2",
        "ring_q2",
        "pinky_q2",
    ]

    constants = RobotConstants()

    visual_stack = []

    disable_self_collisions = True

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.constants.ARM_STIFFNESS,
            damping=self.constants.ARM_DAMPING,
            force_limit=self.constants.ARM_FORCE_LIMIT,
            normalize_action=self.constants.NORMALIZE_PD_ARMJOINT,
        )

        arm_pd_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.constants.ARM_STIFFNESS,
            damping=self.constants.ARM_DAMPING,
            force_limit=self.constants.ARM_FORCE_LIMIT,
            normalize_action=self.constants.NORMALIZE_PD_ARMJOINT,
            use_delta=True,
        )

        hand_pd_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.constants.HAND_STIFFNESS,
            damping=self.constants.HAND_DAMPING,
            force_limit=50,
            normalize_action=self.constants.NORMALIZE_PD_HANDJOINT,
            use_delta=True,
        )

        arm_pd_pose_ee = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=self.constants.ARM_POSE_PD_EE_POS_LOWER,
            pos_upper=self.constants.ARM_POSE_PD_EE_POS_UPPER,
            rot_lower=self.constants.ARM_POSE_PD_EE_ROT_LOWER,
            rot_upper=self.constants.ARM_POSE_PD_EE_ROT_UPPER,
            stiffness=self.constants.ARM_STIFFNESS,
            damping=self.constants.ARM_DAMPING,
            ee_link="ee_link",
            urdf_path=self.urdf_path,
            frame="root_translation:root_aligned_body_rotation",
            normalize_action=self.constants.NORMALIZE_PDEE,
        )

        hand_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.constants.HAND_STIFFNESS,
            damping=self.constants.HAND_DAMPING,
            force_limit=50,
            normalize_action=self.constants.NORMALIZE_PD_HANDJOINT,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, hand=hand_pd_joint_pos),
            pd_joint_delta_pos=dict(arm=arm_pd_delta_pos, hand=hand_pd_delta_pos),
            arm_pd_ee_pose_hand_pd_joint_pos=dict(arm=arm_pd_pose_ee, hand=hand_pd_joint_pos),
        )

        return copy.deepcopy(controller_configs)

    def is_static(self, threshold=1e-2):
        qvel = self.robot.get_qvel()[..., :-10]
        return torch.max(torch.abs(qvel), dim=-1)[0] < threshold

    def _after_init(self):
        hand_primary_link_names = [
            "thumb_L1",
            "thumb_L2",
            "index_L1",
            "middle_L1",
            "ring_L1",
            "pinky_L1",
        ]

        self.hand_primary_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_primary_link_names
        )

        hand_front_link_names = [
            "thumb_L2",
            "index_L2",
            "middle_L2",
            "ring_L2",
            "pinky_L2",
        ]
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_front_link_names
        )

        finger_tip_link_names = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]
        self.finger_tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), finger_tip_link_names
        )

        hand_contact_order = [
            "thumb_base",
            "index_L1",
            "index_L2",
            "index_tip",
            "middle_L1",
            "middle_L2",
            "middle_tip",
            "pinky_L1",
            "pinky_L2",
            "pinky_tip",
            "ring_L1",
            "ring_L2",
            "ring_tip",
            "thumb_L1",
            "thumb_L2",
            "thumb_tip",
        ]
        self.hand_contact_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_contact_order
        )

        wrist_link_name = "ee_link"

        self.wrist_link = sapien_utils.get_obj_by_name(self.robot.get_links(), wrist_link_name)

        self.num_envs = self.scene.num_envs

        self.disable_finger_self_collision()

        self.pos_low = self.constants.ARM_POSE_PD_EE_POS_LOWER
        self.pos_high = self.constants.ARM_POSE_PD_EE_POS_UPPER
        self.rot_low = self.constants.ARM_POSE_PD_EE_ROT_LOWER
        self.rot_high = self.constants.ARM_POSE_PD_EE_ROT_UPPER

    def get_joint_limits(self):
        qlimits = self.robot.get_qlimits().cpu().numpy()[0]
        return qlimits

    def get_finger_pos(self):
        pos = []
        for link in self.hand_contact_links:
            pos.append(link.pose.p)

        return torch.stack(pos, dim=1).float().to(self.device)

    def get_finger_tip_pos(self, flatten=True):
        pos = []
        for link in self.finger_tip_links:
            pos.append(link.pose.p)

        pos = torch.stack(pos, dim=1).float().to(self.device)
        if flatten:
            return pos.flatten(start_dim=1)
        return pos

    def disable_finger_self_collision(self):
        """

        This function disable self collision and collision between index, middle, ring, pinky.
        Thumb's self collision is disabled but collision with other fingers is enabled.

        """

        print("[bold green][INFO] Disabling Finger Self Collision For Left Hand[/bold green]")

        index_finger_links = [
            "index_L1",
            "index_L2",
            "index_tip",
        ]

        middle_finger_links = [
            "middle_L1",
            "middle_L2",
            "middle_tip",
        ]
        ring_finger_links = [
            "ring_L1",
            "ring_L2",
            "ring_tip",
        ]
        pinky_finger_links = [
            "pinky_L1",
            "pinky_L2",
            "pinky_tip",
        ]

        thumb_finger_links = [
            "thumb_L1",
            "thumb_L2",
            "thumb_tip",
        ]

        for link in index_finger_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link)
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link in middle_finger_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link)
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link in ring_finger_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link)
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link in pinky_finger_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link)
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        for link in thumb_finger_links:
            link = sapien_utils.get_obj_by_name(self.robot.get_links(), link)
            link.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def get_joint_names_in_order(self):
        return self.arm_joint_names + self.hand_joint_names

    def visualize_actor_link_contact(self, actor):
        contact_vector = self.get_actor_grasping_vector(actor)

        self.visualize_link_contact(contact_vector)

    def get_wrist_pose(self):
        p = self.wrist_link.pose.p.detach().cpu().numpy()
        q = self.wrist_link.pose.q.detach().cpu().numpy()
        return p, q

    def get_wrist_raw_pose(self):
        return self.wrist_link.pose.raw_pose

    def reset(self):
        kf = self.keyframes["rest"]
        self.robot.set_qpos(kf.qpos)
        self.robot.set_pose(kf.pose)


@register_agent()
class XArm7AbilityRight(XArm7AbilityBase):
    uid = "xarm7_right"
    urdf_path = (
        f"{ASSETS_ROOT}/robot/combined/xarm7_ability/xarm7_ability_right_hand_glb_nmm_cs.urdf"
    )
    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose([0, -0.4, 0], [1, 0, 0, 0]),
            qpos=np.array(
                [
                    -0.03141593,
                    0.13439035,
                    0.03141593,
                    0.23911011,
                    3.14159265,
                    1.46433124,
                    -0.00349066,
                ]
                + [0] * 10
            ),
        )
    )


@register_agent()
class XArm7AbilityLeft(XArm7AbilityBase):
    uid = "xarm7_left"
    urdf_path = (
        f"{ASSETS_ROOT}/robot/combined/xarm7_ability/xarm7_ability_left_hand_glb_nmm_cs.urdf"
    )
    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose([0, 0.4, 0], [1, 0, 0, 0]),
            qpos=np.array(
                [
                    -0.03141593,
                    0.13439035,
                    0.03141593,
                    0.23911011,
                    3.14159265,
                    1.46433124,
                    -0.00349066,
                ]
                + [0] * 10
            ),
        )
    )


def main():
    import mani_skill.examples.demo_robot as demo_robot_script

    demo_robot_script.main()


if __name__ == "__main__":
    main()
