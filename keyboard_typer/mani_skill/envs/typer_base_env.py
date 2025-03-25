from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils import sapien_utils

from keyboard_typer.constants import ASSETS_ROOT
from keyboard_typer.utils.keyboard.keyboard import BlackKeyboard, Keyboard
from keyboard_typer.utils.scene_builder.table import TableSceneBuilder
from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from keyboard_typer.curriculum.stages import StageConfig


MAX_EPISODE_STEPS = 100


@dataclass
class TyperEnvBaseConfig:
    robot_uids: Union[str, tuple] = "xarm7_left"
    bimanual: bool = False
    initial_agent_poses: Union[sapien.Pose, list[sapien.Pose]] = None
    fix_keyboard: bool = True
    kb_debug: bool = False


def create_keyboard(scene: sapien.Scene, kb_manager: Keyboard, config: TyperEnvBaseConfig, device):
    urdf_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/mobility_merged.urdf"

    scale = kb_manager.get_scale(0.44)  # The longest side is 0.44
    kb_manager.scale = scale

    builder = scene.create_urdf_loader()
    builder.scale = scale
    builder.fix_root_link = config.fix_keyboard
    builder = builder.parse(str(urdf_path))["articulation_builders"][0]
    builder.disable_self_collisions = True

    keyboard_initial_pose = torch.tensor(
        [
            config.keyboard_initial_pose[0],
            config.keyboard_initial_pose[1],
            config.keyboard_initial_pose[2] * scale * kb_manager.kb_height,
        ],
        device=device,
    )
    builder.initial_pose = sapien.Pose(p=keyboard_initial_pose.cpu().numpy())

    keyboard = builder.build(name="keyboard")
    return keyboard, keyboard_initial_pose


@register_env("TyperBaseEnv-v1", max_episode_steps=MAX_EPISODE_STEPS)
class TyperBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm7_right", "xarm7_left"]

    def __init__(self, *args, config=TyperEnvBaseConfig(), robot_uids="", **kwargs):
        self.config = config
        self.kb_manager = BlackKeyboard()
        super().__init__(*args, robot_uids=self.config.robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, initial_agent_poses=self.config.initial_agent_poses)

    def _load_scene(self, options: dict):
        """Load the table and keyboard into the scene."""
        # Build table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Create keyboard
        self.keyboard, self.keyboard_initial_pose = create_keyboard(
            self.scene, self.kb_manager, self.config, self.device
        )

        for link in self.keyboard.get_links():
            link.set_disable_gravity(True)

        for ac_joint in self.keyboard.get_active_joints():
            # ac_joint.set_drive_properties(stiffness=10, damping=0, force_limit=1e-6)
            ac_joint.set_drive_properties(stiffness=100, damping=100, force_limit=3e-5)
        # active_joints = self.keyboard.get_active_joints()
        # if self.num_envs > 1:
        #     self.keyboard.set_joint_drive_targets(
        #         torch.zeros(len(active_joints), device=self.device),
        #         joint_indices=torch.arange(
        #             len(active_joints), device=self.device, dtype=torch.int32
        #         ),
    
        # Visualization markers (goal and TCP visualization)
        self.goal_viz = actors.build_sphere(
            self.scene,
            radius=0.005,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self.tcp_viz = actors.build_sphere(
            self.scene,
            radius=0.009,
            color=[1, 0, 0, 1],
            name="tcp_viz",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        num_envs = len(env_idx)

        # We observed some keys will drop and maintain at 0.001 instead of 0, we set activation threshold to 0.002 to just be safe
        # when kp go beyond 40, the contact between the hand and the key is unstable

        if self.config.kb_debug:
            self.keyboard_pos = self.keyboard_initial_pose + torch.tensor(
                [0.005, 0.015, 0.1], device=self.device
            )  
        else:
            self.keyboard_pos = self.keyboard_initial_pose + torch.tensor(
                [0.0, 0.0, 0.1], device=self.device
            )
        self.keyboard.set_pose(Pose.create_from_pq(self.keyboard_pos))

        self.keyboard.set_qpos(torch.zeros((num_envs, self.keyboard.dof[0]), device=self.device))

        if self.config.bimanual:
            for agent in self.agent.agents:
                agent.reset()
                # agent.index_poke()
        else:
            self.agent.reset()
            # self.agent.index_poke()

        self.time_step = 0

        self.goal_viz.set_pose(Pose.create_from_pq(self.keyboard_initial_pose))
        self.tcp_viz.set_pose(Pose.create_from_pq(self.keyboard_initial_pose))

    @property
    def _default_human_render_camera_configs(self):
        # pose_1 = Pose.create_from_pq(
        #     p=[0.303772, -0.30858272, 0.394293], q=[0.821503, 0.00156823, 0.570198, -0.0022596]
        # )
        # pose_2 = Pose.create_from_pq(
        #     p=[0.597359, -0.2656287, 0.53965], q=[0.00287366, -0.334428, 0.00101589, 0.942416]
        # )
        # pose_2 = Pose.create_from_pq(
        #     p=[0.712071, 0.0022968, 0.148511], q=[0.0192958, -0.0189956, 0.000362143, 0.999633]
        # )

        pose_1 = Pose.create_from_pq(
            p=[0.70426, 0.33465, 0.321123], q=[0.00349396, 0.243284, 0.0008865, -0.969949]
        )
        pose_2 = Pose.create_from_pq(
            p=[0.495599, 0.336736, 0.350851], q=[0.74583, -0.00343674, 0.666116, 0.0038441]
        )

        return [
            CameraConfig(
                "render_camera",
                pose=pose_2,
                width=512,
                height=512,
                fov=1.71,
                near=0.1,
                far=100,
            ),
            CameraConfig(
                "render_camera_2",
                pose=pose_1,
                width=512,
                height=512,
                fov=1.36,
                near=0.1,
                far=100,
            ),
        ]

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 1, 1, torch.pi / 2, 0.01, 100)]

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=250,
            control_freq=50,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 36,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 12,
                found_lost_pairs_capacity=2**28,
            ),
        )

    def compute_normalized_dense_reward(self, obs, action, info):
        return 0


if __name__ == "__main__":
    config = TyperEnvBaseConfig()

    n_envs = 1

    env = gym.make(
        "TyperBaseEnv-v1",
        config=config,
        render_mode="human",
        num_envs=n_envs,
        obs_mode="state_dict",
        parallel_in_single_scene=False,
    )
    env.reset()

    action = env.action_space.sample()
    while True:
        env.unwrapped.render_human()

        env.step(action)
