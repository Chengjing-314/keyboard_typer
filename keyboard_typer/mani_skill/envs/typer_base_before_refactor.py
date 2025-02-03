# python3 keyboard_typer/mani_skill/envs/typer_env.py

from dataclasses import dataclass
from typing import Union

import gymnasium as gym
import sapien
import torch
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils import sapien_utils

from keyboard_typer.utils.rotations import quaternion_batch_distance

from keyboard_typer.constants import ASSETS_ROOT
from keyboard_typer.mani_skill.agents import XArm7AbilityLeft, XArm7AbilityRight
from keyboard_typer.utils.keyboard.keyboard import BlackKeyboard
from keyboard_typer.utils.scene_builder.table import TableSceneBuilder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPISODE_STEPS = 100


@dataclass
class TyperEnvConfig:
    robot_uids: Union[str, tuple] = "xarm7_right"
    bimanual: bool = False
    initial_agent_poses: Union[sapien.Pose, list[sapien.Pose]] = None
    fix_keyboard: bool = True
    tcp_threshold: float = 0.01
    actuation_threshold: float = 0.002
    rest_threshold: float = 0.001
    time_steps: int = 1
    look_forward_horizon: int = 5


@register_env("TyperBaseEnv-v0", max_episode_steps=MAX_EPISODE_STEPS)
class TyperBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm7_right", "xarm7_left"]
    agent: Union[XArm7AbilityLeft, XArm7AbilityRight]
    MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

    def __init__(self, *args, config=TyperEnvConfig(), robot_uids="", **kwargs):
        self.config = config
        self.kb_manager = BlackKeyboard()
        self.step_per_key = self.MAX_EPISODE_STEPS // self.config.time_steps
        super().__init__(*args, robot_uids=self.config.robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, initial_agent_poses=self.config.initial_agent_poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        urdf_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/mobility_merged.urdf"

        self.scale = self.kb_manager.get_scale(0.45)
        self.kb_manager.scale = self.scale
        builder = self.scene.create_urdf_loader()
        builder.scale = self.scale
        builder.fix_root_link = self.config.fix_keyboard
        builder = builder.parse(str(urdf_path))["articulation_builders"][0]
        builder.disable_self_collisions = True
        self.keyboard_initial_pose = torch.tensor(
            [0.3, 0.1, self.scale * self.kb_manager.kb_height * 0.5], device=self.device
        )
        builder.initial_pose = sapien.Pose(p=self.keyboard_initial_pose.cpu().numpy())
        self.keyboard = builder.build(name="keyboard")

        self.goal_site = actors.build_sphere(
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
        # Ensure the Keyboard behave properly
        for ac_joint in self.keyboard.get_active_joints():
            ac_joint.set_drive_properties(stiffness=40.0, damping=5.0)
        active_joints = self.keyboard.get_active_joints()
        if self.num_envs > 1:
            self.keyboard.set_joint_drive_targets(
                torch.zeros(len(active_joints), device=self.device),
                joint_indices=torch.arange(
                    len(active_joints), device=self.device, dtype=torch.int32
                ),
            )

        keyboard_pos = torch.zeros((num_envs, 3), device=self.device)
        keyboard_pos += self.keyboard_initial_pose
        self.keyboard_pos = keyboard_pos
        self.keyboard.set_pose(Pose.create_from_pq(self.keyboard_pos))

        # * DOF return a list of dof, might need adjustment if we add more keyboard
        self.keyboard.set_qpos(torch.zeros((num_envs, self.keyboard.dof[0]), device=self.device))

        key_pos = self.kb_manager.get_default_key_pos()

        key_pos = torch.tensor(key_pos, device=self.device).unsqueeze(0).repeat(
            num_envs, 1, 1
        ) + self.keyboard_pos.unsqueeze(1)

        if self.config.bimanual:
            for agent in self.agent.agents:
                agent.reset()
        else:
            self.agent.reset()

        self.target_key_press, self.target_key_indices = (
            self.kb_manager.generate_t_time_step_key_press(self.config.time_steps, num_envs)
        )

        # Need better code to produce key press
        self.target_key_indices = torch.tensor(
            np.array(self.target_key_indices).T, device=self.device
        )

        self.time_step = 0

        # Query key position and set goal for viz
        # key_pos: num_envs x num_keys x 3
        # target_key_indices: num_time_steps x num_envs x 3

        batch_indices = torch.arange(num_envs).unsqueeze(-1).expand(-1, self.config.time_steps)

        self.target_key_pos = key_pos[batch_indices, self.target_key_indices]

        self.default_wrist_rotation = torch.tensor(
            self.agent.get_wrist_pose()[1][0], device=self.device
        )

        # We update the goal pose after step_per_key steps
        goal_pos = Pose.create_from_pq(
            self.target_key_pos[torch.arange(num_envs), self.time_step // self.step_per_key]
            + torch.tensor([0, 0, 0.05], device=self.device)
        )
        __import__("IPython").embed(header="typer_base_env.py:159")
        self.goal_site.set_pose(goal_pos)

        # Randomize target finger
        # ignore thumb
        # self.target_finger = torch.randint(
        #     0,
        #     1,
        #     (self.num_envs, self.config.time_steps),
        #     device=self.device,
        # )

        # we dont randomize the target finger first
        self.target_finger = torch.ones(
            (self.num_envs, self.config.time_steps),
            device=self.device,
        ).long()

        target_finger_pos = self.agent.get_finger_tip_pos(flatten=False)
        target_finger_pos = target_finger_pos[
            torch.arange(num_envs),
            self.target_finger[:, self.time_step // self.step_per_key],
        ]

        self.tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))

    def _get_obs_extra(self, info: dict):
        if not self.config.bimanual:
            finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
        else:
            raise NotImplementedError

        num_envs = len(finger_tip_pos)
        target_finger_pos = finger_tip_pos[
            torch.arange(num_envs),
            self.target_finger[:, self.time_step // self.step_per_key],
        ]
        finger_tip_pos = finger_tip_pos.flatten(start_dim=1)

        num_indices = self.target_key_indices.shape[-1]
        start_idx = self.time_step // self.step_per_key
        end_idx = start_idx + self.config.look_forward_horizon
        z_offset = 0.05
        if end_idx > num_indices:
            extra = end_idx - num_indices
            padded_target_key_pos = torch.cat(
                [
                    self.target_key_pos[:, start_idx:]
                    + torch.tensor([0, 0, z_offset], device=self.device),
                    torch.zeros_like(self.target_key_pos[:, -1], device=self.device)
                    .unsqueeze(1)
                    .repeat(1, extra, 1),
                ],
                dim=1,
            )
        else:
            padded_target_key_pos = self.target_key_pos[:, start_idx:end_idx] + torch.tensor(
                [0, 0, z_offset], device=self.device
            )

        self.target_key_pos_t = padded_target_key_pos
        target_key_pos = self.target_key_pos[
            :, self.time_step // self.step_per_key
        ] + torch.tensor([0, 0, 0.05], device=self.device)

        tcp_distance = torch.norm(target_finger_pos - target_key_pos, dim=-1)

        # keyboard_pose = self.keyboard.pose.raw_pose

        # Key qpos
        # target_key_qpos = self.keyboard.qpos[
        #     torch.arange(num_envs),
        #     self.target_key_indices[:, self.time_step // self.step_per_key],
        # ]

        all_key_qpos = self.keyboard.qpos

        wrist_pose = self.agent.get_wrist_raw_pose()

        target_finger = self.target_finger[:, self.time_step // self.step_per_key]

        obs = dict(
            target_finger_idx=target_finger,
            finger_tip_pos=finger_tip_pos,
            target_key_pos=target_key_pos,
            tcp_distance=tcp_distance,
            # wrist_pose=wrist_pose,
            # all_key_qpos=all_key_qpos,
            # keyboard_pose=keyboard_pose,
        )

        return obs

    def compute_normalized_dense_reward(self, obs, action, info):
        finger_tip_pos = self.agent.get_finger_tip_pos(flatten=False)
        target_finger_pos = finger_tip_pos[
            torch.arange(len(finger_tip_pos)),
            self.target_finger[:, self.time_step // self.step_per_key],
        ]
        target_key_pos = target_key_pos = self.target_key_pos[
            :, self.time_step // self.step_per_key
        ] + torch.tensor([0, 0, 0.05], device=self.device)
        tcp_distance_reward = torch.exp(
            -5 * torch.norm(target_finger_pos - target_key_pos, dim=-1)
        )
        over_all_distance_reward = torch.norm(  # noqa
            finger_tip_pos - self.target_key_pos_t[:, 0].unsqueeze(1), dim=-1
        ).mean(dim=-1)

        pressed_keys = (self.keyboard.qpos > self.config.actuation_threshold).float()

        correct_keys = torch.zeros_like(pressed_keys)
        correct_keys[
            torch.arange(len(correct_keys)),
            self.target_key_indices[:, self.time_step // self.step_per_key],
        ] = 1.0

        wrong_key_pressed = (pressed_keys - correct_keys).clamp(min=0).sum(dim=-1)
        correct_key_press = (pressed_keys * correct_keys).sum(dim=-1)

        key_actuation_reward = 100 * (1 - wrong_key_pressed) * correct_key_press
        wrong_key_penality = -wrong_key_pressed

        self.time_step = min(self.time_step + 1, self.MAX_EPISODE_STEPS - 1)

        if self.time_step < self.MAX_EPISODE_STEPS:
            self.goal_site.set_pose(Pose.create_from_pq(target_key_pos))

            self.tcp_viz.set_pose(Pose.create_from_pq(target_finger_pos))

        # smaller velocity when close to the target

        # vel_reward = torch.norm(self.agent.robot.qvel) / (tcp_distance_reward + 1e-6)

        self.reward_dict = dict(
            tcp_distance_reward=tcp_distance_reward.mean(dim=-1).item(),
            # over_all_distance_reward=over_all_distance_reward.mean(dim=-1).item(),
            over_all_distance_reward=0,
            # rotation_distance_reward=rotation_distance_reward.mean(dim=-1).item(),
            rotation_distance_reward=0,  # ignore for now
            # key_actuation_reward=key_actuation_reward.mean(dim=-1).item(),
            key_actuation_reward=0,
            # velocity_penalty=vel_reward.mean(dim=-1).item(),
            velocity_penalty=0,
            # wrong_key_penality=wrong_key_penality.mean(dim=-1).item(),
            wrong_key_penality=0,
        )

        # return -(
        #     tcp_distance_reward
        #     # + 0.3 * over_all_distance_reward
        #     # + 0.5 * rotation_distance_reward
        #     - key_actuation_reward
        #     # + 0.1 * vel_reward
        #     + wrong_key_penality
        # )

        return tcp_distance_reward + key_actuation_reward - wrong_key_penality

    def get_reward_details(self):
        return self.reward_dict

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])

        pose_1 = Pose.create_from_pq(
            p=[0.303772, -0.00858272, 0.394293], q=[0.821503, 0.00156823, 0.570198, -0.0022596]
        )

        pose_2 = Pose.create_from_pq(
            p=[0.570552, 0.071466, 0.823753], q=[0.00162786, -0.425034, 0.000762582, 0.905176]
        )

        pose_2 = Pose.create_from_pq(
            p=[0.712071, 0.0722968, 0.148511], q=[0.0192958, -0.0189956, 0.000362143, 0.999633]
        )

        return [
            CameraConfig(
                "render_camera",
                pose=pose_2,
                width=512,
                height=512,
                fov=1.36,
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


def main():
    config = TyperEnvConfig()

    # config.robot_uids = ("xarm7_right", "xarm7_left")
    # config.bimanual = True
    # config.initial_agent_poses = [None, None]

    n_envs = 1

    env = gym.make(
        "TyperBaseEnv-v0",
        config=config,
        render_mode="human",
        num_envs=n_envs,
        obs_mode="state_dict",
        parallel_in_single_scene=False,
    )
    env.reset()
    import time

    action = env.action_space.sample()  # use to check if the pd controller is working
    ctr = 0
    while True:
        env.unwrapped.render_human()

        env.step(action)

        # obs, _, _, _, _ = env.step(action=action)
        # __import__("IPython").embed(header="typer_env.py:404")
        # ctr += 1
        # if ctr > 100:
        #     __import__("IPython").embed(header="typer_env.py:406")
        # env.step(None)
        # begin = time.time()
        # _, _, _, _, info = env.step(env.action_space.sample())
        #
        # end = time.time()
        #
        # print(f"fps: {n_envs / (end - begin)}")


if __name__ == "__main__":
    main()
