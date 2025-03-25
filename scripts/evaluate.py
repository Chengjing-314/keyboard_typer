from dataclasses import dataclass, field
import os

import gymnasium as gym
from keyboard_typer.curriculum.stages import StageConfig
from keyboard_typer.mani_skill.envs.typer_env import TyperEnv, TyperEnvConfig
from keyboard_typer.rl_lib import (
    Agent,
    AgentConfig,
    ManiSkillVectorEnv,
    PPOConfig,
    PPO_typer,
)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro


@dataclass
class Args:
    env_id: str = "TyperEnv-v0"
    num_eval_envs: int = 1
    seed: int = 0
    obs_mode: str = "state"
    typer_config: TyperEnvConfig = field(default_factory=lambda: TyperEnvConfig())
    agent_config: AgentConfig = field(default_factory=lambda: AgentConfig())
    stage_config: StageConfig = field(default_factory=lambda: StageConfig())
    exp_name: str = "TyperEnv"

    trainer: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            exp_name="TyperEnv",
            exp_version="v1",
            total_timesteps=5000_0000,
            eval_freq=10,
        )
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.trainer.exp_name = args.exp_name
    args.trainer.obs_mode = args.obs_mode
    args.agent_config.obs = args.obs_mode

    args.agent_config.load_from = (
        "/data/chengjingyuan/keyboard_typer/logs/new_eva_base_seed1/ckpts/ckpt_771.pt"
    )

    eval_env_kwargs = dict(
        obs_mode=args.obs_mode,
        # control_mode="pd_joint_delta_pos",
        control_mode="arm_delta_pos_hand_pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
    )

    fake_env = gym.make(
        args.env_id,
        config=args.typer_config,
        num_envs=args.num_eval_envs,
        stage_config=args.stage_config,
        **eval_env_kwargs,
    )
    if isinstance(fake_env.action_space, gym.spaces.Dict):
        fake_env = FlattenActionSpaceWrapper(fake_env)

    eval_env = gym.make(
        args.env_id,
        config=args.typer_config,
        num_envs=args.num_eval_envs,
        stage_config=args.stage_config,
        **eval_env_kwargs,
    )
    if isinstance(eval_env.action_space, gym.spaces.Dict):
        eval_env = FlattenActionSpaceWrapper(eval_env)

    eval_output_dir = os.path.join(args.trainer.exp_root, args.trainer.exp_name, "eval_videos")
    print(f"Saving eval videos to {eval_output_dir}")
    eval_env = RecordEpisode(
        eval_env,
        output_dir=eval_output_dir,
        save_trajectory=False,
        trajectory_name="trajectory",
        max_steps_per_video=200, #! This is hardcoded at the moment
        video_fps=30,
    )
    eval_env = ManiSkillVectorEnv(
        eval_env, args.num_eval_envs, **eval_env_kwargs # ignore_terminations=False 
    )
    fake_env = ManiSkillVectorEnv(fake_env, args.num_eval_envs, **eval_env_kwargs)

    agent = Agent(args.agent_config, eval_env)

    trainer = PPO_typer(config=args.trainer, agent=agent, env=fake_env, eval_env=eval_env)

    traj = trainer.evaluate(save_traj=True)
    
    traj = traj.squeeze(1)
    
    import numpy as np

    def interpolate_joint_angles(angles, x):

        """
        Interpolates between each consecutive pair of joint angles.
        
        Parameters:
          angles: np.array of shape (n, 17) -- original joint angles
          x: int -- number of intermediate steps between each consecutive pair
          
        Returns:
          np.array of shape ( (n-1)*(x+1) + 1, 17 ) with interpolated joint angles.
        """
        n, num_joints = angles.shape
        interpolated = []  # to store the new sequence of joint angles

        for i in range(n - 1):
            start = angles[i]
            end = angles[i + 1]
            # Create x+2 points including both endpoints.
            # We remove the last point to avoid duplicates except for the final pair.
            t_values = np.linspace(0, 1, x + 2)
            for t in t_values[:-1]:
                interp_point = (1 - t) * start + t * end
                interpolated.append(interp_point)
        # Append the last point of the original sequence.
        interpolated.append(angles[-1])
        return np.array(interpolated)

    traj_interpolated = interpolate_joint_angles(traj, 10)
    np.save("interpolated_trajectory.npy", traj_interpolated)
    np.save("original_trajectory.npy", traj)
        

    eval_env.close()
    fake_env.close()
