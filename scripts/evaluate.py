import os
from dataclasses import dataclass, field

import gymnasium as gym

import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from keyboard_typer.mani_skill.envs.typer_env import TyperEnv, TyperEnvConfig  # noqa
from keyboard_typer.rl_lib import PPO_typer, PPOConfig, Agent, AgentConfig, ManiSkillVectorEnv
from keyboard_typer.curriculum.stages import StageConfig


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
        "/data/chengjingyuan/keyboard_typer/logs/fixed_hand/ckpts/ckpt_171.pt"
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
        max_steps_per_video=args.trainer.num_eval_steps,
        video_fps=30,
    )
    eval_env = ManiSkillVectorEnv(
        eval_env, args.num_eval_envs, ignore_terminations=True, **eval_env_kwargs
    )
    fake_env = ManiSkillVectorEnv(fake_env, args.num_eval_envs, **eval_env_kwargs)

    agent = Agent(args.agent_config, eval_env)

    trainer = PPO_typer(config=args.trainer, agent=agent, env=fake_env, eval_env=eval_env)

    trainer.evaluate(save_traj=True)

    eval_env.close()
