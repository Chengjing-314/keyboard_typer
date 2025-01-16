import os
from dataclasses import dataclass, field

import gymnasium as gym

import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from keyboard_typer.mani_skill.envs.typer_env import TyperEnv, TyperEnvConfig  # noqa
from keyboard_typer.rl_lib import PPO_typer, PPOConfig, Agent, AgentConfig, ManiSkillVectorEnv


@dataclass
class Args:
    env_id: str = "TyperEnv-v0"
    num_envs: int = 512
    num_eval_envs: int = 2
    seed: int = 0
    typer_config: TyperEnvConfig = field(default_factory=lambda: TyperEnvConfig())
    agent_config: AgentConfig = field(default_factory=lambda: AgentConfig())
    exp_name: str = "TyperEnv"

    trainer: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            exp_name="TyperEnv",
            exp_version="v1",
            total_timesteps=5000_0000,
        )
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.trainer.exp_name = args.exp_name

    # env setup
    env_kwargs = dict(
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
    )
    env = gym.make(args.env_id, config=args.typer_config, num_envs=args.num_envs, **env_kwargs)
    eval_env = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
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
    env = ManiSkillVectorEnv(env, args.num_envs, **env_kwargs)
    eval_env = ManiSkillVectorEnv(eval_env, args.num_eval_envs, **env_kwargs)
    assert isinstance(env.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(args.agent_config, env)

    trainer = PPO_typer(config=args.trainer, agent=agent, env=env, eval_env=eval_env)
    trainer.train(seed=args.seed)

    env.close()
    eval_env.close()
