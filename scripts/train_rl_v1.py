## CUDA_VISIBLE_DEVICES=5 python3 scripts/train_rl_v1.py --exp_name
## CUDA_VISIBLE_DEVICES=6 python3 scripts/train_rl_v1.py --exp_name t85

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
    num_envs: int = 512
    num_eval_envs: int = 16
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
            total_timesteps=8000_0000,
            eval_freq=10,
        )
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.trainer.exp_name = args.exp_name
    args.trainer.obs_mode = args.obs_mode
    args.agent_config.obs = args.obs_mode
    args.seed = 1

    # Task parameters
    args.stage_config.actuation_reward_weight = 12.0  # was 12 for first exp
    args.stage_config.distance_reward_weight = 15.0
    # Regularization parameters
    args.stage_config.hand_qpos_reward_weight = 5.0
    args.stage_config.penetration_penalty_weight = 2.0
    args.stage_config.action_regularization_weight = 1.0
    args.stage_config.wrong_key_penality_weight = 4.0
    args.stage_config.penetration_zone = 1e-3

    # args.agent_config.load_from = (
    #     "/data/chengjingyuan/keyboard_typer/logs/fixed_hand/ckpts/ckpt_141.pt"
    # )

    # args.agent_config.load_from = (
    #     "/data/chengjingyuan/keyboard_typer/logs/reset_init_scratch/ckpts/ckpt_181.pt"
    # )

    # env setup
    train_env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode="pd_joint_delta_pos",
        # control_mode="arm_delta_pos_hand_pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
    )

    eval_env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode="pd_joint_delta_pos",
        # control_mode="arm_delta_pos_hand_pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
    )
    env = gym.make(
        args.env_id,
        config=args.typer_config,
        num_envs=args.num_envs,
        stage_config=args.stage_config,
        **train_env_kwargs,
    )
    eval_env = gym.make(
        args.env_id,
        config=args.typer_config,
        num_envs=args.num_eval_envs,
        stage_config=args.stage_config,
        **eval_env_kwargs,
    )
    # env = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    # eval_env = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
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
    env = ManiSkillVectorEnv(env, args.num_envs, **train_env_kwargs)
    eval_env = ManiSkillVectorEnv(eval_env, args.num_eval_envs, **eval_env_kwargs)
    assert isinstance(env.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(args.agent_config, env)

    trainer = PPO_typer(config=args.trainer, agent=agent, env=env, eval_env=eval_env)
    trainer.train(seed=args.seed, stage_config=args.stage_config)

    env.close()
    eval_env.close()
