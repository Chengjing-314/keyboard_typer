import torch
import gymnasium as gym

from keyboard_typer.mani_skill.envs.typer_env import TyperEnvConfig
from keyboard_typer.curriculum.stages import StageConfig
import rich

console = rich.get_console()


def main():
    config = TyperEnvConfig(kb_debug=True)
    stage_config = StageConfig()

    env = gym.make(
        "TyperEnv-v0",
        config=config,
        stage_config=stage_config,
        render_mode="human",
        num_envs=1,
        obs_mode="state_dict",
        parallel_in_single_scene=False,
        control_mode="pd_joint_pos",
    )
    env.reset()
    report_metric(env)


def report_metric(env, pressed=0.0025):
    weight = 1.2
    action = torch.tensor(
        [
            -0.03141593,
            0.13439035,
            0.03141593,
            0.23911011,
            3.1415927,
            1.4643313,
            -0.00349066,
            0.0,
            0.305 * weight,
            0.0,
            0.0,
            0.0,
            0.0,
            1.395 * weight,
            0.0,
            0.0,
            0.0,
        ]
    )
    action_raise = torch.tensor(
        [
            -0.03141593,
            0.13439035,
            0.03141593,
            0.23911011,
            3.1415927,
            1.4643313,
            -0.00349066,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    press_start, actuated_start = -1, -1
    recover_steps = -1
    for i in range(int(1e3)):
        if i < 5e2:
            env.step(action)
            if env.unwrapped.keyboard.qpos[0, 43] > 1e-5 and press_start == -1:
                press_start = i
            if env.unwrapped.keyboard.qpos[0, 43] > pressed and actuated_start == -1:
                actuated_start = i
        else:
            env.step(action_raise)
            if env.unwrapped.keyboard.qpos[0, 43] < pressed and recover_steps == -1:
                recover_steps = i - 5e2
    console.line()
    console.log(f"Press start: {press_start}")
    console.log(f"Actuated start: {actuated_start}")
    console.log(f"Recover steps: {recover_steps}")
    console.line()


if __name__ == "__main__":
    main()
