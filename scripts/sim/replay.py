from keyboard_typer.mani_skill.envs.typer_env import TyperEnvConfig
from keyboard_typer.curriculum.stages import StageConfig
import gymnasium as gym 
import numpy as np



def main():

    config = TyperEnvConfig()
    stage_config = StageConfig()
    n_envs = 1

    env = gym.make(
        "TyperEnv-v0",
        config=config,
        stage_config=stage_config,
        render_mode="human",
        num_envs=n_envs,
        obs_mode="state_dict",
        parallel_in_single_scene=False,
        control_mode="pd_joint_pos",
    )
    env.reset()
    
    intepolated_trajctory = np.load("/data/chengjingyuan/keyboard_typer/interpolated_trajectory.npy")
    original_trajectory = np.load("/data/chengjingyuan/keyboard_typer/original_trajectory.npy")
    
    rest_qpos = np.array(env.unwrapped.agent.keyframes['rest'].qpos)
    
    reset_idx = []
    
    for i, qpos in enumerate(original_trajectory):
        if np.allclose(qpos, rest_qpos, atol=1e-3):
            reset_idx.append(i)
   
    for qpos in original_trajectory:
        env.step(qpos)
        env.render_human()



if __name__ == "__main__":
    main()
