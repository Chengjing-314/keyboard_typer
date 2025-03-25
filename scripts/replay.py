from keyboard_typer.mani_skill.envs.typer_env import TyperEnvConfig
from keyboard_typer.curriculum.stages import StageConfig
from keyboard_typer.constants import PROJECT_ROOT
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
    
    episode_ids = np.load(PROJECT_ROOT / "episode_ids.npy")
    
    traj = np.load(PROJECT_ROOT / "traj.npy").squeeze(1)
    
    
    episode_traj = traj[episode_ids[1] + 1:episode_ids[2],:]    

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
    
    episode_traj = interpolate_joint_angles(episode_traj, 1000)

    np.save(PROJECT_ROOT / "interpolated_trajectory.npy", episode_traj)
    
    env.reset()
    for qpos in episode_traj:
        env.step(qpos)
        env.render_human()
    
    env.close() 



if __name__ == "__main__":
    main()
