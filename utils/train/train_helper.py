import numpy as np
import torch


def get_action_dimension(reward_type, ignore_kinematics_limit=False):

    if reward_type == "hand_qpos_only":
        if ignore_kinematics_limit:
            return 32 - 2 * 6
        else:
            return 32 - 2 * 6 - 2 * 4
    elif reward_type == "standard":
        if ignore_kinematics_limit:
            return 32
        else:
            return 32 - 2 * 4
    elif reward_type == "hand_qpos_with_wrist_residual":
        if ignore_kinematics_limit:
            return 32
        else:
            return 32 - 2 * 4
    elif reward_type == "wrist_pos_only":
        if ignore_kinematics_limit:
            return 32
        else:
            return 32 - 2 * 4
    else:
        raise ValueError("Invalid reward type")


def get_more_free_gpu():
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        memory_available = [int(x) for x in result.stdout.strip().split("\n")]
        return np.argmax(memory_available)

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_flattend_obs_extra_dict(obs_dict):

    extra_dict = obs_dict["extra"]

    obs = []

    for k, v in extra_dict.items():
        #print(k, v.shape)
        obs.append(v)

    obs = torch.cat(obs, dim=-1)

    return obs
