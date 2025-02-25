from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import gymnasium as gym
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import VectorEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.structs.types import Array
from torch.distributions.normal import Normal
from keyboard_typer.mani_skill.envs.typer_env import MAX_EPISODE_STEPS
import wandb


if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv

console = rich.get_console()


class ManiSkillVectorEnv(VectorEnv):
    """
    Gymnasium Vector Env implementation for ManiSkill environments running on the GPU for parallel simulation and optionally parallel rendering

    Note that currently this also assumes modeling tasks as infinite horizon (e.g. terminations is always False, only reset when timelimit is reached)

    Args:
        env: The environment created via gym.make / after wrappers are applied. If a string is given, we use gym.make(env) to create an environment
        num_envs: The number of parallel environments. This is only used if the env argument is a string
        env_kwargs: Environment kwargs to pass to gym.make. This is only used if the env argument is a string
        auto_reset (bool): Whether this wrapper will auto reset the environment (following the same API/conventions as Gymnasium).
            Default is True (recommended as most ML/RL libraries use auto reset)
        ignore_terminations (bool): Whether this wrapper ignores terminations when deciding when to auto reset. Terminations can be caused by
            the task reaching a success or fail state as defined in a task's evaluation function. Default is False, meaning there is early stop in
            episode rollouts. If set to True, this would generally for situations where you may want to model a task as infinite horizon where a task
            stops only due to the timelimit.
    """

    def __init__(
        self,
        env: BaseEnv | str,
        num_envs: int = None,
        auto_reset: bool = True,
        max_episode_steps: int = None,
        ignore_terminations: bool = False,
        **kwargs,
    ):
        if isinstance(env, str):
            self._env = gym.make(env, num_envs=num_envs, **kwargs)
        else:
            self._env = env
            num_envs = self.base_env.num_envs
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        super().__init__(
            num_envs,
            self._env.unwrapped.single_observation_space,
            self._env.unwrapped.single_action_space,
        )

        self.returns = torch.zeros(self.num_envs, device=self.base_env.device)
        self.return_components: dict[str, torch.Tensor] = {}
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps is None and self.base_env.spec.max_episode_steps is not None:
            self.max_episode_steps = self.base_env.spec.max_episode_steps
        if self.max_episode_steps is None:
            # search wrappers to find where max episode steps may have been defined
            self.max_episode_steps = gym_utils.find_max_episode_steps_value(env)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict | None = dict(),
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.base_env.device)
            mask[env_idx] = True
            self.returns[mask] = 0
            for key in self.return_components:
                self.return_components[key][mask] = 0
        else:
            self.returns *= 0
            for key in self.return_components:
                self.return_components[key] *= 0
        return obs, info

    def step(self, actions: Array | dict) -> tuple[Array, Array, Array, Array, dict]:
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        self.returns += rew

        if "reward_components" in infos:
            for key, value in infos["reward_components"].items():
                if key not in self.return_components:
                    self.return_components[key] = torch.zeros_like(self.returns)
                self.return_components[key] += value

        infos["episode"] = dict(r=self.returns, r_components=self.return_components)
        if self.max_episode_steps is not None:
            truncations: torch.Tensor = self.base_env.elapsed_steps >= self.max_episode_steps
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False
        dones = torch.logical_or(terminations, truncations)
        infos["real_next_obs"] = (
            obs  # not part of standard API but makes some RL code slightly less complicated
        )
        if dones.any():
            infos["episode"]["r"] = self.returns.clone()
            infos["episode"]["r_components"] = {
                key: value.clone() for key, value in self.return_components.items()
            }
            final_obs = obs
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            obs, _ = self.reset(options=dict(env_idx=env_idx))
            infos["episode"]["_r"] = dones
            infos["final_info"] = infos.copy()
            # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
            infos["final_observation"] = final_obs
            # NOTE (stao): that adding masks like below is a bit redundant and not necessary
            # but this is to follow the standard gymnasium API
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE (stao): Unlike gymnasium, the code here does not add masks for every key in the info object.
        return obs, rew, terminations, truncations, infos

    def close(self):
        return self._env.close()

    def call(self, name: str, *args, **kwargs):
        function = getattr(self.env, name)
        return function(*args, **kwargs)

    def get_attr(self, name: str):
        raise RuntimeError("To get an attribute get it from the .env property of this object")

    def render(self):
        return self.base_env.render()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray:
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


@dataclass
class AgentConfig:
    obs: Literal["state"] = "state_dict"
    latent_dim: int | None = None
    load_from: str | None = None


class Agent(nn.Module):
    def __init__(self, config: AgentConfig, env: VectorEnv):
        super().__init__()
        self.config = config
        if self.config.latent_dim is None and self.config.obs == "state":
            latent_dim = np.array(env.unwrapped.single_observation_space.shape).prod()
        else:
            latent_dim = self.config.latent_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(env.unwrapped.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(env.unwrapped.single_action_space.shape)) * -0.5
        )

        if self.config.load_from is not None:
            self.load_state_dict(torch.load(self.config.load_from, map_location="cpu"))

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


@dataclass
class PPOConfig:
    use_cuda: bool = True
    exp_root: str = "logs"
    exp_name: str = "default"
    exp_version: str | None = None
    save_model: bool = True
    use_wandb: bool = True
    wandb_project: str = "keyboard_typer"

    total_timesteps: int = 5_500_000
    num_steps: int = MAX_EPISODE_STEPS
    num_eval_steps: int = MAX_EPISODE_STEPS
    num_minibatches: int = 32
    update_epochs: int = 4
    eval_freq: int = 8

    learning_rate: float = 3e-4
    anneal_lr: bool = False

    gamma: float = 0.99
    gae_lambda: float = 0.9
    finite_horizon_gae: bool = True

    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    target_kl: float = 0.1
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    obs_mode = "state"

    def __post_init__(self):
        if self.exp_version is None:
            if os.path.exists(os.path.join(self.exp_root, self.exp_name)):
                versions = os.listdir(os.path.join(self.exp_root, self.exp_name))
                if len(versions) == 0:
                    self.exp_version = "version_0"
                else:
                    __import__("IPython").embed(header="rl_lib.py:326")
                    versions = [int(v.split("_")[-1]) for v in versions]
                    self.exp_version = f"version_{max(versions) + 1}"
            else:
                self.exp_version = "version_0"


class PPO:
    def __init__(
        self,
        config: PPOConfig,
        agent: Agent,
        env: ManiSkillVectorEnv,
        eval_env: ManiSkillVectorEnv,
    ):
        self.config = config

        self.env = env
        self.eval_env = eval_env

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu"
        )
        self.agent = agent.to(self.device)

    def train(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        batch_size = int(self.env.num_envs * self.config.num_steps)
        minibatch_size = batch_size // self.config.num_minibatches
        num_iterations = self.config.total_timesteps // batch_size

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=self.config.exp_name,
                save_code=True,
                monitor_gym=True,
            )

        optimizer = optim.Adam(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        if isinstance(self.env.unwrapped.single_observation_space, gym.spaces.dict.Dict):
            obs = DictArray(
                (self.config.num_steps, self.env.num_envs),
                self.env.unwrapped.single_observation_space,
                device=self.device,
            )
        else:
            obs = torch.zeros(
                (self.config.num_steps, self.env.num_envs)
                + self.env.unwrapped.single_observation_space.shape
            ).to(self.device)
        actions = torch.zeros(
            (self.config.num_steps, self.env.num_envs)
            + self.env.unwrapped.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        rewards = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        dones = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        values = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=seed)
        eval_obs, _ = self.eval_env.reset(seed=seed)
        next_done = torch.zeros(self.env.num_envs, device=self.device)
        eps_lens = np.zeros(self.env.num_envs)
        console.rule()
        console.log(
            f"{num_iterations=} num_envs={self.env.num_envs} num_eval_envs={self.eval_env.num_envs}"
        )
        console.log(f"{batch_size=} {minibatch_size=} update_epochs={self.config.update_epochs}")
        console.log(
            f"single_observation_space.shape={self.env.unwrapped.single_observation_space.shape} single_action_space.shape={self.env.unwrapped.single_action_space.shape}"
        )
        console.rule()
        action_space_low, action_space_high = (
            torch.from_numpy(self.env.unwrapped.single_action_space.low).to(self.device),
            torch.from_numpy(self.env.unwrapped.single_action_space.high).to(self.device),
        )

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), action_space_low, action_space_high)

        for iteration in range(1, num_iterations + 1):
            console.log(f"Epoch: {iteration}, {global_step=}")
            final_values = torch.zeros(
                (self.config.num_steps, self.env.num_envs), device=self.device
            )
            self.agent.eval()
            if iteration % self.config.eval_freq == 1:
                # evaluate
                console.log("Evaluating...")
                self.eval_env.reset()
                returns = []
                return_components = {}
                eps_lens = []
                successes = []
                failures = []
                for i in range(self.config.num_eval_steps):
                    with torch.no_grad():
                        action = self.agent.get_action(eval_obs, deterministic=True)
                        eval_obs, _, eval_terminations, eval_truncations, eval_infos = (
                            self.eval_env.step(action)
                        )
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            eps_lens.append(
                                eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy()
                            )
                            returns.append(
                                eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy()
                            )
                            if "reward_components" in eval_infos["final_info"]:
                                for key, value in eval_infos["final_info"]["episode"][
                                    "r_components"
                                ].items():
                                    if key not in return_components:
                                        return_components[key] = []
                                    return_components[key].append(value[mask].cpu().numpy())
                            if "success" in eval_infos:
                                successes.append(
                                    eval_infos["final_info"]["success"][mask].cpu().numpy()
                                )
                            if "fail" in eval_infos:
                                failures.append(
                                    eval_infos["final_info"]["fail"][mask].cpu().numpy()
                                )
                returns = np.concatenate(returns)
                return_components = {
                    key: np.concatenate(value) for key, value in return_components.items()
                }
                eps_lens = np.concatenate(eps_lens)
                console.log(
                    f"Evaluated {self.config.num_eval_steps * self.eval_env.num_envs} steps resulting in {len(eps_lens)} episodes"
                )
                if len(successes) > 0:
                    successes = np.concatenate(successes)
                    if self.config.use_wandb:
                        wandb.log({"eval/success_rate": successes.mean()}, step=global_step)
                    console.log(f"eval_success_rate={successes.mean()}")
                if len(failures) > 0:
                    failures = np.concatenate(failures)
                    if self.config.use_wandb:
                        wandb.log({"eval/fail_rate": failures.mean()}, step=global_step)
                    console.log(f"eval_fail_rate={failures.mean()}")

                console.log(f"eval_episodic_return={returns.mean()}")
                for key, value in return_components.items():
                    console.log(f"eval_episodic_return_{key}={value.mean()}")
                if self.config.use_wandb:
                    wandb.log({"eval/episodic_return": returns.mean()}, step=global_step)
                    wandb.log({"eval/episodic_length": eps_lens.mean()}, step=global_step)
                    for key, value in return_components.items():
                        wandb.log({f"eval/episodic_return_{key}": value.mean()}, step=global_step)
            if self.config.save_model and iteration % self.config.eval_freq == 1:
                model_path = os.path.join(
                    self.config.exp_root,
                    self.config.exp_name,
                    self.config.exp_version,
                    f"ckpt_{iteration}.pt",
                )
                torch.save(self.agent.state_dict(), model_path)
                console.log(f"model saved to {model_path}")

            # Annealing the rate if instructed to do so.
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.config.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            rollout_time = time.time()
            for step in range(0, self.config.num_steps):
                global_step += self.env.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(
                    clip_action(action)
                )
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
                rewards[step] = reward.view(-1)

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    episodic_return = final_info["episode"]["r"][done_mask].cpu().numpy().mean()
                    if "success" in final_info:
                        if self.config.use_wandb:
                            wandb.log(
                                {
                                    "charts/success_rate": final_info["success"][done_mask]
                                    .cpu()
                                    .numpy()
                                    .mean()
                                },
                                step=global_step,
                            )
                    if "fail" in final_info:
                        if self.config.use_wandb:
                            wandb.log(
                                {
                                    "charts/fail_rate": final_info["fail"][done_mask]
                                    .cpu()
                                    .numpy()
                                    .mean()
                                },
                                step=global_step,
                            )
                    if self.config.use_wandb:
                        wandb.log({"charts/episodic_return": episodic_return}, step=global_step)
                        wandb.log(
                            {
                                "charts/episodic_length": final_info["elapsed_steps"][done_mask]
                                .cpu()
                                .numpy()
                                .mean()
                            },
                            step=global_step,
                        )

                    if isinstance(infos["final_observation"], dict):
                        for k in infos["final_observation"]:
                            infos["final_observation"][k] = infos["final_observation"][k][
                                done_mask
                            ]
                        final_values[
                            step, torch.arange(self.env.num_envs, device=self.device)[done_mask]
                        ] = self.agent.get_value(infos["final_observation"]).view(-1)
                    else:
                        final_values[
                            step, torch.arange(self.env.num_envs, device=self.device)[done_mask]
                        ] = self.agent.get_value(infos["final_observation"][done_mask]).view(-1)
            rollout_time = time.time() - rollout_time
            # bootstrap value according to termination and truncation
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = (
                        next_not_done * nextvalues + final_values[t]
                    )  # t instead of t+1
                    # next_not_done means nextvalues is computed from the correct next_obs
                    # if next_not_done is 1, final_values is always 0
                    # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                    if self.config.finite_horizon_gae:
                        """
                        See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                        1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                        lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                        lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                        lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                        We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                        """
                        if t == self.config.num_steps - 1:  # initialize
                            lam_coef_sum = 0.0
                            reward_term_sum = 0.0  # the sum of the second term
                            value_term_sum = 0.0  # the sum of the third term
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + self.config.gae_lambda * lam_coef_sum
                        reward_term_sum = (
                            self.config.gae_lambda * self.config.gamma * reward_term_sum
                            + lam_coef_sum * rewards[t]
                        )
                        value_term_sum = (
                            self.config.gae_lambda * self.config.gamma * value_term_sum
                            + self.config.gamma * real_next_values
                        )

                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[
                            t
                        ]
                    else:
                        delta = rewards[t] + self.config.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = (
                            delta
                            + self.config.gamma
                            * self.config.gae_lambda
                            * next_not_done
                            * lastgaelam
                        )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                returns = advantages + values

            # flatten the batch
            if isinstance(self.env.unwrapped.single_observation_space, gym.spaces.dict.Dict):
                b_obs = obs.reshape((-1,))
            else:
                b_obs = obs.reshape((-1,) + self.env.unwrapped.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.unwrapped.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            self.agent.train()
            b_inds = np.arange(batch_size)
            clipfracs = []
            update_time = time.time()
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                        ]

                    if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    v_loss *= self.config.vf_coef

                    entropy_loss = entropy.mean() * self.config.ent_coef
                    loss = pg_loss - entropy_loss + v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break
            update_time = time.time() - update_time

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.config.use_wandb:
                wandb.log(
                    {
                        "losses/total_loss": loss.item(),
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "charts/update_time": update_time,
                        "charts/rollout_time": rollout_time,
                        "charts/rollout_fps": self.env.num_envs
                        * self.config.num_steps
                        / rollout_time,
                    },
                    step=global_step,
                )
            console.log("SPS:", int(global_step / (time.time() - start_time)))

        if self.config.save_model:
            model_path = os.path.join(
                self.config.exp_root,
                self.config.exp_name,
                self.config.exp_version,
                "final_ckpt.pt",
            )
            torch.save(self.agent.state_dict(), model_path)
            console.log(f"model saved to {model_path}")

    def predict(self, obs):
        pass


class PPO_typer(PPO):
    def __init__(
        self,
        config: PPOConfig,
        agent: Agent,
        env: ManiSkillVectorEnv,
        eval_env: ManiSkillVectorEnv,
    ):
        super().__init__(config, agent, env, eval_env)

    def get_reward_detail(self, eval=False):
        if eval:
            reward_dict = self.eval_env._env.unwrapped.get_reward_details()
        else:
            reward_dict = self.env._env.unwrapped.get_reward_details()

        return reward_dict

    def reset_reward(self):
        self.tcp_distance_reward = 0
        self.over_all_distance_reward = 0
        self.key_actuation_reward = 0
        self.rotation_distance_reward = 0
        self.velocity_penalty = 0
        self.wrong_key_penalty = 0

    def update_reward(self, reward_dict):
        self.tcp_distance_reward += reward_dict["tcp_distance_reward"]
        self.over_all_distance_reward += reward_dict["over_all_distance_reward"]
        self.key_actuation_reward += reward_dict["key_actuation_reward"]
        self.rotation_distance_reward += reward_dict["rotation_distance_reward"]
        self.velocity_penalty += reward_dict["velocity_penalty"]
        self.wrong_key_penalty += reward_dict["wrong_key_penalty"]

    def process_obs(self, obs: dict):
        agent_dict = obs["agent"]
        extra_dict = obs["extra"]

        # normalization to [-1, 1]
        qlimits = self.env._env.unwrapped.agent.robot.qlimits[0]  # dof * 2
        agent_dict["qpos"] = (
            2 * (agent_dict["qpos"] - qlimits[:, 0]) / (qlimits[:, 1] - qlimits[:, 0]) - 1
        )

        # torch concat
        agent_tensor = [v for v in agent_dict.values()]
        extra_tensor = [v.unsqueeze(-1) if v.ndim == 1 else v for v in extra_dict.values()]

        return torch.cat(agent_tensor + extra_tensor, dim=-1).float()

    def normalize_qpos(self, obs: torch.Tensor):
        qlimits_low, qlimits_high = (
            self.env._env.unwrapped.agent.robot.qlimits[0][:, 0],
            self.env._env.unwrapped.agent.robot.qlimits[0][:, 1],
        )

        obs[:, : self.env.unwrapped.single_action_space.shape[0]] = (
            2
            * (obs[:, : self.env.unwrapped.single_action_space.shape[0]] - qlimits_low)
            / (qlimits_high - qlimits_low)
        )

        return obs

    def train(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        batch_size = int(self.env.num_envs * self.config.num_steps)
        minibatch_size = batch_size // self.config.num_minibatches
        num_iterations = self.config.total_timesteps // batch_size

        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=self.config.exp_name,
                save_code=True,
                monitor_gym=True,
            )

        optimizer = optim.Adam(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        if isinstance(self.env.unwrapped.single_observation_space, gym.spaces.dict.Dict):
            obs = DictArray(
                (self.config.num_steps, self.env.num_envs),
                self.env.unwrapped.single_observation_space,
                device=self.device,
            )
        else:
            obs = torch.zeros(
                (self.config.num_steps, self.env.num_envs)
                + self.env.unwrapped.single_observation_space.shape
            ).to(self.device)
        actions = torch.zeros(
            (self.config.num_steps, self.env.num_envs)
            + self.env.unwrapped.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        rewards = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        dones = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)
        values = torch.zeros((self.config.num_steps, self.env.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=seed)
        eval_obs, _ = self.eval_env.reset(seed=seed)

        # next_obs = self.normalize_qpos(next_obs)
        # eval_obs = self.normalize_qpos(eval_obs)

        next_done = torch.zeros(self.env.num_envs, device=self.device)
        eps_lens = np.zeros(self.env.num_envs)
        console.rule()
        console.log(
            f"{num_iterations=} num_envs={self.env.num_envs} num_eval_envs={self.eval_env.num_envs}"
        )
        console.log(f"{batch_size=} {minibatch_size=} update_epochs={self.config.update_epochs}")
        console.log(
            f"single_observation_space.shape={self.env.unwrapped.single_observation_space.shape} single_action_space.shape={self.env.unwrapped.single_action_space.shape}"
        )
        console.rule()
        action_space_low, action_space_high = (
            torch.from_numpy(self.env.unwrapped.single_action_space.low).to(self.device),
            torch.from_numpy(self.env.unwrapped.single_action_space.high).to(self.device),
        )

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), action_space_low, action_space_high)

        for iteration in range(1, num_iterations + 1):
            console.log(f"Epoch: {iteration}, {global_step=}")
            final_values = torch.zeros(
                (self.config.num_steps, self.env.num_envs), device=self.device
            )
            self.agent.eval()
            if iteration % self.config.eval_freq == 1:
                # evaluate
                console.log("Evaluating...")
                self.eval_env.reset()
                returns = []
                return_components = {}
                eps_lens = []
                successes = []
                failures = []
                for i in range(self.config.num_eval_steps):
                    with torch.no_grad():
                        action = self.agent.get_action(eval_obs, deterministic=True)
                        eval_obs, _, eval_terminations, eval_truncations, eval_infos = (
                            self.eval_env.step(action)
                        )
                        # eval_obs = self.normalize_qpos(eval_obs)
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            eps_lens.append(
                                eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy()
                            )
                            returns.append(
                                eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy()
                            )
                            if "reward_components" in eval_infos["final_info"]:
                                for key, value in eval_infos["final_info"]["episode"][
                                    "r_components"
                                ].items():
                                    if key not in return_components:
                                        return_components[key] = []
                                    return_components[key].append(value[mask].cpu().numpy())
                            if "success" in eval_infos:
                                successes.append(
                                    eval_infos["final_info"]["success"][mask].cpu().numpy()
                                )
                            if "fail" in eval_infos:
                                failures.append(
                                    eval_infos["final_info"]["fail"][mask].cpu().numpy()
                                )
                returns = np.concatenate(returns)
                return_components = {
                    key: np.concatenate(value) for key, value in return_components.items()
                }
                eps_lens = np.concatenate(eps_lens)
                console.log(
                    f"Evaluated {self.config.num_eval_steps * self.eval_env.num_envs} steps resulting in {len(eps_lens)} episodes"
                )
                if len(successes) > 0:
                    successes = np.concatenate(successes)
                    if self.config.use_wandb:
                        wandb.log({"eval/success_rate": successes.mean()}, step=global_step)
                    console.log(f"eval_success_rate={successes.mean()}")
                if len(failures) > 0:
                    failures = np.concatenate(failures)
                    if self.config.use_wandb:
                        wandb.log({"eval/fail_rate": failures.mean()}, step=global_step)
                    console.log(f"eval_fail_rate={failures.mean()}")

                console.log(f"eval_episodic_return={returns.mean()}")
                for key, value in return_components.items():
                    console.log(f"eval_episodic_return_{key}={value.mean()}")
                if self.config.use_wandb:
                    wandb.log({"eval/episodic_return": returns.mean()}, step=global_step)
                    wandb.log({"eval/episodic_length": eps_lens.mean()}, step=global_step)
                    for key, value in return_components.items():
                        wandb.log({f"eval/episodic_return_{key}": value.mean()}, step=global_step)
            if self.config.save_model and iteration % self.config.eval_freq == 1:
                model_path = os.path.join(
                    self.config.exp_root,
                    self.config.exp_name,
                    "ckpts",
                    f"ckpt_{iteration}.pt",
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.agent.state_dict(), model_path)
                console.log(f"model saved to {model_path}")

            # Annealing the rate if instructed to do so.
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.config.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            self.reset_reward()

            rollout_time = time.time()
            for step in range(0, self.config.num_steps):
                global_step += self.env.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.env.step(
                    clip_action(action)
                )
                # next_obs = self.normalize_qpos(next_obs)
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
                rewards[step] = reward.view(-1)

                reward_dict = self.get_reward_detail()
                self.update_reward(reward_dict=reward_dict)

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    episodic_return = final_info["episode"]["r"][done_mask].cpu().numpy().mean()
                    if "success" in final_info:
                        if self.config.use_wandb:
                            wandb.log(
                                {
                                    "charts/success_rate": final_info["success"][done_mask]
                                    .cpu()
                                    .numpy()
                                    .mean()
                                },
                                step=global_step,
                            )
                    if "fail" in final_info:
                        if self.config.use_wandb:
                            wandb.log(
                                {
                                    "charts/fail_rate": final_info["fail"][done_mask]
                                    .cpu()
                                    .numpy()
                                    .mean()
                                },
                                step=global_step,
                            )
                    if self.config.use_wandb:
                        wandb.log({"charts/episodic_return": episodic_return}, step=global_step)
                        wandb.log(
                            {
                                "charts/episodic_length": final_info["elapsed_steps"][done_mask]
                                .cpu()
                                .numpy()
                                .mean()
                            },
                            step=global_step,
                        )

                    if isinstance(infos["final_observation"], dict):
                        for k in infos["final_observation"]:
                            infos["final_observation"][k] = infos["final_observation"][k][
                                done_mask
                            ]
                        final_values[
                            step, torch.arange(self.env.num_envs, device=self.device)[done_mask]
                        ] = self.agent.get_value(infos["final_observation"]).view(-1)
                    else:
                        final_values[
                            step, torch.arange(self.env.num_envs, device=self.device)[done_mask]
                        ] = self.agent.get_value(infos["final_observation"][done_mask]).view(-1)
            rollout_time = time.time() - rollout_time
            # bootstrap value according to termination and truncation
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = (
                        next_not_done * nextvalues + final_values[t]
                    )  # t instead of t+1
                    # next_not_done means nextvalues is computed from the correct next_obs
                    # if next_not_done is 1, final_values is always 0
                    # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                    if self.config.finite_horizon_gae:
                        """
                        See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                        1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                        lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                        lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                        lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                        We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                        """
                        if t == self.config.num_steps - 1:  # initialize
                            lam_coef_sum = 0.0
                            reward_term_sum = 0.0  # the sum of the second term
                            value_term_sum = 0.0  # the sum of the third term
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + self.config.gae_lambda * lam_coef_sum
                        reward_term_sum = (
                            self.config.gae_lambda * self.config.gamma * reward_term_sum
                            + lam_coef_sum * rewards[t]
                        )
                        value_term_sum = (
                            self.config.gae_lambda * self.config.gamma * value_term_sum
                            + self.config.gamma * real_next_values
                        )

                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[
                            t
                        ]
                    else:
                        delta = rewards[t] + self.config.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = (
                            delta
                            + self.config.gamma
                            * self.config.gae_lambda
                            * next_not_done
                            * lastgaelam
                        )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                returns = advantages + values

            # flatten the batch
            if isinstance(self.env.unwrapped.single_observation_space, gym.spaces.dict.Dict):
                b_obs = obs.reshape((-1,))
            else:
                b_obs = obs.reshape((-1,) + self.env.unwrapped.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.unwrapped.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            self.agent.train()
            b_inds = np.arange(batch_size)
            clipfracs = []
            update_time = time.time()
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                        ]

                    if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    v_loss *= self.config.vf_coef

                    entropy_loss = entropy.mean() * self.config.ent_coef
                    loss = pg_loss - entropy_loss + v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break
            update_time = time.time() - update_time

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.config.use_wandb:
                wandb.log(
                    {
                        "losses/total_loss": loss.item(),
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "charts/update_time": update_time,
                        "charts/rollout_time": rollout_time,
                        "charts/rollout_fps": self.env.num_envs
                        * self.config.num_steps
                        / rollout_time,
                    },
                    step=global_step,
                )
                wandb.log(
                    {
                        "rewards/tcp_distance": self.tcp_distance_reward / self.config.num_steps,
                        "rewards/over_all_distance": self.over_all_distance_reward
                        / self.config.num_steps,
                        "rewards/key_actuation": self.key_actuation_reward / self.config.num_steps,
                        "rewards/rotation_distance": self.rotation_distance_reward
                        / self.config.num_steps,
                        "rewards/velocity_penalty": self.velocity_penalty / self.config.num_steps,
                        "rewards/wrong_key_penalty": self.wrong_key_penalty
                        / self.config.num_steps,
                    },
                    step=global_step,
                )
            console.log("SPS:", int(global_step / (time.time() - start_time)))

        if self.config.save_model:
            model_path = os.path.join(
                self.config.exp_root,
                self.config.exp_name,
                "ckpts",
                "final_ckpt.pt",
            )
            torch.save(self.agent.state_dict(), model_path)
            console.log(f"model saved to {model_path}")

    def log_extra(self, global_step):
        pass
