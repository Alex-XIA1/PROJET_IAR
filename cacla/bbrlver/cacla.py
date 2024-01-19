import hydra
import torch
from torch import nn
from omegaconf import DictConfig

from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent
from bbrl.agents.agent import Agent
from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.critics import NamedCritic
from bbrl_algos.models.loggers import Logger
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best
from bbrl_algos.models.envs import get_eval_env_agent
from bbrl.utils.chrono import Chrono
from bbrl_algos.models.exploration_agents import AddGaussianNoise
from torch.distributions import Normal
from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

from bbrl_algos.models.shared_models import build_mlp, build_alt_mlp

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


class CaclaActor(ContinuousDeterministicActor):
    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        #print(obs)
        action = self.model(obs)
        self.set(("action_mean", t), action)


class CaclaGaussianExplorer(AddGaussianNoise):
    def forward(self, t, **kwargs):
        act = self.get(("action_mean", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)

class CaclaCritic(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )
        self.is_q_function = False

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.name}/v_values", t), critic)




# Create the REINFORCE Agent
def create_cacla_agent(cfg, train_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()

    # Critic
    critic = TemporalAgent(
            CaclaCritic(obs_size, cfg.algorithm.architecture.critic_hidden_size)
        ) 
    # print_agent = PrintAgent()
    actor = CaclaActor(
        obs_size,
        cfg.algorithm.architecture.actor_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.act,
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = CaclaGaussianExplorer(
        cfg.algorithm.action_noise,
        seed=cfg.algorithm.seed.explorer,
    )

    tr_agent = Agents(train_env_agent, actor, noise_agent)  # TODO : add OU noise
    # Get an agent that is executed on a complete workspace
    return TemporalAgent(tr_agent), critic


# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference
    # print(f"reward:{reward}, V:{v_value}, MB:{must_bootstrap}")
    target = (
        reward[1:]
        + cfg.algorithm.discount_factor
        * v_value[1:].detach()
        * must_bootstrap[1:].int()
    )
    #print(reward.max())
    td = target - v_value[:-1]

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(f"target:{target}, td:{td}, cl:{critic_loss}")
    return critic_loss, td.detach()


def compute_actor_loss(actions, actions_mean, td):
    actions = actions[:-1]
    actions_mean = actions_mean[:-1]
    td = torch.flatten(td)
    error =  torch.flatten((actions - actions_mean))
    error = error[td > 0]
    actor_loss = error**2

    return actor_loss.mean()

def run_reinforce(cfg, logger, trial=None):
    best_reward = float("-inf")

    # 2) Create the environment agent
    env_agent = get_eval_env_agent(cfg)

    # 3) create CACLA Agent
    (
        cacla_agent,
        critic_agent,
    ) = create_cacla_agent(cfg, env_agent)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, cacla_agent, critic_agent)

    # 8) Training loop
    nb_steps = 0
    var0 = 1
    beta = 0.001
    for episode in range(cfg.algorithm.nb_episodes):
        # print_agent.reset()
        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        # Configure the workspace to the right dimension.
        train_workspace = Workspace()
        cacla_agent(train_workspace, t=0, stop_variable="env/done")
        # Get relevant tensors (size are timestep x n_envs x ....)
        terminated, reward, action, action_mean = train_workspace[
            "env/terminated",
            "env/reward",
            "action",
            "action_mean"
        ]
        critic_agent(train_workspace, t = 0, stop_variable="env/done")
        v_value = train_workspace["critic/v_values"]
        for i in range(cfg.algorithm.n_envs_eval):
            nb_steps += len(action[:, i])

        # Determines whether values of the critic should be propagated
        must_bootstrap = ~terminated

        critic_loss, td = compute_critic_loss(cfg, reward, must_bootstrap, v_value)
        actor_loss = compute_actor_loss(action, action_mean, td)

        # Log losses
        logger.log_losses(nb_steps, critic_loss, 0, actor_loss)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        if actor_loss.item() > 0:
            actor_optimizer.zero_grad()
            # var0 = (1-beta)*var0 + beta*(td**2)
            # n_update = torch.ceil(td/torch.sqrt(var0))
            n_update = 1
            actor_loss = actor_loss * n_update
            actor_loss.backward()
            actor_optimizer.step()

        # Compute the cumulated reward on final_state
        rewards = train_workspace["env/cumulated_reward"][-1]
        mean = rewards.mean()
        logger.log_reward_losses(rewards, nb_steps)
        print(
            f"epsiode: {episode}, reward: {mean:.02f}, best: {best_reward:.02f}"
        )
        if cfg.save_best and mean > best_reward:
            best_reward = mean
            policy = cacla_agent.agent.agents[1]
            critic = critic_agent.agent
            save_best(
                policy,
                cfg.gym_env.env_name,
                mean,
                "./reinforce_best_agents/",
                "reinforce",
            )
            if cfg.plot_agents and False:
                plot_policy(
                    policy,
                    env_agent,
                    best_reward,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                    stochastic=False,
                )
                plot_critic(
                    critic,
                    env_agent,
                    best_reward,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                )


@hydra.main(
    config_path="./configs/",
    # config_name="reinforce_debugv.yaml",
    config_name="cacla_cartpole.yaml",
    # version_base="1.1",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_reinforce)
    else:
        logger = Logger(cfg_raw)
        run_reinforce(cfg_raw, logger)


if __name__ == "__main__":
    main()
