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

from bbrl_algos.models.envs import get_env_agents

from bbrl_algos.models.shared_models import build_mlp, build_alt_mlp

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use("TkAgg")


class CaclaActor(ContinuousDeterministicActor):
    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        #print(obs)
        action = self.model(obs)
        self.set(("action", t), action)


class CaclaGaussianExplorer(AddGaussianNoise):
    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action_mean", t), act)
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
def create_cacla_agent(cfg, train_env_agent, eval_env_agent):
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
        name="action_selector"
    )

    tr_agent = Agents(train_env_agent, actor, noise_agent)  # TODO : add OU noise
    ev_agent = Agents(eval_env_agent, actor, noise_agent)  # TODO : add OU noise
    # Get an agent that is executed on a complete workspace
    return TemporalAgent(tr_agent), TemporalAgent(ev_agent), critic


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
    #print('Loss')
    #print(v_value.shape, reward.shape, must_bootstrap.shape)
    target = (
        reward[:-1]
        + cfg.algorithm.discount_factor
        * v_value[:-1].detach()
        * must_bootstrap[:-1].int()
    )
    #print(reward.max())
    #print('==================')
    td = (target - v_value[1]).squeeze()
    #print('td: ', td.shape)

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(f"target:{target}, td:{td}, cl:{critic_loss}")
    return critic_loss, td.detach()


def compute_actor_loss(actions, actions_mean, td):
    actions = actions[0].squeeze()
    actions_mean = actions_mean[0].squeeze()
    td = torch.flatten(td)
    #print('act: ', actions.shape, actions_mean.shape)
    error =  torch.flatten((actions - actions_mean))
    error = error[td > 0]
    actor_loss = error**2
    return actor_loss.mean()

def run_cacla(cfg, logger, trial=None):
    best_reward = float("-inf")
    if cfg.collect_stats:
        directory = "./dqn_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "dqn_" + cfg.gym_env.env_name + ".data"
        fo = open(filename, "wb")
        stats_data = []

    # 1) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)
    print(train_env_agent.envs[0])
    print(eval_env_agent.envs[0])

    # 3) create CACLA Agent
    (
        train_agent, eval_agent, v_agent
    ) = create_cacla_agent(cfg, train_env_agent, eval_env_agent)
    train_workspace = Workspace()

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, train_agent, v_agent)
    nb_steps = 0
    tmp_steps_eval = 0

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
            )

        transition_workspace: Workspace = train_workspace.get_transitions(
                    filter_key="env/done"
                )

        # Only get the required number of steps
        steps_diff = cfg.algorithm.n_steps - nb_steps
        if transition_workspace.batch_size() > steps_diff:
            for key in transition_workspace.keys():
                transition_workspace.set_full(
                    key, transition_workspace[key][:, :steps_diff]
                )

        nb_steps += transition_workspace.batch_size()

        # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
        v_agent(transition_workspace, t=0, n_steps=2, choose_action=False)

        v_value, terminated, reward, action, action_mean = transition_workspace[
            "critic/v_values",
            "env/terminated",
            "env/reward",
            "action",
            "action_mean"
        ]

        # Determines whether values of the critic should be propagated
        # True if the task was not terminated.
        must_bootstrap = ~terminated

        critic_loss, td = compute_critic_loss(cfg, reward, must_bootstrap, v_value)
        actor_loss = compute_actor_loss(action, action_mean, td)

        # Store the loss
        logger.add_log("critic_loss", critic_loss, nb_steps)

        critic_optimizer.zero_grad()
        critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(
        #     v_agent.parameters(), cfg.algorithm.max_grad_norm
        # )
        critic_optimizer.step()
        
        if actor_loss.item() > 0:
            actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     train_agent.parameters(), cfg.algorithm.max_grad_norm
            # )
            actor_optimizer.step()
        


        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                choose_action=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            logger.log_reward_losses(rewards, nb_steps)
            mean = rewards.mean()

            if mean > best_reward:
                best_reward = mean

            print(
                f"nb_steps: {nb_steps}, reward: {mean:.02f}, best: {best_reward:.02f}"
            )

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    best_reward,
                    "./dqn_best_agents/",
                    "dqn",
                )
                if False and cfg.plot_agents:
                    critic = eval_agent.agent.agents[1]
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        input_action="policy",
                    )
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots2/",
                        cfg.gym_env.env_name,
                        input_action=None,
                    )
            if cfg.collect_stats:
                stats_data.append(rewards)

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    if cfg.collect_stats:
        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)
        print(np.shape(stats_data))
        np.savetxt(filename, stats_data.numpy())
        fo.flush()
        fo.close()

    return best_reward

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
        run_cacla(cfg_raw, logger)


if __name__ == "__main__":
    main()
