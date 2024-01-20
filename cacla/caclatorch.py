import os
import sys
import math

import time
from typing import Tuple, Optional

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime
OmegaConf.register_new_resolver(
    "current_time", lambda: strftime("%Y%m%d-%H%M%S"), replace=True
)
from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, make_env, record_video
from datetime import datetime
from datetime import date
from IPython import display as ipythondisplay

import gymnasium
from gymnasium import logger as gymlogger
# from gym.wrappers import Monitor # deprecated 2023 - https://stackoverflow.com/questions/71520568/importerror-cannot-import-name-monitor-from-gym-wrappers
from gymnasium.wrappers.record_video import RecordVideo
gymlogger.set_level(40) #error only
import numpy as np
import math
from torch.optim import SGD, Adam

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, spaces
from gymnasium.wrappers import TimeLimit

import sys
sys.modules[__name__]


"""
Continuous action version of the classic cart-pole system implemented by Rich
Sutton et al.
"""

class ContinuousEnvCACLA(CartPoleEnv):
    """Continuous version  of the CartPole-v1 environment"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            self.min_action, self.max_action, shape=(1,), dtype=np.float64
        )

    def step(self, action):
        if action > self.max_action:
            action = np.array(self.max_action)
        elif action < self.min_action:
            action = np.array(self.min_action)
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * float(action)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

class ContinuousEnvArticle(ContinuousEnvCACLA):
    """Continuous version  of the CartPole-v1 environment"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_action = -1
        self.max_action = 1

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        x = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))
        self.state = 0, 0, x, 0
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}
    
try:
  gymnasium.envs.register(
      id='CartpoleEnvCacla',
      entry_point='__main__:ContinuousEnvCACLA',
      max_episode_steps=500
  )
except:
    print("Except")
    pass
try:
  gymnasium.envs.register(
      id='CartpoleEnvArticle',
      entry_point='__main__:ContinuousEnvArticle',
      max_episode_steps=500
  )
except:
    print("Except")
    pass



class Cacla(nn.Module):
    def __init__(self, in_dim, 
                 h_dim, activation=nn.Tanh,
                 discount_factor=0.95, 
                 gaussian_noise=0.01, 
                 var=False, 
                 exploration='gaussian',
                 transform= lambda x: x,
                transform_critic = lambda x: x,
                transform_actor =  lambda x: x):
        super(Cacla,self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            activation(),
            nn.Linear(h_dim, 1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            activation(),
            nn.Linear(h_dim, 1),
        )
        # transform
        self.transform = transform
        self.transform_critic = transform_critic
        self.transform_actor = transform_actor
        # learning hyperparaters
        self.discount_factor = discount_factor
        self.vart = 1
        self.beta = 0.001
        self.with_var = var

        # exploration parameters
        self.gaussian_noise = gaussian_noise
        self.eps = 1
        self.eps_decay = 0.99
        self.eps_min = 0.01
        self.exploration = 'gaussian'
    
    def forward(self, x):
        x = self.transform(x)
        x = torch.tensor(x).unsqueeze(0)
        return self.transform_critic(self.critic(x)), self.transform_actor(self.actor(x))
    
    def actor_forward(self, x):
        x = self.transform(x)
        x = torch.from_numpy(x).unsqueeze(0)
        return self.transform_actor(self.actor(x))
                            
    def critic_forward(self, x):
        x = self.transform(x)
        x = torch.from_numpy(x).unsqueeze(0)
        return self.transform_critic(self.critic(x))

    # Loss computation
    def compute_critic_loss(self, value, target):
        delta = (target - value).detach()
        loss =  F.mse_loss(value, target)
        n_update = 1
        if self.with_var and delta > 0:
            n_update = torch.ceil(delta/np.sqrt(self.vart)).item()
            self.vart = (1-self.beta)*self.vart +self.beta*loss.detach()
        return loss, int(n_update)
    
    def compute_actor_loss(self, value, target):
        return F.mse_loss(value, target)
    
    # Exploration
    def sample(self, x):
        if self.exploration == 'gaussian':
            return self.sample_gaussian(x)
        elif self.exploration == 'epsilon':
            return self.sample_epsilon(x)
        else: 
            print('Exploration unknown: ', self.exploration)       
    
    def sample_gaussian(self, x):
        return (x + np.random.normal(0,self.gaussian_noise))
    
    def sample_epsilon(self, x, inf=-1, sup=1):
        if torch.rand(1) > self.eps: return x
        else: return np.random.rand() * (sup - inf) - inf


def test(eval_env, model, n_test = 10, noise_std = 0.3):
    cum_reward = 0
    with torch.no_grad():
        for _ in range(n_test):
            done = False
            truncated = False
            obs0,_ = eval_env.reset()
            while not done and not truncated:
                a0 = model.actor_forward(obs0)  
                obs0, reward, done,truncated,_ = eval_env.step(a0) 
                cum_reward += reward
    return cum_reward/n_test

def addnoise(x, std):
    return x + np.random.normal(0,std)
    
def train(eval_env, 
          train_env,
          model,
          optim, 
          step_max=102400, 
          step_eval=1024, 
          n_test=10, 
          noise_std=0.3):
    obs ,_ = train_env.reset()
    scores = []
    optim_critic, optim_actor = optim
    count = 0
    # boucle d'apprentissage
    for it in range(step_max+1):
        v0, a0 = model(obs)
        obs0 = obs
        action = model.sample(a0).detach()
        act = torch.clip(action,-1.,1.)
        obs, reward, done, truncated,reste = train_env.step(addnoise(act, noise_std))
        
        # Compute losses
        with torch.no_grad(): v1 = model.critic_forward(obs)
        target_v = addnoise(reward, noise_std) + model.discount_factor*v1
        optim_critic.zero_grad()
        critic_loss, n_update = model.compute_critic_loss(v0, target=target_v)
        critic_loss.backward()
        optim_critic.step()
        if (target_v - v0).item() > 0: 
            for _ in range(n_update):
                optim_actor.zero_grad()
                F.mse_loss(model.actor_forward(obs0), action).backward()
                optim_actor.step()
        if done or truncated: obs,_ = train_env.reset()
        if it % (100*(1<<count)) == 0:
        #if it % 1024 == 0:
            #if it % step_eval == 0:
            count += 1
            perf = test(eval_env, model, n_test, noise_std)
            scores.append((it, perf))
            print(f'{it = } | reward {perf}')
    return scores


def make_env(env_name, max_episode_steps= 500):
    return TimeLimit(gymnasium.make(env_name), max_episode_steps)

# Make env
env_name = 'CartpoleEnvArticle'
#env_name = 'CartpoleEnvArticle'
max_episode = 500
train_env = make_env(env_name, max_episode_steps=max_episode)
eval_env = make_env(env_name, max_episode_steps=max_episode)

all_perfs = []
n_runs = 20

# Model hyperparameters
discount_factor = 0.95
noise_std= 0.3
exploration = 'gaussian'
var = True
varname = 'var' if var else 'novar'
path = f'log/{exploration}/32_{varname}_normobs_gamma{int(discount_factor*100)}_std{noise_std}'

obsmax = np.array([0.21, 2.14, 0.27, 3.32], dtype=np.float32)
vmax, vmin = 5, 5
amax, amin = -1, 1

def norm_obs(x, xmax):
    return x / xmax

def norm_out(v, vmin, vmax):
    return 2*(v - vmin) / (vmax-vmax)-1

if not os.path.exists(path): 
    os.mkdir(path)

for run in range(1,n_runs+1):
    print(f'======= {run = } / {n_runs} =========')
    cacla = Cacla(in_dim=4, h_dim=32, 
                  discount_factor=discount_factor, 
                  gaussian_noise=0.1,
                  activation = nn.Tanh,
                 var=var,
                 exploration=exploration,
                 #transform=partial(norm_obs, xmax=obsmax),
                 #transform_critic = partial(norm_out, vmin=vmin, vmax=vmax),
                 #transform_actor = partial(norm_out, vmin=amin, vmax=amax)
                 )
    optim_crit = SGD(cacla.critic.parameters(), lr=0.01)
    optim_act = SGD(cacla.actor.parameters(), lr=0.01)
    optim = (optim_crit, optim_act)
    scores = train(eval_env, train_env, cacla, optim, 
                   step_max=102400,
                  n_test=10,
                  noise_std= noise_std)
    scores = np.array(scores)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt(path+f'/{timestr}.log', scores)