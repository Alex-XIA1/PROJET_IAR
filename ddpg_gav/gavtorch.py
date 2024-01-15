import os
import sys
from pathlib import Path
import math

from moviepy.editor import ipython_display as video_display
import time
from tqdm.auto import tqdm
from typing import Tuple, Optional
from functools import partial

from omegaconf import OmegaConf
import torch
import bbrl_gymnasium

import copy
from abc import abstractmethod, ABC
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
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
from IPython.display import HTML
from torch.optim import SGD, Adam

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, spaces
from gymnasium.wrappers import TimeLimit

import sys
sys.modules[__name__]

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
    

try:
  gymnasium.envs.register(
      id='CartpoleEnvCacla',
      entry_point='__main__:ContinuousEnvCACLA',
      max_episode_steps=500
  )
except:
    print("Except")
    pass




# Gradient Ascent value algorithm de l'article
class Gav(nn.Module):
    def __init__(self, in_dim, h_dim, activation=nn.Tanh, discount_factor=0.95, gaussian_noise=0.01, lrate = 0.01, decay=0.99, exploration="gaussian"):
        super(Gav,self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(in_dim+1, h_dim),
            activation(),
            nn.Linear(h_dim, 1)
        )
        self.critic_optim = SGD(self.critic.parameters(), lr=lrate)
        
        self.actor = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            activation(),
            nn.Linear(h_dim, 1),
            #activation()
        )
        self.actor_optim = SGD(self.actor.parameters(), lr=lrate)
        
        self.discount_factor = discount_factor
        self.gaussian_noise = gaussian_noise
        self.e = 1
        self.prob_update = 1
        self.decay_rate = decay
        self.exploration = exploration
    
    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0)
        # le critic prend aussi l'action
        act = self.actor(x)
        # gaussian exploration
        if self.exploration=="gaussian": act = self.sample(act)
        # egreedy exploration
        elif self.exploration=="egreedy" and torch.rand(1) < self.e: act = self.greed()
        x = torch.hstack((x,act))
        return self.critic(x), act
    
    def sample(self, x):
        return (x + np.random.normal(0,self.gaussian_noise))
    
    # nombre entre -1 et 1
    def greed(self):
        return (2 * torch.rand(1) - 1).unsqueeze(0)
    
    def decaye(self):
        self.e*=0.999
        self.e = max(0.01,self.e)
    
    def decay(self):
        self.prob_update*= self.decay_rate
        self.prob_update = max(0.01, self.prob_update)

def test(eval_env, model, n_test = 10):
    cum_reward = 0
    with torch.no_grad():
        for _ in range(n_test):
            done, truncated = False, False
            obs0,_ = eval_env.reset()
            while not done and not truncated:
                _, a0 = model(obs0)  
                obs0, reward, done,truncated,_ = eval_env.step(a0) 
                cum_reward += reward
    return cum_reward/n_test
    
def train(eval_env, train_env, model, eval_interval=5000, step_max=100000, exploration="gaussian"):
    it = 0
    obs,_ = train_env.reset()
    scores = np.array([0])
    count = 0
    while it <= step_max:
        it += 1
        model.critic_optim.zero_grad()
            
        q0, action = model(obs)
        
        obs, reward, done, truncated, reste = train_env.step(action+np.random.normal(0,0.1)) 
        reward = model.sample(reward+np.random.normal(0,0.1))
        q1, _ = model(obs)
        delta = reward + model.discount_factor*q1.detach() - q0
        (delta**2).backward()
        model.critic_optim.step()
        if torch.rand(1) > model.prob_update: 
            model.actor_optim.zero_grad()
            (-model(obs)[0]).backward()
            model.actor_optim.step()
        model.decay()
        # pour diminuer exponentiellement
        model.decaye()
        
        if done or truncated: obs,_ = train_env.reset()
        if it % (eval_interval*(1<<count)) == 0 :
            count += 1
            perf = test(eval_env, model)
            print(f'{it = } | reward {perf}')
            scores = np.hstack((scores,perf))
    return scores



# Initialisation
n_runs = 20
final = []
explo = "gaussian"
dfact = 0.80
for run in range(n_runs):
    train_env = gymnasium.make('CartpoleEnvCacla')
    train_env = TimeLimit(train_env, max_episode_steps=500)
    eval_env = gymnasium.make('CartpoleEnvCacla')
    eval_env = TimeLimit(eval_env, max_episode_steps=500)
    print(f'======= {run = } / {n_runs-1} =========')
    gav = Gav(in_dim=4, h_dim=12, 
              discount_factor=dfact, 
              gaussian_noise=0.1,exploration =explo)
    scores = train(eval_env, train_env, gav, eval_interval=100, step_max=102400)
    print(scores.shape)
    final.append(scores)

final = np.array(final)
date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
np.savetxt("gav"+explo+"_"+str(int(dfact*100))+"_"+date+".txt",final)