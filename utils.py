"""
Continuous action version of the classic cart-pole system implemented by Rich
Sutton et al.
"""
import math

import numpy as np

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, spaces
from gymnasium.wrappers import TimeLimit

import sys
sys.modules[__name__]

class ContinuousEnvArticle(CartPoleEnv):
    """Continuous version  of the CartPole-v1 environment"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            self.min_action, self.max_action, shape=(1,), dtype=np.float64
        )
        
        self.x_threshold = 1

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

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
"""
Continuous action version of the classic cart-pole system implemented by Rich
Sutton et al.
"""
import math

import numpy as np

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



try:
  gymnasium.envs.register(
      id='CartpoleEnvArticle',
      entry_point='__main__:ContinuousEnvArticle',
      max_episode_steps=500
  )
except:
    print("Except")
    pass

