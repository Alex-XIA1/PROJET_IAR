import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense


class Cacla_Keras:
    def __init__(self, input_dim, output_dim, alpha, beta, gamma, exploration_factor):
        """
        initializes CACLA reinforcement learning algorithm.
        """
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.gamma = gamma
        self.exploration_factor = exploration_factor
        self.lr_decay = 1

        self.alpha = alpha
        self.beta = beta

        # creates neural networks.
        self.actor = self._create_actor(input_dim, output_dim, alpha)
        self.critic = self._create_critic(input_dim, 1, beta)

    def update_lr(self, lr_decay):
        """
        :param lr_decay: decay for both actor and critic
        changes learning rate for actor and critic based on lr_decay.
        """
        keras.backend.set_value(self.critic.optimizer.lr,
                                keras.backend.get_value(self.critic.optimizer.lr) * lr_decay)
        keras.backend.set_value(self.actor.optimizer.lr,
                                keras.backend.get_value(self.actor.optimizer.lr) * lr_decay)
        self.alpha *= lr_decay
        self.beta *= lr_decay

    def update_exploration(self, exploration_decay=None):
        """
        updates the exploration factor.
        :param exploration_decay: exploration_factor multiplier. if None, default value is used.
        """
        self.exploration_factor = 0.3

    @staticmethod
    def sample(action, explore):
        """
        :param action: default action predicted by actor
        :param explore: exploration factor
        :return: explored action, normally distributed around default action.
        """
        return np.random.normal(action, explore)

    @staticmethod
    def _create_actor(input_dim, output_dim, learning_rate):
        """
        Creates actor. Uses 1 hidden layers with number of neurons 5 * input_dim (~40).
        initializes weights to some small value.
        """
        l1_size = 12

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_dim, activation="tanh"))
        model.add(Dense(output_dim, activation="linear"))

        optim = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optim)
        return model

    @staticmethod
    def _create_critic(input_dim, output_dim, learning_rate):
        """
        See self._create_actor.
        """
        l1_size = 12

        model = Sequential()
        model.add(Dense(l1_size, input_dim=input_dim, activation="tanh"))
        model.add(Dense(output_dim, activation='linear'))

        optim = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optim)
        return model