from __future__ import print_function

import numpy as np

from snn_mab import SpikingBanditAgent, BanditException
from ltl.logging_tools import configure_loggers
from ltl.optimizees.optimizee import Optimizee
from collections import namedtuple, OrderedDict


BanditParameters = \
    namedtuple('BanditParameters', ['n_arms', 'n_pulls', 'n_samples', 'seed',
                                    'max_learning_rate', 'learning_rule',
                                    'establish_connection'])


class BanditOptimizee(Optimizee):
    def __init__(self, traj, parameters, dp=False, logger=None):
        super(BanditOptimizee, self).__init__(traj)
        seed = np.uint32(parameters.seed)
        self.random_state = np.random.RandomState(seed=seed)
        self.parameters = parameters
        self.n_pulls = parameters.n_pulls
        self.n_arms = parameters.n_arms
        self.n_samples = parameters.n_samples
        self.n_tries = 10
        self.dp = dp

        self.agent = SpikingBanditAgent(logger)
        self.establish_connection = parameters.establish_connection
        self.learning_rule = parameters.learning_rule
        self.keys = list(self.learning_rule.default_hyperparameters.keys())
        self.shapes = \
            list(np.shape(self.learning_rule.default_hyperparameters[a]) for a in self.keys)
        self.keys.extend(['action_inhibition', 'stim_inhibition'])
        self.shapes.extend([[], []])

        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('seed', seed)

    def get_params(self):
        params = []
        for param in [self.parameters.n_arms,
                      self.parameters.n_pulls,
                      self.parameters.n_samples,
                      self.parameters.seed,
                      self.parameters.max_learning_rate]:
            params.append({type(param).__name__: dict(param.__asdict())})
        return OrderedDict((['agent', params]), )

    def create_individual(self):
        individual = dict()
        for i, k in enumerate(self.keys):
            individual[k] = np.float64(self.random_state.uniform(0, 1, self.shapes[i]))
        return individual

    def bounding_func(self, individual):
        for k, v in individual.items():
            individual[k] = np.clip(v, 0, 1)
        return individual

    def simulate(self, traj):
        configure_loggers(exactly_once=True)

        action_inhibition = traj.individual.action_inhibition * (-63)
        stim_inhibition = traj.individual.stim_inhibition * (-63)
        agent_hyperparameters = \
            dict(action_inhibition=action_inhibition, stim_inhibition=stim_inhibition)

        learning_rule_hyperparameters = dict(
            learning_rate=traj.individual.learning_rate,
            learning_rate_decay=traj.individual.learning_rate_decay,
            weight_prior=traj.individual.weight_prior)
        learning_rule_hyperparameters['learning_rate'] = \
            learning_rule_hyperparameters['learning_rate'] * self.parameters.max_learning_rate
        learning_rule = self.learning_rule(hyperparameters=learning_rule_hyperparameters)

        with self.establish_connection() as connector:
            l = []
            bandit_probabilities = self.random_state.rand(self.n_tries, self.n_samples, 2)
            if self.dp:
                bandit_probabilities[:, :, 1] = 1. - bandit_probabilities[:, :, 0]
            for i in range(self.n_tries):
                try:
                    r = self.agent.play_bandit_batch(bandit_probabilities[i].reshape((-1,)), self.n_pulls,
                                                     self.n_samples,
                                                     agent_hyperparameters,
                                                     learning_rule, connector)
                    traj.f_add_result('$set.$.bandit_probabilities', bandit_probabilities[i])
                    traj.f_add_result('$set.$.reward_action', r[1]['a_r'])
                    return (-r[0],)
                except BanditException as e:
                    print(e)
                except RuntimeError as e:
                    print(e)
            return (-50,)
