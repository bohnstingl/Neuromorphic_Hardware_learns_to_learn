__author__ = 'Tom Schaul, tom@idsia.ch'


from pybrain_local.rl.learners.learner import Learner


class MetaLearner(Learner):
    """ Learners that make use of other Learners, or learn how to learn. """