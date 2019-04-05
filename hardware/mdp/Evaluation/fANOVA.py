from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from fanova import fANOVA
import fanova.visualizer
import pickle
import sys
import numpy as np

n_input_neurons = 5
n_hidden_neurons = 7
n_output_neurons = 1
size_w1 = n_input_neurons * n_hidden_neurons
size_w2 = n_hidden_neurons * n_output_neurons

ann_parameters = np.array([0.51502079, 0.49740018, 0.20023772, 0.28295012, 0.20839904, 0.20295959,
 0.58402285, 0.1,        0.18168052, 0.13090999, 0.78121004, 0.72488774,
 0.24002499, 0.68832076, 0.10375894, 0.21000279, 0.2803778,  0.83113176,
 0.18101726, 0.36358029, 0.60059928, 0.9,        0.37448422, 0.40058301,
 0.27326555, 0.43463887, 0.66596137, 0.13436804, 0.72311331, 0.81158696,
 0.5622522,  0.1461575,  0.48496516, 0.72801292, 0.32241905, 0.82438525,
 0.78395973, 0.78074635, 0.41544295, 0.86644921, 0.35569846, 0.47646374,
 0.11145061, 0.41872347, 0.31664511, 0.87257462, 0.88002756, 0.23176495,
 0.79220276])
w1 = ann_parameters[:size_w1].reshape((n_input_neurons, n_hidden_neurons))
b1 = ann_parameters[size_w1:(size_w1 + n_hidden_neurons)]
w2 = ann_parameters[(size_w1 + n_hidden_neurons):(size_w1 + n_hidden_neurons + size_w2)].reshape((n_hidden_neurons, n_output_neurons))

f = pickle.load(open(sys.argv[1], 'rb'))
inputs = f['inputs']
outputs = f['outputs']

import ipdb
ipdb.set_trace()
outputs = np.zeros(inputs.shape[1])

cs = ConfigurationSpace()
cs.add_hyperparameter(UniformFloatHyperparameter("0", 0., 1., default_value=0.5))
cs.add_hyperparameter(CategoricalHyperparameter("1", [0., 1.], default_value=1.))
cs.add_hyperparameter(CategoricalHyperparameter("2", [0., 1.], default_value=1.))
cs.add_hyperparameter(UniformFloatHyperparameter("3", 0., 1., default_value=0.5))
cs.add_hyperparameter(UniformFloatHyperparameter("4", 0., 1., default_value=0.5))
param = cs.get_hyperparameters()

f = fANOVA(X, Y, config_space=cs, n_trees=30)

print('Total importances')
res = f.quantify_importance((0,))
print(res[(0,)]['total importance'])
res = f.quantify_importance((1,))
print(res[(1,)]['total importance'])
res = f.quantify_importance((2,))
print(res[(2,)]['total importance'])
res = f.quantify_importance((3,))
print(res[(3,)]['total importance'])
res = f.quantify_importance((4,))
print(res[(4,)]['total importance'])

print('Pair-wise importance')
print(f.get_most_important_pairwise_marginals((0, 1, 2, 3, 4)))
