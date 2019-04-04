import pydls as dls
import json
import helpers as hp
import pylogging
import Utils as utils
import numpy as np
import struct
import mdptoolbox
import mdptoolbox.example
import fractional as frac
import os
import sys
import pickle
sys.path.append("../LTL")
sys.path.append("..")

from ltl.optimizees.optimizee import Optimizee
from NetworkMaze import DLSNetwork

if __name__ == '__main__':
    
    if len(sys.argv) != 9:
        raise Exception('Wrong amount parameters')

    eta = np.float64(sys.argv[1])
    exitatory = np.float64(sys.argv[2])
    gamma = np.float64(sys.argv[3])
    inhibitory = np.float64(sys.argv[4])
    lam = np.float64(sys.argv[5])
    rescaleFreq = np.float64(sys.argv[6])
    weightLower = np.float64(sys.argv[7])
    weightUpper = np.float64(sys.argv[8])

    weightLower = np.int64(63 * weightLower)
    weightUpper = np.int64(63 * weightUpper)
    rescaleFreq = np.int64(2000 * rescaleFreq)
    exitatory = np.int64(63 * exitatory)
    inhibitory = np.int64(63 * inhibitory)

    nStates = 9
    nActions = 4
    maxIteration = 3999
    multipleRuns = 1
    use32BitParams = False

    if nStates + nActions > 32:
        raise Exception("The network is too big for the chip")

    network = DLSNetwork(nStates, nActions, multipleRuns)

    fitnesses = []
    for i in range(20):
        
        network.Run(gamma, lam, eta, maxIteration, hdf=False, use32BitParams=use32BitParams, weightLower=weightLower, weightUpper=weightUpper, rescaleFreq=rescaleFreq, verbose=False, exitatory=exitatory, inhibitory=inhibitory)

        #Get the fitness of the network
        fitness = network.ComputeFitness()[0]
        fitnesses.append(fitness)
    
    fitnesses = np.array(fitnesses)
    
    print '%.7f' % fitnesses.mean()
