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
import subprocess
sys.path.append("../LTL")
sys.path.append("..")

from ltl.optimizees.optimizee import Optimizee
from NetworkMDP import DLSNetwork

class DLSMDPOptimizee(Optimizee):
    
    def __init__(self, traj=None):
        
        super(DLSMDPOptimizee, self).__init__(traj)
        
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)

        '''
        self.nStates = 2
        self.nActions = 4
        self.maxIteration = 1999
        self.multipleRuns = 1

        if self.nStates + self.nActions > 32:
            raise Exception("The network is too big for the chip")

        #Create and run the network
        self.network = DLSNetwork(self.nStates, self.nActions, self.multipleRuns)
        '''

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
    
        wlow = np.float64(np.random.random())
        # Define the first solution candidate randomly
        return {'gamma': np.float64(np.random.random()),
               #'lam': np.float64(np.random.random()),
               'weightLower': wlow,
               'weightUpper': np.float64(np.random.uniform(wlow, 1.)),
               'eta': np.float64(np.random.random()),
               'exitatory' : np.float64(np.random.random()),
               'inhibitory' : np.float64(np.random.uniform(0.1, 1.)),
               'rescaleFreq': np.float64(np.random.uniform(0.0129, 1.))}
        
    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        
        return {'gamma': np.float64(np.clip(individual['gamma'], a_min=0., a_max=1.)),
                #'lam': np.float64(np.clip(individual['lam'], a_min=0., a_max=1.)),
                'weightLower': np.float64(np.clip(individual['weightLower'], a_min=0., a_max=1.)),
                'weightUpper': np.float64(np.clip(individual['weightUpper'], a_min=individual['weightLower'], a_max=1.)),
                'eta': np.float64(np.clip(individual['eta'], a_min=0., a_max=1.)),
                'exitatory' : np.float64(np.clip(individual['exitatory'], a_min=0., a_max=1.)),
                'inhibitory' : np.float64(np.clip(individual['inhibitory'], a_min=0.1, a_max=1.)),
                'rescaleFreq': np.float64(np.clip(individual['rescaleFreq'], a_min=0.0129, a_max=1.))}
        
    def get_params(self):
        """
        Get the important parameters of the optimizee. This is used by :class:`ltl.recorder`
        for recording the optimizee parameters.

        :return: a :class:`dict`
        """
        return None
    
    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~pypet.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """

        '''
        gamma = np.array(traj.individual.gamma)
        #lam = np.array(traj.individual.lam)
        lam = 0.5
        eta = np.array(traj.individual.eta)

        weightLower = np.int64(63 * np.array(traj.individual.weightLower))
        weightUpper = np.int64(63 * np.array(traj.individual.weightUpper))
        rescaleFreq = np.int64(2000 * np.array(traj.individual.rescaleFreq))
        use32BitParams = False
        
        exitatory = np.int64(63 * np.array(traj.individual.exitatory))
        inhibitory = np.int64(63 * np.array(traj.individual.inhibitory))
        #exitatory = None
        #inhibitory = None        

        fitnesses = []
        for i in range(20):
            
            self.network.Run(gamma, lam, eta, self.maxIteration, hdf=False, use32BitParams=use32BitParams, weightLower=weightLower, weightUpper=weightUpper, rescaleFreq=rescaleFreq, verbose=False, exitatory=exitatory, inhibitory=inhibitory)

            #Get the fitness of the network
            fitness = self.network.ComputeFitness()[0]
            fitnesses.append(fitness)
        
        fitnesses = np.array(fitnesses)
        return (fitnesses.mean(),)
        '''

        board = '07'
        #lam = np.array(traj.individual.lam)
        lam = 0.5
    
        cnt = 0
        while(cnt < 100):
            try:

                if cnt > 0 and (cnt % 3 == 0):
                    #Wait for a input not loose the already collected data
                    tst = input('Waiting for an input since the execution is not able to proceed!')

                cmd = ['srun', '-p', 'dls', '--gres', board, 'python', './NetworkMDP_OptimizeePart.py', '%.8f' % np.array(traj.individual.eta), '%.8f' % np.array(traj.individual.exitatory), '%.8f' % np.array(traj.individual.gamma), '%.8f' % np.array(traj.individual.inhibitory), '%.8f' % lam, '%.8f' % np.array(traj.individual.rescaleFreq), '%.8f' % np.array(traj.individual.weightLower), '%.8f' % np.array(traj.individual.weightUpper)]

                result = np.float64(subprocess.check_output(cmd).split('\n')[-2])

                return (result,)

            except:
                cnt += 1

        exit()

        
