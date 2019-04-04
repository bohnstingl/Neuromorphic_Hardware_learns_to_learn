'''
Created on Sep 7, 2017

@author: thomas
'''
import nest
import numpy as np
import nest.raster_plot
import nest.voltage_trace
import numpy as np
import os
import sys
sys.path.append("../LTL")
sys.path.append("..")
from ltl.optimizees.optimizee import Optimizee
from ltl.logging_tools import configure_loggers
from StateAction import StateAction

class StateActionOptimizee(Optimizee):


    def __init__(self, traj=None):
        
        super(StateActionOptimizee, self).__init__(traj)
        
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
    
    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        return {'gamma': np.float64(np.random.random()),
                'lam': np.float64(np.random.normal()),
                'eta': np.float64(np.random.random()),
                'scalingFct': np.float64(np.random.random()),
                'weightPrior': np.float64(np.random.random()),
                'decay': np.float64(np.random.random())}
        
    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'gamma': np.float64(np.clip(individual['gamma'], a_min=0., a_max=1.)),
                'lam': np.float64(np.clip(individual['lam'], a_min=0., a_max=1.)),
                'eta': np.float64(np.clip(individual['eta'], a_min=0., a_max=1.)),
                'scalingFct': np.float64(np.clip(individual['scalingFct'], a_min=0., a_max=1.)),
                'weightPrior': np.float64(np.clip(individual['weightPrior'], a_min=0., a_max=1.)),
                'decay': np.float64(np.clip(individual['decay'], a_min=0., a_max=1.))}
        
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

        lam = None
        gamma = np.array(traj.individual.gamma)
        lam = np.array(traj.individual.lam)
        eta = np.array(traj.individual.eta)
        decay = np.array(traj.individual.decay)
    
        ####################################################
        # Parameters                                       #
        ####################################################
        nStates = 2
        nActions = 4
        nInhibit = 10
        wStateAction = (150 + 350 * np.array(traj.individual.weightPrior))
        wActionExhibition = 300.
        wActionInhibit = -400.
        delay = 5.
        readtimeout = 0.
        simtime = 50.
        NRep = 2000
        scalingFactor = np.float64(200 + 3800 * np.array(traj.individual.scalingFct))
        
        # Plasticity parameters for the case of weight dependency
        plast_params = {
              'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
              'gamma' : gamma,
              'eta' : eta,
              'decay' : decay}          # discount factor
        
        plast_params_lam = {
              'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
              'gamma' : gamma,# discount factor
              'lambda' : lam,
              'eta' : eta,
              'decay' : decay}
    
        fitnesses = []
        for i in range(20):

            if lam != None:
                plast_params = plast_params_lam 
                
            stateActionNW = StateAction(nStates, nActions, wStateAction, scalingFactor, lam=(lam != None))
        
            #Create the network
            stateActionNW.CreateNetwork(nInhibit, wStateAction, wActionExhibition, wActionInhibit)
            
            #Get random initial state
            initialState = np.random.randint(0, nStates)
        
            #Run the training procedure
            stateActionNW.Run(initialState, plast_params, NRep, simtime, delay, readtimeout, hdf=False)
            
            fitness = stateActionNW.ComputeFitness()[0]

            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses)
        return (fitnesses.mean(),)
        
