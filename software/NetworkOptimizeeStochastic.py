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
from experiments.Network_Experiment_Stochastic import Network

class NetworkOptimizeeStochastic(Optimizee):


    def __init__(self, traj=None):
        
        super().__init__(traj)
        
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        
    ####################################################
    #Function for Optimizee                            #
    #Hyperparameters:                                  #
    #J_EI                                              #
    #J_IE                                              #
    #J_in                                              #
    ####################################################
    
    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        return {'J_EI': np.float64(np.random.normal(480., 25)),
                'J_IE': np.float64(np.random.normal(-280., 25)),
                'J_in': np.float64(np.random.normal(20., 5.)),
                'baseline': np.float64(np.random.normal(40., 5.))}
        
    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'J_EI': np.float64(np.clip(individual['J_EI'], a_min=360., a_max=660.)),
                'J_IE': np.float64(np.clip(individual['J_IE'], a_min=-400., a_max=-150.)),
                'J_in': np.float64(np.clip(individual['J_in'], a_min=4., a_max=50.)),
                'baseline': np.float64(np.clip(individual['baseline'], a_min=20., a_max=40.))}
        
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
        configure_loggers(exactly_once=True)  # logger configuration is here since this function is paralellised

        J_EI = np.array(traj.individual.J_EI)
        J_IE = np.array(traj.individual.J_IE)
        J_in = np.array(traj.individual.J_in)
        baseline = np.array(traj.individual.baseline)
        
        #Train the network and test it afterwards
        network = Network()
    
        ####################################################
        # Parameters                                       #
        ####################################################
        NRep = 5000
        presentationTime = 100.
        simtime = presentationTime + 50.
        nInputNeurons = 80
        nStateNeurons = 4
        nInhibitoryNeurons = 50
        
        plast_params_nessler = {
          'w_max':   100., #??      # Max weight of plastic synapses // here, it should be relatively high (why?)
          'eta':     0.01, #??      # learning rate
          'c':  1.0,       # exponent of weight dependency
          'sigma': 10., #ms for STDP time window
          'baseline': baseline}
        
        ####################################################
        
        N_pat = 4 # Number of different patterns
        network.get_spike_patterns(nInputNeurons, N_pat, PLOT=False)
    
        
        plast_params = plast_params_nessler
        network.CreateNetwork(plast_params, nInputNeurons, nStateNeurons, nInhibitoryNeurons, J_in, J_EI, J_IE)
            
        #for given number of input presentations
        #iterator = tqdm.tqdm(range(0, NRep))
        #for z in iterator:
        for z in range(0, NRep):
                        
            network.PresentInput(np.random.randint(0, N_pat), presentationTime, demo=True)
            
            network.Simulate(simtime)
            
        return network.TestNetwork(None, presentationTime, simtime, plot=False, extensive=True, demo=True)
        
        
        
        
