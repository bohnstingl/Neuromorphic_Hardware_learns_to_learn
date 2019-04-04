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
from experiments.Network import Network

class NetworkOptimizee(Optimizee):


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
        return {'J_EI': np.float64(np.random.normal(30., 6.)),
                'J_IE': np.float64(np.random.normal(-19., 6.)),
                'J_in': np.float64(np.random.normal(20., 5.))}
        
    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'J_EI': np.float64(np.clip(individual['J_EI'], a_min=0., a_max=150.)),
                'J_IE': np.float64(np.clip(individual['J_IE'], a_min=-100., a_max=0.)),
                'J_in': np.float64(np.clip(individual['J_in'], a_min=0., a_max=150.))}
        
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
        
        #Train the network and test it afterwards
        network = Network()
    
        ####################################################
        # Parameters                                       #
        ####################################################
        J_noise = 28.1
        observationValues = [224, 56, 14, 195]
        NRep = 300
        presentationTime = 100.
        simtime = presentationTime + 50.
        nInputNeurons = 8
        nStateNeurons = 4
        nInhibitoryNeurons = 20
        WDEP = False
        
        # Plasticity parameters for the case of no weight dependency
        plast_params_nowdep = {
              'w_max':   20., #??      # Max weight of plastic synapses // on the order or tens
              'eta':     0.1, #??      # learning rate
              'A_neg':   .2, #??      # LTD factor
              'A_pos':   1.15, #??      # LTP factor
              'A_decay': 0.}       # weight decay factor [Not used]
        
        # Plasticity parameters for the case of weight dependency
        plast_params_wdep = {
              'w_max':   200., #??      # Max weight of plastic synapses // here, it should be relatively high (why?)
              'eta':     0.1, #??      # learning rate
              'alpha':  0.5,       # exponent of weight dependency
              'A_decay': 0.}       # weight decay factor
        
        if WDEP:
            plast_params = plast_params_wdep
        else:
            plast_params = plast_params_nowdep  
        
        ####################################################
        
        #Create the network
        network.CreateNetwork(nInputNeurons, nStateNeurons, 5, nInhibitoryNeurons, J_in, J_noise, J_EI, J_IE)
            
        #for given number of input presentations
        #iterator = tqdm.tqdm(range(0, NRep))
        #for z in iterator:
        for z in range(0, NRep):
                        
            #draw input
            rnd = np.random.randint(0, len(observationValues))
            pattern = observationValues[rnd]
            
            #present input
            network.PresentInput(np.array([pattern], dtype=np.uint8), presentationTime)
            
            network.Simulate(simtime, WDEP, plast_params)
    
        #network.PlotEvaluation()
            
        return network.TestNetwork(observationValues, presentationTime, simtime, plot=False, extensive=True)
        
        
        
        
