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
sys.path.append("../Hardware")
import MazeGenerator3 as Generator
from ltl.optimizees.optimizee import Optimizee
from ltl.logging_tools import configure_loggers
from StateActionMaze import StateAction
from pybrain3_local.rl.environments.mazes import Maze, MDPMazeTask
from pybrain3_local.rl.environments.mazes.tasks import FourByThreeMaze
from pybrain3_local.rl.learners.valuebased import ActionValueTable
from pybrain3_local.rl.agents import LearningAgent
from pybrain3_local.rl.learners import Q, QLambda, SARSA #@UnusedImport
from pybrain3_local.rl.experiments import Experiment

class StateActionMazeOptimizee(Optimizee):


    def __init__(self, traj=None):
        
        super(StateActionMazeOptimizee, self).__init__(traj)
        
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
        wStateAction = (150 + 500 * np.array(traj.individual.weightPrior))
        wActionInhibit = 50.
        delay = 5.
        readtimeout = 0.
        simtime = 30.
        NRep = 10000
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
    
        #Check if inhibition yielded problems / better spiking behavior
    
        if lam != None:
            plast_params = plast_params_lam
        
        envmatrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 0, 0, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1, 0, 1],
                              [1, 0, 0, 1, 0, 1, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
        goal = (7, 7)
        
        fitnesses = []
        
        for i in range(10):
    
            envmatrix, goal = Generator.GenerateMaze(3, 3)
        
            gridRow, gridCol = envmatrix.shape
            nStates = gridRow * gridCol 
            nActions = 4
            termianlState = goal[0] * gridRow + goal[1]
            maze = Maze(envmatrix, goal)
            
            stateActionNW = StateAction(maze, nStates, nActions, wStateAction, scalingFactor, lam=(lam != None))
        
            #Create the network
            stateActionNW.CreateNetwork(wStateAction, wActionInhibit)
            
            #Get the initial state of the maze
            initialState = stateActionNW.getState()
        
            #Run the training procedure
            stateActionNW.Run(initialState, plast_params, NRep, simtime, delay, readtimeout, hdf=False)
            
            fitness = stateActionNW.ComputeFitness()[0]

            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses)

        print(fitnesses.mean())

        return (fitnesses.mean(),)
        
