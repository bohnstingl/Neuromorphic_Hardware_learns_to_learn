'''
Created on Sep 13, 2017

@author: thomas
'''

import nest
import nest.raster_plot
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox
import mdptoolbox_local.example
import tqdm
import pylab
import sys, time
from pybrain3_local.rl.environments.mazes import Maze, MDPMazeTask
from pybrain3_local.rl.environments.mazes.tasks import FourByThreeMaze
from pybrain3_local.rl.learners.valuebased import ActionValueTable
from pybrain3_local.rl.agents import LearningAgent
from pybrain3_local.rl.learners import Q, QLambda, SARSA #@UnusedImport
from pybrain3_local.rl.experiments import Experiment
import datetime as dt
import math
import sys
sys.path.append("..")
sys.path.append("../Hardware")
import EvaluateMaze
import MazeGenerator3 as Generator

class StateAction(object):
    '''
    classdocs
    '''

    def __init__(self, maze, nStates, nActions, weightOffset, scalingFactor, evaluationIter=100, lam=False, weightParams=None):
        '''
        Constructor
        '''
        
        nest.ResetKernel()
        nest.hl_api.set_verbosity('M_ERROR')  # Do not print stuff during simulation
        nest.SetKernelStatus({'print_time': False,
                              'local_num_threads': 1})  # Number of threads used 

        self.nStates = nStates
        self.nActions = nActions
        
        # create task
        self.env = MDPMazeTask(maze)
        
        self.moveCntRst = np.inf
        self.moveCnt = self.moveCntRst

        self.Q = np.zeros((nStates, nActions))
        self.wins = 0
        self.eta = None
        
        self.weightOffset = weightOffset
        self.scalingFactor = scalingFactor
        
        if lam != 0.:
            self.eligibilityTraces = np.zeros((self.nStates, self.nActions))
            self.lam = True
        else:
            self.lam = False
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.eventsTimes = []
        self.eventAddresses = []
        
        self.QDistance = []
        self.currentIteration = 0
        self.evaluationIter = evaluationIter
        
        self.discreteWeights = False
        
        if weightParams != None:
            self.discreteWeights = True
            self.minWeight = weightParams['minWeight']
            self.weightStep = (weightParams['maxWeight'] - weightParams['minWeight']) / weightParams['resolution']

    def getState(self):
        
        return self.env.getObservation()[0]
    
    def CreateNetwork(self, wStateAction, wActionInhibit):   
       
        #Action neurons are 
        self.stateNeurons, self.actionNeurons, self.stateSpikes, self.actionSpikes, self.stateActionConnection = self.createNetwork(wStateAction, wActionInhibit)
           
    def createNetwork(self, wStateAction, wActionInhibit):
        
        nest.SetDefaults('iaf_psc_exp',
                 {'C_m': 30.0,  
                  'tau_m': 30.0,
                  'I_e': 0.0,
                  'E_L': -60.0,
                  'V_th': -55.0,
                  'tau_syn_ex': 3.0,
                  'tau_syn_in': 2.0,
                  'V_reset': -60.0})
        
        #Create the state neurons
        stateNeurons = nest.Create('spike_generator', self.nStates)
        actionNeurons = nest.Create('iaf_psc_exp', self.nActions)

        stateSpikes = nest.Create('spike_detector', 1)
        actionSpikes = nest.Create('spike_detector', 1)
        
        #Connect the actions to the state neurons
        nest.Connect(stateNeurons, actionNeurons,
                        {'rule': 'all_to_all'},
                        {'model': 'static_synapse',
                        'delay': 1.,
                        'weight': wStateAction})#{"distribution": "normal", "mu": wStateAction, "sigma": 0.7 * abs(wStateAction)}})
        
        #Create inhibitory connections between the action neurons
#         nest.Connect(stateNeurons, actionNeurons,
#                         {'rule': 'all_to_all'},
#                         {'model': 'static_synapse',
#                         'delay': 1.,
#                         'weight': {"distribution": "normal", "mu": -wActionInhibit, "sigma": 0.7 * abs(wActionInhibit)}})
        
        #Connect the spike detectors
        nest.Connect(stateNeurons, stateSpikes)
        nest.Connect(actionNeurons, stateSpikes)
        nest.Connect(actionNeurons, actionSpikes)
        stateActionConnection = nest.GetConnections(stateNeurons, actionNeurons)
        
        return stateNeurons, actionNeurons, stateSpikes, actionSpikes, stateActionConnection
     
    def PresentPattern(self, state, delay):
        
        binStateVector = np.zeros(self.nStates)
        binStateVector[state] = 1
        for ind, value in enumerate(binStateVector):
            handledStateNeuron = [self.stateNeurons[ind]]
            nest.SetStatus(handledStateNeuron,{'origin': nest.GetKernelStatus('time')})
            nest.SetStatus(handledStateNeuron, {'spike_times': []})
            if value == 1:
                nest.SetStatus(handledStateNeuron, {'spike_times': [delay]})

    def ReadOut(self, simtime, timeout):
        #Returns real action: 0 ... nActions - 1

        actionEvents = nest.GetStatus(self.actionSpikes,'events')[0]
        allEvents = nest.GetStatus(self.stateSpikes,'events')[0]
        self.eventsTimes.extend(allEvents['times'])
        self.eventAddresses.extend(allEvents['senders'])

        #Get the action of the first spike after the given timeout
        actionSpikeTimes = actionEvents['times']
        actionSenders = actionEvents['senders']
        
        #Set the spike times before the timeout to infinity to avoid early spiking
        currentTime = nest.GetKernelStatus('time')
        actionSpikeTimes[actionSpikeTimes < (timeout + currentTime - simtime)] = np.inf
        #index = np.argmin(actionSpikeTimes)
        
        '''Check if some action has spiked. Otherwise, pick a random action'''
        if len(actionSpikeTimes) > 0:
            index = np.random.choice(np.where(actionSpikeTimes == np.array(actionSpikeTimes).min())[0])
            actionTime = actionSpikeTimes[index]
            actionSender = actionSenders[index] - self.actionNeurons[0]
        else:
            actionSender = np.random.randint(0, self.nActions)
        
        return actionSender

    def GetValueFromProbVect(self, probVect):
        
        rndValue = np.random.rand()
        
        index = 0
        probSum = 0.
        for value in probVect:
            probSum += value
            if rndValue <= probSum:
                break
            else:
                index += 1

        return index

    def PerformAction(self, action, state):

        #Perform the action with the environment
        self.env.performAction([action])
        nextState = self.getState()
        r = self.env.getReward()
        
        if r == 1.:
            self.wins += 1

        return nextState, r
    
    def update_weights_td1(self, plast_params):
        #Weight update according to Q-learning TD(1)
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        gamma = plast_params['gamma']
        
        if self.eta == None:
            if 'eta' in plast_params:
                self.eta = plast_params['eta']
        
        oldState = self.states[self.currentIteration - 1]
        oldAction = self.actions[self.currentIteration - 1]
        oldReward = self.rewards[self.currentIteration - 1]
        
        newState = self.states[self.currentIteration]
        newAction = self.actions[self.currentIteration]
        
        #Get synapse for old state/action pair
        synapse = None
        newSynapse = None
        for syn in self.stateActionConnection:
            if (syn[0] - 1) == oldState and ((syn[1] - self.actionNeurons[0]) % self.nActions) == oldAction:
                synapse = syn
                break

        weight = (nest.GetStatus([synapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
        
        #Get the new synapse and its weights
        for syn in self.stateActionConnection:
            if (syn[0] - 1) == newState and ((syn[1] - self.actionNeurons[0]) % self.nActions) == newAction:
                newSynapse = syn
                break
        newWeight = (nest.GetStatus([newSynapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
        
        #Perform the Q-learning TD(1) update step
        #Learning rate schedule overtaken from QLearning library
        #mdptoolbox.mdp.QLearning
       
        delta = oldReward + gamma * newWeight - weight
        #delta1 = oldReward + gamma * self.Q[newState, :].max() - self.Q[oldState, oldAction]
        
        if self.eta == None:
            dQ = (1 / math.sqrt(self.currentIteration + 2)) * delta
        else:
            dQ = self.eta * delta
        
        self.Q[oldState, oldAction] = self.Q[oldState, oldAction] + dQ
        weight = (weight + dQ)  * self.scalingFactor + self.weightOffset
        
        if weight < 0.:
            weight = 0.
            
        if weight > w_max:
            weight = w_max
            
        nest.SetStatus([synapse], 'weight', weight)

        #Check for eta decay
        if 'decay' in plast_params:
            self.eta = self.eta * plast_params['decay']
        
    def update_weights_tdlam(self, plast_params):
        #Weight update according to Q-learning TD(1)
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        gamma = plast_params['gamma']
        lam = plast_params['lambda']
        
        if self.eta == None:
            if 'eta' in plast_params:
                self.eta = plast_params['eta']
        
        oldState = self.states[self.currentIteration - 1]
        oldAction = self.actions[self.currentIteration - 1]
        oldReward = self.rewards[self.currentIteration - 1]
        
        newState = self.states[self.currentIteration]
        newAction = self.actions[self.currentIteration]
        
        #Get synapse for old state/action pai
        synapse = self.getSynapse(oldState, oldAction)
        weight = (nest.GetStatus([synapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
        
        #Get the new synapse and its weights
        newSynapse = self.getSynapse(newState, newAction)
        newWeight = (nest.GetStatus([newSynapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
        
        d = oldReward + gamma * newWeight - weight
        
        #Update all weigupdate_weights_td1hts according to the rule
        for state in range(self.nStates):
            for action in range(self.nActions):
                
                #Handle the eligibility traces
                if state == oldState and action == oldAction:
                    self.eligibilityTraces[state, action]  = gamma * lam * self.eligibilityTraces[state, action] + 1
                else:
                    self.eligibilityTraces[state, action]  = gamma * lam * self.eligibilityTraces[state, action]
                
                synapse = self.getSynapse(state, action)
                weight = (nest.GetStatus([synapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
                
                if self.eta == None:
                    dQ = (1 / math.sqrt(self.currentIteration + 2)) * d * self.eligibilityTraces[state, action]
                else:
                    dQ = self.eta * d * self.eligibilityTraces[state, action]
                
                self.Q[oldState, oldAction] = self.Q[oldState, oldAction] + dQ
                weight = (weight + dQ)  * self.scalingFactor + self.weightOffset
                
                if self.discreteWeights:
                    weight = np.round((weight - self.minWeight) / self.weightStep) * self.weightStep + self.minWeight                
                
                if weight < 0.:
                    weight = 0.
                    
                if weight > w_max:
                    weight = w_max
                
                nest.SetStatus([synapse], 'weight', weight)

        #Check for eta decay
        if 'decay' in plast_params:
            self.eta = self.eta * plast_params['decay']
                
    def getSynapse(self, state, action):
        for syn in self.stateActionConnection:
            if (syn[0] - 1) == state and ((syn[1] - self.actionNeurons[0]) % self.nActions) == action:
                return syn
        
    def CompareQ(self):
        
        self.QDistance.append(np.sum((self.ql.Q - self.Q) * (self.ql.Q - self.Q)))
        
    def DetectCycle(self):
        
        #Check if a cycle occured. Return False if no cycle occured and True if a cycle occured
        
        #Currently only a counter for maximum number of trials, before new round starts
        self.moveCnt -= 1
        if self.moveCnt == 0:
            self.moveCnt = self.moveCntRst
            return True
        else:
            return False
        
    def GetQTable(self):
        
        return self.Q
    
    def ComputePolicy(self):
        
        preferredActions = np.ones((self.nStates, 2)) * -1
        
        for stat in nest.GetStatus(self.stateActionConnection):
            
            if preferredActions[stat['source'] - 1][1] < stat['weight']:
                preferredActions[stat['source'] - 1][0] = stat['target'] - self.actionNeurons[0]
                preferredActions[stat['source'] - 1][1] = stat['weight']
                
        policy = np.zeros(self.nStates, dtype=np.int)
        for i, state in enumerate(preferredActions):
            policy[i] =  int(state[0])
            
        return policy
    
    def Run(self, initialState, plast_params, NRep, simtime, delay, readtimeout, hdf=True):
        '''This funciton represents the training loop for the network'''

        self.hdf = hdf
        
        '''Store some information into the hdf file'''
        if self.hdf:
            import h5py

            folder = "HDFs/Maze/"

            rn = np.random.randint(2**32)          
        
            path = folder + "/Maze_" + dt.datetime.now().strftime('%Y_%m_%d_%H-%M-%S') + str(rn) + ".hdf5"
            self.hdfFile = h5py.File(path, "w")
            self.hdfFile.create_dataset('date', data=(dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S').encode(), ))            
            self.hdfFile.create_dataset('gamma', data=(plast_params['gamma'], ))
            if 'lambda' in plast_params:
                self.hdfFile.create_dataset('lam', data=(plast_params['lambda'], ))
            if 'eta' in plast_params:
                self.hdfFile.create_dataset('eta', data=(plast_params['eta'], ))
                
            self.hdfFile.create_dataset('maze', data=self.env.env.mazeTable)
            self.hdfFile.create_dataset('goalX', data=(self.env.env.goal[0], ))
            self.hdfFile.create_dataset('goalY', data=(self.env.env.goal[1], ))
            self.hdfFile.create_dataset('trialsPerIteration', data=(NRep, ))
            self.hdfFile.create_dataset('nActions', data=(self.nActions, ))
            self.hdfFile.create_dataset('nStates', data=(self.nStates, ))
            self.hdfFile.create_dataset('platform', data=(b"SW", ))
            
            '''In software simulations only one run will be performed'''
            self.hdfFile.create_dataset('multipleRuns', data=(1, ))
            self.hdfFile.create_dataset('iterationsOnChip', data=(1, ))
            
        if hdf:
            #Create a group for each new run
            group = self.hdfFile.create_group('Run_0')
            
        #Measure the total execution time
        startTime = time.time()
        
        #for given number of input presentations
        iterator = tqdm.tqdm(range(0, NRep))
        for z in iterator:
    
            #Present the state pattern as a single spike at the given delay
            self.PresentPattern(initialState, delay)
            
            #Simulate the network (do the readout and perform the action)
            nextState, reward = self.Simulate(simtime, initialState, readtimeout, plast_params)
            initialState = nextState
            
        '''Sort the lists'''
        indices = np.argsort(self.eventsTimes)
        addresses = np.array(np.array(self.eventAddresses)[indices])
        times = np.sort(self.eventsTimes)
        
        if hdf:
            group.create_dataset('spikeTimes0', data=times)
            group.create_dataset('spikeAddresses0', data=addresses)
            group.create_dataset('states0', data=self.states)
            group.create_dataset('actions0', data=self.actions)
            group.create_dataset('policy0', data=self.ComputePolicy())
            group.create_dataset('weights0', data=nest.GetStatus(self.stateActionConnection, 'weight'))
            group.create_dataset('Qtable0', data=self.Q)
            group.create_dataset('Wins0', data=(self.wins,))
                
        
        if hdf:
            self.hdfFile.create_dataset('simulationTime', data=(time.time() - startTime, ))
            print('Closing file')
            self.hdfFile.close()
        
    def Simulate(self, simtime, state, readtimeout, plast_params, train=True, epsilon=True):
    
        #Reset the spike detectors
        nest.ResetNetwork()
        nest.Simulate(simtime)
        
        #Returns the action Neuron ID. ACtion % nActions yields the
        #chosen action 0 based -> 0, 1, 2, 3
        action = self.ReadOut(simtime, readtimeout)
        
        #Epsilon schedule overtaken from QLearning library
        #mdptoolbox.mdp.QLearning
        #Check if a random step should be taken
        
        if epsilon:
            rnd = np.random.rand()
            if rnd > (1 - (1 / math.log(self.currentIteration + 2))):#self.eps:
                action = np.random.randint(0, self.nActions)
        
        #Add the action/state to the lists
        self.states.append(state)
        self.actions.append(action)
        
        #State is also represented as the state neuron ID
        nextState, reward = self.PerformAction(action, state)
            
        self.rewards.append(reward)

        #Skip the q-table update in the first iteration
        if train:
            if self.currentIteration > 0:
                #Update synapses before the reward
                
                if self.lam:
                    self.update_weights_tdlam(plast_params)
                else:
                    self.update_weights_td1(plast_params)
                
            self.currentIteration += 1
    
            #if (self.currentIteration % self.evaluationIter) == 0:
            #    self.CompareQ()
                
            #Check for cycle or final state
            cycle = self.DetectCycle()
            if cycle:
                self.env.env.reset()
                return self.env.getObservation(), 0
            else:
                return nextState, reward
        else:
            return nextState, reward

    def PlotEvaluation(self):
        
        plt.figure()
        plt.plot(self.rewards)
        plt.ylabel('Reward signal')
        plt.xlabel('Pattern presentations')
        #plt.savefig('rewards.png')
        
        plt.figure()
        plt.plot(self.actions)
        plt.ylabel('Performed actions')
        plt.xlabel('Pattern presentations')
        #plt.savefig('actions.png')

        plt.figure()
        plt.plot(self.QDistance)
        plt.ylabel('Squared distance to optimal Q values')
        plt.xlabel('Pattern presentations')
        
        #Print Q table
#         preferredActions = np.ones((self.nStates, 2)) * -1
#         
#         for stat in nest.GetStatus(self.stateActionConnection):
#             
#             if preferredActions[stat['source'] - 1][1] < stat['weight']:
#                 preferredActions[stat['source'] - 1][0] = stat['target'] - self.actionNeurons[0]
#                 preferredActions[stat['source'] - 1][1] = stat['weight']
                
        print('Learnt Q-table')
        print(self.Q)
                
        print('weights')
        print(nest.GetStatus(self.stateActionConnection, 'weight'))
        
        print('Learned policy: ')
#         policy = '('
#         for state in preferredActions:
#             policy += str(int(state[0])) + ', '
        policy = self.ComputePolicy()
        print(policy)
        
        
        #print('Optiomal policy: \n' + str(self.ql.policy))
        #print('===================================')
        #print('Q function: \n' + str(self.ql.Q))
        
        print("===================================")
        print("Value iteration")
        vi = mdptoolbox_local.mdp.ValueIteration(self.P, self.R, 0.95)
        vi.run()
        print("Policy: \n" + str(vi.policy))
        
        #print(self.P)
        #print(self.R)

        plt.show()
        
    def ComputeFitness(self):
        
        policy = self.ComputePolicy()
        #difference = EvaluateMaze.ComparePolicy(self.env.env.mazeTable, self.env.env.goal, self.nStates, self.nActions, platform=0, toComparePolicy = policy, plot=False, fixedPolicy=[])#, fixedPolicy=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        
        #s = (difference, )
        #print(s)
        #return s

        print(self.wins)
        #Optimize on amount of wins
        return (-self.wins,)
        
def plot_policy(ax, envMatrix, termState, gridRows, gridCols, Q_table):

    #Maze is flipped due to environment N down S up
    def next_state(state, a):
        if a == 0:
            return state + gridCols
        elif a == 1:
            return state + 1
        elif a == 2:
            return state - gridCols 
        elif a == 3:
            return state - 1
        else:
            raise ValueError('Unkown action {}'.format(a))
        
    def transformState(state):
        
        x = int(state % gridCols)
        y = int(state / gridCols)
        
        return x, y

    nStates, nActions = Q_table.shape
    ax.pcolor(envMatrix, cmap=plt.cm.jet)
    
    x, y = transformState(termState)
    ax.annotate('G',xy=(x + .2, y + 0.2),color='black',size=25)
    
    for state in range(nStates):
        
        x, y = transformState(state)
        
        #Fill The wall elements
        if envMatrix[y, x] == 1.:
            ax.fill(x, y, "w")
            continue
        
        if state < gridCols or state % gridCols == 0 or state % gridCols == (gridCols - 1) or state >= (gridRows - 1) * gridCols or state == termState:
            continue
        
        action = np.random.choice(np.where(np.max(Q_table[state,:]) == Q_table[state,:])[0])
        nextState = next_state(state, action)
        xn, yn = transformState(nextState)
        ax.arrow(x + .5, y + .5, (xn - x), (yn - y), head_width=.1, color='white')
        #ax.annotate(str(state), xy=(x + .2, y + .2), color='black',size=25)
    
    ax.set_title('Policy')
      
      
if __name__ == '__main__':
    
    ####################################################
    # Parameters                                       #
    ####################################################  
    wStateAction = 475
    wActionInhibit = 50.
    delay = 5.
    readtimeout = 0.
    simtime = 30.
    NRep = 100000
    NRepTest = 10000
    scalingFactor = 972
    lam = True
    train = True
    epsilon = False
    weightParams = {
        'maxWeight' : 400,
        'minWeight' : 50,
        'resolution' : 2**6}
    
    # Plasticity parameters for the case of weight dependency
    plast_params = {
          'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'gamma' : 0.94}             # 0.03705, #learning rate 
    
    plast_params_lam = {
          'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'eta' :  np.random.rand(),
          'gamma' : np.random.rand(),# discount factor
          'lambda' : np.random.rand(),
          'decay' : np.random.rand()}

    #Check if inhibition yielded problems / better spiking behavior

    if lam:
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

    envmatrix, goal = Generator.GenerateMaze(3, 3)
    
#     envmatrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                           [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                           [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                           [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#  
#     gridRow, gridCol = envmatrix.shape
#     nStates = 400
#     nActions = 4
#     goal = (19, 1)
#     termianlState = goal[0] * gridRow + goal[1]

    if len(sys.argv) == 2:
        import h5py
        #A file is given. Perform the network with the same R and P
        hdf5File = h5py.File(sys.argv[1],'r')
        envmatrix = np.array(hdf5File['maze'][:])
        #maze[:, :] = maze[::-1, ::-1]
        goalStateX = hdf5File['goalX'][0]
        goalStateY = hdf5File['goalY'][0]
        
        #Check if the file is a SW or a HW simulation
        if hdf5File['platform'][0] != 'SW':
            envmatrix = EvaluateMaze.ExpandMaze(envmatrix)
            goalStateX += 1
            goalStateY += 1
        
        goal = (goalStateX, goalStateY)
            

    gridRow, gridCol = envmatrix.shape
    nStates = gridRow * gridCol 
    nActions = 4
    termianlState = goal[0] * gridRow + goal[1]
    maze = Maze(envmatrix, goal)
        
    stateActionNW = StateAction(maze, nStates, nActions, wStateAction, scalingFactor, lam=lam, weightParams=None)

    #Get the initial state of the maze
    initialState = stateActionNW.getState()

    #Create the network
    stateActionNW.CreateNetwork(wStateAction, wActionInhibit)
    
    #Perform the training
    stateActionNW.Run(initialState, plast_params, NRep, simtime, delay, readtimeout)
    
    stateActionNW.ComputeFitness()
    
    #fig, ax = plt.subplots()
    #plot_policy(ax, envmatrix, termianlState, gridRow, gridCol, stateActionNW.GetQTable().reshape(nStates, nActions))
    #plt.show()
