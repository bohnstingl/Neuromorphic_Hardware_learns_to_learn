'''
Created on Sep 13, 2017

@author: thomas
'''

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import sys
from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.environments.mazes.tasks import FourByThreeMaze
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, QLambda, SARSA #@UnusedImport
from pybrain.rl.experiments import Experiment
from BrainscaleSEnv import BrainscalesEnvironment
from pymarocco import PyMarocco, Defects

class StateAction(object):
    '''
    classdocs
    '''

    def __init__(self, maze, nStates, nActions, weightOffset, scalingFactor, evaluationIter=100, lam=False, weightParams=None):
        '''
        Constructor
        '''
 
        self.nStates = nStates
        self.nActions = nActions
        
        # create task
        self.env = MDPMazeTask(maze)
        
        self.moveCntRst = np.inf
        self.moveCnt = self.moveCntRst

        self.Q = np.zeros((nStates, nActions))
        
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
        
        self.QDistance = []
        self.currentIteration = 0
        self.evaluationIter = evaluationIter
        
        self.discreteWeights = False
        
        if weightParams != None:
            self.discreteWeights = True
            self.minWeight = weightParams['minWeight']
            self.weightStep = (weightParams['maxWeight'] - weightParams['minWeight']) / weightParams['resolution']
            
        #Initialize the hardware
        self.brainscales_environment = BrainscalesEnvironment()
        self.hwEnv = self.brainscales_environment.get_env()

    def getState(self):
        
        return self.env.getObservation()[0]
    
    def CreateNetwork(self, wStateAction, wActionInhibit):   
       
        #Action neurons are 
        self.stateNeurons, self.actionNeurons, self.stateActionConnection = self.createNetwork(wStateAction, wActionInhibit)
           
    def createNetwork(self, wStateAction, wActionInhibit):
        
        self.hwEnv.reset()
        
#         nest.SetDefaults('iaf_psc_exp',
#                  {'C_m': 30.0, #pF 
#                   'tau_m': 30.0, #ms
#                   'I_e': 0.0, #pA
#                   'E_L': -60.0, #mV
#                   'V_th': -55.0, #mV
#                   'tau_syn_ex': 3.0, #ms
#                   'tau_syn_in': 2.0, #ms
#                   'V_reset': -60.0 #mV})
        
        
        '''Think about parameters to use for the conductance based thing'''
        neuronParam = {'cm': 0.03,  # [nF]
                       'tau_m': 30.0,  # [ms]
                       #'v_rest': -20.,  # [mV]
                       'v_thresh': -55.0,  # [mV]
                       'v_reset': -60.,  # [mV]
                       'tau_syn_E': 3.0,  # [ms]
                       'tau_syn_I': 2.0,  # [ms]
                       #'tau_refrac': 0.1,  # [ms]
                       #'e_rev_I' : -100., #[mV]
                       'e_rev_E' : 60., #[mV]
                       }
        
        #Create the state neurons
        #stateNeurons = nest.Create('spike_generator', self.nStates)
        #actionNeurons = nest.Create('iaf_psc_exp', self.nActions)
        
        # Create excitatory and inhibitory populations
        stateNeurons = self.brainscales_environment.Population(self.nStates, self.hwEnv.SpikeSourceArray)
        actionNeurons = self.brainscales_environment.Population(self.nActions, self.hwEnv.IF_cond_exp, neuronParam)
        
        stateNeurons.record()
        actionNeurons.record()

        #stateSpikes = nest.Create('spike_detector', 1)
        #actionSpikes = nest.Create('spike_detector', 1)
        
#         #Connect the actions to the state neurons
#         nest.Connect(stateNeurons, actionNeurons,
#                         {'rule': 'all_to_all'},
#                         {'model': 'static_synapse',
#                         'delay': 1.,
#                         'weight': wStateAction})#{"distribution": "normal", "mu": wStateAction, "sigma": 0.7 * abs(wStateAction)}})
        
        #weight_random = RandomDistribution("normal", parameters=(J_EE, sh_w * J_EE), boundaries=(0., np.inf))
        connector = self.hwEnv.AllToAllConnector(allow_self_connections=False, weights=1)
        projection = self.brainscales_environment.Projection(stateNeurons, actionNeurons, connector, wStateAction,
                                                             target='excitatory', label="exex")
        
        #Connect the spike detectors
        #nest.Connect(stateNeurons, stateSpikes)
        #nest.Connect(actionNeurons, actionSpikes)
        #stateActionConnection = nest.GetConnections(stateNeurons, actionNeurons)
        
        #Perform the BrainscaleS part for mapping
        self.brainscales_environment.run_mapping(20000.)

        # call at least once
        self.brainscales_environment.set_sthal_params()

        self.brainscales_environment.marocco.skip_mapping = True
        self.brainscales_environment.marocco.backend = PyMarocco.Hardware
        self.brainscales_environment.marocco.hicann_configurator = PyMarocco.HICANNv4Configurator

        self.brainscales_environment.set_hardware_weights()
        
        return stateNeurons, actionNeurons, projection
     
    def PresentPattern(self, state, delay):
        
        binStateVector = np.zeros(self.nStates)
        binStateVector[state] = 1
        for ind, value in enumerate(binStateVector):
            #handledStateNeuron = [self.stateNeurons[ind]]  
            handledStateNeuron = self.brainscales_environment.PopulationView(self.stateNeurons, [ind])          
            handledStateNeuron.set('origin', self.hwEnv.get_current_time())
            handledStateNeuron.set('spike_times', [])
            if value == 1:
                handledStateNeuron.set('spike_times', [delay])

    def ReadOut(self, simtime, timeout):
        #Returns real action: 0 ... nActions - 1
        
        print(self.self.actionNeurons.getSpikes())
        exit(0)

        actionEvents = nest.GetStatus(self.actionSpikes,'events')[0]

        #Get the action of the first spike after the given timeout
        actionSpikeTimes = actionEvents['times']
        actionSenders = actionEvents['senders']
        
        #Set the spike times before the timeout to infinity to avoid early spiking
        currentTime = nest.GetKernelStatus('time')
        actionSpikeTimes[actionSpikeTimes < (timeout + currentTime - simtime)] = np.inf
        #index = np.argmin(actionSpikeTimes)
        index = np.random.choice(np.where(actionSpikeTimes == np.array(actionSpikeTimes).min())[0])
        actionTime = actionSpikeTimes[index]
        actionSender = actionSenders[index] - self.actionNeurons[0]
        
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

        return nextState, r
    
    def update_weights_td1(self, plast_params):
        #Weight update according to Q-learning TD(1)
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        gamma = plast_params['gamma']
        
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
        
        dQ = (1 / math.sqrt(self.currentIteration + 2)) * delta
        self.Q[oldState, oldAction] = self.Q[oldState, oldAction] + dQ
        weight = (weight + dQ)  * self.scalingFactor + self.weightOffset
        
        #dQ1 = (1 / math.sqrt(self.currentIteration + 2)) * delta1
        #self.Q1[oldState, oldAction] = self.Q1[oldState, oldAction] + dQ1
        
#         if self.discreteWeights:
#             weight = np.round((weight - self.minWeight) / self.weightStep) * self.weightStep + self.minWeight
        
        if weight < 0.:
            weight = 0.
            
        if weight > w_max:
            weight = w_max
            
        nest.SetStatus([synapse], 'weight', weight)
        
    def update_weights_tdlam(self, plast_params):
        #Weight update according to Q-learning TD(1)
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        gamma = plast_params['gamma']
        lam = plast_params['lambda']
        
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
        self.eligibilityTraces[oldState, oldAction] += 1
        
        #Update all weigupdate_weights_td1hts according to the rule
        for state in range(self.nStates):
            for action in range(self.nActions):
                synapse = self.getSynapse(state, action)
                weight = (nest.GetStatus([synapse], 'weight')[0] - self.weightOffset) / self.scalingFactor
                
                dQ = (1 / math.sqrt(self.currentIteration + 2)) * d * self.eligibilityTraces[state, action]
                self.Q[oldState, oldAction] = self.Q[oldState, oldAction] + dQ
                weight = (weight + dQ)  * self.scalingFactor + self.weightOffset
                
                if self.discreteWeights:
                    weight = np.round((weight - self.minWeight) / self.weightStep) * self.weightStep + self.minWeight                
                
                if weight < 0.:
                    weight = 0.
                    
                if weight > w_max:
                    weight = w_max
                
                nest.SetStatus([synapse], 'weight', weight)
                self.eligibilityTraces[state, action] = gamma * lam * self.eligibilityTraces[state, action]
        
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
        
    def Simulate(self, simtime, state, readtimeout, plast_params, train=True):
    
        #Reset the spike detectors
        self.hwEnv.run(simtime)
        
        #Returns the action Neuron ID. ACtion % nActions yields the
        #chosen action 0 based -> 0, 1, 2, 3
        action = self.ReadOut(simtime, readtimeout)
        
        #Epsilon schedule overtaken from QLearning library
        #mdptoolbox.mdp.QLearning
        #Check if a random step should be taken
        
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
    
            if (self.currentIteration % self.evaluationIter) == 0:
                self.CompareQ()
                
            #Check for cycle or final state
            cycle = self.DetectCycle()
            if cycle:
                self.env.reset()
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
        preferredActions = np.ones((self.nStates, 2)) * -1
        
        for stat in nest.GetStatus(self.stateActionConnection):
            
            if preferredActions[stat['source'] - 1][1] < stat['weight']:
                preferredActions[stat['source'] - 1][0] = stat['target'] - self.actionNeurons[0]
                preferredActions[stat['source'] - 1][1] = stat['weight']
                
        print('Learnt Q-table')
        print(self.Q)
                
        print('weights')
        print(nest.GetStatus(self.stateActionConnection, 'weight'))
        
        print('Learned policy: ')
        policy = '('
        for state in preferredActions:
            policy += str(int(state[0])) + ', '
        print(policy + ')')
        
        
        print('Optiomal policy: \n' + str(self.ql.policy))
        print('===================================')
        print('Q function: \n' + str(self.ql.Q))
        
        print("===================================")
        print("Value iteration")
        vi = mdptoolbox.mdp.ValueIteration(self.P, self.R, 0.95)
        vi.run()
        print("Policy: \n" + str(vi.policy))
        
        #print(self.P)
        #print(self.R)

        plt.show()
        
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
        if envMatrix[x, y] == 1.:
            ax.fill(x, y, "w")
            continue
        
        if state < gridCols or state % gridCols == 0 or state % gridCols == (gridCols - 1) or state >= (gridRows - 1) * gridCols or state == termState:
            continue
        
        action = np.random.choice(np.where(np.max(Q_table[state,:]) == Q_table[state,:])[0])
        nextState = next_state(state, action)
        xn, yn = transformState(nextState)
        ax.arrow(x + .5, y + .5, (xn - x), (yn - y), head_width=.1, color='white')
    
    ax.set_title('Policy')
      
      
if __name__ == '__main__':
    
#     nest.ResetKernel()
#     nest.set_verbosity('M_ERROR')  # Do not print stuff during simulation
#     nest.SetKernelStatus({'print_time': False,
#                           'local_num_threads': 8})  # Number of threads used
    
    ####################################################
    # Parameters                                       #
    ####################################################  
    wStateAction = 300.
    wActionInhibit = 50.
    delay = 5.
    readtimeout = 1.
    simtime = 30.
    NRep = 10
    NRepTest = 1000
    scalingFactor = 100.
    lam = True
    weightParams = {
        'maxWeight' : 400,
        'minWeight' : 50,
        'resolution' : 2**6}
    
    # Plasticity parameters for the case of weight dependency
    plast_params = {
          'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'eta':     .01, #??      # learning rate
          'gamma' : .95}          # discount factor
    
    plast_params_lam = {
          'w_max':   500., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'eta':     .01, #??      # learning rate
          'gamma' : 0.95,# discount factor
          'lambda' : 0.90}

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
 
    gridRow, gridCol = envmatrix.shape
    nStates = 81
    nActions = 4
    goal = (7, 7)
    termianlState = goal[0] * gridRow + goal[1]
    maze = Maze(envmatrix, goal)
        
    stateActionNW = StateAction(maze, nStates, nActions, wStateAction, scalingFactor, lam=lam, weightParams=None)

    #Get the initial state of the maze
    initialState = stateActionNW.getState()

    #Create the network
    stateActionNW.CreateNetwork(wStateAction, wActionInhibit)
    
#     pylab.gray()
#     pylab.ion()

    #for given number of input presentations
    iterator = tqdm.tqdm(range(0, NRep))
    for z in iterator:

        #Present the state pattern as a single spike at the given delay
        stateActionNW.PresentPattern(initialState, delay)
        
        #Simulate the network (do the readout and perform the action)
        nextState, reward = stateActionNW.Simulate(simtime, initialState, readtimeout, plast_params)
        initialState = nextState
        
#         pylab.pcolor(stateActionNW.GetQTable().reshape(81,4).max(1).reshape(9,9))
#         pylab.pcolor(stateActionNW.GetQTable().reshape(25,4).max(1).reshape(5,5))
#         pylab.draw()

    #stateActionNW.PlotEvaluation()
#     plt.figure()
#     plt.pcolor(stateActionNW.GetQTable().reshape(25,4).max(1).reshape(5,5))
#     plt.show()
    
    fig, ax = plt.subplots()
    plot_policy(ax, envmatrix, termianlState, gridRow, gridCol, stateActionNW.GetQTable().reshape(nStates, nActions))
    plt.show()
    
#     #Finished training. perform Testing now with final weights
#     rewardNW = [0]
#     initialState = np.random.randint(0, nStates)
#     state = initialState
#     iterator = tqdm.tqdm(range(0, NRepTest))
#     for z in iterator:
#         #Present the state pattern as a single spike at the given delay
#         stateActionNW.PresentPattern(state, delay)
#         
#         #Simulate the network (do the readout and perform the action)
#         nextState, reward = stateActionNW.Simulate(simtime, state, readtimeout, plast_params, False)
#         state = nextState
#         rewardNW.append(reward + rewardNW[-1])
#     
#     #Test reference implementations now
#     policies = []
#     states = [initialState, initialState]
#     vi = mdptoolbox.mdp.ValueIteration(P, R, 0.95, max_iter=100000)
#     vi.run()
#     policies.append(vi.policy)
#     ql = mdptoolbox.mdp.QLearning(P, R, 0.95)
#     ql.run()
#     policies.append(ql.policy)
#     
#     rewardRef = [[0],[0]]
#     iterator = tqdm.tqdm(range(0, NRepTest))
#     for z in iterator:
#         for ref in range(2):
#             nextState, reward = stateActionNW.PerformAction(policies[ref][states[ref]], states[ref])
#             states[ref] = nextState
#             rewardRef[ref].append(reward + rewardRef[ref][-1])
#         
#     # Create plots with pre-defined labels.
#     # Alternatively, you can pass labels explicitly when calling `legend`.
#     fig, ax = plt.subplots()
#     ax.plot(rewardNW, label='Network')
#     ax.plot(rewardRef[0], label='Value Iteration')
#     ax.plot(rewardRef[1], label='Q Learning')
#     
#     # Now add the legend with some customizations.
#     legend = ax.legend(loc='upper center', shadow=True)
#     
#     # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
#     frame = legend.get_frame()
#     frame.set_facecolor('0.90')
#     
#     # Set the fontsize
#     for label in legend.get_texts():
#         label.set_fontsize('large')
#     
#     for label in legend.get_lines():
#         label.set_linewidth(1.5)  # the legend line width
#     plt.show()
    