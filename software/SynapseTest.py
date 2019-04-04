'''
Created on Sep 13, 2017

@author: thomas
'''

import nest
import nest.raster_plot
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt
import pylab
import mdptoolbox
import mdptoolbox_local.example
import tqdm
import math
import time

class StateAction(object):
    '''
    classdocs
    '''

    
    def __init__(self, nStates, nActions, weightOffset, scalingFactor, inhibit, evaluationIter=100, lam=False, weightParams=None):
        '''
        Constructor
        '''
        self.nActions = nActions
        self.nStates = nStates
        self.inhibit = inhibit
        
        self.weightOffset = weightOffset
        self.scalingFactor = scalingFactor
        
        if lam != 0.:
            self.eligibilityTraces = np.zeros((nStates, nActions))
            self.lam = True
        else:
            self.lam = False
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.evaluationIter = evaluationIter
        
        self.discreteWeights = False
        
        if weightParams != None:
            self.discreteWeights = True
            self.minWeight = weightParams['minWeight']
            self.weightStep = (weightParams['maxWeight'] - weightParams['minWeight']) / weightParams['resolution']

    def CreateNetwork(self, nInhibit, wStateAction, wActionExhibition, wActionInhibit):
       
        #Action neurons are 
        self.stateNeurons, self.actionNeurons, self.stateSpikes, self.actionSpikes, self.stateActionConnection, self.voltmeter = self.createNetwork(nInhibit, wStateAction, wActionExhibition, wActionInhibit)
           
    def createNetwork(self, nInhibit, wStateAction, wActionExhibition, wActionInhibit):
        
        nest.SetDefaults('iaf_psc_exp',
                 {'C_m': 30.0,
                  'tau_m': 20.0,
                  'I_e': 0.0,
                  'E_L': -70.0,#-60.0,
                  'V_th': -10.0,
                  'tau_syn_ex': 3.0,
                  'tau_syn_in': 2.0,
                  'V_reset': -70.0})
        
        #Warning, consider a different model where the state neuron are also iaf and not spike generators
        
        #Create the state neurons
        stateNeurons = nest.Create('spike_generator', self.nStates)
        actionNeurons = nest.Create('iaf_psc_exp', self.nActions)
        voltmeter = nest.Create('voltmeter', 1)
        #stateNeuronsInput = nest.Create('spike_generator', self.nStates)
        inhibitoryNeurons = nest.Create('iaf_psc_exp', self.nActions)

        stateSpikes = nest.Create('spike_detector', 1)
        actionSpikes = nest.Create('spike_detector', 1)
        inhibitionSpikes = nest.Create('spike_detector', 1)
        
        #Connect the actions to the state neurons
#         nest.Connect(stateNeuronsInput, stateNeurons,
#                         {'rule': 'one_to_one'},
#                         {'model': 'static_synapse',
#                         'delay': 1.,
#                         'weight': 200.})#{"distribution": "normal", "mu": wStateAction, "sigma": 0.7 * abs(wStateAction)}})
        
        #Connect the actions to the state neurons
        nest.Connect(stateNeurons, actionNeurons,
                        {'rule': 'all_to_all'},
                        {'model': 'static_synapse',
                        'delay': 1.,
                        'weight': {"distribution": "normal", "mu": wStateAction, "sigma": 0.1 * abs(wStateAction)}})
        
        if self.inhibit:
            #Create inhibitory connections between the action neurons
            nest.Connect(actionNeurons, inhibitoryNeurons,
                            {'rule': 'all_to_all'},
                            {'model': 'static_synapse',
                            'delay': .1,
                            'weight': wActionExhibition})
                         
            nest.Connect(inhibitoryNeurons, actionNeurons,
                            {'rule': 'all_to_all'},
                            {'model': 'static_synapse',
                            'delay': .1,
                            'weight': wActionInhibit})
        
        #Connect the spike detectors
        nest.Connect(stateNeurons, stateSpikes)
        nest.Connect(actionNeurons, actionSpikes)
        nest.Connect(voltmeter, actionNeurons)
        stateActionConnection = nest.GetConnections(stateNeurons, actionNeurons)
                
        return stateNeurons, actionNeurons, stateSpikes, actionSpikes, stateActionConnection, voltmeter
     
    def PresentPattern(self, state, delay):
        
        binStateVector = np.zeros(self.nStates)
        binStateVector[state] = 1
        for ind, value in enumerate(binStateVector):
            handledStateNeuron = [self.stateNeurons[ind]]
            nest.SetStatus(handledStateNeuron,{'origin': nest.GetKernelStatus('time')})
            nest.SetStatus(handledStateNeuron, {'spike_times': []})
            if value == 1:
                nest.SetStatus(handledStateNeuron, {'spike_times': [0.1]})

    def ReadOut(self, simtime, timeout):
        #Returns real action: 0 ... nActions - 1

        actionEvents = nest.GetStatus(self.actionSpikes,'events')[0]
        stateEvents = nest.GetStatus(self.stateSpikes,'events')[0]
        
        nest.voltage_trace.from_device(self.voltmeter)
        plt.show()
        
        print(actionEvents)
        print(len(list(set(actionEvents['senders']))))
        
    def Simulate(self, simtime, state, readtimeout, plast_params, train=True):
    
        #Reset the spike detectors
        start_time = time.time()
        nest.ResetNetwork()
        nest.Simulate(simtime)
        #print("--- %s seconds for network ---" % (time.time() - start_time))
        
        #Returns the action Neuron ID. ACtion % nActions yields the
        #chosen action 0 based -> 0, 1, 2, 3
        
        start_time = time.time()
        action = self.ReadOut(simtime, readtimeout)
      
      
if __name__ == '__main__':
    
    nest.ResetKernel()
    nest.hl_api.set_verbosity('M_ERROR')  # Do not print stuff during simulation
    nest.SetKernelStatus({'print_time': False,
                          'local_num_threads': 8})  # Number of threads used
    
    ####################################################
    # Parameters                                       #
    ####################################################
    nStates = 4
    nActions = 10
    nInhibit = 10
    wStateAction = 3000.
    wActionExhibition = 500.
    wActionInhibit = -500.
    delay = 10.
    readtimeout = 0.
    simtime = 100.
    NRep = 10000
    NRepTest = 10000
    scalingFactor = 50
    lam = True
    inhibit = False
    weightParams = {
        'maxWeight' : 400,
        'minWeight' : 50,
        'resolution' : 2**6}
    
    # Plasticity parameters for the case of weight dependency
    plast_params = {
          'w_max':   300., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'gamma' : .95}          # discount factor
    
    plast_params_lam = {
          'w_max':   300., #??     # Max weight of plastic synapses // here, it should be relatively high (why?)
          'gamma' : 0.95,# discount factor
          'lambda' : 0.1}
    
    #Check if inhibition yielded problems / better spiking behavior

    if lam:
        plast_params = plast_params_lam
        
    stateActionNW = StateAction(nStates, nActions, wStateAction, scalingFactor, inhibit, lam=lam)#, weightParams=weightParams)

    #Create the network
    stateActionNW.CreateNetwork(nInhibit, wStateAction, wActionExhibition, wActionInhibit)
    
    #Get random initial state
    initialState = np.random.randint(0, nStates)
    
    rewardNWDuringTraining = [0]

    #Present the state pattern as a single spike at the given delay
    stateActionNW.PresentPattern(initialState, delay)
    
    #Simulate the network (do the readout and perform the action)
    stateActionNW.Simulate(simtime, initialState, readtimeout, plast_params)
