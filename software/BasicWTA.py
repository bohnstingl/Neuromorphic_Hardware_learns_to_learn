'''
Created on Sep 4, 2017

@author: thomas
'''

import nest
import numpy as np
import mdptoolbox_local.example
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt
import tqdm

def createNetwork(nObservationNeurons, nStateNeurons, nNoiseNeurons, nInhibitoryNeurons, wWTAInhibit, wInhibitWTA):
    
    nest.SetDefaults('iaf_psc_exp',
             {'C_m': 30.0,  
              'tau_m': 30.0,
              'I_e': 0.0,
              'E_L': -70.0,
              'V_th': -55.0,
              'tau_syn_ex': 3.0,
              'tau_syn_in': 2.0,
              'V_reset': -70.0})

    #Use standard STDP synapses for the beginning
    
    #Create the state neurons
    stateNeurons = nest.Create('iaf_psc_exp', nStateNeurons)
    noiseNeurons = nest.Create('poisson_generator', nNoiseNeurons)
    inp = nest.Create('poisson_generator', nObservationNeurons)
    inputNeurons = nest.Create('iaf_psc_exp', nObservationNeurons)
    inhibitoryNeurons = nest.Create('iaf_psc_exp', nInhibitoryNeurons)
    
    stateSpikes = nest.Create('spike_detector', nStateNeurons)
    inputSpikes = nest.Create('spike_detector', nObservationNeurons)
    inhibitorySpikes = nest.Create('spike_detector', nInhibitoryNeurons)
    
    
    #STDP parameters overtaken from Experiment 3 of PoBC
    syn_param = {"tau_plus": 30., #Time constant of STDP window potentiation in ms 
                 "lambda": 0.005, #Step size 
                 "alpha": 1.1, #Asymmetry parameter (scales depressing increments as alpha*lambda
                 "mu_plus": 0., #Weight dependence exponent potentiation
                 "mu_minus": 0., #Weight dependence exponent depression 
                 "Wmax": 10. * 85., #Maximum allowed weight
                 'weight': 85. #Initial weight of the synapses
                }
    
    # connect inputs to neurons
    #sufficient to let the input neurons spike
    nest.Connect(inp, inputNeurons,
                    {'rule': 'one_to_one'},
                    {'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 200.})
    
    nest.Connect(stateNeurons, inhibitoryNeurons,
                    {'rule': 'all_to_all'},
                    {'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': {'distribution': 'uniform',
                                'low': 2.5 * wWTAInhibit,
                                'high': 7.5 * wWTAInhibit}})
    
    nest.Connect(inhibitoryNeurons, stateNeurons,
                    {'rule': 'all_to_all'},
                    {'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': {"distribution": "normal", "mu": wInhibitWTA, "sigma": 0.7 * abs(wInhibitWTA)}})
    
    
    nest.Connect(stateNeurons, stateSpikes)
    nest.Connect(inputNeurons, inputSpikes)
    nest.Connect(inhibitoryNeurons, inhibitorySpikes)
    
    nest.CopyModel("stdp_synapse", "stdp_syn", syn_param)
    nest.Connect(inputNeurons, stateNeurons, {"rule": "all_to_all"}, 
                syn_spec="stdp_syn")
    
    #Modify synaptic connections to have slightly different weights
#     conns = nest.GetConnections(inputNeurons, stateNeurons)
#     w = np.array(nest.GetStatus(conns,'weight'))
#     w = w + np.random.uniform(-15, 15., len(w))
#     nest.SetStatus(conns,'weight',w)
    
    nest.Connect(noiseNeurons, stateNeurons,
                    {'rule': 'all_to_all'},
                    {'model': 'static_synapse',
                    'weight': {'distribution': 'uniform',
                                'low': 1 * WPoissionInit,
                                'high': 4 * WPoissionInit}})
    
    return inputNeurons, inp, stateNeurons, inputSpikes, stateSpikes

def presentInput(inputNeurons, inputValue, presentationTime, nRep):
    
    #generate the input for the given vector
    rateVect = np.unpackbits(inputValue) * 70.
    startTime = nRep * (presentationTime + 50.)
    for index, rate in enumerate(rateVect):
        nest.SetStatus([inputNeurons[index]],{"rate":rate,
                                              "start":startTime + 0.1,
                                              "stop":startTime + 1 * presentationTime})
        
    pass

def updateExcitability(weights):
    
    #Update the excitabilities according to the paper
    
    
    # Get network events
    events = nest.GetStatus(stateSpikes,'events')[0]
    conns = nest.GetConnections(noiseNeurons, stateNeurons)
    w = np.array(nest.GetStatus(conns,'weight'))
    w = weights
    nest.SetStatus(conns,'weight',w)
#     active_input_neurons = list(events['senders'])
#     
#      events = nest.GetStatus(spikes_E,'events')[0]
#       active_wta_neurons = list(events['senders'])
#       if len(active_wta_neurons): # Update only if there was a network spike
#         # slight weight decay
#         if A_decay>0:
#           conns = nest.GetConnections(nodes_inp, nodes_E)
#           w = np.array(nest.GetStatus(conns,'weight'))
#           w -= eta*A_decay
#           w[w<0.] = 0.
#           nest.SetStatus(conns,'weight',w)
#         # first depress all incoming weights
#         conns = nest.GetConnections(nodes_inp, active_wta_neurons)
#         w = np.array(nest.GetStatus(conns,'weight'))
#         w = w-eta*A_neg
#         w[w<0.] = 0.
#         nest.SetStatus(conns,'weight',w)
#         # Now potentiate active inputs
#         if len(active_input_neurons):
#            conns = nest.GetConnections(active_input_neurons, active_wta_neurons)
#            w = np.array(nest.GetStatus(conns,'weight'))
#            w = w+eta*A_pos
#            w[w>w_max] = w_max
#            nest.SetStatus(conns,'weight',w)
    pass

def updateWeights():
    
    pass

def testNetwork(inputNeurons, stateSpikes, NRep, presentationTime, simtime):
    
    #present all four patterns and check the state spikes
    #start at Nrep * simtime
    
    presentInput(inputNeurons, np.array(observationValues[0], dtype=np.uint8), presentationTime, NRep)
    nest.Simulate(simtime)
    
    nest.raster_plot.from_device(stateSpikes, hist=True, title='State spikes for pattern 0')
    plt.xlim([NRep * simtime, (NRep + 1) * simtime])
    
    NRep = NRep + 1
    presentInput(inputNeurons, np.array(observationValues[1], dtype=np.uint8), presentationTime, NRep)
    nest.Simulate(simtime)
    
    nest.raster_plot.from_device(stateSpikes, hist=True, title='State spikes for pattern 1')
    plt.xlim([NRep * simtime, (NRep + 1) * simtime])
    
    NRep = NRep + 1
    presentInput(inputNeurons, np.array(observationValues[2], dtype=np.uint8), presentationTime, NRep)
    nest.Simulate(simtime)
    
    nest.raster_plot.from_device(stateSpikes, hist=True, title='State spikes for pattern 2')
    plt.xlim([NRep * simtime, (NRep + 1) * simtime])
    
    NRep = NRep + 1
    presentInput(inputNeurons, np.array(observationValues[3], dtype=np.uint8), presentationTime, NRep)
    nest.Simulate(simtime)
    
    nest.raster_plot.from_device(stateSpikes, hist=True, title='State spikes for pattern 3')
    plt.xlim([NRep * simtime, (NRep + 1) * simtime])
    plt.show()
        
    
def evaluateNetwork(printOutput):
    
    
    if printOutput:
        #Resolve state spikes to last pattern
#         nest.raster_plot.from_device(inputSpikes, hist=True)
#         plt.show()
         
        #try:
#             nest.raster_plot.from_device(stateSpikes, hist=True)
#             plt.show()
#             nest.raster_plot.from_device(inhibitorySpikes, hist=True)
#             plt.show()
        conns = nest.GetConnections(inputNeurons, stateNeurons)
        w = np.array(nest.GetStatus(conns,'weight'))
        #print(nest.GetStatus([conns[0]],'weight'))
        #stateSpikes = nest.Create('spike_detector', 4)
        #events = nest.GetStatus(stateSpikes,'events')
        
        
        #events = nest.GetStatus(stateSpikes,'events')
        #nest.SetStatus(stateSpikes, 'events', ({'senders' : np.array([]), 'times' : np.array([])},
        #                                       {'senders' : np.array([]), 'times' : np.array([])},
        #                                       {'senders' : np.array([]), 'times' : np.array([])},
        #                                       {'senders' : np.array([]), 'times' : np.array([])}))
        #events2 = nest.GetStatus(stateSpikes,'events')[0]
        added = len(Wrec[0][0])
        for i in range(4):
            if added == 0:
                Wrec[i] = np.append(Wrec[i], np.array([nest.GetStatus(in_conns[i],'weight')]), axis=1)
            else:
                Wrec[i] = np.append(Wrec[i], np.array([nest.GetStatus(in_conns[i],'weight')]), axis=0)
            #in_conns[i] = nest.GetConnections(inputNeurons, [stateNeurons[i]])
        #print(w)
        #print('Weights updated')
    
    #returns the fitness
    pass

nest.ResetKernel()
nest.hl_api.set_verbosity('M_ERROR')  # Do not print stuff during simulation
nest.SetKernelStatus({'print_time': False,
                      'local_num_threads': 8})  # Number of threads used

WPoissionInit = 5.
J_EI = 40.
J_IE = -75.0
weight = [0.1, 5., 10., 20.]
observationValues = [178, 23, 85, 202]
NRep = 100
presentationTime = 200.
simtime = presentationTime + 50.

#Create the network
inputNeurons, inputGeneratorNeurons, stateNeurons, inputSpikes, stateSpikes = createNetwork(8, 4, 5, 10, J_EI, J_IE)


Wrec = []
in_conns = []
for i in range(4):
    in_conns.append(nest.GetConnections(inputNeurons, [stateNeurons[i]]))
    Wrec.append(np.array([[]]))
    
#for given number of input presentations
iterator = tqdm.tqdm(range(0, NRep))
for z in iterator:
            
    if (z % 100) == 0 and z != 0:
        p = True
        testNetwork(inputGeneratorNeurons, stateSpikes, NRep, presentationTime, simtime)

    else:
        p = False
    #updateExcitability(w)
    
    #draw input
    rnd = np.random.randint(0, 4)
    pattern = observationValues[rnd]
    
    #present input
    presentInput(inputGeneratorNeurons, np.array([pattern], dtype=np.uint8), presentationTime, z)
    
    nest.Simulate(simtime)
    evaluateNetwork(p)
#     nest.raster_plot.from_device(inputSpikes, hist=True, title='Input spikes for pattern ' + str(rnd))
#     nest.raster_plot.from_device(stateSpikes, hist=True, title='State spikes for pattern ' + str(rnd))
#     plt.show()
#     plt.close()
    #updateWeights()
    
# plot weight evolution
plt.figure()
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(Wrec[i])
    plt.colorbar()
    
testNetwork(inputGeneratorNeurons, stateSpikes, NRep, presentationTime, simtime)


    
