'''
Created on Jul 30, 2017

@author: thomas
'''

import nest
import numpy as np
import mdptoolbox_local.example
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt

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
def GetValueFromProbVect(probVect):
        
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

#Create the state neurons
nActions = 3
nStates = 2
rewardOverall = 0
nRewardNeuronsPerAction = 10
Nrep = 30000
P, R = mdptoolbox_local.example.rand(nStates, nActions, is_sparse=False)
# R = np.array([[[ 0.574877,0.35767517],
#   [ 0.,-0.27974414]],
#  [[ 0.10381131,-0.],
#   [-0.23645307,0.]],
#  [[-0.,-0.92181376],
#   [ 0.9469583,-0.75116803]]])
# P = np.array([[[ 0.65345319,0.34654681],
#   [ 0.,1.]],
#  [[ 1.,0.],
#   [ 1.,0.]],
#  [[ 0.,1.],
#   [ 0.75691315,0.24308685]]])
#R = np.array([[-0.2, -1], [0.2, -1], [-0.2, 0.3]])
#P = np.array([[0, 1], [1, 1], [0, 0]])
actionNeurons = nest.Create('iaf_psc_exp', nActions * nStates)
excitatoryNoise = nest.Create('poisson_generator', nActions * nStates)
parrotNeuron = nest.Create('parrot_neuron', nStates)
parrotNeuronReward = nest.Create('parrot_neuron', nActions * nStates * nRewardNeuronsPerAction)
inputNeuron = nest.Create('poisson_generator', nStates)
rewardNeuron = nest.Create('spike_generator', nActions * nStates * nRewardNeuronsPerAction)
#rewardNeuron = nest.Create('poisson_generator', 3)
parrotSpikes = nest.Create('spike_detector', nStates)
actionSpike = nest.Create('spike_detector', nActions)
rewardSpike = nest.Create('spike_detector', nActions * nStates * nRewardNeuronsPerAction)
volTrans = nest.Create('volume_transmitter', nActions * nStates)


nest.Connect(inputNeuron, parrotNeuron,
             {'rule' : 'one_to_one'})
weights = np.random.uniform(180, 220, nStates * nActions)
weightEvolution = []
nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse1',
              {'weight': weights[0],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[0]})

nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse2',
              {'weight': weights[1],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[1]})

nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse3',
              {'weight': weights[2],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[2]})

nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse4',
              {'weight': weights[3],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[3]})

nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse5',
              {'weight': weights[4],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[4]})

nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse6',
              {'weight': weights[5],
              'A_plus': 1.,
              'A_minus': 1.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 3000.,
              'b': 0.0001,
              'Wmin': 0.,
              'Wmax': 250.,
              'vt': volTrans[5]})

nest.Connect([parrotNeuron[0]], [actionNeurons[0]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse1'})
nest.Connect([parrotNeuron[0]], [actionNeurons[1]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse2'})
nest.Connect([parrotNeuron[0]], [actionNeurons[2]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse3'})
nest.Connect([parrotNeuron[1]], [actionNeurons[3]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse4'})
nest.Connect([parrotNeuron[1]], [actionNeurons[4]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse5'})
nest.Connect([parrotNeuron[1]], [actionNeurons[5]],
             {'rule': 'one_to_one'},
             { 'model': 'layer1_stdp_synapse6'})

nest.Connect(excitatoryNoise, actionNeurons,
             {'rule' : 'one_to_one'},
             {'model' : 'static_synapse',
              'weight' : 50.})

nest.Connect(rewardNeuron, parrotNeuronReward,
             {'rule' : 'one_to_one'})
nest.Connect(parrotNeuronReward[0:nRewardNeuronsPerAction], [volTrans[0]],
             {'rule': 'all_to_all'})
nest.Connect(parrotNeuronReward[nRewardNeuronsPerAction:2*nRewardNeuronsPerAction], [volTrans[1]],
             {'rule' : 'all_to_all'})
nest.Connect(parrotNeuronReward[2*nRewardNeuronsPerAction:3*nRewardNeuronsPerAction], [volTrans[2]],
             {'rule': 'all_to_all'})
nest.Connect(parrotNeuronReward[3*nRewardNeuronsPerAction:4*nRewardNeuronsPerAction], [volTrans[3]],
             {'rule': 'all_to_all'})
nest.Connect(parrotNeuronReward[4*nRewardNeuronsPerAction:5*nRewardNeuronsPerAction], [volTrans[4]],
             {'rule': 'all_to_all'})
nest.Connect(parrotNeuronReward[5*nRewardNeuronsPerAction:6*nRewardNeuronsPerAction], [volTrans[5]],
             {'rule': 'all_to_all'})

nest.Connect(parrotNeuron, parrotSpikes)
nest.Connect(actionNeurons, actionSpike)
nest.Connect(parrotNeuronReward, rewardSpike)

conns = nest.GetConnections(parrotNeuron, actionNeurons)
active = False
initialState = 0

nest.SetStatus(excitatoryNoise, {'rate' : 100.})

for trials in range(Nrep):
    nest.ResetNetwork()
    #print(nest.GetStatus(conns,'weight'))
    weightEvolution.append(list(nest.GetStatus(conns,'weight')))
    #print(nest.GetStatus(conns,'n'))
    #print(nest.GetStatus(conns,'c'))
    nest.SetStatus(inputNeuron, {'rate' : 0.})
    nest.SetStatus([inputNeuron[initialState]], {'rate' : 100.})
    #nest.SetStatus(rewardNeuron, {'spike_times' : [25.]})
    nest.Simulate(50)
    #nest.raster_plot.from_device(parrotSpikes, hist=True, title='Parrot spikes')
    #nest.raster_plot.from_device(actionSpike, hist=True, title='State spikes')
    #plt.show()
    #print(nest.GetStatus(conns,'n'))
    #print(nest.GetStatus(conns,'c'))

    if active:
        #nest.raster_plot.from_device(rewardSpike, hist=True, title='Reward Spikes')
        #print(nest.GetStatus(conns,'n'))
        #plt.show()
        #import ipdb
        #ipdb.set_trace()
        active = False

    #Get first spike to determin action
    events = nest.GetStatus(actionSpike,'events')[0]
    
    #If there is no action to select skip
    if len(events['times']) != 0:
        action = ((events['senders'][np.argmin(events['times'])] - 1) % 3) + 1
        print('Selected action was: ' + str(action))
        nextState = GetValueFromProbVect(P[action - 1, initialState, :])
        reward = R[action - 1, initialState, nextState]
        #reward = R[action - 1, initialState]
        #nextState = P[action - 1, initialState]
        print('Reward was: ' + str(reward))
        print('Next state is: ' + str(nextState))
        
        rewardOverall += reward
    
        #print(nest.GetStatus(conns,'weight'))
        #print(nest.GetStatus(conns,'n'))
        #print(nest.GetStatus(conns,'c'))
    
        #Set the reward firing but a bit delayed
        nest.SetStatus(rewardNeuron, {'spike_times' : []})
        #nest.Simulate(20)
        if reward > 0.0:
            #import ipdb
            #ipdb.set_trace()P
            nest.SetStatus(rewardNeuron, {'spike_times' : []})
            nest.SetStatus(rewardNeuron[(initialState * nActions * nRewardNeuronsPerAction) + (action-1) * nRewardNeuronsPerAction:(initialState * nActions * nRewardNeuronsPerAction) + action*nRewardNeuronsPerAction], {'spike_times' : np.arange(20.0, 80.0, np.around(1. / reward, decimals=1))})
            nest.SetStatus(rewardNeuron, {'origin': nest.GetKernelStatus('time')})
            active = True
            
        initialState = nextState
        
backupweights = nest.GetStatus(conns,'weight')

policy = []
for state in range(nStates):
    policy.append(np.argmax(backupweights[len(policy) * nActions: (len(policy) + 1) * nActions]))

#Once trained readout weights and use them for perform test
#print('State transitions')
#print(P)
#print('Rewards')
print(R)
print(policy)
#print(rewardOverall)
#print(backupweights)

weightEvolution = np.array(weightEvolution)
plt.figure()
for i in range(nActions * nStates):
    plt.subplot(1,nActions * nStates,i+1)
    plt.plot(weightEvolution[:, i])

plt.show()

ql = mdptoolbox_local.mdp.PolicyIterationModified(P, R, 0.96)
ql.run()
#print(ql.Q)
print(ql.policy)