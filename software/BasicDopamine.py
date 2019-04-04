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

#Create the state neurons
stateNeurons = nest.Create('iaf_psc_exp', 1)
parrotNeuron = nest.Create('parrot_neuron', 1)
parrotNeuronReward = nest.Create('parrot_neuron', 1)
inputNeuron = nest.Create('poisson_generator', 1)
rewardNeuron = nest.Create('spike_generator', 1)
parrotSpikes = nest.Create('spike_detector', 1)
stateSpike = nest.Create('spike_detector', 1)
volTrans = nest.Create('volume_transmitter', 1)


nest.Connect(inputNeuron, parrotNeuron)
nest.CopyModel('stdp_dopamine_synapse',
               'layer1_stdp_synapse',
              {'weight': 150.,
              'A_plus': 10000.,
              'A_minus': 1000.,
              'tau_plus': 10.,
              'tau_c': 10.,
              'tau_n': 300.,
              'b': 10.,
              'Wmin': 0.,
              'Wmax': 200.,
              'vt': volTrans[0]})
nest.CopyModel('stdp_synapse',
               'layer2_stdp_synapse',
              {'weight': 150.,
              'alpha': 1.5,
              'tau_plus': 10.,
              'Wmax': 200.})
nest.Connect(parrotNeuron, stateNeurons,
             {'rule': 'all_to_all'},
             { 'model': 'layer1_stdp_synapse'})

nest.Connect(rewardNeuron, parrotNeuronReward)
nest.Connect(parrotNeuronReward, volTrans,
             {'rule': 'one_to_one'},
             {'model' : 'static_synapse',
              'weight' : 1.,
              'delay' : 0.1,})
nest.Connect(parrotNeuron, parrotSpikes)
nest.Connect(stateNeurons, stateSpike)

conns = nest.GetConnections(parrotNeuron, stateNeurons)
print(nest.GetStatus([conns[0]],'weight'))
print(nest.GetStatus([conns[0]],'n'))
print(nest.GetStatus([conns[0]],'c'))
nest.SetStatus(inputNeuron, {'rate' : 50.})
nest.SetStatus(rewardNeuron, {'spike_times' : [25.]})
nest.Simulate(26.2)
print(nest.GetStatus([conns[0]],'n'))
print(nest.GetStatus([conns[0]],'c'))
nest.Simulate(325)

print(nest.GetStatus([conns[0]],'weight'))
print(nest.GetStatus([conns[0]],'n'))
print(nest.GetStatus([conns[0]],'c'))
#nest.raster_plot.from_device(parrotSpikes, hist=True, title='Parrot spikes')
#nest.raster_plot.from_device(stateSpike, hist=True, title='State spikes')
#plt.show()

