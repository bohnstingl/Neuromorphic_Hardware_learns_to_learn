'''
Created on Jul 30, 2017

@author: thomas
'''

import nest

nest.SetDefaults('iaf_psc_exp',
                 {'C_m': 30.0,  
                  'tau_m': 30.0,
                  'I_e': 0.0,
                  'E_L': -70.0,
                  'V_th': -55.0,
                  'tau_syn_ex': 3.0,
                  'tau_syn_in': 2.0,
                  'V_reset': -70.0})
    
stateNeurons = nest.Create('iaf_psc_exp', 1)
inputNeurons = nest.Create('spike_generator', 1)
inp = nest.Create('parrot_neuron', 1)
spikeInp = nest.Create('spike_detector', 1)
spikeState = nest.Create('spike_detector', 1)
spikeInput = nest.Create('spike_detector', 1)
spikeNessler = nest.Create('spike_detector', 1)
nesslerNeuron = nest.Create('pp_psc_delta_Nessler', 1,
                            {'dead_time': 2.0,
                             't_ref_remaining': 0.,
                             'E_sfa': 1.2,
                             'E_sfa_Max': 10.,
                             'eta': 1.0,
                             'is_excitable': True,
                             'baseline': 2.2,})

nest.CopyModel('stdp_synapse_Nessler_Simple',
               'test_synapse',
               {'weight': 35., 'delay': 1.,
                'baseline': 30.,
                'eta': 1.,
                'c' : 1.,
                'Wmax': 500.,})

nest.Connect(inp, stateNeurons,
            {'rule': 'all_to_all'},
            {'model': 'test_synapse'})

nest.Connect(inputNeurons, inp,
            {'rule': 'all_to_all'},
            {'model': 'static_synapse',
             'weight': 300.})
nest.Connect(inputNeurons, nesslerNeuron,
            {'rule': 'all_to_all'},
            {'model': 'static_synapse',
             'weight': 300.})

nest.Connect(inputNeurons, spikeInp)
nest.Connect(stateNeurons, spikeState)
nest.Connect(inp, spikeInput)
nest.Connect(nesslerNeuron, spikeNessler)


#E_sfa resets itself
#Excitability update does not work

nest.GetStatus(nest.GetConnections(inp, stateNeurons),'weight')
nest.Simulate(50)

nest.SetStatus(inputNeurons, {'spike_times': [3.]})
nest.SetStatus(inputNeurons, {'origin': nest.GetKernelStatus('time')})
nest.GetStatus(spikeInp, 'events')[0]
nest.Simulate(50)
nest.Simulate(50)

nest.ResetNetwork()

#nest.SetStatus(inputNeurons, {'spike_times': [28.]})
nest.SetStatus(inputNeurons, {'spike_times': [3.]})
nest.SetStatus(inputNeurons, {'origin': nest.GetKernelStatus('time')})
nest.Simulate(25)
#nest.ResetNetwork()


print('here')
