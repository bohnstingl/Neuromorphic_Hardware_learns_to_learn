'''
Created on Sep 3, 2017

@author: thomas
'''

import nest
import numpy as np
import mdptoolbox_local.example
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nest.ResetKernel()
    nest.hl_api.set_verbosity('M_ERROR')  # Do not print stuff during simulation
    nest.SetKernelStatus({'print_time': False,
                          'local_num_threads': 8})  # Number of threads used
    
    nest.SetDefaults('aeif_cond_alpha',
                 {'C_m' : 150.,
                  't_ref' : 20., 
                  'V_reset' : -70.6,
                  'E_L' : -70.6,
                  'g_L' : 30.,
                  'I_e' : 0.0,
                  'a' : 4.,
                  'b' : 800.5,
                  'Delta_T' : 2.,
                  'tau_w' : 144.,
                  'V_th' : -50.4,
                  'V_peak' : 20.})
    node = nest.Create('aeif_cond_alpha', 1)
    
    inp2 = nest.Create('spike_generator', 10, {'spike_times' : [0.1, 100., 200., 300., 400., 450.]})
    noise_E = nest.Create('poisson_generator', 1, {'rate': 50.})
    inp = nest.Create('step_current_generator')
    nest.SetStatus(inp, {'amplitude_times' : [0.1],
                        'amplitude_values' : [600.0]})
    voltmeter = nest.Create('voltmeter', 1)
    spikes = nest.Create('spike_detector', 1)
    nest.Connect(noise_E, node, {'rule': 'all_to_all'},
                {'model': 'static_synapse',
                'weight': 80.})
    nest.Connect(voltmeter, node)
    nest.Connect(node, spikes)
    
    nest.Simulate(1500)
    
    nest.raster_plot.from_device(spikes, hist=True)
    plt.figure()
    nest.voltage_trace.from_device(voltmeter)
    plt.title('Membrane potentials before learning')
    plt.show()
