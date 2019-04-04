'''
Created on Sep 4, 2017

@author: thomas
'''

import nest
import numpy as np
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt
import tqdm

class Network:
    
    Wrec = []
    in_conns = []
    recordInterval = 100
    nCurrentPresentation = 0
    
    def __init__(self):
        
        nest.ResetKernel()
        nest.hl_api.set_verbosity('M_ERROR')  # Do not print stuff during simulation
        nest.SetKernelStatus({'print_time': False,
                              'local_num_threads': 4})  # Number of threads used
               
    def CreateNetwork(self, nObservationNeurons, nStateNeurons, nNoiseNeurons, nInhibitoryNeurons, wInput, wStateNoise, wWTAInhibit, wInhibitWTA):   
       
        self.inputNeurons, self.stateNeurons, self.inputSpikes, self.stateSpikes = self.createNetwork(nObservationNeurons, 
                                                                                                      nStateNeurons, 5, 
                                                                                                      nInhibitoryNeurons, 
                                                                                                      wInput, wStateNoise, wWTAInhibit, wInhibitWTA)
       
        for i in range(nStateNeurons):
            self.in_conns.append(nest.GetConnections(self.inputNeurons, [self.stateNeurons[i]]))
            self.Wrec.append(np.array([[]]))
           
    def createNetwork(self, nObservationNeurons, nStateNeurons, nNoiseNeurons, nInhibitoryNeurons, wInput, wStateNoise, wWTAInhibit, wInhibitWTA):
        
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
        inputNeurons = nest.Create('poisson_generator', nObservationNeurons)
        inhibitoryNeurons = nest.Create('iaf_psc_exp', nInhibitoryNeurons)
        
        stateSpikes = nest.Create('spike_detector', nStateNeurons)
        inputSpikes = nest.Create('spike_detector', nObservationNeurons)
        inhibitorySpikes = nest.Create('spike_detector', nInhibitoryNeurons)
        
        # connect inputs to neurons
        nest.Connect(inputNeurons, stateNeurons,
                    {'rule': 'all_to_all'},
                    {'model': 'static_synapse',
                    'delay': 1,
                    'weight': {'distribution': 'uniform',
                                'low': 2.5 * wInput,
                                'high': 7.5 * wInput}})
            
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
        
        #Connect the spike detectors
        nest.Connect(stateNeurons, stateSpikes)
        nest.Connect(inputNeurons, inputSpikes)
        nest.Connect(inhibitoryNeurons, inhibitorySpikes)
        
        # connect noise generators to neurons
        # Each excitatory neuron gets excitatory and inhibitory noise
        # this leads to random fluctuations of the membrane potential
        noise_E = nest.Create('poisson_generator', nStateNeurons, {'rate': 100.})
        noise_I = nest.Create('poisson_generator', nStateNeurons, {'rate': 100.})
        nest.CopyModel('static_synapse_hom_w',
                       'excitatory_noise',
                       {'weight': wStateNoise, 'delay': 1.})
        nest.Connect(noise_E, stateNeurons,
                     {'rule': 'one_to_one'},
                     {'model': 'excitatory_noise'})
        nest.CopyModel('static_synapse_hom_w',
                       'inhibitory_noise',
                       {'weight': -wStateNoise, 'delay': 1.})
        nest.Connect(noise_I, stateNeurons,
                     {'rule': 'one_to_one'},
                     {'model': 'inhibitory_noise'})
        
    #     nest.Connect(noiseNeurons, stateNeurons,
    #                     {'rule': 'all_to_all'},
    #                     {'model': 'static_synapse',
    #                     'weight': {'distribution': 'uniform',
    #                                 'low': 1 * WPoissionInit,
    #                                 'high': 4 * WPoissionInit}})
        
        return inputNeurons, stateNeurons, inputSpikes, stateSpikes
    
    def PresentInput(self, inputValue, presentationTime):
        
        #generate the input for the given vector
        rateVect = np.unpackbits(inputValue) * 70.
        for nrn, r in zip(self.inputNeurons,rateVect):
            nest.SetStatus([nrn],{'rate':r})
            
        # Set the times to have the time origin right
        nest.SetStatus(self.inputNeurons,{'origin': nest.GetKernelStatus('time')})
    
    def updateExcitability(self, weights):
        
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
    
    def update_weights(self, plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E):
        # Update the weights of the excitatory neurons based on pre-post pairings
        # Depress if there was a post-spike but no pre
        # Potentiate if there was pre and post spike
        # If A_decay>0 in plast_params, decay all incoming weights to neuron if the
        #              the neuron did not spike, but some neuron spiked in the network
        # plast_params..Parameters for the updates, see below in main part
        # spikes_inp... The spikes of input neurons
        # spikes_E..... The spikes of excitatory neurons
        # nodes_X...... GIDs of inp and E
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        eta   = plast_params['eta']
        A_neg = plast_params['A_neg']
        A_pos = plast_params['A_pos']
        A_decay = plast_params['A_decay']
        
        # Get network events
        events = nest.GetStatus(spikes_inp,'events')[0]
        active_input_neurons = list(events['senders'])
        events = nest.GetStatus(spikes_E,'events')[0]
        active_wta_neurons = list(events['senders'])
        if len(active_wta_neurons): # Update only if there was a network spike
            # slight weight decay
            if A_decay>0:
                conns = nest.GetConnections(nodes_inp, nodes_E)
                w = np.array(nest.GetStatus(conns,'weight'))
                w -= eta*A_decay
                w[w<0.] = 0.
                nest.SetStatus(conns,'weight',w)
                
        # first depress all incoming weights
        conns = nest.GetConnections(nodes_inp, active_wta_neurons)
        w = np.array(nest.GetStatus(conns,'weight'))
        w = w-eta*A_neg
        w[w<0.] = 0.
        nest.SetStatus(conns,'weight',w)
        # Now potentiate active inputs
        if len(active_input_neurons):
            conns = nest.GetConnections(active_input_neurons, active_wta_neurons)
            w = np.array(nest.GetStatus(conns,'weight'))
            w = w+eta*A_pos
            w[w>w_max] = w_max
            nest.SetStatus(conns,'weight',w)
            
    def update_weights_wdep(self, plast_params,spikes_inp, spikes_E, nodes_inp, nodes_E):
        # Update the weights of the excitatory neurons based on pre-post pairings
        # Depress if there was a post-spike but no pre by (w/w_max)^alpha
        # Potentiate if there was pre and post spike by ((w_max-w)/w_max)^alpha
        # If A_decay>0 in plast_params, decay all incoming weights to neuron if the
        #              the neuron did not spike, but some neuron spiked in the network
        # plast_params..Parameters for the updates, see below in main part
        # spikes_inp... The spikes of input neurons
        # spikes_E..... The spikes of excitatory neurons
        # nodes_X...... GIDs of inp and E
    
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        eta   = plast_params['eta']
        alpha = plast_params['alpha']
        A_decay = plast_params['A_decay']
        
        # Get network events
        events = nest.GetStatus(spikes_inp,'events')[0]
        active_input_neurons = list(events['senders'])
        events = nest.GetStatus(spikes_E,'events')[0]
        active_wta_neurons = list(events['senders'])
        if len(active_wta_neurons): # Update only if there was a network spike
            # slight weight decay
            if A_decay>0:
                conns = nest.GetConnections(nodes_inp, nodes_E)
                w = np.array(nest.GetStatus(conns,'weight'))
                w -= eta*A_decay
                w[w<0.] = 0.
                nest.SetStatus(conns,'weight',w)
            # Backup active input weights
            if len(active_input_neurons):
                conns_act = nest.GetConnections(active_input_neurons, active_wta_neurons)
                w_act = np.array(nest.GetStatus(conns_act,'weight'))
            # depress all incoming weights
            conns = nest.GetConnections(nodes_inp, active_wta_neurons)
            w = np.array(nest.GetStatus(conns,'weight'))
            w = w - eta*(w/w_max)**alpha
            w[w<0] = 0.
            nest.SetStatus(conns,'weight',w)
            # Undo depression for active inputs
            if len(active_input_neurons):
                nest.SetStatus(conns_act,'weight',w_act)
            # Now potentiate active inputs
            if len(active_input_neurons):
                w = np.array(nest.GetStatus(conns_act,'weight'))
                w = w+eta*((w_max-w)/w_max)**alpha
                w[w>w_max] = w_max
                nest.SetStatus(conns_act,'weight',w)
    
    def TestNetwork(self, observationValues, presentationTime, simtime, plot=False, extensive=False):
        
        #present all four patterns and check the state spikes
        firstState = []
        fitness = []        

        if extensive:
            zRep = 20.
            for patternNr in range(len(observationValues)):
                for it in range(0, int(zRep)):
                    nest.ResetNetwork()
                    self.PresentInput(np.array(observationValues[patternNr], dtype=np.uint8), presentationTime)
                    nest.Simulate(simtime)
                
                    #get first spike after input presentation
                    events = nest.GetStatus(self.stateSpikes,'events')[0]
                    firstState.append(events['senders'][np.argmin(events['times'])] - 1)

                print('States for patern ' + str(patternNr) + ' were: ' + str(firstState))
                patternOccurences = np.bincount(firstState) - zRep / len(observationValues)
                print(np.bincount(firstState))                
                print(patternOccurences)
                fitness.append(patternOccurences[np.argmax(patternOccurences)] * (zRep / (zRep - zRep / len(observationValues))) * (100. / zRep))
                print(fitness[patternNr])
                
                firstState.clear()

            fitness = np.mean(fitness)

            if plot:
                for patternNr in range(len(observationValues)):
                    nest.ResetNetwork()
                    self.PresentInput(np.array(observationValues[patternNr], dtype=np.uint8), presentationTime)
                    nest.Simulate(simtime)
                    
                    nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
                plt.show()

        else:

            for patternNr in range(len(observationValues)):
                nest.ResetNetwork()
                self.PresentInput(np.array(observationValues[patternNr], dtype=np.uint8), presentationTime)
                nest.Simulate(simtime)
                
                if plot:
                    nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
            
                #get first spike after input presentation
                events = nest.GetStatus(self.stateSpikes,'events')[0]
                firstState.append(events['senders'][np.argmin(events['times'])])
        
            plt.show()
            print('States were: ' + str(list(set(firstState))))
            fitness = len(list(set(firstState))) * 100. / len(observationValues)

        return fitness
    
    def Simulate(self, simtime, WDEP, plast_params):
        
        nest.ResetNetwork()
        nest.Simulate(simtime)

        if (self.nCurrentPresentation % self.recordInterval) == 0 and self.nCurrentPresentation > 0:
            self.SaveWeights() 
            
        if WDEP:
            self.update_weights_wdep(plast_params, self.inputSpikes, self.stateSpikes, self.inputNeurons, self.stateNeurons)
        else:
            self.update_weights(plast_params, self.inputSpikes, self.stateSpikes, self.inputNeurons, self.stateNeurons)
        
    def SaveWeights(self):
        
        
        #Resolve state spikes to last pattern
        #nest.raster_plot.from_device(inputSpikes, hist=True)
        #plt.show()
         
        #nest.raster_plot.from_device(stateSpikes, hist=True)
        #plt.show()
        #nest.raster_plot.from_device(inhibitorySpikes, hist=True)
        #plt.show()
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
        added = len(self.Wrec[0][0])
        for i in range(len(self.stateNeurons)):
            if added == 0:
                self.Wrec[i] = np.append(self.Wrec[i], np.array([nest.GetStatus(self.in_conns[i],'weight')]), axis=1)
            else:
                self.Wrec[i] = np.append(self.Wrec[i], np.array([nest.GetStatus(self.in_conns[i],'weight')]), axis=0)
            #in_conns[i] = nest.GetConnections(inputNeurons, [stateNeurons[i]])
        #print(w)
        #print('Weights updated')
    
    pass

    def PlotEvaluation(self):

        # plot weight evolution
        plt.figure()
        for i in range(len(self.stateNeurons)):
            plt.subplot(1,len(self.stateNeurons),i+1)
            plt.imshow(self.Wrec[i])
            plt.colorbar()
        
        plt.figure()
        for i in range(len(self.stateNeurons)):
            plt_nrn = [self.stateNeurons[i]]
            conns = nest.GetConnections(self.inputNeurons, plt_nrn)
            w = np.array(nest.GetStatus(conns,'weight'))
            plt.plot(np.array(range(len(self.inputNeurons)))+1,w)
            plt.xlabel('input neuron')
            plt.ylabel('weights to state neuron')
            
        plt.show()    
    
    
if __name__ == '__main__':
    
    network = Network()
    
    ####################################################
    # Parameters                                       #
    ####################################################
    WPoissionInit = 5.
    J_EI = 30.
    J_IE = -19.0
    J_noise = 28.1
    J_in = 50
    weight = [0.1, 5., 10., 20.]
    observationValues = [224, 56, 14, 195]
    NRep = 5000
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
        
    print(network.TestNetwork(observationValues, presentationTime, simtime, plot=True, extensive=False))

    train = input('Want to train?')
    if train.lower() != 'y':
        exit()
        
    #for given number of input presentations
    iterator = tqdm.tqdm(range(0, NRep))
    for z in iterator:
    #for z in range(0, NRep):
        
        #draw input
        rnd = np.random.randint(0, len(observationValues))
        pattern = observationValues[rnd]
        
        #present input
        network.PresentInput(np.array([pattern], dtype=np.uint8), presentationTime)
        
        network.Simulate(simtime, WDEP, plast_params)

    network.PlotEvaluation()
        
    print(network.TestNetwork(observationValues, presentationTime, simtime, plot=True, extensive=True))


    
