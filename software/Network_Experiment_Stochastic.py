'''
Created on Sep 4, 2017

@author: thomas
'''

import nest
import numpy as np
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt

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
               
    def CreateNetwork(self, plast_params, nObservationNeurons, nStateNeurons, nInhibitoryNeurons, wInput, wWTAInhibit, wInhibitWTA):   
       
        self.inputNeurons, self.inputParrotNeurons, self.stateNeurons, self.inputSpikes, self.inputParrotSpikes, self.stateSpikes, self.inhibitorySpikes = self.createNetwork(plast_params,
                                                                                                      nObservationNeurons, 
                                                                                                      nStateNeurons, 
                                                                                                      nInhibitoryNeurons, 
                                                                                                      wInput, wWTAInhibit, wInhibitWTA)
       
        for i in range(nStateNeurons):
            self.in_conns.append(nest.GetConnections(self.inputParrotNeurons, [self.stateNeurons[i]]))
            self.Wrec.append(np.array([[]]))
           
    def createNetwork(self, plast_params, nObservationNeurons, nStateNeurons, nInhibitoryNeurons, wInput, wWTAInhibit, wInhibitWTA):
        
        # Unpack Plasticity parameters
        w_max = plast_params['w_max']
        eta   = plast_params['eta']
        c = plast_params['c']
        sigma = plast_params['sigma']
        baseline = plast_params['baseline']
        
        nest.SetDefaults('pp_psc_delta_Nessler',
                 {'C_m': 30.0,  
                  'tau_m': 30.0,
                  'with_reset': True,
                  't_ref_remaining': 0.,
                  'dead_time': 2.0,
                  'E_sfa': 0.,
                  'V_m' : -70.,
                  'c_1': 0.,
                  'c_2': 1.,
                  'c_3': 1.,
                  'is_excitable': False, #Set to false for the beginning
                  })
        
        nest.SetDefaults('iaf_psc_exp',
                 {'C_m': 30.0,  
                  'tau_m': 30.0,
                  'I_e': 0.0,
                  'E_L': -70.0,
                  'V_th': -55.0,
                  'tau_syn_ex': 3.0,
                  'tau_syn_in': 2.0,
                  'V_reset': -70.0})
        
        #Create the state neurons
        stateNeurons = nest.Create('pp_psc_delta_Nessler', nStateNeurons)
        inputNeurons = nest.Create('poisson_generator', nObservationNeurons)
        inputParrotNeurons = nest.Create('parrot_neuron', nObservationNeurons)
        inhibitoryNeurons = nest.Create('iaf_psc_exp', nInhibitoryNeurons)
        
        stateSpikes = nest.Create('spike_detector', nStateNeurons)
        inputSpikes = nest.Create('spike_detector', nObservationNeurons)
        parrotSpikes = nest.Create('spike_detector', nObservationNeurons)
        inhibitorySpikes = nest.Create('spike_detector', nInhibitoryNeurons)
        
        # connect inputs to neurons
        nest.Connect(inputParrotNeurons, stateNeurons,
                    {'rule': 'all_to_all'},
                    {'model': 'stdp_synapse_Nessler_Simple',
                    'delay': 1.,
                    'weight': {"distribution": "uniform", "low": 0.5 * wInput, "high": 1.5 * abs(wInput)},
                    'Wmax': w_max,
                    'c': c,
                    'sigma': sigma,
                    'eta': eta,
                    'baseline': baseline})
        
        nest.Connect(inputNeurons, inputParrotNeurons,
                     {'rule': 'one_to_one'})
            
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
        nest.Connect(inputParrotNeurons, parrotSpikes)
        nest.Connect(inhibitoryNeurons, inhibitorySpikes)
        
        return inputNeurons, inputParrotNeurons, stateNeurons, inputSpikes, parrotSpikes, stateSpikes, inhibitorySpikes
    
    def PresentInput(self, inputValue, presentationTime, demo=False):
        
        if demo:
            self.set_pattern(self.inputNeurons, inputValue)
        else:

            #generate the input for the given vector
            rateVect = np.unpackbits(inputValue) * 70.
            for nrn, r in zip(self.inputNeurons,rateVect):
                nest.SetStatus([nrn],{'rate':r})
                
            # Set the times to have the time origin right
            nest.SetStatus(self.inputNeurons,{'origin': nest.GetKernelStatus('time')})

    def get_spike_patterns(self, N_in, N_pat, Rmax = 25.0, Rmin = 0.01, PLOT=True):
        
        step = N_in/N_pat-2  # defines the means of the Gaussian rate profiles
        offs = N_in/N_pat/2+2 # defines the offset of the means
        rates=[]
        idxs = np.array(range(N_in))
        
        for i in range(N_pat):
            rr = 50.0*np.exp(-(idxs-(offs+i*step))**2/50.0)
            rr[rr <= Rmin] = Rmin
            rates.append(rr)
            
        if PLOT:
            plt.figure()
            leg = []
            idx = 0
            for rr in rates:
                plt.plot(rr)
                leg.append('pattern ' + str(idx))
                idx += 1
            plt.xlabel('input neuron index')
            plt.ylabel('firing rate [Hz]')
            plt.legend(leg)
            plt.title('Input rate patterns')

        self.rates = rates
        return rates

    def set_pattern(self, nodes_inp, pat_idx):
        # Set the rates of the Poisson_generators nodes_inp to rates[pat_idx]
        # and set their time origin to the global simulation time
        for nrn, r in zip(nodes_inp,self.rates[pat_idx]):
            nest.SetStatus([nrn],{'rate':r})
        # set input times
        nest.SetStatus(nodes_inp,{'origin': nest.GetKernelStatus('time')})
        # Note: When we do consecutive simulations in nest, the internal simulation time
        #       is continued from the last one.
        #       nest.GetKernelStatus('time') returns the current time of the simulation
        #       We use this to set the origin of the input nodes (Poisson generator) such that
        #       the spikes are emitted there relative to this origin (i.e. at correct times)
        #       See also below where we define 'nodes_inp'
    
    def TestNetwork(self, observationValues, presentationTime, simtime, plot=False, extensive=False, demo=True):
        
        #present all four patterns and check the state spikes
        firstState = []
        fitness = []  
        specializedNeuron = []

        if extensive:
            zRep = 20.
            
            if demo:
                for patternNr in range(len(self.rates)):
                    for it in range(0, int(zRep)):
                        nest.ResetNetwork()
                        self.PresentInput(patternNr, presentationTime, demo=True)
                        nest.Simulate(simtime)
                    
                        #get first spike after input presentation
                        events = nest.GetStatus(self.stateSpikes,'events')[0]
                        firstState.append(events['senders'][np.argmin(events['times'])] - 1)

                    #print('States for patern ' + str(patternNr) + ' were: ' + str(firstState))
                    patternOccurences = np.bincount(firstState) - zRep / len(self.inputNeurons)
                    #print(np.bincount(firstState))                
                    #print(patternOccurences)
                    neuronID = np.argmax(patternOccurences)
                    f = patternOccurences[np.argmax(patternOccurences)] * (zRep / (zRep - zRep / len(self.inputNeurons))) * (100. / zRep)
                    if neuronID not in specializedNeuron:
                        fitness.append(f)
                    else:
                        fitness.append(-400)
                    specializedNeuron.append(np.argmax(patternOccurences))
                    
                    #print(fitness[patternNr])
                    
                    firstState = []

                fitness = np.mean(fitness)

                if plot:
                    for patternNr in range(len(self.rates)):
                        nest.ResetNetwork()
                        self.PresentInput(patternNr, presentationTime, demo=True)
                        nest.Simulate(simtime)
                        
                        if len(self.stateSpikes) > 0:
                            nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
                        else:
                            nest.ResetNetwork()
                            self.PresentInput(patternNr, presentationTime, demo=True)
                            nest.Simulate(simtime)
                            nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
                    plt.show()
            else:
                for patternNr in range(len(observationValues)):
                    for it in range(0, int(zRep)):
                        nest.ResetNetwork()
                        self.PresentInput(np.array(observationValues[patternNr], dtype=np.uint8), presentationTime)
                        nest.Simulate(simtime)
                    
                        #get first spike after input presentation
                        events = nest.GetStatus(self.stateSpikes,'events')[0]
                        firstState.append(events['senders'][np.argmin(events['times'])] - 1)

                    print('States for patern ' + str(patternNr) + ' were: ' + str(firstState))
                    patternOccurences = np.bincount(firstState) - zRep / len(self.inputNeurons)
                    print(np.bincount(firstState))                
                    print(patternOccurences)
                    fitness.append(patternOccurences[np.argmax(patternOccurences)] * (zRep / (zRep - zRep / len(self.inputNeurons))) * (100. / zRep))
                    print(fitness[patternNr])
                    
                    firstState = []

                fitness = np.mean(fitness)

                if plot:
                    for patternNr in range(len(observationValues)):
                        nest.ResetNetwork()
                        self.PresentInput(np.array(observationValues[patternNr], dtype=np.uint8), presentationTime)
                        nest.Simulate(simtime)
                        
                        nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
                    plt.show()

        else:

            for patternNr in range(len(self.rates)):
                nest.ResetNetwork()
                self.PresentInput(patternNr, presentationTime, demo=True)
                nest.Simulate(simtime)
                
                if plot:
                    nest.raster_plot.from_device(self.stateSpikes, hist=True, title='State spikes for pattern ' + str(patternNr + 1))
                    nest.raster_plot.from_device(self.inputSpikes, hist=True, title='Input spikes for pattern ' + str(patternNr + 1))
                    nest.raster_plot.from_device(self.inputParrotSpikes, hist=True, title='Parrot spikes for pattern ' + str(patternNr + 1))
            
                #get first spike after input presentation
                events = nest.GetStatus(self.stateSpikes,'events')[0]
                firstState.append(events['senders'][np.argmin(events['times'])])
            
            nest.raster_plot.from_device(self.inhibitorySpikes, hist=True, title='Inhibitory spikes')
        
            plt.show()
            print('States were: ' + str(list(set(firstState))))
            fitness = len(list(set(firstState))) * 100. / len(observationValues)

        print(fitness)
        return fitness
    
    def Simulate(self, simtime):
        
        nest.ResetNetwork()
        nest.Simulate(simtime)
        
        #Send a single keep alive spike with all neurons which have not spiked in the current interval
        #eventSenders = nest.GetStatus(self.inputParrotSpikes, 'events')[0]['senders']
        #inactiveNeurons = np.array([ind for ind in range(len(self.inputParrotSpikes)) if (ind + self.inputParrotNeurons[0]) not in eventSenders]) + self.keepAliveNeurons[0]
        #nest.SetStatus(inactiveNeurons.tolist(), {'spike_times' : [0.1]})
        #nest.SetStatus(self.keepAliveNeurons, {'origin': nest.GetKernelStatus('time')})
        #nest.Simulate(3)

        if (self.nCurrentPresentation % self.recordInterval) == 0 and self.nCurrentPresentation > 0:
            self.SaveWeights()
        
        self.nCurrentPresentation += 1
        
    def SaveWeights(self):
        
        added = len(self.Wrec[0][0])
        for i in range(len(self.stateNeurons)):
            if added == 0:
                self.Wrec[i] = np.append(self.Wrec[i], np.array([nest.GetStatus(self.in_conns[i],'weight')]), axis=1)
            else:
                self.Wrec[i] = np.append(self.Wrec[i], np.array([nest.GetStatus(self.in_conns[i],'weight')]), axis=0)
    
    pass

    def PlotEvaluation(self):

        # plot weight evolution
        plt.figure()
        for i in range(len(self.stateNeurons)):
            plt.subplot(1,len(self.stateNeurons),i+1)
            plt.imshow(self.Wrec[i])
            plt.colorbar()
            plt.savefig('weightEvolution.png')
        
        plt.figure()
        for i in range(len(self.stateNeurons)):
            plt_nrn = [self.stateNeurons[i]]
            conns = nest.GetConnections(self.inputParrotNeurons, plt_nrn)
            w = np.array(nest.GetStatus(conns,'weight'))
            plt.plot(np.array(range(len(self.inputParrotNeurons)))+1,w)
            plt.xlabel('input neuron')
            plt.ylabel('weights to state neuron')
            plt.savefig('weightsToInput.png')

        plt.figure()
        plt_nrn = [self.stateNeurons[0]]
        conns = nest.GetConnections(self.inputParrotNeurons, plt_nrn)
        w = np.array(nest.GetStatus(conns,'weight'))
        plt.plot(np.array(range(len(self.inputParrotNeurons)))+1,w)
        plt.xlabel('input neuron')
        plt.ylabel('weights to state neuron')
        plt.savefig('weightsOfNeuron0.png')
            
        plt.show()    
    
    
if __name__ == '__main__':
    
    network = Network()
    
    ####################################################
    # Parameters                                       #
    ####################################################
    J_EI = 20.
    J_IE = -20.0
    J_in = 50.0
    observationValues = [224, 56, 14, 195]
    NRep = 500
    presentationTime = 100.
    simtime = presentationTime + 50.
    nInputNeurons = 80
    nStateNeurons = 4
    nInhibitoryNeurons = 50

    N_pat = 4 # Number of different patterns
    network.get_spike_patterns(nInputNeurons, N_pat, PLOT=False)

    plast_params_nessler = {
          'w_max':   100., #??      # Max weight of plastic synapses // here, it should be relatively high (why?)
          'eta':     0.1, #??      # learning rate
          'c':  1.0,       # exponent of weight dependency
          'sigma': 10., #ms for STDP time window
          'baseline': 40.}
    
    plast_params = plast_params_nessler
    
    ####################################################
    
    #Create the network
    network.CreateNetwork(plast_params, nInputNeurons, nStateNeurons, nInhibitoryNeurons, J_in, J_EI, J_IE)
        
    print(network.TestNetwork(observationValues, presentationTime, simtime, plot=True, extensive=False))    

    exit()
    #train = input('Want to train?')
    #if train.lower() != 'y':
    #    exit()
    
    #Introduce a single spike into all input neurons so that the synapses can work
        
    #for given number of input presentations
    import tqdm
    iterator = tqdm.tqdm(range(0, NRep))
    for z in iterator:
    #for z in range(0, NRep):
        
        #draw input
        rnd = np.random.randint(0, len(observationValues))
        pattern = observationValues[rnd]
        
        #present input
        inputValue = np.array([pattern], dtype=np.uint8)
                
        network.PresentInput(np.random.randint(0, N_pat), presentationTime, demo=True)
        #network.PresentInput(np.random.randint(0, len(observationValues)), presentationTime, demo=True)
        
        network.Simulate(simtime)

    network.PlotEvaluation()
        
    print(network.TestNetwork(observationValues, presentationTime, simtime, plot=True, extensive=True, demo=True))


    
