import pydls as dls
import json
#import helpers as hp
import helpers_pydls as hp
import pylogging
import Utils as utils
import numpy as np
import struct
import mdptoolbox
import mdptoolbox.example
import fractional as frac
import h5py
import datetime as dt
import dict_conv
import os
import EvaluateMDP
import time

class DLSNetwork:
    
    def __init__(self, nStates, nActions, multipleRuns):
        
        self.nStates = nStates
        self.nActions = nActions
        
        self.recurrentSpikeAddress = 50
        self.recurrentSpikeCumulationTime = 100
        self.chip = dls.Chip()
        self.multipleRuns = multipleRuns

        #Hold lists to enable overall evaluation for LTL
        self.accumulatedStates = []
        self.accumulatedActions = []
        self.accumulatedRewards = []
        
    def CreateNetwork(self, weightMatrix=[]):
        '''Wire the network'''
        
        #Network layout
        if weightMatrix == []:
            print 'weights are none'
        else:
            print 'Weights are not none'
        
        #The address of this spike determines which state neuron was firing
        #syn address 0 - (nStates - 1) -> addresses for state neurons
        #syn address nStates - (nstates + nActions) -> addresses for action neurons
        
        #This makes up the recurrent connections. The diagonal elements contain the recurrent connections
        #Synram contains the synapses -> access with row and column
        #The state neurons are connected on the diagonal elements in the synram
        for stateNeuronID in range(self.nStates):
            syn = self.chip.synram.get(dls.Synapse_row(stateNeuronID), dls.Synapse_column(stateNeuronID))
            syn.weight(63)  # 6bit max weight to ensure spiking
            syn.address(self.recurrentSpikeAddress)  # address
            self.chip.synram.set(dls.Synapse_row(stateNeuronID), dls.Synapse_column(stateNeuronID), syn)

        #This makes up the off-diagonal elements
        for stateNeuronID in range(self.nStates):
            neuronID = stateNeuronID + 1

            if neuronID == self.nStates:
                neuronID = 0
            
            syn = self.chip.synram.get(dls.Synapse_row(neuronID), dls.Synapse_column(stateNeuronID))
            syn.weight(63)  # 6bit max weight to ensure spiking
            syn.address(stateNeuronID)  # address
            self.chip.synram.set(dls.Synapse_row(neuronID), dls.Synapse_column(stateNeuronID), syn)
            
        #The action neurons get fully connected to the state neurons
        for actionNeuron in range(self.nActions):
            actionNeuronID = self.nStates + actionNeuron
            for stateNeuronID in range(self.nStates):                
                syn = self.chip.synram.get(dls.Synapse_row(stateNeuronID), dls.Synapse_column(actionNeuronID))
                
                if weightMatrix == []:
                    syn.weight(20)  # 6bit #initialise the weights somewhere in the middle
                else:
                    print str(stateNeuronID) + ' | ' + str(actionNeuronID) + ':' + str(weightMatrix[stateNeuronID][actionNeuronID])
                    syn.weight(weightMatrix[stateNeuronID][actionNeuronID])
                syn.address(self.recurrentSpikeAddress)  # address
                self.chip.synram.set(dls.Synapse_row(stateNeuronID), dls.Synapse_column(actionNeuronID), syn)
                self.chip.rate_counter.enable(dls.Neuron_index(actionNeuronID), True)

        #Configure the action synapse drivers as inhibitory
        for rest in range(self.nStates, 32):
            self.chip.syndrv_config.senx(dls.Synapse_row(rest), False)
            self.chip.syndrv_config.seni(dls.Synapse_row(rest), True)

        #Configure inhibitory weights for action neurons
        #Here they are only inhibitory for the state neurons
        for actionNeuron in range(self.nActions):
            actionNeuronID = self.nStates + actionNeuron
            for neuronID in range(self.nStates):# + self.nActions):
                syn = self.chip.synram.get(dls.Synapse_row(actionNeuronID), dls.Synapse_column(neuronID))
                syn.weight(63)  # 6bit #initialise the weights somewhere in the middle
                syn.address(self.recurrentSpikeAddress)  # address
                self.chip.synram.set(dls.Synapse_row(actionNeuronID), dls.Synapse_column(neuronID), syn)  
        
        #Set the pulse length to 1
        self.chip.syndrv_config.pulse_length(1)
    
        #For now leave the first neuron empty to not mix up external inputs with state neuron spikes
        #Enable the output config for the state and action neurons
        for neuronID in range(self.nStates + self.nActions):
            neuron = self.chip.neurons.get(dls.Neuron_index(neuronID))
            neuron.enable_out(True)
            self.chip.neurons.set(dls.Neuron_index(neuronID), neuron)

        self.chip.rate_counter.clear_on_read(True)
            
        '''Think about addressing and timing of 'recurrent' spikes'''
        self.fpga_conf = dls.Config_reg()
        self.fpga_conf.spike_router_enable = True
        self.router = dls.Spike_router_bypass(self.recurrentSpikeCumulationTime, self.recurrentSpikeAddress)

        pass
    
    def loadConfig(self):
        '''Load the configs'''
        
        if not self.useCalibration:
            
            if dls.get_allocated_board_ids()[0] == 'B291698':
                with open('cap_mem_29.json', 'r') as f:
                    self.capmem_defaults = json.load(f)

                with open('dac_default.json', 'r') as f:
                    self.dac_default = json.load(f)

                # set config for state neuron
                for neuron_ind in range(dls.Neuron_index.num_neurons):
                    for k, v in self.capmem_defaults['neuron_params'][neuron_ind].items():
                        key = dict_conv.conversion_dict[k]
                        hp.fill_cap_mem_cell(self.chip.cap_mem, neuron_ind, key, v)
                self.chip.cap_mem.set(dls.Cap_mem_row(0),
                                 dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                                 self.capmem_defaults['global_params']['v_reset'])

            else:
                with open('calibration_20.json') as f:
                    self.capmem_defaults = json.load(f)
            
                with open('dac_07_chip_20.json', 'r') as f:
                    self.dac_default = json.load(f)

                # set config for state neuron
                for neuron_ind in range(dls.Neuron_index.num_neurons):
                    for k, v in self.capmem_defaults['neuron_params'][neuron_ind].items():
                        key = dict_conv.conversion_dict[k]
                        hp.fill_cap_mem_cell(self.chip.cap_mem, neuron_ind, key, v)
                self.chip.cap_mem.set(dls.Cap_mem_row(0),
                                 dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                                 self.capmem_defaults['global_params']['v_reset'])
            
                # set config for state neuron
                for stateNeuron in range(self.nStates):
                    for k, v in self.capmem_defaults['neuron_params'][stateNeuron].items():
                        key = dict_conv.conversion_dict[k]
                        if k == 'i_refr':
                            v = 1022  # minimal refractory period
                        # elif k == 'v_syn_in':
                        #     v = 1022  # maximal inhibition
                        elif k == 'v_thresh':
                            v = self.capmem_defaults['neuron_params'][31]['v_leak'] + 100
                        hp.fill_cap_mem_cell(self.chip.cap_mem, stateNeuron, key, v)
                
                pass
            
        else:
            path = ''
            if dls.get_allocated_board_ids()[0] == 'B291698':
                path = 'configs/board_29/calibration.json'
            elif dls.get_allocated_board_ids()[0] == '07':
                path = 'configs/board_07/calibration.json'
        
            with open(path) as f:
                self.calibrated_config = json.load(f)
            
            #Set the calibration for the action neurons
            for neuron_ind in range(32):
                for k, v in self.calibrated_config['neuron_params'][neuron_ind].items():
                    key = dict_conv.conversion_dict[k]
                    hp.fill_cap_mem_cell(self.chip.cap_mem, neuron_ind, key, v)
            
            #Set the configuration for the state neurons. The state neurons should have the standard
            #Spiking behavior
            state_neuron_calib = dict_conv.convert_from_camel_case(self.capmem_defaults)
            for stateNeuronID in range(self.nStates):
                for k, v in state_neuron_calib.items():
                    key = dict_conv.conversion_dict[k]
                    hp.fill_cap_mem_cell(self.chip.cap_mem, stateNeuronID, key, v)
             
    def loadPPUProgram(self):
        '''Load the PPU program'''

        builder = dls.Dls_program_builder()  # capmem einpendeln
        builder.set_time(0)
        builder.set_chip(self.chip)
        builder.wait_for(100000)
        builder.halt()
        
        # Load the program
        program = dls.Ppu_program()
        program.read_from_file("PPU/bin/mdp.raw")

        #Prepare the mailbox
        mailbox = dls.Mailbox()
        utils.load_mailbox(mailbox, "PPU/MailboxContentTwoStep")

        # Setup synram control register
        # These are magic numbers which configure the timing how the synram is
        # written.
        synram_config_reg = dls.Synram_config_reg()
        synram_config_reg.pc_conf(1)
        synram_config_reg.w_conf(1)
        synram_config_reg.wait_ctr_clear(1)

        # PPU control register
        ppu_control_reg_start = dls.Ppu_control_reg()
        ppu_control_reg_start.inhibit_reset(True)

        ppu_control_reg_end = dls.Ppu_control_reg()
        ppu_control_reg_end.inhibit_reset(False)

        # Playback memory program
        spike_builder = dls.Dls_program_builder()
        spike_builder.set_synram_config_reg(synram_config_reg)
        spike_builder.set_mailbox(mailbox)
        spike_builder.set_ppu_program(program)
        spike_builder.set_ppu_control_reg(ppu_control_reg_end)
        spike_builder.set_ppu_control_reg(ppu_control_reg_start)
        spike_builder.set_time(0)
        #spike_builder.wait_for(1000)
        #spike_builder.fire(2**32 - 1, 0)
        for i in range(1):
            spike_builder.wait_for(10**8)
        status_handle = spike_builder.get_ppu_status_reg()
        spike_builder.set_ppu_control_reg(ppu_control_reg_end)
        mailbox_handle = spike_builder.get_mailbox()
        synram_handle = spike_builder.get_synram()
        spike_builder.halt()

        return builder, spike_builder, mailbox_handle, synram_handle, status_handle
    
    def GenerateProblemMatrices(self, symmetric = True):
        
        P = np.zeros((self.nActions, self.nStates, self.nStates))
        
        #Edit the State-matrix
        P[0][0][0] = 0.
        P[1][0][0] = 0.
        P[0][0][1] = np.random.random()
        P[0][0][2] = 1. - P[0][0][1]

        if symmetric:
            P[1][0][2] = P[0][0][1]
            P[1][0][1] = P[0][0][2]
        else:
            P[1][0][2] = np.random.random()
            P[1][0][1] = 1. - P[1][0][2]

        P[0][1][0] = 1.
        P[0][1][1:3] = 0.
        P[1][1][0] = 1.
        P[1][1][1:3] = 0.
        P[0][2][0] = 1.
        P[0][2][1:3] = 0.
        P[1][2][0] = 1.
        P[1][2][1:3] = 0.

        #Edit the reward matrix
        R = np.zeros((self.nActions, self.nStates, self.nStates))
        
        #draw random rewards for the last transitions
        R[0][1][0] = np.random.random()
        R[1][1][0] = 1. - R[0][1][0]

        if symmetric:
            R[1][2][0] = R[0][1][0]
            R[0][2][0] = R[1][1][0]
        else:
            R[0][2][0] = np.random.random()
            R[1][2][0] = 1. - R[0][2][0]

        return P, R
        
    def Run(self, gamma, lam, eta, maxIteration, hdf=True, use32BitParams=False, weightUpper=None, weightLower=None):

        while True:

            path = "HDFs/TwoStep/TwoStep_" + dt.datetime.now().strftime('%Y_%m_%d_%H-%M-%S') + ".hdf5"
            cnt = 0
            #try:
            #Create an hdf5 file to store the results of the execution
            if hdf:
                hdf5File = h5py.File(path, "w")

            #Generate the MDP problem
            self.P, self.R = self.GenerateProblemMatrices()

            self.P = np.array([[[0., 0.7, 0.3], [1., 0., 0.], [1., 0., 0.]],
                               [[0., 0.3, 0.7], [1., 0., 0.], [1., 0., 0.]]])
                
            self.R = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])

            #Sanity check
            if self.P.shape != (self.nActions, self.nStates, self.nStates) or self.R.shape != (self.nActions, self.nStates, self.nStates):
                raise Exception('The MDP matrices are malformed')
        
            self.R = (self.R + 1) / 2.0
            if hdf:
                hdf5File.create_dataset('P', data=self.P)
                hdf5File.create_dataset('R', data=self.R)
            
            rndSeed = np.random.randint(2**32)

            #The floating point numbers can take values between +- 65536
            maxPValue = (2**16) - 1
            currentMax = self.P.max()
            self.P = self.P * (maxPValue / currentMax)

            #Prepare the mailbox file
            with open("PPU/MailboxContentTwoStep", 'wb') as f:
                f.write(struct.pack('>I', rndSeed))
                flat_list_P = [np.uint32(it) for sublist in self.P.tolist() for item in sublist for it in item]
                flat_list_R = [np.int8(frac.to_fractional(it)) for sublist in self.R.tolist() for item in sublist for it in item]
                f.write(struct.pack('>%uI' % len(flat_list_P), *flat_list_P))
                f.write(struct.pack('>%ub' % len(flat_list_R), *flat_list_R))
                f.write(struct.pack('>I', np.uint32(maxPValue)))
                f.write(struct.pack('>I', np.uint32(2000)))
                
                #Differentiate whether the 32bit parameters should be used or not
                if use32BitParams:
                    f.write(struct.pack('>I', np.uint32(frac.to_fractional(gamma, precision=32))))
                    f.write(struct.pack('>I', np.uint32(frac.to_fractional(lam, precision=32))))
                    f.write(struct.pack('>I', np.uint32(frac.to_fractional(eta, precision=32))))
                else:
                    f.write(struct.pack('>b', np.int8(frac.to_fractional(gamma))))
                    f.write(struct.pack('>b', np.int8(frac.to_fractional(lam))))
                    f.write(struct.pack('>b', np.int8(frac.to_fractional(eta))))
                
                #In case the weight shift should be used write the additional parameter to the mailbox
                if weightLower != None:
                    f.write(struct.pack('>B', np.uint8(weightLower)))
                    f.write(struct.pack('>B', np.uint8(weightUpper)))
    
            
            #load the config
            self.loadConfig()

        	#Create the network structure
            self.CreateNetwork()

            if hdf:
                #Write the important parameters into the hdf5 file
                hdf5File.create_dataset('date', data=(dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ))            
                hdf5File.create_dataset('gamma', data=(gamma, ))
                hdf5File.create_dataset('lam', data=(lam, ))
                hdf5File.create_dataset('eta', data=(eta, ))
                hdf5File.create_dataset('trialsPerIteration', data=(2000, ))
                hdf5File.create_dataset('boardID', data=(dls.get_allocated_board_ids()[0], ))
                hdf5File.create_dataset('CapMemKeys', data=[str(i) for i in self.capmem.keys()])
                hdf5File.create_dataset('CapMemValues', data=[str(i) for i in self.capmem.values()])
                hdf5File.create_dataset('DACConfKeys', data=[str(i) for i in self.dac.keys()])
                hdf5File.create_dataset('DACConfValues', data=[str(i) for i in self.dac.values()])
                hdf5File.create_dataset('nActions', data=(self.nActions, ))
                hdf5File.create_dataset('nStates', data=(self.nStates, ))
                hdf5File.create_dataset('platform', data=('HW', ))
                hdf5File.create_dataset('multipleRuns', data=(self.multipleRuns, ))
                hdf5File.create_dataset('iterationsOnChip', data=((maxIteration / 2000) + 1, ))
                if weightLower != None:
                    hdf5File.create_dataset('weightLower', data=(weightLower, ))
                    hdf5File.create_dataset('weightUpper', data=(weightUpper, ))

            #Start the measurement
            startTime = time.time()
        
            self.builder, self.spikes_builder, self.mailbox_handle, self.synram_handle, self.status_handle = self.loadPPUProgram()

            with dls.connect(dls.get_allocated_board_ids()[0]) as c:
                dls.soft_reset(c)
                dls.set_config_reg(c, self.fpga_conf)  # chip config is only with chip, this is FPGA config
                hp.setup_dac(c, self.dac_default)  # soft reset ? resets dac?
                dls.set_spike_router(c, self.router)  # set spike router
            
                self.builder.transfer(c, 0)  # connection, 0 is program address
                self.builder.execute(c, 0)  # triggers execution on FPGA
                self.builder.fetch(c)
                
                for run in range(self.multipleRuns):
                  group = hdf5File.create_group('Run_' + str(run))
                  accumulatedStatesPerRun = []
	                accumulatedActionsPerRun = []
	                accumulatedRewardsPerRun = []
	                weights = []
                #Don't change the network structure and just reupload the PPU program
	                for i in range((maxIteration / 2000) + 1):
	                    self.chip = dls.Chip()
	                    self.loadConfig()
	                    self.CreateNetwork(weights)
	                    self.builder, self.spikes_builder, self.mailbox_handle, self.synram_handle = self.loadPPUProgram()
	                    self.builder.transfer(c, 0)  # connection, 0 is program address
	                    self.builder.execute(c, 0)  # triggers execution on FPGA
	                    self.builder.fetch(c)
	                    self.spikes_builder.transfer(c, 0)
	                    self.spikes_builder.execute(c, 0)
	                    self.spikes_builder.fetch(c)
                    
                    #Switch the spike router off and perform the readout
                    #self.fpga_conf = dls.Config_reg()
                    #self.fpga_conf.spike_router_enable = False
                    #dls.set_config_reg(c, self.fpga_conf)

                    #Evaluate the mailbox for states, actions weights and compute the rewards
	                    spike_times, spike_address, states, actions, rewards, policy, weights, Q_table = self.EvaluateNetwork(2000, verbose=False)
	                    if hdf:
	                        group.create_dataset('spikeTimes' + str(i), data=spike_times)
	                        group.create_dataset('spikeAddresses' + str(i), data=spike_address)
	                        group.create_dataset('states' + str(i), data=states)
	                        group.create_dataset('actions' + str(i), data=actions)
	                        group.create_dataset('rewards' + str(i), data=rewards)
	                        group.create_dataset('policy' + str(i), data=policy)
	                        group.create_dataset('weights' + str(i), data=weights)
	                        group.create_dataset('Qtable' + str(i), data=Q_table)

											accumulatedStatesPerRun.extend(states)
	                		accumulatedActionsPerRun.extend(actions)
	                		accumulatedRewardsPerRun.extend(rewards)
                  self.accumulatedStates.extend(states)
                  self.accumulatedActions.extend(actions)
                  self.accumulatedRewards.extend(rewards)
                    
                    #Switch the spike router back on
                    #self.fpga_conf = dls.Config_reg()
                    #self.fpga_conf.spike_router_enable = True
                    #self.router = dls.Spike_router_bypass(self.recurrentSpikeCumulationTime, self.recurrentSpikeAddress)
                    #dls.set_spike_router(c, self.router)
                    
            if hdf:
                hdf5File.create_dataset('simulationTime', data=(time.time() - startTime, ))
                print('Closing file')
                hdf5File.close()
            return
        
            #except:
            #    print 'Network problems'

            #    if cnt > 4:
            #        exit()
                
            #    #Delete the file
            #    os.remove(path)

            #    time.sleep(5)
            #    cnt += 1
    
    def EvaluateNetwork(self, maxIteration, verbose=True):
        '''Collect the results and evaluate the network'''
        
        #Print the spikes
        spike_train = self.spikes_builder.get_spikes()

        spike_times = []
        spike_address = []
        for spike in spike_train:
            spike_times.append(spike.time)
            spike_address.append(spike.address)

        #print str(len(spike_train)) + ' were sent'

        mailbox_result = self.mailbox_handle.get()

        if verbose:        
            for spike in spike_train:
                print 'Adr ' + str(spike.address) + ' Time: ' + str(spike.time) 

            #Print the mailbox content
            if False:
                utils.print_mailbox_string(mailbox_result)
            else:
                utils.print_mailbox(mailbox_result)

        #Read the mailbox and collect the results
        stateOffset = 0x000
        actionOffset = 0x800
        iterationCounterOffset = 0xffc
    
        states = utils.convertByteListToInt8(utils.readRange_mailbox(mailbox_result, stateOffset, stateOffset + maxIteration), False)
        actions = utils.convertByteListToInt8(utils.readRange_mailbox(mailbox_result, actionOffset, actionOffset + maxIteration), False)
        iterationCounter = utils.convertByteListToInt(utils.readRange_mailbox(mailbox_result, iterationCounterOffset, iterationCounterOffset + 4), False)

        #if iterationCounter[0] != 2000:
        #    print 'Iteration counter was wrong! ' + str(iterationCounter[0])
        #    raise Exception('Iteration counter was wrong!')
        
        #Compute the rewards based on the state and action pair
        rewards = [0]
        for i in range(1, len(states)):
            rewards.append(self.R[actions[i-1]][states[i-1]][states[i]])

        if verbose:
            print 'Executed iterations: ' + str(iterationCounter[0])

            print rewards

        #Read the qvalues directly from the weights (synram)
        synram = self.synram_handle.get()
        #Discard the first line, since this just contains the input information

        policy = []
        Q_table = []
        for state in range(self.nStates):
            weights = [synram.get(dls.Synapse_row(state), dls.Synapse_column(self.nStates + actionNeuron)).weight() for actionNeuron in range(self.nActions)]
            Q_table.append(weights)
            maxQIndex = np.argmax(weights)         
            policy.append(maxQIndex)

        if verbose:
            print policy

        weights = []
        for row in range(32):
            rowStr = ''
            weights.append([synram.get(dls.Synapse_row(row), dls.Synapse_column(col)).weight() for col in range(32)])            
            for col in range(32):
                rowStr += '{:2d}, '.format(synram.get(dls.Synapse_row(row), dls.Synapse_column(col)).weight())
            
            if verbose:    
                print rowStr

        weights = np.array(weights)
         
        return spike_times, spike_address, states, actions, rewards, policy, weights, Q_table

    def ComputeFitness(self):
        averageCumregret = EvaluateMDP.averageCumRegret(self.accumulatedStates, self.accumulatedActions, self.accumulatedRewards, self.R, self.multipleRuns)
        s = (np.sum(averageCumregret), )
        print s
        return s

    
if __name__ == '__main__':
    
    # activate logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    log = pylogging.get("main")
    
    ####################
    #### Parameters ####
    ####################
    nStates = 3
    nActions = 2
    gamma = 0.95 # discount factor
    lam = 0.12
    eta = 0.012
    maxIteration = 1999
    useCalibration = False
    multipleRuns = 1
    
    if nStates + nActions > 32:
        raise Exception("The network is too big for the chip")
    
    #It is crucial to create the network before the capmem is uploaded
    network = DLSNetwork(nStates, nActions, useCalibration, multipleRuns)
    
    network.Run(gamma, lam, eta, maxIteration)

    #network.EvaluateNetwork(maxIteration)
