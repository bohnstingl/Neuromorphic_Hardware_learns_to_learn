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
        counti = 0
        for actionNeuron in range(self.nActions):
            #if actionNeuron != 2:
            #    continue
            
            actionNeuronID = self.nStates + actionNeuron
            for stateNeuronID in range(self.nStates):                
                syn = self.chip.synram.get(dls.Synapse_row(stateNeuronID), dls.Synapse_column(actionNeuronID))
                
                if weightMatrix == []:
                    syn.weight(20 + 3 * counti)  # 6bit #initialise the weights somewhere in the middle
                else:
                    print str(stateNeuronID) + ' | ' + str(actionNeuronID) + ':' + str(weightMatrix[stateNeuronID][actionNeuronID])
                    syn.weight(weightMatrix[stateNeuronID][actionNeuronID])
                counti += 1

                syn.address(self.recurrentSpikeAddress)  # address
                self.chip.synram.set(dls.Synapse_row(stateNeuronID), dls.Synapse_column(actionNeuronID), syn)
                self.chip.rate_counter.enable(dls.Neuron_index(stateNeuronID), False)

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
            #neuron.enable_out(True)
            self.chip.neurons.set(dls.Neuron_index(neuronID), neuron)

        self.chip.rate_counter.clear_on_read(True)
            
        '''Think about addressing and timing of 'recurrent' spikes'''
        self.fpga_conf = dls.Config_reg()
        self.fpga_conf.spike_router_enable = True
        self.router = dls.Spike_router_bypass(self.recurrentSpikeCumulationTime, self.recurrentSpikeAddress)

        pass
    
    def loadConfig(self):
        '''Load the configs'''
        
        boardCalibMapping = {'B291698' : {'dac' : 'dac_default.json',
        								  'cap' : 'cap_mem_29.json'},
							 '07' : {'dac' : 'dac_07_chip_20.json',
								     'cap' : 'calibration_20.json'},
							 'B201319' : {'dac' : 'dac_B201319_chip_21.json',
										  'cap' : 'calibration_24.json'},
							 'B201330' : {'dac' : 'dac_B201330_chip_22.json',
										  'cap' : 'calibration_22.json'}}
        													        	
        with open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['cap'], 'r') as f:
            self.capmem = json.load(f)

        with open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['dac'], 'r') as f:
            self.dac = json.load(f)
            
        # set config for state neuron
        for neuron_ind in range(dls.Neuron_index.num_neurons):
            for k, v in self.capmem['neuron_params'][neuron_ind].items():
                key = dict_conv.conversion_dict[k]
                hp.fill_cap_mem_cell(self.chip.cap_mem, neuron_ind, key, v)
        self.chip.cap_mem.set(dls.Cap_mem_row(0),
                         dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                         self.capmem['global_params']['v_reset'])
                         
        if dls.get_allocated_board_ids()[0] != 'B291698':

            # set config for state neurons
            for stateNeuron in range(self.nStates):
                for k, v in self.capmem['neuron_params'][stateNeuron].items():
                    key = dict_conv.conversion_dict[k]
                    if k == 'i_refr':
                        v = 1022  # minimal refractory period
                    # elif k == 'v_syn_in':
                    #     v = 1022  # maximal inhibition
                    elif k == 'v_thresh':
                        v = self.capmem['neuron_params'][stateNeuron]['v_leak'] + 100
                    hp.fill_cap_mem_cell(self.chip.cap_mem, stateNeuron, key, v)
            
    def loadPPUProgram(self):
        '''Load the PPU program'''

        builder = dls.Dls_program_builder()  # capmem einpendeln
        builder.set_time(0)
        builder.set_chip(self.chip)
        builder.wait_for(100000)
        builder.halt()
        
        # Load the program
        program = dls.Ppu_program()
        program.read_from_file("PPU/bin/rate.raw")

        #Prepare the mailbox
        mailbox = dls.Mailbox()
        utils.load_mailbox(mailbox, "PPU/MailboxContentMDP")

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
        #spike_builder.set_mailbox(mailbox)
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
        
    def Run(self, gamma, lam, eta, maxIteration, hdf=True, use32BitParams=False, weightUpper=None, weightLower=None):

        while True:
            path = "HDFs/MDP/MDP_" + dt.datetime.now().strftime('%Y_%m_%d_%H-%M-%S') + ".hdf5"
            cnt = 0
            try:
                #Create an hdf5 file to store the results of the execution
                if hdf:
                    hdf5File = h5py.File(path, "w")

                #Generate the MDP problem
                self.P, self.R = mdptoolbox.example.rand(self.nStates, self.nActions, is_sparse=False)#, rewardRange=(-0.3, -0.8))

                #self.P = np.array([[[1., 0.], [0., 1.]],
                #                   [[1., 0.], [0., 1.]],
                #                   [[0., 1.], [0., 1.]],
                #                   [[1., 0.], [1., 0.]]])
                    
                #self.R = np.array([[[-1., -1.], [-1., -1.]],
                #                   [[-1., -1.], [-1., -1.]],
                #                   [[-1., 1.], [-1., -1.]],
                #                   [[-1., -1.], [1., -1.]]])

                #self.P = np.array([[[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.]],
                #                   [[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]],
                #                   [[0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]],
                #                   [[1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.], [1., 0., 0., 0., 0.]]])
                    
                #self.R = np.array([[[-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., 1.], [-1., -1., -1., -1., -1.]],
                #                   [[-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., 1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.]],
                #                   [[-1., 1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.]],
                #                   [[-1., -1., -1., -1., -1.], [-1., -1., 1., -1., -1.], [-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.], [1., -1., -1., -1., -1.]]])
                
                self.P = np.array([[[1., 0.], [0., 1.]],
                                   [[1., 0.], [0., 1.]],
                                   [[0., 1.], [0., 1.]],
                                   [[1., 0.], [1., 0.]]])
                    
                self.R = np.array([[[-1., -1.], [-1., -1.]],
                                   [[-1., -1.], [-1., -1.]],
                                   [[-1., 1.], [-1., -1.]],
                                   [[-1., -1.], [1., -1.]]]) 

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
                with open("PPU/MailboxContentMDP", 'wb') as f:
                    f.write(struct.pack('>I', rndSeed))
                    flat_list_P = [np.uint32(it) for sublist in self.P.tolist() for item in sublist for it in item]
                    flat_list_R = [np.int8(frac.to_fractional(it)) for sublist in self.R.tolist() for item in sublist for it in item]
                    f.write(struct.pack('>%uI' % len(flat_list_P), *flat_list_P))
                    f.write(struct.pack('>%ub' % len(flat_list_R), *flat_list_R))
                    f.write(struct.pack('>I', np.uint32(maxPValue)))
                    f.write(struct.pack('>I', np.uint32(200)))
                    
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

                #Handle the case when more than 2000 iterations are needed. Dont disconnect from the board,
                #just delete the mailbox and start over again with the last weights
                
                with dls.connect(dls.get_allocated_board_ids()[0]) as c:
                    dls.soft_reset(c)
                    dls.set_config_reg(c, self.fpga_conf)  # chip config is only with chip, this is FPGA config
                    hp.setup_dac(c, self.dac)  # soft reset ? resets dac?
                    dls.set_spike_router(c, self.router)  # set spike router

                    self.builder.transfer(c, 0)  # connection, 0 is program address
                    self.builder.execute(c, 0)  # triggers execution on FPGA
                    self.builder.fetch(c)

                    
                    
                    for run in range(self.multipleRuns):
	
                        if hdf:
                            #Create a group for each new run
                            group = hdf5File.create_group('Run_' + str(run))
                        accumulatedStatesPerRun = []
                        accumulatedActionsPerRun = []
                        accumulatedRewardsPerRun = []

                        weights = []
                        #weights = np.array([[0, 0, 0, 0, 0, 0],
                        #                    [0, 0, 0, 0, 0, 0]])         
                        #Don't change the network structure and just reupload the PPU program
                        for i in range((maxIteration / 2000) + 1):
	                        
                            #load the config
                            self.chip = dls.Chip()
                            self.loadConfig()
                            self.CreateNetwork(weights)
                            self.builder, self.spikes_builder, self.mailbox_handle, self.synram_handle, self.status_handle = self.loadPPUProgram()

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
                            spike_times, spike_address, states, actions, rewards, policy, weights, Q_table = self.EvaluateNetwork(2000, verbose=True)

                            if hdf:
                                group.create_dataset('spikeTimes' + str(i), data=spike_times)
                                group.create_dataset('spikeAddresses' + str(i), data=spike_address)
                                group.create_dataset('states' + str(i), data=states)
                                group.create_dataset('actions' + str(i), data=actions)
                                group.create_dataset('rewards' + str(i), data=rewards)
                                group.create_dataset('policy' + str(i), data=policy)
                                group.create_dataset('weights' + str(i), data=weights)
                                group.create_dataset('Qtable' + str(i), data=Q_table)
	
                            #Add the states, actions and rewards for multiple iterations together
                            accumulatedStatesPerRun.extend(states)
                            accumulatedActionsPerRun.extend(actions)
                            accumulatedRewardsPerRun.extend(rewards)
	                    		
                        #Add the all the states, actions and rewards from a single run together
                        self.accumulatedStates.append(states)
                        self.accumulatedActions.append(actions)
                        self.accumulatedRewards.append(rewards)
	                        
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

            except Exception as e:
                print 'Network problems ' + str(cnt)
                print e

            #    if cnt > 4:
            #        exit()
                
                #Delete the file
            #    if hdf:
            #        os.remove(path)


                exit()

                time.sleep(5)
                cnt += 1
    
    def EvaluateNetwork(self, maxIteration, verbose=True):
        '''Collect the results and evaluate the network'''
        
        #Print the spikes
        spike_train = self.spikes_builder.get_spikes()

        spike_times = []
        spike_address = []
        for spike in spike_train:
            spike_times.append(spike.time)
            spike_address.append(spike.address)
            #print 'A ' + str(spike.address) + ' T: ' + str(spike.time)

        print str(len(spike_train)) + ' were sent'
        #print list(set(spike_address))

        mailbox_result = self.mailbox_handle.get()
        status = self.status_handle.get()

        if status.sleep() != True:
            print 'PPU did not finish!'

        if verbose:        
            for spike in spike_train:
                print 'Adr ' + str(spike.address) + ' Time: ' + str(spike.time) 

            #Print the mailbox content
            if True:
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

        synram = self.synram_handle.get()

        policy = []
        Q_table = []
        for state in range(self.nStates):
            weights = [synram.get(dls.Synapse_row(state), dls.Synapse_column(self.nStates + actionNeuron)).weight() for actionNeuron in range(self.nActions)]
            Q_table.append(weights)
            maxQIndex = np.argmax(weights)         
            policy.append(maxQIndex)

        weights = []
        for row in range(32):
            rowStr = ''
            weights.append([synram.get(dls.Synapse_row(row), dls.Synapse_column(col)).weight() for col in range(32)])            
            for col in range(32):
                rowStr += '{:2d}, '.format(synram.get(dls.Synapse_row(row), dls.Synapse_column(col)).weight())
            
            if verbose:    
                print rowStr

        weights = np.array(weights)

        if verbose:
            print policy
         
        return spike_times, spike_address, states, actions, rewards, policy, weights, Q_table

    def ComputeFitness(self):
        
        #Compute the average over multiple runs
        averageCumregret, _, _ = EvaluateMDP.averageCumRegret(np.array(self.accumulatedStates), np.array(self.accumulatedActions), np.array(self.accumulatedRewards), self.R, self.multipleRuns)
        
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
    nStates = 2
    nActions = 4
    gamma = 0.95#0.568 #0.94
    lam = 0.12
    eta = 0.012#0.754   #0.38
    maxIteration = 1999
    multipleRuns = 1
    weightLower = 10
    weightUpper = 35
    use32BitParams = False
    
    if nStates + nActions > 32:
        raise Exception("The network is too big for the chip")
    
    #It is crucial to create the network before the capmem is uploaded
    network = DLSNetwork(nStates, nActions, multipleRuns)
    
    '''
    #load the config
    network.loadConfig()

    #Create the network structure
    network.CreateNetwork()

    builder, spikes_builder, mailbox_handle, synram_handle = network.loadPPUProgram()

    with dls.connect(dls.get_allocated_board_ids()[0]) as c:
        dls.soft_reset(c)
        dls.set_config_reg(c, network.fpga_conf)  # chip config is only with chip, this is FPGA config
        hp.setup_dac(c, network.dac_default)  # soft reset ? resets dac?
        dls.set_spike_router(c, network.router)  # set spike router

        builder.transfer(c, 0)  # connection, 0 is program address
        builder.execute(c, 0)  # triggers execution on FPGA
        builder.fetch(c)

        #Don't change the network structure and just reupload the PPU program
        spikes_builder.transfer(c, 0)
        spikes_builder.execute(c, 0)
        spikes_builder.fetch(c)

    #Print the spikes
    spike_train = spikes_builder.get_spikes()

    spike_times = []
    spike_address = []
    for spike in spike_train:
        spike_times.append(spike.time)
        spike_address.append(spike.address)
        #print 'A ' + str(spike.address) + ' T: ' + str(spike.time)

    print str(len(spike_train)) + ' were sent'
    print list(set(spike_address))
    exit()
    '''   
 
    network.Run(gamma, lam, eta, maxIteration, use32BitParams=use32BitParams, weightLower=weightLower, weightUpper=weightUpper)

    network.ComputeFitness()
    
