import pydls as dls
import json
#import helpers as hp
import helpers_pydls as hp
import pylogging
import Utils as utils
import numpy as np
import struct
import fractional as frac
import pylab
import matplotlib.pyplot as plt
import h5py
import datetime as dt
import dict_conv
import os
import EvaluateMaze
import time
import MazeGenerator2 as Generator
import sys
import io
import traceback

class DLSNetwork:
    
    def __init__(self, nStates, nActions, multipleRuns):
        
        self.nStates = nStates
        self.nActions = nActions
        
        self.recurrentSpikeAddress = 50
        self.recurrentSpikeCumulationTime = 100
        self.chip = dls.Chip()
        self.multipleRuns = multipleRuns
        self.winCounter = 0
        self.inhibitory = None

        boardCalibMapping = {'B291698' : {'dac' : 'dac_default.json',
        								  'cap' : 'cap_mem_29.json'},
							 '07' : {'dac' : 'dac_07_chip_20.json',
								     'cap' : 'calibration_20.json'},
							 'B201319' : {'dac' : 'dac_B201319_chip_21.json',
										  'cap' : 'calibration_24.json'},
							 'B201330' : {'dac' : 'dac_B201330_chip_22.json',
										  'cap' : 'calibration_22.json'}}
        													        	
        f = open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['cap'], 'r')
        self.capmem = json.load(f)
        f.close()

        f = open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['dac'], 'r')
        self.dac = json.load(f)
        f.close()

        self.ConnectToDLS()

    def ConnectToDLS(self):

        self.chip = dls.Chip()
        self.loadConfig()
        self.CreateNetwork()

        builder = dls.Dls_program_builder()  # capmem einpendeln
        builder.set_time(0)
        builder.set_chip(self.chip)
        builder.wait_for(100000)
        builder.halt()

        #Connect to DLS and keep connection open
        self.connection = dls.connect(dls.get_allocated_board_ids()[0])
   
        dls.soft_reset(self.connection)
        dls.set_config_reg(self.connection, self.fpga_conf)  # chip config is only with chip, this is FPGA config
        hp.setup_dac(self.connection, self.dac)  # soft reset ? resets dac?
        dls.set_spike_router(self.connection, self.router)  # set spike router

        builder.transfer(self.connection, 0)  # connection, 0 is program address
        builder.execute(self.connection, 0)  # triggers execution on FPGA
        builder.fetch(self.connection)
        
    def CreateNetwork(self, weightMatrix=[], exitatory=None, inhibitory=None):
        '''Wire the network'''
        
        #Network layout
        
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
            #if actionNeuron != 2:
            #    continue
            actionNeuronID = self.nStates + actionNeuron
            for stateNeuronID in range(self.nStates):                
                syn = self.chip.synram.get(dls.Synapse_row(stateNeuronID), dls.Synapse_column(actionNeuronID))
                
                if weightMatrix == []:
                    if exitatory != None:
                        syn.weight(exitatory)  # 6bit #initialise the weights somewhere in the middle
                    else:
                        syn.weight(35)
                else:
                    print str(stateNeuronID) + ' | ' + str(actionNeuronID) + ':' + str(weightMatrix[stateNeuronID][actionNeuronID])
                    syn.weight(weightMatrix[stateNeuronID][actionNeuronID])

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
            for neuronID in range(self.nStates):
                syn = self.chip.synram.get(dls.Synapse_row(actionNeuronID), dls.Synapse_column(neuronID))
                syn.weight(63)
                syn.address(self.recurrentSpikeAddress)  # address
                self.chip.synram.set(dls.Synapse_row(actionNeuronID), dls.Synapse_column(neuronID), syn)

            for neuronID in range(self.nStates, self.nStates + self.nActions):
                syn = self.chip.synram.get(dls.Synapse_row(actionNeuronID), dls.Synapse_column(neuronID))
                if inhibitory != None or self.inhibitory != None:
                    if self.inhibitory == None:
                        self.inhibitory = inhibitory
                    syn.weight(self.inhibitory)# 6bit #initialise the weights somewhere in the middle
                else:
                    syn.weight(63)
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
            
    def loadPPUProgram(self, mailbox_file):
        '''Load the PPU program'''

        builder = dls.Dls_program_builder()  # capmem einpendeln
        builder.set_time(0)
        builder.set_chip(self.chip)
        builder.wait_for(100000)
        builder.halt()
        
        # Load the program
        program = dls.Ppu_program()
        program.read_from_file("PPU/bin/maze.raw")

        #Prepare the mailbox
        mailbox = dls.Mailbox()
        #utils.load_mailbox(mailbox, "PPU/MailboxContentMaze")
        utils.load_mailbox_from_virtual_file(mailbox, mailbox_file)

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
        
    def Run(self, gamma, lam, eta, maxIteration, hdf=True, use32BitParams=False, weightUpper=None, weightLower=None, rescaleFreq=None, verbose=True, exitatory=None, inhibitory=None):

        self.chip = dls.Chip()
        self.inhibitory = None

        cnt = 0
        while True:

            rn = np.random.randint(2**32)
                
            path = "HDFs/Maze/Maze_" + dt.datetime.now().strftime('%Y_%m_%d_%H-%M-%S') + str(rn) + ".hdf5"    
            
            try:
                #Set the win counter to 0 every time a exception occurs
                self.winCounter = 0

                if len(sys.argv) == 2:
                    #A file is given. Perform the network with the same R and P
                    hdf5File = h5py.File(sys.argv[1],'r')
                    envmatrix = np.array(hdf5File['maze'][:])
                    #maze[:, :] = maze[::-1, ::-1]
                    self.goalStateX = hdf5File['goalX'][0]
                    self.goalStateY = hdf5File['goalY'][0]
                    
                    #Check if the file is a SW or a HW simulation
                    if hdf5File['platform'][0] == 'SW':
                        self.maze = EvaluateMaze.RemoveWalls(envmatrix)
                        self.goalStateX -= 1
                        self.goalStateY -= 1
                    else:
                        self.maze = envmatrix

                    self.goalState = (int(self.goalStateX), int(self.goalStateY))
                else:
                    self.maze, self.goalState = Generator.GenerateMaze(2, 2, withoutWall=True)
                    self.goalStateX = self.goalState[0]
                    self.goalStateY = self.goalState[1]

                #Create an hdf5 file to store the results of the execution
                if hdf:
                    hdf5File = h5py.File(path, "w")

                #Generate the maze
                # x | down
                # y ->
                #self.maze = np.array([[0, 0, 0],
                #        	          [0, 1, 0],
                #                      [0, 1, 0]])

                #self.gridRow, self.gridCol = self.maze.shape

                #self.goalStateX = 2
                #self.goalStateY = 0
                #self.goalState = self.goalStateX * self.gridCol + self.goalStateY

                rndSeed = np.random.randint(2**32)

                f = io.BytesIO()

                #Prepare the mailbox file
                #with open("PPU/MailboxContentMaze", 'wb') as f:
                f.write(struct.pack('>I', rndSeed))
                flat_list = [item for sublist in self.maze for item in sublist]
                f.write(struct.pack('>%uB' % len(flat_list), *flat_list))
                f.write(struct.pack('>B', np.uint8(self.goalStateX)))
                f.write(struct.pack('>B', np.uint8(self.goalStateY)))
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
                    f.write(struct.pack('>I', np.uint32(rescaleFreq)))
                
                #load the config
                self.loadConfig()

                #Create the network structure
                self.CreateNetwork()

                if hdf:
                    #Write the important parameters into the hdf5 file
                    hdf5File.create_dataset('date', data=(dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), ))
                    hdf5File.create_dataset('maze', data=self.maze)
                    hdf5File.create_dataset('goalX', data=(self.goalStateX, ))
                    hdf5File.create_dataset('goalY', data=(self.goalStateY, ))
                    hdf5File.create_dataset('gamma', data=(gamma, ))
                    hdf5File.create_dataset('lam', data=(lam, ))
                    hdf5File.create_dataset('eta', data=(eta, ))
                    hdf5File.create_dataset('trialsPerIteration', data=(2000, ))
                    hdf5File.create_dataset('boardID', data=(dls.get_allocated_board_ids()[0], ))
                    hdf5File.create_dataset('CapMemKeys', data=[str(i) for i in self.capmem.keys()])
                    hdf5File.create_dataset('CapMemValues', data=[str(i) for i in self.capmem.values()])
                    hdf5File.create_dataset('DACConfKeys', data=[str(i) for i in self.dac.keys()])
                    hdf5File.create_dataset('platform', data=('HW', ))
                    hdf5File.create_dataset('DACConfValues', data=[str(i) for i in self.dac.values()])
                    hdf5File.create_dataset('multipleRuns', data=(self.multipleRuns, ))
                    hdf5File.create_dataset('iterationsOnChip', data=((maxIteration / 2000) + 1, ))
                    if weightLower != None:
                        hdf5File.create_dataset('weightLower', data=(weightLower, ))
                        hdf5File.create_dataset('weightUpper', data=(weightUpper, ))
                        hdf5File.create_dataset('rescaleFreq', data=(rescaleFreq, ))

                #Start the measurement
                startTime = time.time()
            
                self.builder, self.spikes_builder, self.mailbox_handle, self.synram_handle, self.status_handle = self.loadPPUProgram(f.getvalue())

                #Handle the case when more than 2000 iterations are needed. Dont disconnect from the board,
                #just delete the mailbox and start over again with the last weights
                
                #with dls.connect(dls.get_allocated_board_ids()[0]) as c:
                #dls.soft_reset(c)
                #dls.set_config_reg(c, self.fpga_conf)  # chip config is only with chip, this is FPGA config
                #hp.setup_dac(c, self.dac)  # soft reset ? resets dac?
                #dls.set_spike_router(c, self.router)  # set spike router

                self.builder.transfer(self.connection, 0)  # connection, 0 is program address
                self.builder.execute(self.connection, 0)  # triggers execution on FPGA
                self.builder.fetch(self.connection)
                
                for run in range(self.multipleRuns):

                    if hdf:
                        #Create a group for each new run
                        group = hdf5File.create_group('Run_' + str(run))

                    weights = []       
                    #Don't change the network structure and just reupload the PPU program
                    for i in range((maxIteration / 2000) + 1):
                        
                        #load the config
                        self.chip = dls.Chip()
                        self.loadConfig()
                        self.CreateNetwork(weights)
                        self.builder, self.spikes_builder, self.mailbox_handle, self.synram_handle, self.status_handle = self.loadPPUProgram(f.getvalue())

                        self.builder.transfer(self.connection, 0)  # connection, 0 is program address
                        self.builder.execute(self.connection, 0)  # triggers execution on FPGA
                        self.builder.fetch(self.connection)

                        self.spikes_builder.transfer(self.connection, 0)
                        self.spikes_builder.execute(self.connection, 0)
                        self.spikes_builder.fetch(self.connection)

                        #Read and store the mailbox, synram content and the spike trains
                        spike_times, spike_address, states, actions, self.policy, weights, Q_table, winCounter = self.EvaluateNetwork(2000, verbose=verbose)
                        if hdf:
                            group.create_dataset('spikeTimes' + str(i), data=spike_times)
                            group.create_dataset('spikeAddresses' + str(i), data=spike_address)
                            group.create_dataset('states' + str(i), data=states)
                            group.create_dataset('actions' + str(i), data=actions)
                            group.create_dataset('policy' + str(i), data=self.policy)
                            group.create_dataset('weights' + str(i), data=weights)
                            group.create_dataset('Qtable' + str(i), data=Q_table)
                            group.create_dataset('Wins' + str(i), data=(winCounter,))

                        self.winCounter += int(winCounter)
	                        
                if hdf:
                    hdf5File.create_dataset('simulationTime', data=(time.time() - startTime, ))
                    print('Closing file')
                    hdf5File.close()

                f.close()
            
                return
        
            except Exception as e:
                print 'Network problems ' + str(cnt)
                print e
                traceback.print_exc()

                if hdf:
                    os.remove(path)

                if cnt >= 20:
                    exit()

                time.sleep(5)
                cnt += 1

                #Reconnect to chip again
                self.connection.disconnect()
                self.ConnectToDLS()
            

    def PlotPolicy(self, ax):

        #Maze is flipped due to environment N down S up
        def next_state(state, a):
            if a == 0:
                return state + self.gridCols
            elif a == 1:
                return state + 1
            elif a == 2:
                return state - self.gridCols 
            elif a == 3:
                return state - 1
            else:
                raise ValueError('Unkown action {}'.format(a))
            
        def transformState(state):
            
            x = int(state % self.gridCols)
            y = int(state / self.gridCols)
            
            return x, y

        ax.pcolor(self.maze, cmap=plt.cm.jet)
        
        x, y = transformState(self.goalState)
        ax.annotate('G',xy=(x + .2, y + 0.2),color='black',size=25)
        
        for state in range(self.nStates):
            
            x, y = transformState(state)
            
            #Fill The wall elements
            if self.maze[x, y] == 1.:
                ax.fill(x, y, "w")
                continue
            
            if state < self.gridCols or state % self.gridCols == 0 or state % self.gridCols == (self.gridCols - 1) or state >= (self.gridRows - 1) * self.gridCols or state == self.goalState:
                continue
            
            action = np.random.choice(np.where(np.max(self.Q_table[state,:]) == self.Q_table[state,:])[0])
            nextState = next_state(state, action)
            xn, yn = transformState(nextState)
            ax.arrow(x + .5, y + .5, (xn - x), (yn - y), head_width=.1, color='white')
        
        ax.set_title('Policy')
        
    
    def EvaluateNetwork(self, maxIteration, verbose=True):
        '''Collect the results and evaluate the network'''
        
        #Print the spikes
        spike_train = self.spikes_builder.get_spikes()

        spike_times = []
        spike_address = []
        for spike in spike_train:
            spike_times.append(spike.time)
            spike_address.append(spike.address)

        #Print the mailbox content
        mailbox_result = self.mailbox_handle.get()
        status = self.status_handle.get()

        if status.sleep() != True:
            print 'PPU did not finish!'

        if verbose:
            for spike in spike_train:
                print spike

            if False:
                utils.print_mailbox_string(mailbox_result)
            else:
                utils.print_mailbox(mailbox_result)

        synram = self.synram_handle.get()
        for row in range(32):
            rowStr = ''        
            for col in range(32):
                rowStr += '{:2d}, '.format(synram.get(dls.Synapse_row(row), dls.Synapse_column(col)).weight())
            
            print rowStr
        utils.print_mailbox(mailbox_result)

        #Read the mailbox and collect the results
        stateOffset = 0x000
        actionOffset = 0x800
        iterationCounterOffset = 0xffc
        winOffset = 0xff0;
    
        states = utils.convertByteListToInt8(utils.readRange_mailbox(mailbox_result, stateOffset, stateOffset + maxIteration), False)
        actions = utils.convertByteListToInt8(utils.readRange_mailbox(mailbox_result, actionOffset, actionOffset + maxIteration), False)
        iterationCounter = utils.convertByteListToInt(utils.readRange_mailbox(mailbox_result, iterationCounterOffset, iterationCounterOffset + 4), False)
        winCounter = utils.convertByteListToInt(utils.readRange_mailbox(mailbox_result, winOffset, winOffset + 4), False)

        if iterationCounter[0] != 2000:
            print 'Iteration counter was wrong! ' + str(iterationCounter[0])
            raise Exception('Iteration counter was wrong!')

        #Read the qvalues directly from the weights (synram)
        synram = self.synram_handle.get()
        #Discard the first line, since this just contains the input information

        policy = []
        self.Q_table = []
        for state in range(self.nStates):
            weights = [synram.get(dls.Synapse_row(state), dls.Synapse_column(self.nStates + actionNeuron)).weight() for actionNeuron in range(self.nActions)]
            self.Q_table.append(weights)
            maxQIndex = np.argmax(weights)
            policy.append(maxQIndex)

        self.Q_table = np.array(self.Q_table)

        if verbose:
            print 'Executed iterations: ' + str(iterationCounter[0])
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

        return spike_times, spike_address, states, actions, policy, weights, self.Q_table, winCounter[0]
    
    def ComputeFitness(self):

        self.maze = EvaluateMaze.ExpandMaze(self.maze)
        
        #s = EvaluateMaze.ComparePolicy(self.maze, (self.goalStateX + 1, self.goalStateY + 1), self.nStates, self.nActions, toComparePolicy=self.policy, platform=1, plot=False, fixedPolicy=np.array([0., 0., 0., 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.]))
        #print s
        #return (s,)

        print (self.winCounter)               
        return (-self.winCounter,)

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

    nStates = 9
    nActions = 4
    nInhibit = 10
    maxIteration = 5999
    multipleRuns = 1

    eta = 0.9
    exitatory = 0.518610724013
    gamma = 0.4
    inhibitory = 0.7
    lam = 0.292608258461
    rescaleFreq = 0.53706116852
    weightLower = 0.5
    weightUpper = 0.5

    weightLower = np.int64(63 * weightLower)
    weightUpper = np.int64(63 * weightUpper)
    rescaleFreq = np.int64(2000 * rescaleFreq)
    exitatory = np.int64(63 * exitatory)
    inhibitory = np.int64(63 * inhibitory)

    use32BitParams = False
    
    if nStates + nActions > 32:
        raise Exception("The network is too big for the chip")
    
    #It is crucial to create the network before the capmem is uploaded
    network = DLSNetwork(nStates, nActions, multipleRuns)
    
    network.Run(gamma, lam, eta, maxIteration, use32BitParams=use32BitParams, weightLower=weightLower, weightUpper=weightUpper, rescaleFreq=rescaleFreq, exitatory=exitatory, inhibitory=inhibitory)

    network.EvaluateNetwork(maxIteration)

    network.ComputeFitness()

    #fig, ax = plt.subplots()
    #network.PlotPolicy(ax)
    #plt.show()
    #plt.savefig('Maze.png')
