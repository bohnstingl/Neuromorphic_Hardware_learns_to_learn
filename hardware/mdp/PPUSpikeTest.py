import pydls as dls
import json
import pylogging
import Utils as utils
import dict_conv
import helpers_pydls as hp
import numpy as np

# activate logging
pylogging.reset()
pylogging.default_config(
    level=pylogging.LogLevel.INFO,
    fname="",
    print_location=False,
    color=True,
    date_format='RELATIVE')
log = pylogging.get("main")

# __________________________________________________________________________
# set cap_mem values - either calibrated or defaults
boardCalibMapping = {'B291698' : {'dac' : 'dac_default.json',
								  'cap' : 'cap_mem_29.json'},
					 '07' : {'dac' : 'dac_07_chip_20.json',
						     'cap' : 'calibration_20.json'},
					 'B201319' : {'dac' : 'dac_B201319_chip_21.json',
								  'cap' : 'calibration_24.json'},
					 'B201330' : {'dac' : 'dac_B201330_chip_22.json',
								  'cap' : 'calibration_22.json'}}
													        	
with open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['cap'], 'r') as f:
    cap_mem_config = json.load(f)

with open(boardCalibMapping[dls.get_allocated_board_ids()[0]]['dac'], 'r') as f:
    dac_config = json.load(f)

# one neuron
chip = dls.Chip()
for neuron_ind in range(dls.Neuron_index.num_neurons):
    for k, v in cap_mem_config['neuron_params'][neuron_ind].items():
        key = dict_conv.conversion_dict[k]
        hp.fill_cap_mem_cell(chip.cap_mem, neuron_ind, key, v)
chip.cap_mem.set(dls.Cap_mem_row(0), dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                 cap_mem_config['global_params']['v_reset'])

#for syn_row in range(32):
#    for syn_col in range(32):
#        syn = chip.synram.get(dls.Synapse_row(syn_row), dls.Synapse_column(syn_col))
#        syn.weight(63)
#        syn.address(20 if syn_row == 1 else 21)
#        chip.synram.set(dls.Synapse_row(syn_row), dls.Synapse_column(syn_col), syn)

syn = chip.synram.get(dls.Synapse_row(0), dls.Synapse_column(0))
syn.weight(63)
syn.address(20)
chip.synram.set(dls.Synapse_row(0), dls.Synapse_column(0), syn)

for i in range(32):
    
    neuron = chip.neurons.get(dls.Neuron_index(i))
    if i == 0:
        neuron.enable_out(True)
        neuron.mux_readout_mode(neuron.Mux_readout_mode.epsp)
    else:
        neuron.enable_out(False)
    chip.neurons.set(dls.Neuron_index(i), neuron)

#Default value is 8
#chip.syndrv_config.pulse_length(8)
# --------------------------------------------------------------------------

busy_work = dls.Dls_program_builder()
busy_work.set_time(0)
busy_work.set_chip(chip)
busy_work.wait_for(100000)
busy_work.halt()

# Load the program
program = dls.Ppu_program()
program.read_from_file("PPU/bin/spike.raw")

# Setup synram control register
# These are magic numbers which configure the timing how the synram is
# written.
synram_config_reg = dls.Synram_config_reg()
synram_config_reg.pc_conf(1)
synram_config_reg.w_conf(1)
synram_config_reg.wait_ctr_clear(1)

# spike router
fpga_conf = dls.Config_reg()
fpga_conf.spike_router_enable = False

router = dls.Spike_router_bypass(100, 42)  # accumulates spikes in 100 FPGA cycles, address of sent in spikes
# recurrent spikes are all of same address

# PPU control register
ppu_control_reg_start = dls.Ppu_control_reg()
ppu_control_reg_start.inhibit_reset(True)

ppu_control_reg_end = dls.Ppu_control_reg()
ppu_control_reg_end.inhibit_reset(False)

# Playback memory program
spikes_builder = dls.Dls_program_builder()
spikes_builder.set_synram_config_reg(synram_config_reg)
spikes_builder.set_ppu_program(program)
#spikes_builder.set_mailbox(mailbox)
#spikes_builder.fire(2**32 - 1, 20)

spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
spikes_builder.set_ppu_control_reg(ppu_control_reg_start)
spikes_builder.set_time(0)
spikes_builder.wait_until(1000000)
status_handle = spikes_builder.get_ppu_status_reg()
spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
mailbox_handle = spikes_builder.get_mailbox()
spikes_builder.halt()

with dls.connect(dls.get_allocated_board_ids()[0]) as connection:
    dls.soft_reset(connection)
    hp.setup_dac(connection, dac_config)

    busy_work.transfer(connection, 0)
    busy_work.execute(connection, 0)
    busy_work.fetch(connection)

    spikes_builder.transfer(connection, 0)
    spikes_builder.execute(connection, 0)
    spikes_builder.fetch(connection)

    spike_train = spikes_builder.get_spikes()
    spike_mat = np.zeros((len(spike_train), 2), dtype=np.int)
    counter = 0
    for spike in spike_train:
        log.info('Spike @ t = {:d} \tfrom neuron {:2d}'.format(spike.time, spike.address))
        spike_mat[counter, :] = spike.time, spike.address
        counter += 1
    #log.info(spike_mat)
    log.info('Total of {:3d} spikes were received'.format(counter))
    log.info('Finished')

mailbox_result = mailbox_handle.get()    
utils.print_mailbox_string(mailbox_result)

'''
with dls.connect(dls.get_allocated_board_ids()[0]) as c:
    dls.soft_reset(c)
    dls.set_config_reg(c, fpga_conf)  # chip config is only with chip, this is FPGA config
    hp.setup_dac(c, dac_config)  # soft reset ? resets dac?
    dls.set_spike_router(c, router)  # set spike router

    builder.transfer(c, 0)  # connection, 0 is program address
    builder.execute(c, 0)  # triggers execution on FPGA
    builder.fetch(c)
    
    spikes_builder.transfer(c, 0)
    spikes_builder.execute(c, 0)
    spikes_builder.fetch(c)
'''
