import pydls as dls
import json
import helpers as hp
import pylogging
import Utils as utils

# activate logging
pylogging.reset()
pylogging.default_config(
    level=pylogging.LogLevel.INFO,
    fname="",
    print_location=False,
    color=True,
    date_format='RELATIVE')
log = pylogging.get("main")

# Set the weights of all neurons to a different value to test the readout
chip = dls.Chip()
cnt = 63
for neuronID in range(32):
    for connectedNeuron in range(32):
        syn = chip.synram.get(dls.Synapse_row(neuronID), dls.Synapse_column(connectedNeuron))
        syn.weight(cnt)  # 6bit
        syn.address(20)  # address
        chip.synram.set(dls.Synapse_row(neuronID), dls.Synapse_column(connectedNeuron), syn)
        cnt -= 1
        if cnt < 0:
            cnt = 63
        
    neuron = chip.neurons.get(dls.Neuron_index(neuronID))
    neuron.enable_out(True)
    neuron.mux_readout_mode(dls.Neuron.vmem)
    chip.neurons.set(dls.Neuron_index(neuronID), neuron)

# capmem
with open('cap_mem_default.json', 'r') as f:
    capmem_defaults = json.load(f)

with open('dac_default.json', 'r') as f:
    dac_default = json.load(f)
hp.set_cap_mem_values(chip.cap_mem, capmem_defaults)

# spike router
fpga_conf = dls.Config_reg()
fpga_conf.spike_router_enable = False

router = dls.Spike_router_bypass(100, 42)  # accumulates spikes in 100 FPGA cycles, address of sent in spikes
# recurrent spikes are all of same address

builder = dls.Dls_program_builder()  # capmem einpendeln
builder.set_time(0)
builder.set_chip(chip)
builder.wait_for(100000)
builder.halt()

# Load the program
program = dls.Ppu_program()
program.read_from_file("PPU/bin/weight.raw")

# Setup synram control register
# These are magic numbers which configure the timing how the synram is
# written.
#synram_config_reg = dls.Synram_config_reg()
#synram_config_reg.pc_conf(1)
#synram_config_reg.w_conf(1)
#synram_config_reg.wait_ctr_clear(1)

# PPU control register
ppu_control_reg_start = dls.Ppu_control_reg()
ppu_control_reg_start.inhibit_reset(True)

ppu_control_reg_end = dls.Ppu_control_reg()
ppu_control_reg_end.inhibit_reset(False)

# Playback memory program
spikes_builder = dls.Dls_program_builder()
#spikes_builder.set_synram_config_reg(synram_config_reg)
spikes_builder.set_ppu_program(program)
spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
spikes_builder.set_ppu_control_reg(ppu_control_reg_start)
spikes_builder.set_time(0)
spikes_builder.wait_until(10000000)
status_handle = spikes_builder.get_ppu_status_reg()
spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
mailbox_handle = spikes_builder.get_mailbox()
spikes_builder.halt()

with dls.connect(dls.get_allocated_board_ids()[0]) as c:
    dls.soft_reset(c)
    dls.set_config_reg(c, fpga_conf)  # chip config is only with chip, this is FPGA config
    hp.setup_dac(c, dac_default)  # soft reset ? resets dac?
    dls.set_spike_router(c, router)  # set spike router

    builder.transfer(c, 0)  # connection, 0 is program address
    builder.execute(c, 0)  # triggers execution on FPGA
    builder.fetch(c)
    
    spikes_builder.transfer(c, 0)
    spikes_builder.execute(c, 0)
    spikes_builder.fetch(c)
    
#Read the mailbox as the weights are stored there
mailbox_result = mailbox_handle.get()
if False:
    utils.print_mailbox_string(mailbox_result)
else:
    utils.print_mailbox(mailbox_result)
