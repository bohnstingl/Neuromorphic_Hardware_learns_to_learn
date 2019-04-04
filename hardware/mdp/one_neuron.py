import pydls as dls
import json
import helpers as hp
import pylogging

# activate logging
pylogging.reset()
pylogging.default_config(
    level=pylogging.LogLevel.INFO,
    fname="",
    print_location=False,
    color=True,
    date_format='RELATIVE')
log = pylogging.get("main")

# one neuron
chip = dls.Chip()
syn = chip.synram.get(dls.Synapse_row(0), dls.Synapse_column(0))
syn.weight(63)  # 6bit
syn.address(42)  # address
chip.synram.set(dls.Synapse_row(0), dls.Synapse_column(0), syn)
neuron = chip.neurons.get(dls.Neuron_index(0))
neuron.enable_out(True)
neuron.mux_readout_mode(dls.Neuron.vmem)
chip.neurons.set(dls.Neuron_index(0), neuron)

# capmem
with open('cap_mem_default.json', 'r') as f:
    capmem_defaults = json.load(f)

with open('dac_default.json', 'r') as f:
    dac_default = json.load(f)
hp.set_cap_mem_values(chip.cap_mem, capmem_defaults)

# spike router
fpga_conf = dls.Config_reg()
fpga_conf.spike_router_enable = True

router = dls.Spike_router_bypass(100, 42)  # accumulates spikes in 100 FPGA cycles, address of sent in spikes
# recurrent spikes are all of same address

builder = dls.Dls_program_builder()  # capmem einpendeln
builder.set_time(0)
builder.set_chip(chip)
builder.wait_for(100000)
builder.halt()

spikes_builder = dls.Dls_program_builder()
spikes_builder.set_time(0)
spikes_builder.wait_for(1000)
# minimale wartezeit zwischen 2 spikes 48 FPGA zyklen
# sonst stauen sie sich einfach
# kein array
spikes_builder.fire(1, 42)  # row (32bit) as mask, address
spikes_builder.wait_for(10000)
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
    
spike_train = spikes_builder.get_spikes()
import idpdb
ipdb.set_trace()

#Prefilter spike train
for spike in spike_train:
    if spike.time > 100000:
        spike_train.pop(0)

for spike in spike_train:
    print spike.address
    print spike.sender
