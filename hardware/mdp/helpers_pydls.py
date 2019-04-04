import pydls
import numpy
import itertools
import struct

cap_mem_key_map = {
        "vLeak"         : pydls.Cap_mem_row( 0),
        "vThresh"       : pydls.Cap_mem_row( 1),
        "vSynEx"        : pydls.Cap_mem_row( 2),
        "vSynIn"        : pydls.Cap_mem_row( 3),
        "vUnused_4"     : pydls.Cap_mem_row( 4),
        "vUnused_5"     : pydls.Cap_mem_row( 5),
        "vUnused_6"     : pydls.Cap_mem_row( 6),
        "vUnused_7"     : pydls.Cap_mem_row( 7),
        "vUnused_8"     : pydls.Cap_mem_row( 8),
        "iBiasSpkCmp"   : pydls.Cap_mem_row( 9),
        "iBiasDelay"    : pydls.Cap_mem_row(10),
        "iBiasLeak"     : pydls.Cap_mem_row(11),
        "iBiasLeakSd"   : pydls.Cap_mem_row(12),
        "iBiasReadOut"  : pydls.Cap_mem_row(13),
        "iRefr"         : pydls.Cap_mem_row(14),
        "iBiasSynGmEx"  : pydls.Cap_mem_row(15),
        "iBiasSynSdEx"  : pydls.Cap_mem_row(16),
        "iBiasSynResEx" : pydls.Cap_mem_row(17),
        "iBiasSynOffEx" : pydls.Cap_mem_row(18),
        "iBiasSynResIn" : pydls.Cap_mem_row(19),
        "iBiasSynGmIn"  : pydls.Cap_mem_row(20),
        "iUnused_21"    : pydls.Cap_mem_row(21),
        "iBiasSynSdIn"  : pydls.Cap_mem_row(22),
        "iBiasSynOffIn" : pydls.Cap_mem_row(23),
        }

dac_key_map = {
        "cadc_ramp_bias"    : pydls.Dac_channel.cadc_ramp_bias(),
        "cadc_ramp_01"      : pydls.Dac_channel.cadc_ramp_01(),
        "cadc_ramp_slope"   : pydls.Dac_channel.cadc_ramp_slope(),
        "cadc_vbias"        : pydls.Dac_channel.cadc_vbias(),
        "syn_vddresmeas"    : pydls.Dac_channel.syn_vddresmeas(),
        "syn_vstore"        : pydls.Dac_channel.syn_vstore(),
        "syn_vramp"         : pydls.Dac_channel.syn_vramp(),
        "syn_vbias"         : pydls.Dac_channel.syn_vbias(),
        "capmem_ioffset"    : pydls.Dac_channel.capmem_ioffset(),
        "general_purpose_0" : pydls.Dac_channel.general_purpose_0(),
        "general_purpose_1" : pydls.Dac_channel.general_purpose_1(),
        "syn_vreset"        : pydls.Dac_channel.syn_vreset(),
        "syn_coroutbias"    : pydls.Dac_channel.syn_coroutbias(),
        "capmem_ibuf_bias"  : pydls.Dac_channel.capmem_ibuf_bias(),
        "capmem_iref"       : pydls.Dac_channel.capmem_iref(),
        }

def fill_cap_mem_cell(cap_mem, neuron_ind, key, value):
    cap_mem.set(pydls.Cap_mem_row(cap_mem_key_map[key]), pydls.Cap_mem_column(neuron_ind), value)

def fill_cap_mem_row(cap_mem, row, value):
    for index in range(pydls.Neuron_index.num_neurons):
        cap_mem.set(row, pydls.Cap_mem_column(index), value)

def set_cap_mem_values(cap_mem, values):
    for key, row in cap_mem_key_map.items():
        fill_cap_mem_row(cap_mem, row, values[key])
    # Global paramter vReset
    cap_mem.set(
            pydls.Cap_mem_row(0),
            pydls.Cap_mem_column(pydls.Neuron_index.num_neurons),
            values["vReset"])

def set_fixed_indegree(synram, weight, degree, address=0):
    nonzero_synapse = pydls.Synapse()
    nonzero_synapse.address(address)
    nonzero_synapse.weight(weight)
    for col in range(pydls.Neuron_index.num_neurons):
        perm = numpy.random.permutation(pydls.Neuron_index.num_neurons)
        perm = perm[:degree]
        for row in perm:
            synram.set(
                    pydls.Synapse_row(row),
                    pydls.Synapse_column(col),
                    nonzero_synapse)

def set_correlation_switches(synram, config):
    switch = pydls.Synapse()
    switch.config(config)
    for col in range(pydls.Neuron_index.num_neurons):
        synram.set(pydls.Synapse_row(33), pydls.Synapse_column(col), switch)

def set_syndrv_inhibitory(syndrv, indexes):
    for index in indexes:
        syndrv.senx(pydls.Synapse_row(index), False)
        syndrv.seni(pydls.Synapse_row(index), True)

def setup_dac(connection, values):
    dac_control = pydls.Dac_control()
    dac_control.gain = 0
    dac_control.buf = 3
    dac_control.vdo = 0
    pydls.set_dac_control(connection, pydls.dac12, dac_control)
    pydls.set_dac_control(connection, pydls.dac25, dac_control)
    for key, value in values.items():
        pydls.set_dac(connection, dac_key_map[key], value)

def start_ppu(program_builder):
    # Prepare control registers
    toggle_off = pydls.Ppu_control_reg()
    toggle_off.inhibit_reset(False)
    toggle_on = pydls.Ppu_control_reg()
    toggle_on.inhibit_reset(True)

    # Start the ppu by switching the inhibit reset bit off and on again
    program_builder.set_ppu_control_reg(toggle_off)
    program_builder.set_ppu_control_reg(toggle_on)

def stop_ppu(program_builder):
    # Prepare control register
    toggle_off = pydls.Ppu_control_reg()
    toggle_off.inhibit_reset(False)
    toggle_off.force_clock_off(True)

    # Stop the ppu by clearing the inhibit reset bit, plus forcing the clock to
    # be off
    program_builder.set_ppu_control_reg(toggle_off)

def make_synapse_array(synram):
    num_rows = pydls.Synapse_driver.num_drivers
    num_cols = pydls.Neuron_index.num_neurons
    synapse_array = numpy.zeros((num_rows, num_cols, 2), dtype=numpy.uint8)
    for row, col in itertools.product(range(num_rows), range(num_cols)):
        synapse = synram.get(pydls.Synapse_row(row), pydls.Synapse_column(col))
        synapse_array[row, col, 0] = synapse.weight()
        synapse_array[row, col, 1] = synapse.address()
    return synapse_array

def make_spiketrain_array(spiketrain):
    ret = numpy.zeros((2, len(spiketrain)), dtype=int)
    for index, spike in enumerate(spiketrain):
        ret[0, index] = spike.time
        ret[1, index] = spike.address
    return ret

def create_spikes_poisson(num_bins, probability):
    spikes = numpy.random.rand(num_bins, pydls.Neuron_index.num_neurons)
    spikes = (spikes < probability).astype(numpy.uint32)
    factors = numpy.power(2, range(pydls.Neuron_index.num_neurons)).astype(numpy.uint32)
    spike_masks = numpy.dot(spikes, factors)
    return spike_masks

def bytes_in_words(words):
    bytes_in_words = (struct.unpack('BBBB', struct.pack('>I', word)) for word in words)
    return itertools.chain.from_iterable(bytes_in_words)

def set_stdp_calib(synram, calib):
    for row in range(pydls.Synapse_driver.num_drivers):
        for col in range(pydls.Neuron_index.num_neurons):
            synapse = synram.get(pydls.Synapse_row(row), pydls.Synapse_column(col))
            synapse.config(calib[row, col])
            synram.set(pydls.Synapse_row(row), pydls.Synapse_column(col), synapse)
