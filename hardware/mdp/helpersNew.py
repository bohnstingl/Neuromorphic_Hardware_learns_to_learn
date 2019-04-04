import pydlsnew as dls
import pydlsnew.coords as coords
import numpy
import itertools
import struct

cap_mem_key_map = {
        "vLeak"         : coords.Cap_mem_row( 0),
        "vThresh"       : coords.Cap_mem_row( 1),
        "vSynEx"        : coords.Cap_mem_row( 2),
        "vSynIn"        : coords.Cap_mem_row( 3),
        "vUnused_4"     : coords.Cap_mem_row( 4),
        "vUnused_5"     : coords.Cap_mem_row( 5),
        "vUnused_6"     : coords.Cap_mem_row( 6),
        "vUnused_7"     : coords.Cap_mem_row( 7),
        "vUnused_8"     : coords.Cap_mem_row( 8),
        "iBiasSpkCmp"   : coords.Cap_mem_row( 9),
        "iBiasDelay"    : coords.Cap_mem_row(10),
        "iBiasLeak"     : coords.Cap_mem_row(11),
        "iBiasLeakSd"   : coords.Cap_mem_row(12),
        "iBiasReadOut"  : coords.Cap_mem_row(13),
        "iRefr"         : coords.Cap_mem_row(14),
        "iBiasSynGmEx"  : coords.Cap_mem_row(15),
        "iBiasSynSdEx"  : coords.Cap_mem_row(16),
        "iBiasSynResEx" : coords.Cap_mem_row(17),
        "iBiasSynOffEx" : coords.Cap_mem_row(18),
        "iBiasSynResIn" : coords.Cap_mem_row(19),
        "iBiasSynGmIn"  : coords.Cap_mem_row(20),
        "iUnused_21"    : coords.Cap_mem_row(21),
        "iBiasSynSdIn"  : coords.Cap_mem_row(22),
        "iBiasSynOffIn" : coords.Cap_mem_row(23),
        }

dac_key_map = {
        "cadc_ramp_bias"    : coords.Dac_channel.cadc_ramp_bias(),
        "cadc_ramp_01"      : coords.Dac_channel.cadc_ramp_01(),
        "cadc_ramp_slope"   : coords.Dac_channel.cadc_ramp_slope(),
        "cadc_vbias"        : coords.Dac_channel.cadc_vbias(),
        "syn_vddresmeas"    : coords.Dac_channel.syn_vddresmeas(),
        "syn_vstore"        : coords.Dac_channel.syn_vstore(),
        "syn_vramp"         : coords.Dac_channel.syn_vramp(),
        "syn_vbias"         : coords.Dac_channel.syn_vbias(),
        "capmem_ioffset"    : coords.Dac_channel.capmem_ioffset(),
        "general_purpose_0" : coords.Dac_channel.general_purpose_0(),
        "general_purpose_1" : coords.Dac_channel.general_purpose_1(),
        "syn_vreset"        : coords.Dac_channel.syn_vreset(),
        "syn_coroutbias"    : coords.Dac_channel.syn_coroutbias(),
        "capmem_ibuf_bias"  : coords.Dac_channel.capmem_ibuf_bias(),
        "capmem_iref"       : coords.Dac_channel.capmem_iref(),
        }


def fill_cap_mem_row(cap_mem, row, value):
    for index in range(coords.Neuron_index.num_neurons):
        cap_mem.set(row, coords.Cap_mem_column(index), value)


def set_cap_mem_values(cap_mem, values):
    for key, row in cap_mem_key_map.items():
        fill_cap_mem_row(cap_mem, row, values[key])
    # Global paramter vReset
    cap_mem.set(
            coords.Cap_mem_row(0),
            coords.Cap_mem_column(coords.Neuron_index.num_neurons),
            values["vReset"])


def set_syndrv_inhibitory(syndrv, indexes):
    for index in indexes:
        syndrv.senx(coords.Synapse_row(index), False)
        syndrv.seni(coords.Synapse_row(index), True)


def set_dac_values(dac_container, values):
    for key, value in values.items():
        dac_container[dac_key_map[key]] = value


def make_spiketrain_array(spiketrain):
    ret = numpy.zeros((2, len(spiketrain)), dtype=int)
    for index, spike in enumerate(spiketrain):
        ret[0, index] = spike.time
        ret[1, index] = spike.address
    return ret
