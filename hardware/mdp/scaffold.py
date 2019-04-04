import numpy as np
import argparse
import pylogging
import json

import pydlsnew as dls
import pydlsnew.coords as coords
from dls2calib.calibrationdatabase import CalibrationDatabaseDAO
import helpers as hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dac_config', type=argparse.FileType('r'), default=open('dac_07_chip_20.json', 'r'))
    # dstoe: The next argument is not needed anymore, when using the calibration
    # parser.add_argument('--cap_mem_config', type=argparse.FileType('r'), default=open('data/cap_mem_07_chip_20.json', 'r'))
    parser.add_argument('--chip', type=int, default=20)
    parser.add_argument('--weight', type=int, default=10)
    parser.add_argument('--syn_drv', type=int, default=15)
    parser.add_argument('--num_spikes', type=int, default=10)
    parser.add_argument('--spike_delay', type=int, default=200)
    args = parser.parse_args()

    dac_config = json.load(args.dac_config)
    # cap_mem_config = json.load(args.cap_mem_config)

    # Setup logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    log = pylogging.get("main")

    setup = dls.Setup()
    #hp.set_dac_values(setup.board_config.dac_values, dac_config)
    # dstoe: The next line is not needed anymore: The calibration explicitly overrides the values.
    # hp.set_cap_mem_values(setup.chip.cap_mem, cap_mem_config)

    neuron_params = {
        "tau_syn_exc" : 2e-6,
        "tau_syn_inh" : 5e-6,
        "tau_mem"     : 5e-6,
        "tau_ref"     : 5e-6,
        "v_leak"      : 0.800,
        "v_thresh"    : 1.100}

    neuron_global_params = {"v_reset" : 0.600}

    database = CalibrationDatabaseDAO(readonly=True)
    calib = database.load_chip('hardware', args.chip)

    cap_mem_config = {
            "neuron_params" : [
                dls.calibrated_neuron_params(
                    calib,
                    dls.coords.Neuron_index(n),
                    neuron_params)
                for n in range(dls.coords.Neuron_index.num_neurons)],
            "global_params" : dls.calibrated_global_params(calib, neuron_global_params)
            }
    
    with open('calibration_21.json', 'wb') as f:
        json.dump(cap_mem_config, f)
    #
    # with open('./cap_mem_spike_times.json', 'rb') as f:
        # cap_mem_config = json.load(f)
    for neuron_ind in range(dls.coords.Neuron_index.num_neurons):
        setup.chip.cap_mem.neuron_params_from_dict(
            dls.coords.Neuron_index(neuron_ind),
            cap_mem_config['neuron_params'][neuron_ind])
    # dstoe: You're missing to set the global params (which is the reset voltage). You
    # still use the reset voltage given by the cap_mem_config in the command line
    # arguments.
    # Use e.g. setup.chip.cap_mem.global_params_from_dict(cap_mem_config["global_params"])
    setup.chip.cap_mem.global_params_from_dict(cap_mem_config["global_params"])

    # dstoe: The following lines are correct usage of the synapse array. However, there
    # also exists an interface for numpy arrays in the meantime:
    # weights = numpy.full(dls.Synram.get_shape(), 0, dtype=int)
    # addresses = numpy.full(dls.Synram.get_shape(), 21, dtype=int)
    # weights[24, :] = args.weight
    # addresses[args.syn_drv, :] = 20
    # setup.chip.synram.set_weights(weights)
    # setup.chip.synram.set_addresses(addresses)
    for syn_row in range(32):
        for syn_col in range(32):
            syn = setup.chip.synram.get(coords.Synapse_row(syn_row), coords.Synapse_column(syn_col))
            syn.weight(args.weight if syn_row == 24 else 0)
            syn.address(20 if syn_row == args.syn_drv else 21)
            setup.chip.synram.set(coords.Synapse_row(syn_row), coords.Synapse_column(syn_col), syn)

    with open('setup.json', 'w') as f:
        f.write(setup.to_json())

    program = dls.Dls_program_builder()
    
    program.set_time(0)
    program.wait_for(1000)
    for i in range(args.num_spikes):
        program.wait_for(args.spike_delay)
        program.fire(2**32 - 1, 20)
    program.wait_for(1000)
    program.halt()

    log.info('Starting experiment...')

    with dls.connect(dls.get_allocated_board_ids()[0]) as connection:
        setup.do_experiment(connection, program)
    spike_train = program.get_spikes()
    spike_mat = np.zeros((len(spike_train), 2), dtype=np.int)
    counter = 0
    for spike in spike_train:
        log.info('Spike @ t = {:d} \tfrom neuron {:2d}'.format(spike.time, spike.address))
        spike_mat[counter, :] = spike.time, spike.address
        counter += 1
    #log.info(spike_mat)
    log.info('Total of {:3d} spikes were received'.format(counter))
    log.info('Finished')


if __name__ == '__main__':
    main()
