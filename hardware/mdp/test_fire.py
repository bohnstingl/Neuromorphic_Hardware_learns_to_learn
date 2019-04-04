import numpy as np
import argparse
import pylogging
import json

import pydlsnew as dls
import pydlsnew.coords as coords
import helpersNew as hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrated_config', type=argparse.FileType('r'), default=open('calibration_20.json'))
    parser.add_argument('--dac_config', type=argparse.FileType('r'), default=open('dac_07_chip_20.json', 'r'))
    parser.add_argument('--syn_row', type=int, default=31)
    parser.add_argument('--weight', type=int, default=63)
    parser.add_argument('--num_spikes', type=int, default=20)
    parser.add_argument('--delay', type=int, default=100)
    args = parser.parse_args()

    dac_config = json.load(args.dac_config)

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
    hp.set_dac_values(setup.board_config.dac_values, dac_config)

    cap_mem_config = json.load(args.calibrated_config)
    for neuron_ind in range(dls.coords.Neuron_index.num_neurons):
        setup.chip.cap_mem.neuron_params_from_dict(
            dls.coords.Neuron_index(neuron_ind),
            cap_mem_config['neuron_params'][neuron_ind])
    setup.chip.cap_mem.global_params_from_dict(cap_mem_config["global_params"])

    for syn_row in range(32):
        for syn_col in range(32):
            syn = setup.chip.synram.get(coords.Synapse_row(syn_row), coords.Synapse_column(syn_col))
            syn.weight(args.weight)
            syn.address(20 if syn_row == args.syn_row and syn_col != 6 else 21)
            setup.chip.synram.set(coords.Synapse_row(syn_row), coords.Synapse_column(syn_col), syn)

    busy_work = dls.Dls_program_builder()
    busy_work.wait_for(2000)
    busy_work.set_time(0)
    busy_work.fire(2**32 - 1, 20)
    busy_work.wait_for(2000)
    busy_work.halt()

    program = dls.Dls_program_builder()
    
    program.wait_for(1000)
    program.set_time(0)
    for _ in range(args.num_spikes):
        program.wait_for(args.delay)
        #program.fire(1 << syn_row, 20)
        program.fire(2**32 - 1, 20)
    program.wait_for(1000)
    program.halt()

    log.info('Starting experiment...')

    with dls.connect(dls.get_allocated_board_ids()[0]) as connection:
        # for i in range(50):
            # setup.do_experiment(connection, busy_work)
            # for neuron_ind in range(dls.coords.Neuron_index.num_neurons):
                # v = -1 if i % 2 == 0 else 1
                # v_thresh = cap_mem_config['neuron_params'][neuron_ind]['v_thresh']
                # cap_mem_config['neuron_params'][neuron_ind]['v_thresh'] = v_thresh + v
                # setup.chip.cap_mem.neuron_params_from_dict(
                    # dls.coords.Neuron_index(neuron_ind),
                    # cap_mem_config['neuron_params'][neuron_ind])
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
