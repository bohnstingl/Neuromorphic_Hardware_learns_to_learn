import numpy as np
import argparse
import pylogging
import json

import dict_conv
import pydls as dls
import helpers_pydls as hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrated_config', type=argparse.FileType('r'), default=open('calibration_24.json'))
    parser.add_argument('--dac_config', type=argparse.FileType('r'), default=open('dac_B201319_chip_21_david.json', 'r'))
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

    chip = dls.Chip()
    cap_mem_config = json.load(args.calibrated_config)
    for neuron_ind in range(dls.Neuron_index.num_neurons):
        for k, v in cap_mem_config['neuron_params'][neuron_ind].items():
            key = dict_conv.conversion_dict[k]
            hp.fill_cap_mem_cell(chip.cap_mem, neuron_ind, key, v)
    chip.cap_mem.set(dls.Cap_mem_row(0),
                     dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                     cap_mem_config['global_params']['v_reset'])

    for syn_row in range(32):
        for syn_col in range(32):
            syn = chip.synram.get(dls.Synapse_row(syn_row), dls.Synapse_column(syn_col))
            syn.weight(args.weight)
            syn.address(20 if syn_row == args.syn_row and syn_col == 4 else 21)
            chip.synram.set(dls.Synapse_row(syn_row), dls.Synapse_column(syn_col), syn)

    busy_work = dls.Dls_program_builder()
    busy_work.set_time(0)
    busy_work.set_chip(chip)
    busy_work.wait_for(100000)
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

    # Playback memory program
    program = dls.Ppu_program()
    program.read_from_file("PPU/bin/spike.raw")
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


    spikes_builder = dls.Dls_program_builder()
    spikes_builder.set_synram_config_reg(synram_config_reg)
    spikes_builder.set_ppu_program(program)

    spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
    spikes_builder.set_ppu_control_reg(ppu_control_reg_start)
    spikes_builder.set_time(0)
    spikes_builder.wait_until(100000)
    status_handle = spikes_builder.get_ppu_status_reg()
    spikes_builder.set_ppu_control_reg(ppu_control_reg_end)
    mailbox_handle = spikes_builder.get_mailbox()
    spikes_builder.halt()

    log.info('Starting experiment...')

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


if __name__ == '__main__':
    main()
