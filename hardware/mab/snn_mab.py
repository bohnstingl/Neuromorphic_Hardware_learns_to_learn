from __future__ import print_function

import pydls as dls
import pickle
import pypet as pp
import argparse
import struct
import itertools
import json
import pylogging
import numpy as np
import synapses
import helpers_pydls as hp
import dict_conv


state_neuron = 31
mapping = [13, 30]


class BanditException(Exception):
    def __init__(self, reason):
        super(BanditException, self).__init__()
        self.reason = reason


class Connector(object):
    recurrent_address = 21
    recurrent_delay = 100
    stimulate_row = state_neuron

    def __init__(self, calibrated_config, dac_config, pulse_length):
        self.connection = None
        self.dac_config = dac_config

        self.router = dls.Spike_router_bypass(self.recurrent_delay, self.recurrent_address)
        self.chip = dls.Chip()
        self.chip.syndrv_config.pulse_length(pulse_length)
        v_reset = calibrated_config['global_params']['v_reset']

        for neuron_ind in range(32):
            for k, v in calibrated_config['neuron_params'][neuron_ind].items():
                key = dict_conv.conversion_dict[k]
                hp.fill_cap_mem_cell(self.chip.cap_mem, neuron_ind, key, v)
        self.chip.cap_mem.set(dls.Cap_mem_row(0), dls.Cap_mem_column(dls.Neuron_index.num_neurons),
                              v_reset)
        # state neuron recurrent spiking
        for k, v in calibrated_config['neuron_params'][self.stimulate_row].items():
            key = dict_conv.conversion_dict[k]
            if k == 'i_refr':
                v = 1022  # minimal refractory period
            elif k == 'v_thresh':
                v = calibrated_config['neuron_params'][state_neuron]['v_leak'] + 30
            hp.fill_cap_mem_cell(self.chip.cap_mem, self.stimulate_row, key, v)

        # __________________________________________________________________________
        # configure neurons - setup spike counters for clear on read
        for neuron_ind in range(32):
            neuron = self.chip.neurons.get(dls.Neuron_index(neuron_ind))
            if neuron_ind in mapping or neuron_ind == state_neuron:
                neuron.fire_out_mode(dls.Neuron.Fire_out_mode.enable)
                neuron.enable_out(True)
            else:
                neuron.fire_out_mode(dls.Neuron.Fire_out_mode.disable)
                neuron.enable_out(False)
            self.chip.neurons.set(dls.Neuron_index(neuron_ind), neuron)
            self.chip.rate_counter.enable(dls.Neuron_index(neuron_ind), True)
        self.chip.rate_counter.clear_on_read(True)
        # --------------------------------------------------------------------------
        # Setup synram control register
        # These are magic numbers which configure the timing how the synram is
        # written.
        self.synram_config_reg = dls.Synram_config_reg()
        self.synram_config_reg.pc_conf(1)
        self.synram_config_reg.w_conf(1)
        self.synram_config_reg.wait_ctr_clear(1)

        self.fpga_conf = dls.Config_reg()
        self.fpga_conf.spike_router_enable = True

        # PPU control register
        self.ppu_control_reg_start = dls.Ppu_control_reg()
        self.ppu_control_reg_start.inhibit_reset(True)

        self.ppu_control_reg_end = dls.Ppu_control_reg()
        self.ppu_control_reg_end.inhibit_reset(False)

        self.pre_builder = dls.Dls_program_builder()
        self.pre_builder.set_time(0)
        self.pre_builder.set_chip(self.chip)
        self.pre_builder.wait_for(1000000)
        self.pre_builder.halt()
        self.board_id = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        board_ids = dls.get_allocated_board_ids()
        assert(len(board_ids) == 1)
        assert(len(board_ids[0]) != 0)
        self.connection = dls.connect(board_ids[0])
        dls.soft_reset(self.connection)
        dls.set_config_reg(self.connection, self.fpga_conf)
        hp.setup_dac(self.connection, self.dac_config)
        dls.set_spike_router(self.connection, self.router)

        #print('-- before transfer')
        self.pre_builder.transfer(self.connection, 0x0)
        #print('-- before execute')
        self.pre_builder.execute(self.connection, 0x0)
        #print('-- before safefetch')
        # self.pre_builder.fetch(self.connection)
        safe_fetch(self.pre_builder, self.connection)
        #print('-- after safefetch')

        self.board_id = self.board_id if self.board_id is not None else board_ids[0]

    def disconnect(self):
        if self.connection is not None:
            self.connection.disconnect()


class IncrementalLearningRule(object):
    dls_program = dls.Ppu_program()
    dls_program.read_from_file('bin/mabq.raw')
    default_hyperparameters = {'learning_rate': .1,
                               'weight_prior': .7,
                               'learning_rate_decay': .9}

    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters or self.default_hyperparameters

    def extend_words(self, words, seed=1234123):
        words.extend([int(self.hyperparameters['learning_rate'] * (2**32 - 1)),
                      int(self.hyperparameters['weight_prior'] * (2**32 - 1)),
                      seed,
                      int(self.hyperparameters['learning_rate_decay'] * (2**32 - 1))])


class ANNLearningRule(object):
    dls_program = dls.Ppu_program()
    dls_program.read_from_file('bin/mabann.raw')
    n_hidden = 10
    n_input = 5
    default_hyperparameters = {'learning_rate': .1,
                               'ann_parameters': np.random.rand(n_input * n_hidden + 2 * n_hidden)}

    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters or self.default_hyperparameters

    def extend_words(self, words, seed=1234123):
        ann_parameters_machine = []
        words.extend([seed, int(self.hyperparameters['learning_rate'] * (2**32 - 1))])
        for param in self.hyperparameters['ann_parameters']:
            param = (param - .5) * 2**28
            if param >= 2**31 - 1:
                param = 2**31 - 1
            param = struct.unpack('>I', struct.pack('>i', param))[0]
            ann_parameters_machine.append(param)
        words.extend(ann_parameters_machine)


class SpikingBanditAgent(object):
    def __init__(self, logger):
        self.logger = logger
        self.wait = int(1E8)
        self.stimulate_row = state_neuron
        self.hyperparameters = ['action_inhibition', 'stim_inhibition']
        default = [-63, -63]
        #default = [0, 0]
        self.default_hyperparameters = dict(zip(self.hyperparameters, default))

    def play_bandit_batch(self, bandit_probabilities, n_pulls, n_runs,
                          hyperparameters, learning_rule, connector):
        n_batch = 1
        mailbox = dls.Mailbox()
        bandit_probabilities_machine = (bandit_probabilities * 2**32).astype(np.int)
        set_env(mailbox, bandit_probabilities_machine, n_pulls, n_runs, n_batch,
                learning_rule)

        n_arms = int(len(bandit_probabilities) / n_runs)
        action_inhibition = int(hyperparameters['action_inhibition'])
        stim_inhibition = int(hyperparameters['stim_inhibition'])

        weights = np.zeros((32, 32), dtype=np.int)
        addresses = np.zeros((32, 32), dtype=np.int)

        weights[self.stimulate_row, :] = 10
        weights[self.stimulate_row, self.stimulate_row] = 63
        addresses[self.stimulate_row, self.stimulate_row] = connector.recurrent_address

        for i in mapping:
            # weights[i, :n_arms] = action_inhibition
            weights[i, :] = action_inhibition
            weights[i, self.stimulate_row] = stim_inhibition  # same for state neuron
            weights[i, i] = 0
            addresses[i, :] = connector.recurrent_address
            addresses[self.stimulate_row, i] = connector.recurrent_address
        addresses[self.stimulate_row, :n_arms] = connector.recurrent_address
        synapses.setup_synram(weights, addresses, connector.chip)
        pre_builder = dls.Dls_program_builder()
        pre_builder.set_time(0)
        pre_builder.set_chip(connector.chip)
        pre_builder.wait_for(1000000)
        pre_builder.halt()
        #print('-- before tranfer')
        pre_builder.transfer(connector.connection, 0x0)
        #print('-- before execute')
        pre_builder.execute(connector.connection, 0x0)
        #print('-- before safefetch')
        # pre_builder.fetch(connector.connection)
        safe_fetch(pre_builder, connector.connection)
        #print('-- after safefetch')
        # self.logger.info('executed pre_builder')

        # Playback memory program
        builder = dls.Dls_program_builder()
        builder.set_synram_config_reg(connector.synram_config_reg)
        builder.set_mailbox(mailbox)
        builder.set_ppu_program(learning_rule.dls_program)
        builder.set_ppu_control_reg(connector.ppu_control_reg_end)
        builder.set_ppu_control_reg(connector.ppu_control_reg_start)
        builder.set_time(0)
        builder.wait_until(self.wait)
        status_handle = builder.get_ppu_status_reg()
        builder.set_ppu_control_reg(connector.ppu_control_reg_end)
        mailbox_handle = builder.get_mailbox()
        synram_handle = builder.get_synram()
        builder.halt()

        # Transfer execute and copy back results
        #print('-- before transfer')
        builder.transfer(connector.connection, 0x0)
        #print('-- before execute')
        builder.execute(connector.connection, 0x0)
        #print('-- before safefetch')
        # builder.fetch(connector.connection)
        safe_fetch(builder, connector.connection)
        #print('-- after safefetch')
        # self.logger.info('program executed')
        if False:
            synram = synram_handle.get()
            weight_matrix = np.zeros((32, 32), dtype=np.int)
            for row in range(32):
                for col in range(32):
                    syn = synram.get(dls.Synapse_row(row), dls.Synapse_column(col))
                    weight_matrix[row, col] = syn.weight()
                    print('{:2d}'.format(syn.weight()), end=' ')
                print()
        spike_train = builder.get_spikes()
        spike_n = np.zeros((len(spike_train), 2), np.int)
        for i, spike in enumerate(spike_train):
            spike_n[i, 0] = spike.time
            spike_n[i, 1] = spike.address

        # Check status register
        status_reg_result = status_handle.get()
        if status_reg_result.sleep() is not True:
            raise BanditException('PPU did not stop')

        # results
        mailbox_result = mailbox_handle.get()

        a_r = np.zeros((n_runs, n_batch, n_pulls, 2), np.int)
        sampled_probs = np.zeros((n_runs, n_batch, n_arms))
        mailbox_bytes = list(bytes_of_mailbox(mailbox_result))

        all_expected_regrets = []
        for run_index in range(n_runs):
            batch_expected_regrets = []
            for batch_index in range(n_batch):
                for i in range(n_pulls):
                    byte = mailbox_bytes[i + n_pulls * (batch_index + run_index * n_batch)]
                    action = byte & 0x3f
                    if action >= n_arms:
                        if self.logger is not None:
                            self.logger.info('Wrong Action in Mailbox! Ignoring Current Run...')
                        if False:
                            for i, b in enumerate(mailbox_bytes):
                                if i % 16 == 0:
                                    print()
                                print('{:02x}'.format(b), end=' ')
                        raise BanditException('Mailbox has wrong values')
                    reward = 1 if (byte & 0x80) != 0 else 0
                    a_r[run_index, batch_index, i, :] = reward, action

                for i in range(n_arms):
                    a_ind = np.where(a_r[run_index, batch_index, :, 1] == i)[0]
                    p = np.mean(a_r[run_index, batch_index, a_ind, 0])
                    sampled_probs[run_index, batch_index, i] = p

                p_max = np.max(bandit_probabilities[run_index * n_arms : (run_index + 1) * n_arms])
                expected_regret = 0
                for t_action in a_r[run_index, batch_index, :, 1]:
                    expected_regret += p_max - bandit_probabilities[run_index * n_arms + t_action]
                batch_expected_regrets.append(expected_regret)
            all_expected_regrets.append(batch_expected_regrets)

        results = dict(a_r=a_r, sampled_probs=sampled_probs,
                       mailbox_bytes=mailbox_bytes, spikes=spike_n)

        all_expected_regrets = np.array(all_expected_regrets)
        return np.mean(all_expected_regrets), results


def safe_fetch(program, connection):
    program.fetch(connection)
    return
    res_exception_reg = dls.get_exception_reg(connection).res_except
    exception = res_exception_reg.wr_underrun or \
                res_exception_reg.wr_error or \
                res_exception_reg.rd_overflow or \
                res_exception_reg.rd_error
    result_size = dls.get_result_size_reg(connection)
    no_fetch = exception or result_size > 16384 * 1024 / 4
    if not no_fetch:
        program.fetch(connection)
    else:
        raise BanditException('FPGA cannot fetch results')


def load_mailbox(mailbox, filename):
    words = []
    with open(filename, 'rb') as f:
        while True:
            d = f.read(4)
            if len(d) == 0:
                break
            d = d + '\x00' * (4 - len(d))
            word = struct.unpack('>L', d)[0]
            words.append(word)
    for index, word in enumerate(words):
        mailbox.set_word(dls.Address_on_mailbox(index), word)


def bytes_of_mailbox(mailbox):
    words = mailbox.export_words()
    bytes_in_words = (struct.unpack('BBBB', struct.pack('>I', word)) for word in words)
    return itertools.chain.from_iterable(bytes_in_words)


def set_env(mailbox, bandit_probabilities, n_pulls, n_runs, n_batch,
            learning_rule):
    n_arms = len(bandit_probabilities) // n_runs
    words = [n_pulls, n_arms, n_runs, n_batch]
    words.extend(bandit_probabilities)
    learning_rule.extend_words(words)
    for index, word in enumerate(words):
        mailbox.set_word(dls.Address_on_mailbox(index), word)


def inner_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal', type=str, default='../adv/calibration_20.json')
    parser.add_argument('--dac', type=str, default='../adv/dac_07_chip_20.json')
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--pl', type=int, choices=range(32), default=4)
    parser.add_argument('--lr', type=str, choices=['q', 'greedy', 'ann'], default='q')
    parser.add_argument('--generation', type=int, default=-1)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--dependent', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.cal) as f:
        calibrated_config = json.load(f)
    with open(args.dac) as f:
        dac_config = json.load(f)

    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    logger = pylogging.get('main')

    agent = SpikingBanditAgent(logger)

    n_batch = args.n_batch

    agent_hp = agent.default_hyperparameters
    if args.lr == 'q':
        learning_rule = IncrementalLearningRule()
    elif args.lr == 'ann':
        learning_rule = ANNLearningRule()

    if args.load_from != '':
        traj = pp.Trajectory(filename=args.load_from)
        traj.v_auto_load = True
        traj.f_load(index=-1, force=True)
        pop_size = traj.parameters.pop_size
        n_iter = traj.parameters.n_iteration
        max_fitness = -100
        best_individual = None
        if args.generation == -1:
            gen_index = n_iter - 1
        else:
            gen_index = args.generation
        for j in range(pop_size):
            traj.v_idx = gen_index * pop_size + j
            # print(traj.v_idx)
            fitness = traj.results.crun.fitness
            if fitness > max_fitness:
                max_fitness = fitness
                best_individual = dict(traj.parameters.individual.f_get_children())
                best_individual.pop('seed', None)
                for k, v in best_individual.items():
                    best_individual[k] = best_individual[k][traj.v_idx]
        print(best_individual)

        if args.lr == 'q':
            agent_hp = dict(action_inhibition=best_individual['action_inhibition'],
                            stim_inhibition=best_individual['stim_inhibition'])
            lr_hp = dict(learning_rate=best_individual['learning_rate'],
                         learning_rate_decay=best_individual['learning_rate_decay'],
                         weight_prior=best_individual['weight_prior'])
            learning_rule = IncrementalLearningRule(lr_hp)
        elif args.lr == 'ann':
            lr_hp = dict(learning_rate=best_individual['learning_rate'],
                         ann_parameters=best_individual['ann_parameters'])
            agent_hp = agent.default_hyperparameters
            learning_rule = ANNLearningRule(lr_hp)
        else:
            logger.error('Learning rule {:s} not supported yet'.format(args.lr))
            quit()
    bps = []
    ar = []
    regrets = []
    with Connector(calibrated_config, dac_config, args.pl) as connector:
        for i in range(args.n_iter):
            bandit_probabilities = np.random.rand(n_batch, 2)
            if args.dependent:
                bandit_probabilities[:, 1] = 1. - bandit_probabilities[:, 0]
            bandit_probabilities = bandit_probabilities.reshape((-1,))
            try:
                r = agent.play_bandit_batch(bandit_probabilities, 100, n_batch,
                                            agent_hp,
                                            learning_rule, connector)
                regrets.append(r[0])
            except:
                continue
            ar.append(r[1]['a_r'])
            bps.append(bandit_probabilities)
            logger.info('iteration made')
    print(np.mean(regrets))
    if args.verbose:
        spikes = r[1]['spikes']
        logger.info(spikes[:20, :])
        logger.info('')
        logger.info(spikes[-20:, :])
        logger.info('A total of {} spikes was received'.format(spikes.shape[0]))
    if args.out != '':
        with open(args.out, 'wb') as f:
            pickle.dump(dict(bandit_probabilities=bps, a_r=ar), f)
    logger.info('Finished')


if __name__ == '__main__':
    inner_loop()

