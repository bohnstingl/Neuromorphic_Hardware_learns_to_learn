import logging
from enum import Enum
import quantities as pq
import copy

import numpy as np
from pyNN.random import RandomDistribution


import pyhmf as env
from pyhalbe import HICANN
import pyhalbe.Coordinate as C
from pysthal.command_line_util import init_logger
from pymarocco import PyMarocco, Defects
from pymarocco.runtime import Runtime
from pymarocco.coordinates import LogicalNeuron
from pymarocco.results import Marocco

class BrainscalesEnvironment(object):
    def __init__(self, hicann_id=367):
        self.marocco = PyMarocco()
        self.marocco.neuron_placement.default_neuron_size(4)
        self.marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
        self.marocco.merger_routing.strategy(self.marocco.merger_routing.one_to_one)

        self.marocco.bkg_gen_isi = 125
        self.marocco.pll_freq = 125e6

        self.marocco.backend = PyMarocco.Hardware
        self.marocco.calib_backend = PyMarocco.XML
        self.marocco.defects.path = self.marocco.calib_path = "/wang/data/calibration/ITL_2016"
        self.marocco.defects.backend = Defects.XML
        self.marocco.default_wafer = C.Wafer(33)
        self.marocco.param_trafo.use_big_capacitors = True
        self.marocco.input_placement.consider_firing_rate(True)
        self.marocco.input_placement.bandwidth_utilization(0.8)

        self.runtime = Runtime(self.marocco.default_wafer)

        env.setup(marocco=self.marocco, marocco_runtime=self.runtime)
        self.hicann = C.HICANNOnWafer(C.Enum(hicann_id))

        self.projections = dict()
        
    def get_env(self):
        
        return env

    def manual_placement(self, pop, hicann=None):
        if hicann is None:
            hicann = self.hicann
        return self.marocco.manual_placement.on_hicann(pop, hicann)

    def run_mapping(self, run_time=100.):
        self.marocco.skip_mapping = False
        self.marocco.backend = PyMarocco.None

        env.reset()
        env.run(run_time)
        env.reset()

    def configure(self, simulation_callback):
        self.set_sthal_params(self.runtime.wafer(), gmax=1023, gmax_div=1)

        self.skip_mapping = True
        self.marocco.backend = PyMarocco.Hardware
        self.marocco.hicann_configurator = PyMarocco.HICANNv4Configurator
        env.reset()
        simulation_callback()
        env.reset()
        self.marocco.hicann_configurator = PyMarocco.NoResetNoFGConfigurator

    def map_configure(self, run_time=100.):
        self.marocco.skip_mapping = False
        self.marocco.backend = PyMarocco.Hardware
        self.marocco.hicann_configurator = PyMarocco.HICANNv4Configurator

        env.reset()
        env.run(run_time)
        env.reset()

        self.marocco.skip_mapping = True
        self.marocco.hicann_configurator = PyMarocco.NoResetNoFGConfigurator

    def scaling_function(self, nest_like_weight):
        return np.clip(int(np.round((nest_like_weight - 0.0023) / 0.001)), 0, 15)

    def set_sthal_params(self, wafer=None, hicann=None, gmax=1023, gmax_div=1):
        """
        synaptic strength:
        gmax: 0 - 1023, strongest: 1023
        gmax_div: 1 - 15, strongest: 1
        """
        self.gmax = gmax
        self.gmax_div = gmax_div

        if hicann is None:
            hicann = self.hicann

        # for all HICANNs in use
        if wafer is None:
            wafer = self.runtime.wafer()

            fgs = wafer[hicann].floating_gates

            # set parameters influencing the synaptic strength
            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_gmax0, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax1, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax2, gmax)
                fgs.setShared(block, HICANN.shared_parameter.V_gmax3, gmax)

            for driver in C.iter_all(C.SynapseDriverOnHICANN):
                for row in C.iter_all(C.RowOnSynapseDriver):
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.left, gmax_div)
                    wafer[hicann].synapses[driver][row].set_gmax_div(
                        C.right, gmax_div)

            # don't change values below
            for ii in xrange(fgs.getNoProgrammingPasses()):
                cfg = fgs.getFGConfig(C.Enum(ii))
                cfg.fg_biasn = 0
                cfg.fg_bias = 0
                fgs.setFGConfig(C.Enum(ii), cfg)

            for block in C.iter_all(C.FGBlockOnHICANN):
                fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
                fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)

    def Projection(self, source, destination, connector, weights, *args, **kwargs):
        proj = env.Projection(source, destination, connector, **kwargs)
        self.projections[proj] = weights
        return proj

    def Population(self, n_neurons, n_type, neuron_params={}, place_on_same_wafer=True, **kwargs):
        nodes = env.Population(n_neurons, n_type, neuron_params, **kwargs)
        if place_on_same_wafer:
            self.manual_placement(nodes)
        return nodes

    def PopulationView(self, parentPopulation, cellIDs):
        
        populationView = env.PopulationView(parentPopulation, cellIDs)
        
        return populationView
    
    def set_hardware_weights(self):
        for proj, weights in self.projections.items():
            for j, proj_item in enumerate(self.runtime.results().synapse_routing.synapses().find(proj)):
                #proj_item, = self.runtime.results().synapse_routing.synapses().find(proj)
                synapse = proj_item.hardware_synapse()

                proxy = self.runtime.wafer()[synapse.toHICANNOnWafer()].synapses[synapse]
                w = weights
                if str(type(w)).count('RandomDistribution') > 0:
                    w = weights.next()
                proxy.weight = HICANN.SynapseWeight(self.scaling_function(w))

    def set_weight(self, projection, weight):
        self.projections[projection] = weight

    def clear_weights(self):
        self.projections = dict()


# class Tasks(Enum):
#     XOR = 1
#     FADING_MEMORY = 2
# 
# 
# class LSMOptimizee(Optimizee):
#     """
#     
#     Adapted from <http://www.igi.tugraz.at/lehre/PrinciplesOfBrainComputation/SS16/>_.
#     
#     """
# 
#     def __init__(self, traj, task, n_NEST_threads=1):
#         super(LSMOptimizee, self).__init__(traj)
# 
#         #assert task in Tasks
#         self.task = task
# 
#         self.dt_stim_ms = 300.  #[ms]
#         self.stim_len_ms = 50.  #[ms]
# 
#         self.n_NEST_threads = n_NEST_threads
# 
#         # create_individual can be called because __init__ is complete except for traj initialization
#         indiv_dict = self.create_individual()
#         for key, val in indiv_dict.items():
#             traj.individual.f_add_parameter(key, val)
# 
#         # HW initialization
#         self.brainscales_environment = BrainscalesEnvironment()
# 
#         self._initialize()
# 
# 
#     def _initialize(self):
#         # dynamic parameters
#         f0 = 10.
# 
#         def get_u_0(U, D, F):
#             return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))
# 
#         def get_x_0(U, D, F):
#             return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))
# 
#         syn_param_EE = {"tau_facil": 1.,  # facilitation time constant in ms
#                         "tau_rec": 813.,  # recovery time constant in ms
#                         "U": 0.59,  # utilization
#                         "u0": get_u_0(0.59, 813., 1.),
#                         "x0": get_x_0(0.59, 813., 1.),
#                         }
#         self.EE = env.SynapseDynamics(fast=env.TsodyksMarkramMechanism(**syn_param_EE))
#         syn_param_EI = {"tau_facil": 1790.,  # facilitation time constant in ms
#                         "tau_rec": 399.,  # recovery time constant in ms
#                         "U": 0.049,  # utilization
#                         "u0": get_u_0(0.049, 399., 1790.),
#                         "x0": get_x_0(0.049, 399., 1790.),
#                         }
#         self.EI = env.SynapseDynamics(fast=env.TsodyksMarkramMechanism(**syn_param_EI))
# 
#         syn_param_IE = {"tau_facil": 376.,  # facilitation time constant in ms
#                         "tau_rec": 45.,  # recovery time constant in ms
#                         "U": 0.016,  # utilization
#                         "u0": get_u_0(0.016, 45., 376.),
#                         "x0": get_x_0(0.016, 45., 376.),
#                         }
#         self.IE = env.SynapseDynamics(fast=env.TsodyksMarkramMechanism(**syn_param_IE))
# 
#         syn_param_II = {"tau_facil": 21.,  # facilitation time constant in ms
#                         "tau_rec": 706.,  # recovery time constant in ms
#                         "U": 0.25,  # utilization
#                         "u0": get_u_0(0.25, 706., 21.),
#                         "x0": get_x_0(0.25, 706., 21.),
#                         }
#         self.II = env.SynapseDynamics(fast=env.TsodyksMarkramMechanism(**syn_param_II))
# 
#     def create_individual(self):
#         jee, jie = np.random.randint(1, 20, 2).astype(np.float64)
#         return dict(jee=jee, jie=jie)
# 
#     def bounding_func(self, individual):
#         individual = {key: np.float64(np.round(value) if value > 0.01 else 0.01) for key, value in individual.items()}
#         return individual
# 
#     def simulate(self, traj, should_plot=False, debug=False):
# 
#         jee = traj.individual.jee
#         jie = traj.individual.jie
# 
#         assert jee > 0 and jie > 0
# 
#         if self.task == Tasks.XOR:
#             readout_delay = 0.030  # [sec]
#             actual_readout_delay, performance = self._simulate(jee, jie, self.dt_stim_ms, self.stim_len_ms,
#                                                                readout_delay, should_plot, debug)
#             return performance,
#         elif self.task == Tasks.FADING_MEMORY:
#             delta_readout_delay_increase_ms = 1000.  #[sec]
# 
#             # \/\/ Subtract delta_readout_delay_increase to compensate for first addition of delta_readout_delay_increasee
#             dt_stim_ms = self.dt_stim_ms - delta_readout_delay_increase_ms
# 
#             while True:
#                 dt_stim_ms += delta_readout_delay_increase_ms
#                 readout_delay = (dt_stim_ms - self.stim_len_ms - 10.) * 1e-3  # [sec]
#                 logger.info("Running full simulation for readout delay of %f seconds (dt_stim_ms %f ms) to measure "
#                             "fading memory", readout_delay, dt_stim_ms)
#                 actual_readout_delay, performance = self._simulate(jee, jie, dt_stim_ms, self.stim_len_ms,
#                                                                    readout_delay, should_plot, debug)
#                 if actual_readout_delay < readout_delay:
#                     global_readout_delay = (self.dt_stim_ms - self.stim_len_ms - 10.) * 1e-3  # [sec]
#                     logger.info("Returning readout delay %f (actual). Original readout delay was %f. "
#                                 "Global readout delay was %f", actual_readout_delay, readout_delay,
#                                 global_readout_delay)
#                     #\/\/ 'coz the readout delay only increases
#                     delay_difference = (actual_readout_delay - global_readout_delay)
#                     if delay_difference > 0:
#                         new_dt_stim_ms = self.dt_stim_ms + delay_difference * 1e3
#                         logger.info("Permanently Increasing dt_stim_ms to %.4f ms from %.4f ms",
#                                     new_dt_stim_ms, self.dt_stim_ms)
#                         self.dt_stim_ms = new_dt_stim_ms
#                     break
#             return actual_readout_delay,
# 
#     def _simulate(self, jee, jie, dt_stim_ms, stim_len_ms, readout_delay, should_plot, debug):
#         logger.info("Running for %.2f, %.2f", jee, jie)
# 
#         n_runs = 50
#         simtime_ms = dt_stim_ms * (n_runs + 1) - stim_len_ms  # how long shall we simulate [ms] (n_runs presentations of input)
#         if debug:
#             simtime_ms = 20000.
# 
#         logger.debug("Running for %.0f ms", simtime_ms)
# 
#         projections = []
#         weights = []
# 
#         fraction_neurons_recorded = 0.5  # Number of neurons to record from
# 
#         # Network parameters.
#         N_E = 20  # 2000  # number of excitatory neurons
#         N_I = 5  # 500  # number of inhibitory neurons
#         N_neurons = N_E + N_I  # total number of neurons
# 
#         C_E = 2  # int(N_E / 20)  # number of excitatory synapses per neuron
#         C_I = 1  # int(N_E / 20)  # number of inhibitory synapses per neuron
#         C_inp = 10  # int(N_E / 20)  # number of outgoing input synapses per input neuron
# 
#         w_scale = .5 * 1e-3
#         J_EE = w_scale * jee  # strength of E->E synapses [nA]
#         J_EI = J_EE  # strength of E->I synapses [nA]
#         J_IE = w_scale * jie  # strength of inhibitory synapses [nA]
#         J_II = J_IE  # strength of inhibitory synapses [nA]
# 
#         p_rate_hz = 50.0  # this is used to simulate input from neurons around the populations
# 
#         # Create nodes -------------------------------------------------
#         env.reset()
# 
#         neuronParam = {'cm': 0.2,  # [nF]
#                        'tau_m': 20.0,  # [ms]
#                        'v_rest': -20.,  # [mV]
#                        'v_thresh': -10.,  # [mV]
#                        'v_reset': -70.,  # [mV]
#                        'tau_syn_E': 5.0,  # [ms]
#                        'tau_syn_I': 5.0,  # [ms]
#                        'tau_refrac': 0.1,  # [ms]
#                        'e_rev_I' : -100., #[mV]
#                        'e_rev_E' : 60., #[mV]
#                        }
# 
#         # Create excitatory and inhibitory populations
#         nodes = self.brainscales_environment.Population(N_neurons, env.IF_cond_exp, neuronParam)
# 
#         nodes_E = nodes[:N_E]
#         nodes_I = nodes[N_E:]
# 
#         nodes_E.record()
#         nodes_I.record()
# 
#         Rs = 50.  #[Hz]
# 
#         if self.task == Tasks.XOR:
#             inp_spikes, targets = generate_stimuls_xor(dt_stim_ms, stim_len_ms, Rs, simtime_ms)
#         elif self.task == Tasks.FADING_MEMORY:
#             inp_spikes, targets = generate_stimuls_mem(dt_stim_ms, stim_len_ms, Rs, simtime_ms)
#             perf_threshold = 60.  # In percentage
#             delta_readout_delay_decrease = 50. * 1e-3  #[sec]
#         else:
#             raise RuntimeError("Unknown task {}".format(self.task.name))
# 
#         # create two spike generators,
#         # set their spike_times of i-th generator to inp_spikes[i]
# 
#         spike_generators = []
#         spike_generators.append(env.Population(1, env.SpikeSourceArray, {'spike_times': inp_spikes[0]}))
#         spike_generators.append(env.Population(1, env.SpikeSourceArray, {'spike_times': inp_spikes[1]}))
# 
#         # Connect nodes ------------------------------------------------
#         # connect E to E with excitatory synapse model and fixed indegree C_E
#         sh_w = .7
# 
#         weight_random = RandomDistribution("normal", parameters=(J_EE, sh_w * J_EE), boundaries=(0., np.inf))
#         connector = env.FixedNumberPreConnector(C_E, weights=1)
#         projection = self.brainscales_environment.Projection(nodes_E, nodes_E, connector, weight_random,
#                                                              target='excitatory', label="exex")
# 
#         # connect E to I with static synapse model and fixed indegree C_E
#         weight_random = RandomDistribution("normal", parameters=(J_EI, sh_w * J_EI), boundaries=(0., np.inf))
#         connector = env.FixedNumberPreConnector(C_E, weights=1)
#         projection = self.brainscales_environment.Projection(nodes_E, nodes_I, connector, weight_random,
#                                                              target='excitatory', label="exin")
# 
#         # connect I to E with static synapse model and fixed indegree C_I
#         weight_random = RandomDistribution("normal", parameters=(J_IE, sh_w * abs(J_IE)), boundaries=(0., np.inf))
#         connector = env.FixedNumberPreConnector(C_I, weights=1)
#         projection = self.brainscales_environment.Projection(nodes_I, nodes_E, connector, weight_random,
#                                                              target='inhibitory', label="inex")
# 
#         # connect I to I with static synapse model and fixed indegree C_I
#         weight_random = RandomDistribution("normal", parameters=(J_II, sh_w * abs(J_II)), boundaries=(0., np.inf))
#         connector = env.FixedNumberPreConnector(C_I, weights=1)
#         projection = self.brainscales_environment.Projection(nodes_I, nodes_I, connector, weight_random,
#                                                              target='inhibitory', label="inin")
# 
#         # connect input neurons to E-pool
#         # Each input neuron makes C_input synapses
#         # distribute weights uniformly in (2.5*J_EE, 7.5*J_EE)
#         INPUT_ON = True
#         if INPUT_ON:
#             #TODO: WARNING: Workaround for missing poisson spike generator 
#             weight_random = RandomDistribution("uniform", parameters=(1. * J_EE, 2. * J_EE))
#             rd1 = np.random.randint(0, len(nodes), C_inp)
#             conn_population_1 = env.PopulationView(nodes, rd1)
#             rd2 = np.random.randint(0, len(nodes), C_inp)
#             conn_population_2 = env.PopulationView(nodes, rd2)
# 
#             # NOTE: FixedNumberPostConnector doesn't seem to work here for NEST 2.6.0
#             connector = env.AllToAllConnector(allow_self_connections=False, weights=1)
#             projection = self.brainscales_environment.Projection(spike_generators[0], conn_population_1, connector,
#                                                                  weight_random)
#             projection = self.brainscales_environment.Projection(spike_generators[1], conn_population_2, connector,
#                                                                  weight_random)
#         self.brainscales_environment.run_mapping(simtime_ms)
# 
#         # call at least once
#         self.brainscales_environment.set_sthal_params()
# 
#         self.brainscales_environment.marocco.skip_mapping = True
#         self.brainscales_environment.marocco.backend = PyMarocco.Hardware
#         self.brainscales_environment.marocco.hicann_configurator = PyMarocco.HICANNv4Configurator
# 
#         self.brainscales_environment.set_hardware_weights()
# 
#         # simulate
#         env.run(simtime_ms)
# 
#         logger.debug("Done")
# 
#         logger.info("Calculating rates")
#         #compute excitatory rate
#         spike_times_ms_E = nodes_E[:int(fraction_neurons_recorded * N_E)].getSpikes()  # returns spike times in ms
#         spike_times_ms_I = nodes_I[:int(fraction_neurons_recorded * N_I)].getSpikes()  # returns spike times in ms
#         # logger.debug("spike_times_ms_E: \n %s", spike_times_ms_E.shape)
#         n_eventsE = len(spike_times_ms_E)
#         n_eventsI = len(spike_times_ms_I)
# 
#         rate_ex = n_eventsE / simtime_ms / N_E * 1000.0
#         logger.info('Excitatory rate   : %.2f Hz', rate_ex)
# 
#         #compute inhibitory rate
#         rate_in = n_eventsI / simtime_ms / N_I * 1000.0
#         logger.info('Inhibitory rate   : %.2f Hz', rate_in)
#         logger.debug("Done")
# 
#         spike_times_ms_E = convert_spiketrains(spike_times_ms_E)
#         spike_times_ms_I = convert_spiketrains(spike_times_ms_I)
# 
#         if debug or should_plot:
#             num_stims_to_plot = 3
#             logger.debug("Plotting %d stimuli (%.2f ms)", num_stims_to_plot, num_stims_to_plot * dt_stim_ms)
#             logger.debug("Plotting first")
#             plot_spiketrains(spike_times_ms_E, spike_times_ms_I, [0, num_stims_to_plot * dt_stim_ms],
#                              'raster-start.png')
#             logger.debug("Done")
#             logger.debug("Plotting second")
#             plot_spiketrains(spike_times_ms_E, spike_times_ms_I,
#                              [(simtime_ms - num_stims_to_plot * dt_stim_ms), simtime_ms],
#                              'raster-end.png')
#             logger.debug("Done.")
# 
#             if debug:
#                 return 0, 0
# 
#         all_targets = copy.copy(targets)
# 
#         should_rerun_training = True
#         while should_rerun_training:
#             logger.info("Running training with readout delay of %f seconds", readout_delay)
# 
#             rec_times = np.arange(dt_stim_ms / 1000., simtime_ms / 1000., dt_stim_ms / 1000.)
#             rec_times += (stim_len_ms / 1000. + readout_delay)
# 
#             logger.debug("Extract Liquid States...")
#             logger.debug("There are %d times. Min time is %.2f, max time is %.2f", len(rec_times), min(rec_times),
#                          max(rec_times))
# 
#             tau_lsm = 0.020  #[sec]
#             all_states = get_liquid_states(np.array(spike_times_ms_E) * 1e-3, rec_times, tau_lsm)
#             logger.debug("There are %d states and %d targets", len(all_states), len(all_targets))
# 
#             states = all_states[5:, :]  # disregard first 5 stimuli
#             targets = all_targets[5:]
#             logger.debug("(After discarding) There are %d states and %d targets", len(states), len(targets))
#             Nstates = np.size(states, 0)
#             # add constant component to states for bias
#             states = np.hstack([states, np.ones((Nstates, 1))])
# 
#             train_frac = 0.8
#             NUM_TRAIN = 30
#             perf_train = np.zeros(NUM_TRAIN)
#             perf_test = np.zeros(NUM_TRAIN)
#             # logger.debug("Computing Least Squares...")
# 
#             best_params = find_best_hyper_params(states, targets)
# 
#             logger.debug("Best parameters are %s", best_params)
# 
#             for trial in range(NUM_TRAIN):
#                 states_train, states_test, targets_train, targets_test = divide_train_test(states, targets, train_frac)
#                 w = train_readout(states_train, targets_train, best_params)
#                 perf_train[trial] = test_readout(w, states_train, targets_train)
#                 perf_test[trial] = test_readout(w, states_test, targets_test)
# 
#             training_perf_mean, training_perf_std = np.mean(perf_train) * 100, np.std(perf_train * 100)
#             testing_perf_mean, testing_perf_std = np.mean(perf_test) * 100, np.std(perf_test * 100)
# 
#             logger.info("Training performance for %s task was %f %% +- %f and Testing performance was %f %% +- %f "
#                         "for weights jee: %.2f, jie: %.2f and readout delay %f seconds",
#                         'xor', training_perf_mean, training_perf_std, testing_perf_mean, testing_perf_std,
#                         jee, jie, readout_delay)
#             
#             print(testing_perf_mean)
# 
#             should_rerun_training = False
#             # For all other tasks, don't rerun
#             if self.task == Tasks.FADING_MEMORY:
#                 logger.debug("Task was %s. testing_perf_mean was %.2f, perf_threshold %.2f, difference %f",
#                              self.task.name, testing_perf_mean, perf_threshold,
#                              (readout_delay - delta_readout_delay_decrease))
#                 if testing_perf_mean < perf_threshold and (readout_delay - delta_readout_delay_decrease) > 0.:
#                     readout_delay -= delta_readout_delay_decrease
#                     should_rerun_training = True
#                     # continue
# 
#         return readout_delay, testing_perf_mean
# 
#     def end(self):
#         logger.info("End of all experiments. Cleaning up...")
#         # There's nothing to clean up though
# 
# 
# def main():
#     import yaml
#     import os
#     import logging.config
# 
#     from ltl import DummyTrajectory
#     from ltl.paths import Paths
#     from ltl import timed
# 
#     # TODO: Set root_dir_path here
#     paths = Paths('ltl-lsm-xor', dict(run_num='test'), root_dir_path='/wang/users/s2ext_scherr/cluster_home/simulations')
#     with open("bin/logging.yaml") as f:
#         l_dict = yaml.load(f)
#         log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
#         l_dict['handlers']['file']['filename'] = log_output_file
#         logging.config.dictConfig(l_dict)
# 
#     # np.random.seed(42)
# 
#     fake_traj = DummyTrajectory()
#     optimizee = LSMOptimizee(fake_traj, task=Tasks.XOR, n_NEST_threads=15)
# 
#     d = optimizee.create_individual()
#     d['jee'] = 12
#     d['jie'] = 9
#     fake_traj.individual = sdict(d)
# 
#     with timed(logger):
#         testing_performance = optimizee.simulate(fake_traj, should_plot=True)
#     logger.info("Testing performance is %s", testing_performance)
# 
# if __name__ == "__main__":
#     main()

