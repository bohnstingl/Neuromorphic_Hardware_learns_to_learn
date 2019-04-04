from __future__ import with_statement
from __future__ import absolute_import
import logging.config
import os

from pypet import Environment
from pypet import pypetconstants
import sys
import json
import yaml
import pydls as dls
sys.path.append('.')

from bandit_optimizee_ann import BanditOptimizee, BanditParameters
from snn_mab import Connector, ANNLearningRule

from ltl.optimizers.crossentropy.distribution import NoisyGaussian, Gaussian
from ltl.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from ltl.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from ltl.optimizers.gradientdescent import GradientDescentOptimizer, ClassicGDParameters
from ltl.optimizers.simulatedannealing.optimizer import \
    SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from ltl.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
from ltl.paths import Paths
from ltl.recorder import Recorder

import numpy as np
from ltl.logging_tools import create_shared_logger_data, configure_loggers

logger = logging.getLogger(u'bin.ltl-mab')


def main(dependent, optimizer):
    opt = optimizer.upper()
    identifier = '{:05x}'.format(np.random.randint(16**5))
    print('Identifier: ' + identifier)
    allocated_id = '07'  # dls.get_allocated_board_ids()[0]
    board_calibration_map = {'B291698' : {'dac' : 'dac_default.json',
                                          'cap' : 'cap_mem_29.json'},
                             '07' : {'dac' : 'dac_07_chip_20.json',
                                     'cap' : 'calibration_20.json'},
                             'B201319' : {'dac' : 'dac_B201319_chip_21.json',
                                          'cap' : 'calibration_24.json'},
                             'B201330' : {'dac' : 'dac_B201330_chip_22.json',
                                          'cap' : 'calibration_22.json'}}

    dep_name = 'DEP' if dependent else 'IND'
    name = 'MAB_ANN_{}_{}_{}'.format(identifier, opt, dep_name)
    root_dir_path = os.path.expanduser('~/simulations')
    paths = Paths(name, dict(run_no=u'test'), root_dir_path=root_dir_path)

    with open(os.path.expanduser('~/LTL/bin/logging.yaml')) as f:
        l_dict = yaml.load(f)
        log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
        l_dict['handlers']['file']['filename'] = log_output_file
        logging.config.dictConfig(l_dict)

    print("All output logs can be found in directory " + str(paths.logs_path))

    traj_file = os.path.join(paths.output_dir_path, u'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title=u'{} data'.format(name),
                      comment=u'{} data'.format(name),
                      add_time=True,
                      # freeze_input=True,
                      # multiproc=True,
                      # use_scoop=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCK,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers', 'optimizees'],
                              log_levels=['INFO', 'INFO', 'INFO'],
                              log_to_consoles=[True, True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    optimizee_seed = 100

    with open('../adv/' + board_calibration_map[allocated_id]['cap']) as f:
        calibrated_config = json.load(f)
    with open('../adv/' + board_calibration_map[allocated_id]['dac']) as f:
        dac_config = json.load(f)

    class Dummy(object):
        def __init__(self, connector):
            self.connector = connector

        def __enter__(self):
            return self.connector

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class Mgr(object):
        def __init__(self):
            self.connector = None

        def establish(self):
            return Dummy(self.connector)

    max_learning_rate = 1.

    mgr = Mgr()
    optimizee_parameters = \
        BanditParameters(n_arms=2, n_pulls=100, n_samples=40, seed=optimizee_seed,
                         max_learning_rate=max_learning_rate, learning_rule=ANNLearningRule,
                         establish_connection=mgr.establish)
    optimizee = BanditOptimizee(traj, optimizee_parameters, dp=dependent)

    # Add post processing
    optimizer = None
    pop_size = 200
    n_iteration = 60
    if opt == 'CE':
        ce_optimizer_parameters = CrossEntropyParameters(
            pop_size=pop_size, rho=0.06, smoothing=0.3, temp_decay=0,
            n_iteration=n_iteration,
            distribution=NoisyGaussian(noise_magnitude=.2, noise_decay=.925),
            #Gaussian(),#NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
            stop_criterion=np.inf, seed=102)
        ce_optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                             optimizee_fitness_weights=(1,),
                                             parameters=ce_optimizer_parameters,
                                             optimizee_bounding_func=optimizee.bounding_func)
        optimizer = ce_optimizer
    elif opt == 'ES':
        es_optimizer_parameters = EvolutionStrategiesParameters(
            learning_rate=1.8,
            learning_rate_decay=.93,
            noise_std=.03,
            mirrored_sampling_enabled=True,
            fitness_shaping_enabled=True,
            pop_size=int(pop_size / 2),
            n_iteration=n_iteration,
            stop_criterion=np.inf,
            seed=102
        )
        optimizer = EvolutionStrategiesOptimizer(
            traj,
            optimizee.create_individual,
            (1,),
            es_optimizer_parameters,
            optimizee.bounding_func
        )
    elif opt == 'GD':
        gd_parameters = ClassicGDParameters(
            learning_rate=.003,
            exploration_step_size=.1,
            n_random_steps=pop_size,
            n_iteration=n_iteration,
            stop_criterion=np.inf,
            seed=102
        )
        optimizer = GradientDescentOptimizer(
            traj,
            optimizee.create_individual,
            (1,),
            gd_parameters,
            optimizee.bounding_func
        )
    elif opt == 'SA':
        sa_parameters = SimulatedAnnealingParameters(
            n_parallel_runs=pop_size,
            noisy_step=.1,
            temp_decay=.9,
            n_iteration=n_iteration,
            stop_criterion=np.inf,
            seed=102,
            cooling_schedule=AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE
        )
        optimizer = SimulatedAnnealingOptimizer(
            traj,
            optimizee.create_individual,
            (1,),
            sa_parameters,
            optimizee.bounding_func
        )
    elif opt == 'GS':
        n_grid_points = 5
        gs_optimizer_parameters = GridSearchParameters(param_grid={
            'weight_prior': (0, 1, n_grid_points),
            'learning_rate': (0, 1, n_grid_points),
            'stim_inhibition': (0, 1, n_grid_points),
            'action_inhibition': (0, 1, n_grid_points),
            'learning_rate_decay': (0, 1, n_grid_points)
        })
        gs_optimizer = GridSearchOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                           optimizee_fitness_weights=(1,),
                                           parameters=gs_optimizer_parameters)
        optimizer = gs_optimizer
    else:
        exit(1)
    env.add_postprocessing(optimizer.post_process)

    # Add Recorder
    recorder = Recorder(trajectory=traj,
                        optimizee_name='MAB', optimizee_parameters=optimizee_parameters,
                        optimizer_name=optimizer.__class__.__name__,
                        optimizer_parameters=optimizer.get_params())
    recorder.start()

    # Run the simulation with all parameter combinations
    # optimizee.simulate(traj)
    # exit(0)
    with Connector(calibrated_config, dac_config, 3) as connector:
        mgr.connector = connector
        env.run(optimizee.simulate)
    mgr.connector.disconnect()

    ## Outerloop optimizer end
    optimizer.end(traj)
    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == u'__main__':
    dependent = False
    try:
        dependent = int(os.environ['MAB_DEPENDENT']) == 1
    except:
        pass
    optimizer = 'ce'
    try:
        optimizer = os.environ['MAB_OPTIMIZER']
    except:
        pass
    print('dependent: ', dependent)
    print('optimizer: ', optimizer)
    main(dependent, optimizer)

