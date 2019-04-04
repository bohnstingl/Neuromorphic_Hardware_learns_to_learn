import logging.config
import os
import sys
sys.path.append("../LTL")
sys.path.append("..")
from pypet import Environment
from pypet import pypetconstants

from ltl import dict_to_list
from ltl.dataprocessing import get_skeleton_traj, get_var_from_runs, get_var_from_generations
from ltl.optimizees.functions import tools as function_tools
from ltl.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from ltl.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from ltl.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from ltl.paths import Paths
from ltl.recorder import Recorder
from StateActionMazeOptimizee import StateActionMazeOptimizee
import numpy as np
from ltl.logging_tools import create_shared_logger_data, configure_loggers

logger = logging.getLogger('bin.ltl-fun-es')


def main():
    name = 'LTL-MAZE-ES_15x15_TD1'
    try:
        with open('path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation"
        )

    paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, 'data.h5')
    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    env = Environment(trajectory=name, filename=traj_file, file_title=u'{} data'.format(name),
                      comment=u'{} data'.format(name),
                      add_time=True,
                      freeze_input=True,
                      multiproc=True,
                      use_scoop=True,
                      wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      automatic_storing=True,
                      log_stdout=False,  # Sends stdout to logs
                      log_folder=os.path.join(paths.output_dir_path, 'logs')
                      )
    create_shared_logger_data(logger_names=['bin', 'optimizers'],
                              log_levels=['INFO', 'INFO'],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    ## Benchmark function

    optimizee = StateActionMazeOptimizee(traj)

    ## Innerloop simulator

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        #learning_rate_decay=0.95,
        noise_std=0.1,
        mirrored_sampling_enabled=True,
        fitness_shaping_enabled=True,
        pop_size=24,
        n_iteration=30,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(-1.,),
        parameters=parameters,
        optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Add Recorder
    recorder = Recorder(trajectory=traj,
                        optimizee_name='SNN StateAction', optimizee_parameters=['gamma', 'eta'],
                        optimizer_name=optimizer.__class__.__name__,
                        optimizer_parameters=optimizer.get_params())
    recorder.start()

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)
    recorder.end()

    # Finally disable logging and close all log-files
    env.disable_logging()



if __name__ == '__main__':
    main()
