import logging.config
import os
import numpy as np
import sys

from pypet import Environment
from pypet import pypetconstants
from ltl.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
from ltl.paths import Paths
from ltl.recorder import Recorder
from ltl.logging_tools import create_shared_logger_data, configure_loggers
from StateActionOptimizee import StateActionOptimizee

logger = logging.getLogger('bin.ltl-fun-gs')


def main():
    name = 'LTL-MDP-GS'
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

    # NOTE: Innerloop simulator
    optimizee = StateActionOptimizee(traj)

    # NOTE: Outerloop optimizer initialization
    n_grid_divs_per_axis = 50
    parameters = GridSearchParameters(param_grid={
        'gamma': (optimizee.bound[0], optimizee.bound[1], n_grid_divs_per_axis),
        #'lam': (optimizee.bound[0], optimizee.bound[1], n_grid_divs_per_axis),
        'eta': (optimizee.bound[0], optimizee.bound[1], n_grid_divs_per_axis),
    })
    optimizer = GridSearchOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(-1.),
                                    parameters=parameters)

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()


if __name__ == '__main__':
    main()
