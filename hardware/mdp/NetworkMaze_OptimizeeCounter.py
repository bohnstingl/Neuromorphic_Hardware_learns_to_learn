import pydls as dls
import json
import helpers as hp
import pylogging
import Utils as utils
import numpy as np
import struct
import mdptoolbox
import mdptoolbox.example
import fractional as frac
import os
import sys
import pickle
import subprocess
sys.path.append("../LTL")
sys.path.append("..")

from ltl.optimizees.optimizee import Optimizee
if __name__ == '__main__':
    
    cmd = ['srun', '-p', 'dls', '--gres', '07', 'python', './NetworkMaze_OptimizeePart.py', '0.53905861', '0.26051456', '0.7682960', '0.4686936', '0.', '0.46840302', '0.10344514', '0.98030068']

    result = np.float64(subprocess.check_output(cmd).split('\n')[-2])
    
    print result + 1, result - 1
