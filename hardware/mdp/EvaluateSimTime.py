import numpy as np
import pylab
import sys
import os
import tqdm
import collections
import pypet

if __name__ == '__main__':

    import h5py

    analyzeLTLFile = True

    if len(sys.argv) != 2:
        raise Exception('No HDF file given!')

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]) == True:

        print ('Folder mode')
            
        simulationTimes = []
        #Iterate over all given files and handle them
        fileNr = 0
        it = tqdm.tqdm(os.listdir(sys.argv[1]))
        for f in it:
            hdf5File = h5py.File(sys.argv[1] + '/' + f, 'r')
            simulationTime = hdf5File['simulationTime'][0]
            hdf5File.close()
            simulationTimes.append(simulationTime)
        
        simulationTimes = np.array(simulationTimes)
        means = np.mean(simulationTimes)
        variances = np.std(simulationTimes)

        print ('$%.2f' % means + ' \pm ' + '%.2f$' % variances)
        exit()

    #Read the hdf5 file and collect the results
    hdf5File = h5py.File(sys.argv[1], 'r')

    if analyzeLTLFile:
        simulationTime = 0
        traj = pypet.Trajectory(filename=sys.argv[1])
        traj.v_auto_load = True
        traj.f_load(index=-1, force=True)
        it = tqdm.tqdm(range(len(traj)))
        for i in it:
            simulationTime += hdf5File[traj.name]['overview']['runs'][i][4] - hdf5File[traj.name]['overview']['runs'][i][3]
        simulationTimeWithMulticors = hdf5File[traj.name]['overview']['runs'][-1][4] - hdf5File[traj.name]['overview']['runs'][0][3]
        
    else:
        simulationTime = hdf5File['simulationTime'][0]

    hdf5File.close()
    print ('Simtime: ' + str(simulationTime))
    
