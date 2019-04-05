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

    if len(sys.argv) != 3:
        raise Exception('No HDF file given!')

    if len(sys.argv) == 3 and os.path.isdir(sys.argv[2]) == True:

        
        referenceFile = h5py.File(sys.argv[1], 'r')
        isMaze = True
        
        try:
            refMaze = np.array(referenceFile['maze'][:])
            refGoalStateX = referenceFile['goalX'][0]
            refGoalStateY = referenceFile['goalY'][0]

        except:
            isMaze = False
            try:
                refP = np.array(referenceFile['P'][:])
                refR = np.array(referenceFile['R'][:])

            except:
                print('Problem class not detected')
                exit()

        referenceFile.close()

        if isMaze:
            print('Problem class: MAZE')
        else:
            print('Problem class: MDP')

        #Iterate over all given files and handle them
        fileNr = 0
        it = tqdm.tqdm(os.listdir(sys.argv[2]))
        for f in it:
            hdf5File = h5py.File(sys.argv[2] + '/' + f, 'r')
            
            if isMaze:
                maze = np.array(hdf5File['maze'][:])
                goalStateX = hdf5File['goalX'][0]
                goalStateY = hdf5File['goalY'][0]

                if np.array_equal(refMaze, maze) and np.array_equal(refGoalStateX, goalStateX) and np.array_equal(refGoalStateY, goalStateY):
                    print('Found : ' + str(sys.argv[2] + '/' + f))
                    hdf5File.close()
                    exit()

                
            else:
                P = np.array(hdf5File['P'][:])
                R = np.array(hdf5File['R'][:])

                if np.array_equal(refR, R) and np.array_equal(refP, P):
                    print('Found : ' + str(sys.argv[2] + '/' + f))
                    hdf5File.close()
                    exit()
            
            hdf5File.close()

        print('Not found!')
        exit()
    
