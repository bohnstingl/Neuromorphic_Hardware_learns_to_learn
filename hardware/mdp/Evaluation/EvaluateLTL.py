import numpy as np
import pylab
import matplotlib.pyplot as plt
import sys
import time
import pypet as pypet
import matplotlib
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.mlab import griddata
import tqdm
import os.path
import h5py

def GetParameterRanges(individuals):

    runs, n_dim = individuals.shape
    
    ranges = np.zeros((n_dim, 2))

    ranges[:, 0] = individuals.min(0)
    ranges[:, 1] = individuals.max(0)

    return ranges

def PlotFitness(fitnesses, minLen=0, labels=[], averageLength=0, popsize=[]):

    #Plot the fitnesses over iterations
    fig, ax = plt.subplots()
    if type(fitnesses[0]) != np.ndarray:
        if popsize != []:
            x = np.arange(0, len(fitnesses))
            ax.plot(x, fitnesses, '.', alpha=0.3, color='b', label='_nolegend_')
            y = np.arange(0, len(fitnesses) / popsize) * popsize
            ax.plot(y, fitnesses.reshape((-1, popsize)).mean(1))
        else:
            x = np.arange(0, len(traj))
            ax.plot(x, fitnesses)
        
        #ax.set_title('Performance of LTL algorithm')
            
    else:
        amountFiles = fitnesses.shape[0]
        colorTable = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        #import ipdb
        #ipdb.set_trace()
        
        if popsize != []:

            for i in range(amountFiles):
                cl = colorTable[i % len(colorTable)]

                #Get the number of individuals to check that the number is larger than the minLen
                iterations = int(np.ceil(minLen / popsize[i]))
                trajLen = iterations * popsize[i]
                x = np.arange(0, trajLen)

                ax.plot(x, fitnesses[i][:trajLen], '.', alpha=0.2, color=cl, label='_nolegend_')
                y = np.arange(0, trajLen / popsize[i]) * popsize[i]
                ax.plot(y, fitnesses[i][:trajLen].reshape((-1, popsize[i])).mean(1), color=cl)
            
        else:
            #For simplicity just plot the
            x = np.arange(0, minLen)
            for i in range(amountFiles):
                ax.plot(x, fitnesses[i, :])

        #ax.set_title('Comparison of LTL algorithms')

        ax.legend(labels)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Evaluated individuals', fontsize=34, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=34, fontweight='bold')

    #plt.savefig('Publishable_Results/MDP/Random_2_4/TDLam/LTL_Compare_2_4_TDLam.png', dpi = 300, bbox_inches='tight')
    plt.show()

def Plot3DFitness(individuals, fitnesses):
    fig = plt.figure()

    ranges = GetParameterRanges(individuals)

    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(np.linspace(ranges[0,0], ranges[0,1], 100), np.linspace(ranges[1,0], ranges[1,1], 100))

    xi = np.linspace(ranges[0,0], ranges[0,1], 100)
    yi = np.linspace(ranges[1,0], ranges[1,1], 100)
    zi = griddata(individuals[:, 0], individuals[:, 1], fitnesses[:], xi, yi, interp='linear')
    
    #ax.scatter(individuals[:, 0], individuals[:, 1], fitnesses[:])
    
    cset = ax.contourf(x, y, zi, zdir='z', offset=fitnesses.min(), cmap='magma')
    #cset = ax.contourf(x, y, zi, zdir='x', offset=ranges[0,0],cmap=cm.coolwarm)
    #cset = ax.contourf(x, y, zi, zdir='y', offset=ranges[1,0],cmap=cm.coolwarm)

    ax.plot_surface(x, y, zi, rstride=10, cstride=10, alpha=0.6)

    ax.set_xlabel('Eta', fontsize=34, fontweight='bold')
    ax.set_ylabel('Gamma', fontsize=34, fontweight='bold')
    ax.set_zlabel('Fitness', fontsize=34, fontweight='bold')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

    #Plot the 2D fitness projection
    fig = plt.figure()

    ranges = GetParameterRanges(individuals)
    ax = fig.add_subplot(111)
    cset = ax.contourf(x, y, zi, zdir='z', offset=fitnesses.min(), cmap='magma')
    fig.colorbar(cset, orientation='vertical', format='%.0e')
    #ax.set_title('Fitness landscape')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Eta', fontsize=34, fontweight='bold')
    ax.set_ylabel('Gamma', fontsize=34, fontweight='bold')

    plt.show()

def GetPopSize(traj):

    #Check if this is an ES run and mirror sampling is turned on
    factor = 1
    
    try:        
        if traj.parameters.mirrored_sampling_enabled:
            factor = 2
    except:
        pass
    
    try:
        return factor * traj.parameters.pop_size
    except:
        try:
            return factor * traj.parameters.max_pop_size
        except:
            try:
                return factor * traj.parameters.n_random_steps
            except:
                return factor * traj.parameters.n_parallel_runs

def GetIterations(traj):

    return traj.parameters.n_iteration

if __name__ == '__main__':

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 28}

    matplotlib.rc('font', **font)
		
    '''Check if all given parameters are files'''
    allFiles = True
    for f in range(1, len(sys.argv), 2):
        if not os.path.isfile(sys.argv[f]):
	        allFiles = False

    if len(sys.argv) >= 3 and allFiles:

        print('Operating in comparison mode')

        #Readin the full trajectories
        #Determine the lengths of the trajectories
        trajectories = []
        amountFiles = int((len(sys.argv) - 1) / 2)
        labels = [sys.argv[x] for x in range(2, len(sys.argv), 2)]
        #individuals = np.zeros((amountFiles, minLen, 2))
        popsizes = np.zeros(amountFiles, dtype=np.int)
        fitnesses = []

        minLen = np.inf

        fileIndex = 0
        for f in range(1, len(sys.argv), 2):
            fi = sys.argv[f]

            traj = pypet.Trajectory(filename=fi)
            traj.v_auto_load = True
            traj.f_load(index=-1, force=True)

            trajFitnesses = []
            #trajIndividuals = np.zeros((minLen, 2))

            popsizes[fileIndex] = GetPopSize(traj)
            iterations = GetIterations(traj)
            trajLen = iterations * popsizes[fileIndex]

            if trajLen < minLen:
                minLen = trajLen

            #Iterate over all runs and compute the fitness
            li = tqdm.tqdm(xrange(trajLen))
            for i in li:
                traj.v_idx = i
                fit = traj.results.crun.fitness

                if type(fit) == tuple:
                    fit = fit[0]

                if fit < 0:
                    fit = -1 * fit

                trajFitnesses.append(fit)
                #trajIndividuals[i] = traj.individual.eta, traj.individual.gamma

            trajFitnesses = np.array(trajFitnesses)
            fitnesses.append(trajFitnesses)
            #individuals[fileIndex, :, :] = trajIndividuals
            fileIndex += 1
        
        fitnesses = np.array(fitnesses)

        PlotFitness(fitnesses, minLen=minLen, labels=labels, popsize=popsizes)

        exit()

    if len(sys.argv) == 3:
        print('Operating in single representation mode with given trajectory')

        #Load the trajectory    
        traj = pypet.Trajectory(sys.argv[2], filename=sys.argv[1])
        traj.v_auto_load = True
        traj.f_load(index=-1, force=True)

        popsize = GetPopSize(traj)
        iterations = GetIterations(traj)
        trajLen = iterations * popsize
            
        fitnesses = np.zeros(trajLen)
        individuals = np.zeros((trajLen, 2))

        #Iterate over all runs and compute the fitness
        li = tqdm.tqdm(xrange(trajLen))
        for i in li:
            traj.v_idx = i
            fit = traj.results.crun.fitness

            if type(fit) == tuple:
                fit = fit[0]

            if fit < 0:
                fit = -1 * fit

            fitnesses[i] = fit
            individuals[i] = traj.individual.eta, traj.individual.gamma

        PlotFitness(fitnesses, popsize=popsize)

        #Plot the 3D fitness
        Plot3DFitness(individuals, fitnesses)

        exit()

    if len(sys.argv) == 2:
        print('Operating in single representation mode with the last trajectory')

        #Load the trajectory    
        traj = pypet.Trajectory(filename=sys.argv[1])
        traj.v_auto_load = True
        traj.f_load(index=-1, force=True)

        popsize = GetPopSize(traj)
        iterations = GetIterations(traj)
        trajLen = iterations * popsize

        fitnesses = np.zeros(trajLen)
        individuals = np.zeros((trajLen, 2))

        #Iterate over all runs and compute the fitness
        li = tqdm.tqdm(xrange(trajLen))
        for i in li:
            traj.v_idx = i

            fit = traj.results.crun.fitness

            if type(fit) == tuple:
                fit = fit[0]

            if fit < 0:
                fit = -1 * fit

            fitnesses[i] = fit
            individuals[i] = traj.individual.eta, traj.individual.gamma

        PlotFitness(fitnesses, popsize=popsize)

        #Plot the 3D fitness
        Plot3DFitness(individuals, fitnesses)

        exit()
	
    print('Usage EvaluateLTL.py LTL-Folder LTL-MDP-GS_2018_01')
    raise Exception('No HDF file and trajectory name given!')
