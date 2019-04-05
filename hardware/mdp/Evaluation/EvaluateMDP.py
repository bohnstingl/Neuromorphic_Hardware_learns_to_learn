import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import mdptoolbox_local as mdptoolbox
import mdptoolbox_local.example
import os
import tqdm
import collections

def printLatexTableEntry(nwPerf, viPerf, qlPerf, qlfPerf, rndPerf):

    line = '$%.0f' % np.float64(nwPerf.mean(0)[-1]) + ' \pm ' + '%.0f$' % np.float64(np.std(nwPerf, axis=0)[-1])
    line += ' & $%.0f' % np.float64(viPerf.mean(0)[-1]) + ' \pm ' + '%.0f$' % np.float64(np.std(viPerf, axis=0)[-1])
    line += ' & $%.0f' % np.float64(qlPerf.mean(0)[-1]) + ' \pm ' + '%.0f$' % np.float64(np.std(qlPerf, axis=0)[-1])    
    line += ' & $%.0f' % np.float64(qlfPerf.mean(0)[-1]) + ' \pm ' + '%.0f$' % np.float64(np.std(qlfPerf, axis=0)[-1])
    line += ' & $%.0f' % np.float64(rndPerf.mean(0)[-1]) + ' \pm ' + '%.0f$' % np.float64(np.std(rndPerf, axis=0)[-1])

    print (line)

def expectedReward(R, P, oldState):

    cumReward = 0.
    #Combine rewards with probabilities to expected rewards
    for a in range(R.shape[0]):
        for newState in range(R.shape[1]):
            reward = R[a][oldState][newState]
            prob = P[a][oldState][newState]

            cumReward += reward * prob

    return cumReward

#Compute the cumulative regrets between the action taken and the best actions
def cum_regret(states, actions, rewards, R, P):
    
    regrets = []
    max_rewards = []

    for index in range(len(states) - 1):
        oldState = states[index]
        action = actions[index]
        reward = rewards[index]

        max_R = -np.inf
        for a in range(R.shape[0]):
            m = np.max(R[a][oldState][:])
            if m > max_R:
                max_R = m

        max_rewards.append(max_R)
        
        cumReward = expectedReward(R, P, oldState)
        regrets.append(cumReward - reward)

    return np.cumsum(regrets)
    
def averageCumRegret(stateMat, actionMat, rewardMat, R, P, multipleRuns):
	
    #Compute the average cumulative regret
    cumregret = []
    for run in range(multipleRuns):
        cumregret.append(cum_regret(stateMat[run, :], actionMat[run, :], rewardMat[run, :], R, P))
    cumregret = np.array(cumregret)

    #Compute the sum of the cumulative regret for the single runs
    sumCumregret = cumregret.sum(1)
    
    #Get the maximal and the minimal value of the cumregret
    maxCumregret = cumregret[np.argmax(sumCumregret)]
    minCumregret = cumregret[np.argmin(sumCumregret)]

    return cumregret.mean(0), minCumregret, maxCumregret

def cum_reward(states, actions, rewards, R, P):

    cumReward = []

    for index in range(len(states) - 1):
        oldState = states[index]
        action = actions[index]
        reward = rewards[index]

        cumReward.append(reward)

    cumReward = np.array(cumReward)

    #return meanReward, minReward, maxReward
    return np.cumsum(cumReward)

def averageCumReward(stateMat, actionMat, rewardMat, R, P, multipleRuns):
	
    #Compute the average cumulative regret
    cumReward = []
    for run in range(multipleRuns):
        cumReward.append(cum_reward(stateMat[run, :], actionMat[run, :], rewardMat[run, :], R, P))
    cumReward = np.array(cumReward)

    #Compute the sum of the cumulative regret for the single runs
    sumCumreward = cumReward.sum(1)
    
    #Get the maximal and the minimal value of the cumregret
    maxCumreward = cumReward[np.argmax(sumCumreward)]
    minCumreward = cumReward[np.argmin(sumCumreward)]

    return cumReward.mean(0), minCumreward, maxCumreward

def averageRewardPerState(states, rewards, targetState, R, P, multipleRuns, averagingWindow=10):

    targetRewards = []
    maximumReward = np.max(R[:, targetState, :])
    for run in range(multipleRuns):
        r = rewards[run][states[run] == targetState] / maximumReward
        targetRewards.append(np.array(r))

    targetRewards = np.array(targetRewards)
    sumRewards = targetRewards.sum(1)

    #Get the maximal and the minimal value of the cumregret
    maxReward = targetRewards[np.argmax(sumRewards)]
    minReward = targetRewards[np.argmin(sumRewards)]
    meanReward = targetRewards.mean(0)

    #Perform the averaging
    averageTarget = np.zeros((3, targetRewards.shape[1] / averagingWindow))
    cnt = 0

    #This gives only integer average sections. The last section will be discarded
    for lower in range(targetRewards.shape[1] / averagingWindow):
        averageTarget[0][cnt] = meanReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        averageTarget[1][cnt] = minReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        averageTarget[2][cnt] = maxReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        cnt += 1

    return averageTarget[0], averageTarget[1], averageTarget[2], maximumReward

def averageReward(states, rewards, R, P, multipleRuns, averagingWindow=10):
    #This function computes the relative amount of reward per iteration.
    #r from agent and r0 from optimal algo -> return r/r0

    relativeReward = []

    for run in range(multipleRuns):

        relativeRewardPerState = []
        for index in range(len(states[run]) - 1):
            oldState = states[run][index]
            newState = states[run][index + 1]
            reward = rewards[run][index]

            maximumReward = np.max(R[:, oldState, newState])
            minimumReward = np.min(R[:, oldState, newState])

            #This can only happen if the environment is reset to another state
            if reward < minimumReward or reward > maximumReward:
                reward = maximumReward

            k = 1. / (-minimumReward + maximumReward)
            d = 1. - k * maximumReward

            relativeRewardPerState.append(k * reward + d)

        relativeReward.append(relativeRewardPerState)

    relativeReward = np.array(relativeReward)
    sumRewards = relativeReward.sum(1)

    #Get the maximal and the minimal value of the cumregret
    maxReward = relativeReward[np.argmax(sumRewards)]
    minReward = relativeReward[np.argmin(sumRewards)]
    meanReward = relativeReward.mean(0)

    #Perform the averaging
    averageTarget = np.zeros((3, relativeReward.shape[1] / averagingWindow))
    cnt = 0

    #This gives only integer average sections. The last section will be discarded
    for lower in range(relativeReward.shape[1] / averagingWindow):
        averageTarget[0][cnt] = meanReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        averageTarget[1][cnt] = minReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        averageTarget[2][cnt] = maxReward[(lower * averagingWindow):(lower + 1) * averagingWindow].mean(0)
        cnt += 1

    #return meanReward, minReward, maxReward
    return averageTarget[0], averageTarget[1], averageTarget[2]
    

def trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates, averageRnd=False):
        
    #The MDP reference algorithm cannot deal with gamma = 0.
    if gamma == 0.:
        gamma = 0.0001
    if gamma == 1.:
        gamma = 0.9999

    average_runs = 2
    #gamma = 0.99

    cnt = 0
    val = np.zeros((average_runs, iterations))

    while cnt < average_runs:
        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        vi.run()

        viStates, viActions, viRewards = playPolicy(initialState, vi.policy, P, R, iterations)

        val[cnt, :] = viRewards
        cnt += 1
    
    viRewards = np.mean(val[:, :], axis=0)
    
    
    #Solve the MDP with normal Q-Learning
    cnt = 0
    val = np.zeros((average_runs, iterations))
    while cnt < average_runs:
        qlf = mdptoolbox.mdp.QLearning(P, R, gamma, n_iter=10**5)#10**5)
        qlf.run()
        #print qlf.policy
        
        qlfStates = np.array(qlf.states)[:iterations]
        qlfActions = np.array(qlf.actions)[:iterations]
        qlfRewards = np.array(qlf.rewards)[:iterations]

        val[cnt, :] = qlfRewards
        cnt += 1
    
    qlfRewards = np.mean(val[:, :], axis=0)    

    #viStates = qlfStates
    #viActions = qlfActions
    #viRewards = qlfRewards

    #Solve the MDP with Q-Learning and epsilon decay
    cnt = 0
    val = np.zeros((average_runs, iterations))
    while cnt < average_runs:
        ql = mdptoolbox.mdp.QLearningEps(P, R, gamma, 0.30, 0.99, n_iter=10**5)#10**8)
        ql.run()

        qlStates = np.array(ql.states)[:iterations]
        qlActions = np.array(ql.actions)[:iterations]
        qlRewards = np.array(ql.rewards)[:iterations]

        val[cnt, :] = qlRewards
        cnt += 1

    qlRewards = np.mean(val[:, :], axis=0)

    #Solve the MDP with a random Policy, avoid having a random policy with a higher rewards than Q-Learning
    randomCnt = 0
    randRewardsList = np.zeros((1000, iterations))

    while (randomCnt < 1000 and averageRnd) or (randomCnt < 1 and not averageRnd):
        allowed = False
        while not allowed:
            randomPolicy = np.random.randint(0, nActions - 1, nStates)
            randStates, randActions, randRewards = playPolicy(initialState, randomPolicy, P, R, iterations)

            if np.cumsum(randRewards)[-1] < np.cumsum(viRewards)[-1]:
                allowed = True
        
        randRewardsList[randomCnt, :] = randRewards
        randomCnt += 1

    if averageRnd:
        randRewards = np.mean(randRewardsList[:, :], axis=0)
    else:
        randRewards = randRewardsList[0, :]

    return viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards

def playPolicy(state0, policy, P, R, iterations):

    states = []
    actions = []
    rewards = []

    state = state0
    
    for i in range(iterations):
        states.append(state)
        action = policy[state]
        actions.append(action)

        nextState = np.argmax(P[action][state][:])
        reward = R[action][state][nextState]
        rewards.append(reward)

        state = nextState

    return np.array(states), np.array(actions), np.array(rewards)
    
def analyzePerformance(stateSpace, actionSpace, states, actions):
	
        policy = np.ones(stateSpace) * -1
        finishLearning = 0
        thresholdPolicy = collections.deque(maxlen=30)
        stA = np.zeros((stateSpace, actionSpace))
				
        for i in range(len(states)):
            a = actions[i]
            s = states[i]

            policy[s] = a

            stA[s, a] += 1

            #if the policy stayed the same over n iterations, the learning is considered to be finished
            thresholdPolicy.append(np.copy(policy))

            #Check the stack policy for equality
            if i > 30:

                idx = 0
                li = []
                cnt = []
                for j in range(30):
                    pol = thresholdPolicy[j]

                    if li == []:
                        li.append(pol)
                        cnt.append(1)
                        continue

                    onList = False
                    for z in range(len(li)):
                        if np.array_equal(pol, li[z]):
                            onList = True
                            cnt[z] += 1
                            break

                    if not onList:
                        li.append(pol)
                        cnt.append(1)

                cnt = np.array(cnt)
                max = np.max(cnt)
                idx = np.argmax(cnt)
                liE = li[idx]

                if max > 25 and len(np.where(liE == -1)[0]) == 0:
                    finishLearning = i
                    #print stA
                    return finishLearning

        #print stA
        return 2000
		
def HandleHDF5File(f, regret=True):

    #Read the hdf5 file and collect the results
    hdf5File = h5py.File(f,'r')
    iterationsOnChip = int(hdf5File['iterationsOnChip'][0])
    trialsPerIteration = int(hdf5File['trialsPerIteration'][0])
    
    #Use a different gamma value, since the gamma value is handled as an algorithm for the current problem
    gamma = hdf5File['gamma'][0]
    multipleRuns = hdf5File['multipleRuns'][0]
    nActions = hdf5File['nActions'][0]
    nStates = hdf5File['nStates'][0]

    if hdf5File['platform'][0] == 'SW': 
        network = 'Software simulation'
        t = 'time / ms'
        platform = 0
    else:
        network = 'DLS'
        t = 'clock cycles'
        platform = 1

    #Append all collected rewards, actions, states together
    rewards = []
    actions = []
    states  = []
    spikeTimes = []
    spikeAddresses = []
    averageSpeed = []

    #Iterate over multipleRuns
    for run in range(multipleRuns):
        statesPerRun = []
        actionsPerRun = []
        rewardsPerRun = []
        spikeTimesPerRun = []
        spikeAddressesPerRun = []

        #Append and read the states, actions and rewards per iteration
        for i in range(iterationsOnChip):
            statesPerRun.extend(hdf5File['Run_' + str(run)]['states' + str(i)][:])
            actionsPerRun.extend(hdf5File['Run_' + str(run)]['actions' + str(i)][:])

            if platform == 0:
                rewardsPerRun.extend(((hdf5File['Run_' + str(run)]['rewards' + str(i)][:]) + 1.0) / 2.0)
            else:
                rewardsPerRun.extend(hdf5File['Run_' + str(run)]['rewards' + str(i)][:])

            spikeTimesPerRun.extend(hdf5File['Run_' + str(run)]['spikeTimes' + str(i)][:])
            spikeAddressesPerRun.extend(hdf5File['Run_' + str(run)]['spikeAddresses' + str(i)][:])
        	
        averageSpeed.append(analyzePerformance(nStates, nActions, statesPerRun, actionsPerRun))
        
        #Append the results of multiple runs together
        states.append(statesPerRun)
        actions.append(actionsPerRun)
        rewards.append(rewardsPerRun)
        spikeTimes.append(spikeTimesPerRun)
        spikeAddresses.append(spikeAddressesPerRun)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    P = np.array(hdf5File['P'][:])
    R = np.array(hdf5File['R'][:])
    R = (R + 1) / 2.0

    initialState = states[0][0]
    
    hdf5File.close()

    iterations = int(iterationsOnChip * trialsPerIteration)

    if regret:
        cumregret, cumregretLow, cumregretHigh = averageCumRegret(states, actions, rewards, R, P, multipleRuns)
        return platform, iterations, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed
    else:
        average, low, high = averageCumReward(states, actions, rewards, R, P, multipleRuns)
        return platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, average, low, high, averageSpeed

def PlotWithSplitAxis(ax1, ax2, xVect, data):
    
    maxVal100 = 0.
    maxValEnd = 0.

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # plot the same data on both axes
    for idx, d in enumerate(data):

        if d[100] > maxVal100:
            maxVal100 = d[100]

        if d[-1] > maxValEnd:
            maxValEnd = d[-1]

        ax1.plot(x, d, color=colors[idx])
        ax2.plot(x, d, color=colors[idx])

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, maxVal100)
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, maxValEnd)

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')
    ax2.yaxis.tick_left()
    ax2.tick_params(labelright='off')

    return ax1, ax2

def allFolders(parameters):

    for nr, f in enumerate(parameters):
        if nr > 0:
            if os.path.isdir(f) == False:
                return False
    return True

if __name__ == '__main__':

    import h5py
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)

    plotCumulativeReward = True
    splitPlot = True

    if len(sys.argv) < 2:
        raise Exception('No HDF file given!')

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]) == True:

        print ('Testing mode')

        labels = []
        averagePerFile = []
        speedPerFile = []
        viAveragePerFile = []
        qlAveragePerFile = []
        qlfAveragePerFile = []
        rAveragePerFile = []
        results = np.zeros((5, len(os.listdir(sys.argv[1])), 1999))
        
        #Iterate over all given files and handle them
        fileNr = 0
        it = tqdm.tqdm(os.listdir(sys.argv[1]))
        for f in it:
            platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[1] + '/' + f, regret=False)

            labels.append(network + ' ' + str(fileNr))
            
            averagePerFile.append(cumregret)
    
            #Learn the reference algorithms on this issue
            viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates)

            #Store the reference values per file
            viAveragePerFile.append(cum_reward(viStates, viActions, viRewards, R, P))
            qlAveragePerFile.append(cum_reward(qlStates, qlActions, qlRewards, R, P))
            qlfAveragePerFile.append(cum_reward(qlfStates, qlfActions, qlfRewards, R, P))
            rAveragePerFile.append(cum_reward(randStates, randActions, randRewards, R, P))
                
            speedPerFile.append(averageSpeed)

            fileNr += 1

        #Get the maximum and the minimum values of the different algorithms
        averagePerFile = np.array(averagePerFile)
        viAveragePerFile = np.array(viAveragePerFile)
        qlAveragePerFile = np.array(qlAveragePerFile)
        qlfAveragePerFile = np.array(qlfAveragePerFile)
        rAveragePerFile = np.array(rAveragePerFile)
        speedPerFile = np.array(speedPerFile)

        results[0, :, :] = averagePerFile
        results[1, :, :] = viAveragePerFile
        results[2, :, :] = qlAveragePerFile
        results[3, :, :] = qlfAveragePerFile
        results[4, :, :] = rAveragePerFile

        nwMean = np.mean(results[0, :, :], axis=0)
        nwMin = np.min(results[0, :, :], axis=0)
        nwMax = np.max(results[0, :, :], axis=0)
        nwvar = np.var(results[0, :, :], axis=0) / np.sqrt(results[0, :, :].shape[0])

        viMean = np.mean(results[1, :, :], axis=0)
        viMin = np.min(results[1, :, :], axis=0)
        viMax = np.max(results[1, :, :], axis=0)
        vivar = np.var(results[1, :, :], axis=0) / np.sqrt(results[1, :, :].shape[0])

        qlMean = np.mean(results[2, :, :], axis=0)
        qlMin = np.min(results[2, :, :], axis=0)
        qlMax = np.max(results[2, :, :], axis=0)
        qlvar = np.var(results[2, :, :], axis=0) / np.sqrt(results[2, :, :].shape[0])

        qlfMean = np.mean(results[3, :, :], axis=0)
        qlfMin = np.min(results[3, :, :], axis=0)
        qlfMax = np.max(results[3, :, :], axis=0)
        qlfvar = np.var(results[3, :, :], axis=0) / np.sqrt(results[3, :, :].shape[0])

        rMean = np.mean(results[4, :, :], axis=0)
        rMin = np.min(results[4, :, :], axis=0)
        rMax = np.max(results[4, :, :], axis=0)
        rvar = np.var(results[4, :, :], axis=0) / np.sqrt(results[4, :, :].shape[0])

        x = np.arange(1, iterations)
        
        fig, ax = plt.subplots(figsize=(16, 8))

        for fileNr in range(len(os.listdir(sys.argv[1]))):
            random = np.copy(results[4, fileNr, :])
            optimal = np.copy(results[1, fileNr, :] - random)

            oldVal = np.copy(results[0, fileNr, :])
            results[0, fileNr, :] = np.copy((oldVal - random) / optimal)

            valuesToCorrect = np.copy(optimal[optimal==0])
            if valuesToCorrect != []:
                valuesToCorrect = np.int(valuesToCorrect)
                results[0, fileNr, valuesToCorrect] = 0.

            #Update the references
            for i in range(1, 5):
                results[i, fileNr, :] = np.copy((results[i, fileNr, :] - random) / optimal)

        nwMean = np.mean(results[0, :, :], axis=0)
        nwMin = np.min(results[0, :, :], axis=0)
        nwMax = np.max(results[0, :, :], axis=0)
        nwvar = np.var(results[0, :, :], axis=0) / np.sqrt(results[0, :, :].shape[0])

        viMean = np.mean(results[1, :, :], axis=0)
        viMin = np.min(results[1, :, :], axis=0)
        viMax = np.max(results[1, :, :], axis=0)
        vivar = np.var(results[1, :, :], axis=0) / np.sqrt(results[1, :, :].shape[0])

        qlMean = np.mean(results[2, :, :], axis=0)
        qlMin = np.min(results[2, :, :], axis=0)
        qlMax = np.max(results[2, :, :], axis=0)
        qlvar = np.var(results[2, :, :], axis=0) / np.sqrt(results[2, :, :].shape[0])

        qlfMean = np.mean(results[3, :, :], axis=0)
        qlfMin = np.min(results[3, :, :], axis=0)
        qlfMax = np.max(results[3, :, :], axis=0)
        qlfvar = np.var(results[3, :, :], axis=0) / np.sqrt(results[3, :, :].shape[0])

        rMean = np.mean(results[4, :, :], axis=0)
        rMin = np.min(results[4, :, :], axis=0)
        rMax = np.max(results[4, :, :], axis=0)
        rvar = np.var(results[4, :, :], axis=0) / np.sqrt(results[4, :, :].shape[0])

        colors = ['c', 'k', 'k', 'gray']

        c = colors[0]
        ax.plot(x, nwMean, color=c)
        ax.fill_between(x, nwMean - nwvar, nwMean, alpha=0.05, facecolor=c, label='_nolegend_', edgecolors=c, linewidth=2.)
        ax.fill_between(x, nwMean, nwMean + nwvar, alpha=0.05, facecolor=c, label='_nolegend_', edgecolors=c, linewidth=2.)

        c = colors[1]
        ax.errorbar(x, viMean, yerr=vivar, color=c, errorevery=400, elinewidth=None, capsize=4, label='Value Iteration', linestyle='-.')

        c = colors[2]
        ax.errorbar(x, qlMean, yerr=qlvar, color=c, errorevery=400, elinewidth=None, capsize=4, label='Q-Learning', linestyle=':')

        c = colors[3]
        ax.plot(x, rMean, color=c, label='Random-Policy')
        
        ax.legend([network, 'Random-Policy', 'Value Iteration', 'Q-Learning'])
        ax.set_title('No LTL')
        ax.set_xlabel('step')

        ax.set_ylabel('Scaled cumulative reward')

        plt.tick_params(top='off', right='off')
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([-0.2, 1.2])

        ax.xaxis.set_tick_params(labelsize=12.)
        ax.yaxis.set_tick_params(labelsize=12.)
        plt.savefig('CE_new_wo_LTL.png')
        plt.show()

        #Print the table for comparison
        print ('Reward Simulation: ' + str(averagePerFile.mean(0)[-1]) + ' +- ' + str(np.std(averagePerFile, axis=0)[-1]))
        print ('Reward ValueIteration: ' + str(viAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(viAveragePerFile, axis=0)[-1]))
        print ('Reward Q-Learning (eps-decay): ' + str(qlAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(qlAveragePerFile, axis=0)[-1]))
        print ('Reward Q-Learning: ' + str(qlfAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(qlfAveragePerFile, axis=0)[-1]))
        print ('Reward Random-Policy: ' + str(rAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(rAveragePerFile, axis=0)[-1]))
        print ('Average learning speed: ' + str(speedPerFile.mean(0)) + ' +- ' + str(np.std(speedPerFile)))

        printLatexTableEntry(averagePerFile, viAveragePerFile, qlAveragePerFile, qlfAveragePerFile, rAveragePerFile)
    
    elif len(sys.argv) == 2:
        
        print('Normal mode!')

        '''Handle the file'''
        platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[1], regret=False)

        viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates)

        x = np.arange(1, iterations)

        if splitPlot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

            random = cum_reward(randStates, randActions, randRewards, R, P)
            optimal = cum_reward(viStates, viActions, viRewards, R, P) - random
                
            data = []
            data.append((cumregret - random) / optimal)
            
            data.append((cum_reward(viStates, viActions, viRewards, R, P) - random) / optimal)
            data.append((cum_reward(qlStates, qlActions, qlRewards, R, P) - random) / optimal)
            data.append((cum_reward(qlfStates, qlfActions, qlfRewards, R, P) - random) / optimal)
            data.append((cum_reward(randStates, randActions, randRewards, R, P) - random) / optimal)

            ax1, ax2 = PlotWithSplitAxis(ax1, ax2, x, data)

            ax1.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
            ax1.set_xlabel('step')
            ax2.set_xlabel('step')
            ax1.set_ylim([-0.2, 1.2])
            ax2.set_ylim([-0.2, 1.2])
            
            ax1.set_ylabel('Scaled cumulative reward')
            ax2.set_ylabel('Scaled cumulative reward')

            plt.tick_params(top='off', right='off')
            ax1.xaxis.set_tick_params(labelsize=12.)
            ax1.yaxis.set_tick_params(labelsize=12.)            
            ax2.xaxis.set_tick_params(labelsize=12.)
            ax2.yaxis.set_tick_params(labelsize=12.)

            plt.show()
            
            print (analyzePerformance(R.shape[1], R.shape[0], qlStates, qlActions))

        else:
            fig, ax = plt.subplots(figsize=(16, 8))

            #Consider the value iteration rewards as the optimal reward and as 1 all the times
            #In contrast take the random rewards and take them as 0 all the times

            random = cum_reward(randStates, randActions, randRewards, R, P)
            optimal = cum_reward(viStates, viActions, viRewards, R, P) - random

            ax.plot(x, (cumregret - random) / optimal)
            ax.fill_between(x, (cumregretLow - random) / optimal, (cumregret - random) / optimal, alpha=0.5, facecolor='blue', label='_nolegend_')
            ax.fill_between(x, (cumregret - random) / optimal, (cumregretHigh - random) / optimal, alpha=0.5, facecolor='blue', label='_nolegend_')
            
            ax.plot(x, (cum_reward(viStates, viActions, viRewards, R, P) - random) / optimal)
            ax.plot(x, (cum_reward(qlStates, qlActions, qlRewards, R, P) - random) / optimal)
            ax.plot(x, (cum_reward(qlfStates, qlfActions, qlfRewards, R, P) - random) / optimal)
            ax.plot(x, (cum_reward(randStates, randActions, randRewards, R, P) - random) / optimal)

            ax.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
            ax.set_xlabel('step')
            
            ax.set_ylabel('Scaled cumulative reward')

            plt.tick_params(top='off', right='off')
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylim([-0.2, 1.2])
            ax.xaxis.set_tick_params(labelsize=12.)
            ax.yaxis.set_tick_params(labelsize=12.)

            plt.show()

    elif len(sys.argv) > 2 and not allFolders(sys.argv):

        #When comparing multiple hdf files, it is assumed that the operator ensures to have file
        #where the same problem is investigated
        print('Comparison mode with ' + str(len(sys.argv) - 1) + ' HDF files')
    
        labels = []
        averagePerFile = []
        minPerFile = []
        maxPerFile = []

        P = None
        R = None
        iterations = None

        averagePerNetworkDLS = []
        averagePerNetworkSW = []
        
        #Iterate over all given files and handle them
        for fileNr in range(1, len(sys.argv)):

            platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[fileNr], regret=False)

            labels.append(network + ' ' + str(fileNr))

            if network == 'DLS':
                averagePerNetworkDLS.append(cumregret)
            else:
                averagePerNetworkSW.append(cumregret)
            
            averagePerFile.append(cumregret)
            minPerFile.append(cumregretLow)
            maxPerFile.append(cumregretHigh)

        averagePerNetworkDLS = np.array(averagePerNetworkDLS)
        averagePerNetworkSW = np.array(averagePerNetworkSW)
    
        #Learn the reference algorithms on this issue
        viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates)

        x = np.arange(1, iterations)

        fig, ax = plt.subplots(figsize=(16, 8))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        ax.plot(x, averagePerNetworkDLS.mean(axis=0))
        ax.plot(x, averagePerNetworkSW.mean(axis=0))

        print(averagePerNetworkDLS.shape)

        ax.legend(['DLS', 'SW'])
        ax.set_xlabel('step')

        ax.set_ylabel('cumulative reward')

        plt.tick_params(top='off', right='off')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=12.)
        ax.yaxis.set_tick_params(labelsize=12.)

        plt.show()
    
    elif len(sys.argv) > 2 and allFolders(sys.argv):

        def findSameMDP(refR, refP, folder):

            for targetF in os.listdir(folder):
                targetHDF5File = h5py.File(folder + '/' + targetF, 'r')
                P = np.array(targetHDF5File['P'][:])
                R = np.array(targetHDF5File['R'][:])
                R = (R + 1) / 2.0
                targetHDF5File.close()

                if np.array_equal(refR, R) and np.array_equal(refP, P):
                    return targetF

            raise Exception('Same problem could not be found!')

        nFolders = len(sys.argv) - 1

        print ('Testing mode with ' + str(nFolders) + ' folders')
        
        '''The labels depend on the actual executed run'''
        labels = ['DLS (LTL)', 'DLS (no LTL)']
        results = np.zeros((nFolders + 4, len(os.listdir(sys.argv[1])), 1999))
        problemFiles = []

        for folderNr in range(2, len(sys.argv)):
             problemFiles.append([])
        
        #Iterate over all given files and handle them
        for folderNr in range(1, len(sys.argv)):

            viAveragePerFile = []
            qlAveragePerFile = []
            qlfAveragePerFile = []
            rAveragePerFile = []
            averagePerFile = []

            if folderNr == 1:
                it = tqdm.tqdm(os.listdir(sys.argv[folderNr]))
                for f in it:

                    platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[folderNr] + '/' + f, regret=False)
                    
                    averagePerFile.append(cumregret)

                    #Find the same problem in the other folder
                    for targetFolderNr in range(2, len(sys.argv)):
                        problemFile = findSameMDP(R, P, sys.argv[targetFolderNr])
                        problemFiles[folderNr - 2].append(problemFile)

                    #Learn the reference algorithms on this issue
                    viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates, averageRnd=True)

                    #Store the reference values per file
                    viAveragePerFile.append(cum_reward(viStates, viActions, viRewards, R, P))
                    qlAveragePerFile.append(cum_reward(qlStates, qlActions, qlRewards, R, P))
                    qlfAveragePerFile.append(cum_reward(qlfStates, qlfActions, qlfRewards, R, P))
                    rAveragePerFile.append(cum_reward(randStates, randActions, randRewards, R, P))

            else:

                for fileNr in range(len(problemFiles[0])):

                    f = problemFiles[folderNr - 2][fileNr]
                    platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[folderNr] + '/' + f, regret=False)

                    averagePerFile.append(cumregret)

            
            
            #Get the maximum and the minimum values of the different algorithms
            averagePerFile = np.array(averagePerFile)
            viAveragePerFile = np.array(viAveragePerFile)
            qlAveragePerFile = np.array(qlAveragePerFile)
            qlfAveragePerFile = np.array(qlfAveragePerFile)
            rAveragePerFile = np.array(rAveragePerFile)

            results[folderNr - 1, :, :] = np.copy(averagePerFile)
        
            if folderNr == 1:
                
                results[-4, :, :] = np.copy(viAveragePerFile)
                results[-3, :, :] = np.copy(qlAveragePerFile)
                results[-2, :, :] = np.copy(qlfAveragePerFile)
                results[-1, :, :] = np.copy(rAveragePerFile)

        colors = ['C0', 'C0', 'k', 'k', 'gray']
        linestyles = ['-', '-.', '-.', ':', '-']

        for fileNr in range(len(os.listdir(sys.argv[1]))):
            random = np.copy(results[-1, fileNr, :])
            optimal = np.copy(results[-4, fileNr, :] - random)

            #Iterate over all given folders
            for i in range(len(sys.argv) - 1):
                oldVal = np.copy(results[i, fileNr, :])
                results[i, fileNr, :] = np.copy((oldVal - random) / optimal)
                
                valuesToCorrect = np.copy(optimal[optimal==0])
                if valuesToCorrect != []:
                    valuesToCorrect = np.int(valuesToCorrect)
                    results[i, fileNr, valuesToCorrect] = 0.

            #Update the references
            for i in range(len(sys.argv) - 2, len(sys.argv) + 2):
                results[-i, fileNr, :] = np.copy((results[-i, fileNr, :] - random) / optimal)

        import dill
        dill.dump_session('Temp1.db')

        x = np.arange(1, iterations)
        fig, ax = plt.subplots(figsize=(16, 8))

        for i in range(len(labels)):
            c = colors[i]
            l = linestyles[i]
            min = np.min(results[i, :, :], axis=0)
            max = np.max(results[i, :, :], axis=0)
            mean = np.mean(results[i, :, :], axis=0)
            var = np.var(results[i, :, :], axis=0) / np.sqrt(results[i, :, :].shape[0])
            
            ax.plot(x, mean, color=c, linestyle=l)

            if i == 0:
                ax.fill_between(x, mean - var, mean, alpha=0.05, facecolor=c, label='_nolegend_', edgecolors=c, linewidth=2.)
                ax.fill_between(x, mean, mean + var, alpha=0.05, facecolor=c, label='_nolegend_', edgecolors=c, linewidth=2.)

        c = colors[len(labels)]
        l = linestyles[len(labels)]
        min = np.min(results[len(labels), :, :], axis=0)
        max = np.max(results[len(labels), :, :], axis=0)
        mean = np.mean(results[len(labels), :, :], axis=0)
        var = np.var(results[len(labels), :, :], axis=0) / np.sqrt(results[len(labels), :, :].shape[0])
        
        errVi = np.vstack((var, -var))
        ax.errorbar(x, mean, yerr=var, color=c, linestyle=l, label='Value Iteration', errorevery=400, elinewidth=None, capsize=4)

        c = colors[len(labels) + 1]
        l = linestyles[len(labels) + 1]
        min = np.min(results[len(labels) + 1, :, :], axis=0)
        max = np.max(results[len(labels) + 1, :, :], axis=0)
        mean = np.mean(results[len(labels) + 1, :, :], axis=0)
        var = np.var(results[len(labels) + 1, :, :], axis=0) / np.sqrt(results[len(labels) + 1, :, :].shape[0])

        errQl = np.vstack((mean + var, mean-var))
        ax.errorbar(x, mean, yerr=var, color=c, linestyle=l, label='Q-Learning', errorevery=400, elinewidth=None, capsize=4)

        c = colors[len(labels) + 2]
        l = linestyles[len(labels) + 2]
        min = np.min(results[len(labels) + 3, :, :], axis=0)
        max = np.max(results[len(labels) + 3, :, :], axis=0)
        mean = np.mean(results[len(labels) + 3, :, :], axis=0)
        var = np.var(results[len(labels) + 3, :, :], axis=0) / np.sqrt(results[len(labels) + 3, :, :].shape[0])

        ax.plot(x, mean, color=c, linestyle=l, label='Random-Policy')

        labels = labels + (['Random-Policy', 'Value Iteration', 'Q-Learning', ])
        ax.legend(labels)
        ax.set_xlabel('step')

        title = ''
        title += 'scaled '

        ax.set_ylabel(title + 'cumulative reward')

        plt.tick_params(top='off', right='off')
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.xaxis.set_ticks(np.arange(0, 2001, 500))
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.25))
        ax.xaxis.set_tick_params(labelsize=12.)
        ax.yaxis.set_tick_params(labelsize=12.)

        ax.set_ylim([-0.2, 1.2])

        plt.show()

        
    
