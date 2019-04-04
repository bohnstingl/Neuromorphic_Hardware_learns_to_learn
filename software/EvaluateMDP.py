import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt
import mdptoolbox_local as mdptoolbox
import mdptoolbox_local.example
import sys
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
    

def trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates):
        
    #The MDP reference algorithm cannot deal with gamma = 0.
    if gamma == 0.:
        gamma = 0.0001

    #The learners have a hard time to reach the optimum used as a reference here. The reason for this is that the
    #highest reward value per state used as a reference is only given with a very low probability, so that the learner
    #does not see this. The 'handcrafted' policy is optimal for a particular problem, but still did produce a much higher 
    #cumulative regret due to the fact that a much lower reward was given in that state with a much higher probability
    #Compute other strategies to solve the problem
    #Train value iteration and q learning
    #vi = mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=10**5)
    #vi.run()

    #print vi.policy
    #viStates, viActions, viRewards = playPolicy(initialState, vi.policy, P, R, iterations)
    
    #Solve the MDP with normal Q-Learning
    qlf = mdptoolbox.mdp.QLearning(P, R, gamma, n_iter=10**5)#10**8)
    qlf.run()
    #print qlf.policy
    
    qlfStates = np.array(qlf.states)[:iterations]
    qlfActions = np.array(qlf.actions)[:iterations]
    qlfRewards = np.array(qlf.rewards)[:iterations]

    viStates = qlfStates
    viActions = qlfActions
    viRewards = qlfRewards

    #Solve the MDP with Q-Learning and epsilon decay
    ql = mdptoolbox.mdp.QLearningEps(P, R, gamma, 0.30, 0.99, n_iter=10**5)#10**8)
    ql.run()
    #print ql.policy
    
    #qlStates, qlActions, qlRewards = playPolicy(states[0], ql.policy, P, R, iterations)
    qlStates = np.array(ql.states)[:iterations]
    qlActions = np.array(ql.actions)[:iterations]
    qlRewards = np.array(ql.rewards)[:iterations]

    #Solve the MDP with a random Policy
    #handcrafted = [4, 1, 9, 9, 7]
    randomPolicy = np.random.randint(0, nActions - 1, nStates)
    #print randomPolicy
    randStates, randActions, randRewards = playPolicy(initialState, randomPolicy, P, R, iterations)

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
	
		#This function is used to deduce a policy from the current state, actions and rewards.
		#After that the learning speed of the final policy can be investigated
		
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

    '''
    d = .0015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)
    '''

    return ax1, ax2

if __name__ == '__main__':

    import h5py
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 28}

    matplotlib.rc('font', **font)

    plotCumulativeReward = True
    splitPlot = False

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
        
        #Iterate over all given files and handle them
        fileNr = 0
        it = tqdm.tqdm(os.listdir(sys.argv[1]))
        for f in it:
            platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[1] + '/' + f, regret=not plotCumulativeReward)

            labels.append(network + ' ' + str(fileNr))
            
            averagePerFile.append(cumregret)
    
            #Learn the reference algorithms on this issue
            viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates)

            if plotCumulativeReward:
                #Store the reference values per file
                viAveragePerFile.append(cum_reward(viStates, viActions, viRewards, R, P))
                qlAveragePerFile.append(cum_reward(qlStates, qlActions, qlRewards, R, P))
                qlfAveragePerFile.append(cum_reward(qlfStates, qlfActions, qlfRewards, R, P))
                rAveragePerFile.append(cum_reward(randStates, randActions, randRewards, R, P))
            else:
                #Store the reference values per file
                viAveragePerFile.append(cum_regret(viStates, viActions, viRewards, R, P))
                qlAveragePerFile.append(cum_regret(qlStates, qlActions, qlRewards, R, P))
                qlfAveragePerFile.append(cum_regret(qlfStates, qlfActions, qlfRewards, R, P))
                rAveragePerFile.append(cum_regret(randStates, randActions, randRewards, R, P))
                
            speedPerFile.append(averageSpeed)

            fileNr += 1

        #Get the maximum and the minimum values of the different algorithms
        averagePerFile = np.array(averagePerFile)
        viAveragePerFile = np.array(viAveragePerFile)
        qlAveragePerFile = np.array(qlAveragePerFile)
        qlfAveragePerFile = np.array(qlfAveragePerFile)
        rAveragePerFile = np.array(rAveragePerFile)
        speedPerFile = np.array(speedPerFile)

        nwMax = averagePerFile[np.argmax(averagePerFile.sum(1))]
        nwMin = averagePerFile[np.argmin(averagePerFile.sum(1))]
        nwMean = averagePerFile.mean(0)
    
        viMax = viAveragePerFile[np.argmax(viAveragePerFile.sum(1))]
        viMin = viAveragePerFile[np.argmin(viAveragePerFile.sum(1))]
        viMean = viAveragePerFile.mean(0)

        qlMax = qlAveragePerFile[np.argmax(qlAveragePerFile.sum(1))]
        qlMin = qlAveragePerFile[np.argmin(qlAveragePerFile.sum(1))]
        qlMean = qlAveragePerFile.mean(0)

        qlfMax = qlfAveragePerFile[np.argmax(qlfAveragePerFile.sum(1))]
        qlfMin = qlfAveragePerFile[np.argmin(qlfAveragePerFile.sum(1))]
        qlfMean = qlfAveragePerFile.mean(0)

        rMax = rAveragePerFile[np.argmax(rAveragePerFile.sum(1))]
        rMin = rAveragePerFile[np.argmin(rAveragePerFile.sum(1))]
        rMean = rAveragePerFile.mean(0)
    
        x = np.arange(1, iterations)
        
        '''
        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        c = colors[0]
        ax.plot(x, nwMean, color=c)
        #ax.fill_between(x, nwMin, nwMean, alpha=0.05, facecolor=c, label='_nolegend_')
        #ax.fill_between(x, nwMean, nwMax, alpha=0.05, facecolor=c, label='_nolegend_')
        ax.plot(x, nwMin, linestyle='-.', alpha=0.4, color=c, label='_nolegend_')
        ax.plot(x, nwMax, linestyle=':', alpha=0.4, color=c, label='_nolegend_')

        c = colors[1]
        ax.plot(x, viMean, color=c)
        #ax.fill_between(x, viMin, viMean, alpha=0.05, facecolor=c, label='_nolegend_')
        #ax.fill_between(x, viMean, viMax, alpha=0.05, facecolor=c, label='_nolegend_')
        ax.plot(x, viMin, linestyle='-.', alpha=0.4, color=c, label='_nolegend_')
        ax.plot(x, viMax, linestyle=':', alpha=0.4, color=c, label='_nolegend_')

        c = colors[2]
        ax.plot(x, qlMean, color=c)
        #ax.fill_between(x, qlMin, qlMean, alpha=0.05, facecolor=c, label='_nolegend_')
        #ax.fill_between(x, qlMean, qlMax, alpha=0.05, facecolor=c, label='_nolegend_')
        ax.plot(x, qlMin, linestyle='-.', alpha=0.4, color=c, label='_nolegend_')
        ax.plot(x, qlMax, linestyle=':', alpha=0.4, color=c, label='_nolegend_')

        c = colors[3]
        ax.plot(x, qlfMean, color=c)
        #ax.fill_between(x, qlfMin, qlfMean, alpha=0.05, facecolor=c, label='_nolegend_')
        #ax.fill_between(x, qlfMean, qlfMax, alpha=0.05, facecolor=c, label='_nolegend_')
        ax.plot(x, qlfMin, linestyle='-.', alpha=0.4, color=c, label='_nolegend_')
        ax.plot(x, qlfMax, linestyle=':', alpha=0.4, color=c, label='_nolegend_')

        c = colors[4]
        ax.plot(x, rMean, color=c)
        #ax.fill_between(x, rMin, rMean, alpha=0.05, facecolor=c, label='_nolegend_')
        #ax.fill_between(x, rMean, rMax, alpha=0.05, facecolor=c, label='_nolegend_')
        ax.plot(x, rMin, linestyle='-.', alpha=0.4, color=c, label='_nolegend_')
        ax.plot(x, rMax, linestyle=':', alpha=0.4, color=c, label='_nolegend_')
        
        #ax.legend([network, network + ' Min', network + ' Max', 'Value Iteration', 'Value Iteration Min', 'Value Iteration Max','Q-Learning (eps-decay)', 'Q-Learning (eps-decay) Min', 'Q-Learning (eps-decay) Max', 'Q-Learning', 'Q-Learning Min', 'Q-Learning Max', 'Random', 'Random Min', 'Random Max'])
        ax.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
        #ax.set_title('Comparison for MDP (' + str(nStates) + ' x ' + str(nActions) + ')')
        ax.set_xlabel('Step', fontsize=34, fontweight='bold')

        if plotCumulativeReward:
            ax.set_ylabel('Cumulative reward', fontsize=34, fontweight='bold')
        else:
            ax.set_ylabel('Cumulative regret', fontsize=34, fontweight='bold')

        #plt.savefig('MDPEvaluation.png')
        #plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        plt.tick_params(top='off', right='off')
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.show()
        '''

        #Print the table for comparison
        print ('Reward Simulation: ' + str(averagePerFile.mean(0)[-1]) + ' +- ' + str(np.std(averagePerFile, axis=0)[-1]))
        print ('Reward ValueIteration: ' + str(viAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(viAveragePerFile, axis=0)[-1]))
        print ('Reward Q-Learning (eps-decay): ' + str(qlAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(qlAveragePerFile, axis=0)[-1]))
        print ('Reward Q-Learning: ' + str(qlfAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(qlfAveragePerFile, axis=0)[-1]))
        print ('Reward Random-Policy: ' + str(rAveragePerFile.mean(0)[-1]) + ' +- ' + str(np.std(rAveragePerFile, axis=0)[-1]))
        print ('Average learning speed: ' + str(speedPerFile.mean(0)) + ' +- ' + str(np.std(speedPerFile)))

        printLatexTableEntry(averagePerFile, viAveragePerFile, qlAveragePerFile, qlfAveragePerFile, rAveragePerFile)

    elif len(sys.argv) == 2:

        print('Normal mode')

        '''Handle the file'''
        platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[1], regret=not plotCumulativeReward)


        if plotCumulativeReward:
            print('Variance of cum reward: ' + str(cumregretHigh[-1] - cumregretLow[-1]))
        else:
            print('Variance of cum reward: ' + str(cumregretHigh[-1] - cumregretLow[-1]))

        print('Total ' + str(cumregret[-1]))

        neuralData = []

        #Sort the spike trains according to the addresses
        
        li = list(np.arange(1., 5., 0.5))
        li.extend(np.arange(16., 20., 0.5))
        neuralData.append(li)
        neuralData.append(np.arange(8., 12., 0.5))
        neuralData.append([4.1])
        neuralData.append([4.3, 11.3])
        neuralData.append([4.5])
        nStates = 2
        nActions = 3
        
        
        yticks = []
        for i in range(nStates):
            yticks.append('State Neuron ' + str(i))

        for i in range(nActions):
            yticks.append('Action Neuron ' + str(i))

        spAddressNp = np.array(spikeAddresses[0])
        spTimesNp = np.array(spikeTimes[0])
        #for address in set(spikeAddresses[0]):
        #    neuralData.append(spTimesNp[spAddressNp == address])

        neuralData = np.array(neuralData)

        
        #Evaluate spike trains
        plt.eventplot(neuralData, colors=[[0, 0, 0]], linelengths=0.4, linewidth=5.0)
        #plt.title('Rasterplot of first run')
        plt.xlabel('Time / a.u.', fontsize=34, fontweight='bold')
        plt.ylim([-0.95, nStates + nActions - 0.25])
        plt.yticks(np.arange(len(yticks)), yticks, fontsize=34, fontweight='bold')
        plt.tick_params(top='off', right='off')
 
        # Hide the right and top spines
        #ax.spines['right'].set_visible(False)
        #ax.spines['top'].set_visible(False) 
        plt.show()
        
        
        exit()
        viStates, viActions, viRewards, qlfStates, qlfActions, qlfRewards, qlStates, qlActions, qlRewards, randStates, randActions, randRewards = trainReferenceAlgorithms(P, R, initialState, iterations, gamma, nActions, nStates)

        x = np.arange(1, iterations)

        if splitPlot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
                
            data = []
            data.append(cumregret)
            
            if plotCumulativeReward:
                #data.append(cum_reward(viStates, viActions, viRewards, R, P))
                data.append(cum_reward(qlStates, qlActions, qlRewards, R, P))
                data.append(cum_reward(qlfStates, qlfActions, qlfRewards, R, P))
                data.append(cum_reward(randStates, randActions, randRewards, R, P))
            else:
                #data.append(cum_regret(viStates, viActions, viRewards, R, P))
                data.append(cum_regret(qlStates, qlActions, qlRewards, R, P))
                data.append(cum_regret(qlfStates, qlfActions, qlfRewards, R, P))
                data.append(cum_regret(randStates, randActions, randRewards, R, P))

            ax1, ax2 = PlotWithSplitAxis(ax1, ax2, x, data)

            #ax1.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
            ax1.legend([network, 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
            #ax.set_title('Comparison for MDP (' + str(nStates) + ' x ' + str(nActions) + ')')
            ax1.set_xlabel('Step', fontsize=34, fontweight='bold')
            ax2.set_xlabel('Step', fontsize=34, fontweight='bold')
            
            if plotCumulativeReward:
                ax1.set_ylabel('Cumulative reward', fontsize=34, fontweight='bold')
                ax2.set_ylabel('Cumulative reward', fontsize=34, fontweight='bold')
            else:
                ax1.set_ylabel('Cumulative regret', fontsize=34, fontweight='bold')
                ax2.set_ylabel('Cumulative regret', fontsize=34, fontweight='bold')

            #plt.savefig('MDPEvaluation.png')
            plt.tick_params(top='off', right='off')
            
            plt.show()
            
            print (analyzePerformance(R.shape[1], R.shape[0], qlStates, qlActions))

        else:
            fig, ax = plt.subplots()
                
            ax.plot(x, cumregret)
            ax.fill_between(x, cumregretLow, cumregret, alpha=0.5, facecolor='blue', label='_nolegend_')
            ax.fill_between(x, cumregret, cumregretHigh, alpha=0.5, facecolor='blue', label='_nolegend_')
            
            if plotCumulativeReward:
                ax.plot(x, cum_reward(viStates, viActions, viRewards, R, P))
                ax.plot(x, cum_reward(qlStates, qlActions, qlRewards, R, P))
                ax.plot(x, cum_reward(qlfStates, qlfActions, qlfRewards, R, P))
                ax.plot(x, cum_reward(randStates, randActions, randRewards, R, P))
            else:
                ax.plot(x, cum_regret(viStates, viActions, viRewards, R, P))
                ax.plot(x, cum_regret(qlStates, qlActions, qlRewards, R, P))
                ax.plot(x, cum_regret(qlfStates, qlfActions, qlfRewards, R, P))
                ax.plot(x, cum_regret(randStates, randActions, randRewards, R, P))

            ax.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random-Policy'])
            #ax.set_title('Comparison for MDP (' + str(nStates) + ' x ' + str(nActions) + ')')
            ax.set_xlabel('Step', fontsize=34, fontweight='bold')
            
            if plotCumulativeReward:
                ax.set_ylabel('Cumulative reward', fontsize=34, fontweight='bold')
            else:
                ax.set_ylabel('Cumulative regret', fontsize=34, fontweight='bold')

            #plt.savefig('MDPEvaluation.png')
            plt.tick_params(top='off', right='off')
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.show()
    
    elif len(sys.argv) > 2:

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

            platform, iterations, multipleRuns, spikeAddresses, spikeTimes, P, R, gamma, nStates, nActions, network, t, initialState, cumregret, cumregretLow, cumregretHigh, averageSpeed = HandleHDF5File(sys.argv[fileNr], regret=not plotCumulativeReward)

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

        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        ax.plot(x, averagePerNetworkDLS.mean(axis=0))
        ax.plot(x, averagePerNetworkSW.mean(axis=0))

        print(averagePerNetworkDLS.shape)

        ax.legend(['DLS', 'SW'])

        '''
        for fileNr in range(len(averagePerFile)):
            c = colors[fileNr % len(colors)]
            ax.plot(x, averagePerFile[fileNr], color=c)
            #ax.fill_between(x, minPerFile[fileNr], averagePerFile[fileNr], alpha=0.1, facecolor=c, label='_nolegend_')
            #ax.fill_between(x, averagePerFile[fileNr], maxPerFile[fileNr], alpha=0.1, facecolor=c, label='_nolegend_')
        '''
        
        '''
        labels.append('Value Iteration')
        labels.append('Q-Learning (eps-decay)')
        labels.append('Q-Learning')
        labels.append('Random-Policy')
        '''
        
        #ax.legend([network, 'Value Iteration', 'Q-Learning (eps-decay)', 'Q-Learning', 'Random'])
        #ax.legend(labels)
        #ax.set_title('Comparison for MDP (' + str(nStates) + ' x ' + str(nActions) + ')')
        ax.set_xlabel('Step', fontsize=34, fontweight='bold')

        if plotCumulativeReward:
            ax.set_ylabel('Cumulative reward', fontsize=34, fontweight='bold')
        else:
            ax.set_ylabel('Cumulative regret', fontsize=34, fontweight='bold')

        #plt.savefig('MDPEvaluation.png')
        plt.tick_params(top='off', right='off')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
    
    
