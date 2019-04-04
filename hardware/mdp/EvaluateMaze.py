
import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import tqdm
if (sys.version_info > (3, 0)):
    from pybrain3_local.rl.environments.mazes import Maze, MDPMazeTask
    from pybrain3_local.rl.environments.mazes.tasks import FourByThreeMaze
    from pybrain3_local.rl.learners.valuebased import ActionValueTable
    from pybrain3_local.rl.agents import LearningAgent
    from pybrain3_local.rl.learners import Q, QLambda, SARSA #@UnusedImport
    from pybrain3_local.rl.experiments import Experiment
else:
    from pybrain_local.rl.environments.mazes import Maze, MDPMazeTask
    from pybrain_local.rl.environments.mazes.tasks import FourByThreeMaze
    from pybrain_local.rl.learners.valuebased import ActionValueTable
    from pybrain_local.rl.agents import LearningAgent
    from pybrain_local.rl.learners import Q, QLambda, SARSA #@UnusedImport
    from pybrain_local.rl.experiments import Experiment
import time

def ComparePolicy(maze, goal, numStates, numActions, toComparePolicy, platform, plot=True, n_iter=100000, fixedPolicy=[], playSteps=None):

    def transformState(state):
        
        x = int(state % gridCols)
        y = int(state / gridCols)
        
        return x, y

    wins = 0

    gridRows, gridCols = maze.shape

    if platform == 0:
        goalState = goal[0] * gridCols + goal[1]
    else:
        goalState = goal[1] * gridCols + goal[0]

    if fixedPolicy == []:
        environment = Maze(maze, goal)
        controller = ActionValueTable(gridRows * gridCols, numActions)
        controller.initialize(1.)
        #learner = Q()
        learner = QLambda()
        #learner = SARSA()
        agent = LearningAgent(controller, learner)
        task = MDPMazeTask(environment)
        experiment = Experiment(task, agent)

        for i in range(int(n_iter / 100)):
            experiment.doInteractions(100)
            agent.learn()
            agent.reset()
            
            #if playSteps != None and ((i + 1) * 100) == playSteps:
            #    wins = task.env.wins

        task.env.wins = 0
        #Run the algorithm after training to get a baseline
        for i in range(int(playSteps / 100)):
            experiment.doInteractions(100)
            agent.learn()
            agent.reset()

        wins = task.env.wins

        policy = np.zeros(gridRows * gridCols)
        for state in range(gridRows * gridCols):
            x, y = transformState(state)
            if maze[y, x] == 1.:
                continue

            policy[state] = controller.getMaxAction(state)

        #Plot the learned policy
        if plot:
            #fig, ax = plt.subplots()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            PlotPolicy2(ax, policy, controller.params.reshape(gridRows * gridCols, numActions), maze, gridRows, gridCols, goalState)
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.show()

    else:
        policy = fixedPolicy         

    if platform == 0:
        #Compare the two policies SW

        s = 0
        for state in range(gridRows * gridCols):
        
            x, y = transformState(state)
            
            #Fill The wall elements
            if maze[y, x] == 1.:
                continue
            
            if state == goalState:
                continue

            if policy[state] != toComparePolicy[state]:
                s += 1

    else:
        #Compare the two policies HW

        tempPolicy = np.array(toComparePolicy, copy=True)
    
        #Modify policy to have the same notation as in the SW case
        tempPolicy[tempPolicy == 2] = 5
        tempPolicy[tempPolicy == 0] = 2
        tempPolicy[tempPolicy == 5] = 0
        goalState = goal[0] * gridCols + goal[1]

        #Compare the two policies
        s = 0
        for posX in range(1, gridCols - 1):
            for posY in range(1, gridRows - 1):

                x = posX - 1
                y = posY - 1
                state = x * (gridCols - 2) + y
                stateMap = posX * gridCols + posY

                #Fill The wall elements
                if maze[posX, posY] == 1.:
                    continue
                
                if stateMap == goalState:
                    continue

                if policy[stateMap] != tempPolicy[state]:
                    print (stateMap, state)
                    print (policy[stateMap], tempPolicy[state])
                    s += 1

    return s, wins

def PlotPolicy(ax, policy, Q_table, maze, gridRows, gridCols, goalState):

    #Maze is flipped due to environment N down S up
    def next_state(state, a):
        if a == 0:
            return state - gridCols
        elif a == 1:
            return state + 1
        elif a == 2:
            return state + gridCols
        elif a == 3:
            return state - 1
        else:
            raise ValueError('Unkown action {}'.format(a))
        
    def transformState(state):
        
        y = int(state / gridCols)
        x = int(state % gridCols)
        
        return x, y

    colorMap = ax.pcolor(maze, cmap='Reds')#plt.cm.jet)
    
    gridCols += 2
    x, y = transformState(goalState)

    colorAx = colorMap.get_axes()
    colorAx.text(x + .5, y + .5, 'G', ha="center", va="center", color='black', size=100, fontweight='bold')
    #ax.annotate('G',xy=(x + .2, y + .2), color='black',size=25)
    nStates, nActions = Q_table.shape
    g0, g1 = transformState(goalState)

    gridRows, gridCols = maze.shape
    termState = g0 * gridRows + g1

    for posX in range(1, gridCols - 1):
        for posY in range(1, gridRows - 1):

            x = posX - 1
            y = posY - 1
            state = y * (gridCols - 2) + x
            stateMap = posX * gridCols + posY

            #Fill The wall elements
            if maze[posY, posX] == 1.:
                continue
            
            if stateMap == termState:
                continue
    
            action = policy[state]
            nextState = next_state(stateMap, action)
            yn = nextState % gridCols
            xn = nextState / gridCols

            #ax.annotate(str(stateMap), xy=(posX + .2, posY + .2), color='black',size=25)
            ax.arrow(posX + .5, posY + .5, (yn - posY) * 0.0001, (xn - posX) * 0.0001, head_length=.3, head_width=.2, color='black')
    
    #ax.set_title('Policy')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.axis('off')

def PlotPolicy2(ax, policy, Q_table, maze, gridRows, gridCols, goalState):

    #Maze is flipped due to environment N down S up
    def next_state(state, a):
        if a == 0:
            return state + gridCols
        elif a == 1:
            return state + 1
        elif a == 2:
            return state - gridCols 
        elif a == 3:
            return state - 1
        else:
            raise ValueError('Unkown action {}'.format(a))
        
    def transformState(state):
        
        x = int(state % gridCols)
        y = int(state / gridCols)
        
        return x, y

    nStates, nActions = Q_table.shape
    colorMap = ax.pcolor(maze, cmap='Reds')#plt.cm.jet)

    g0, g1 = transformState(goalState)
    termState = g0 * gridRows + g1
    
    x, y = transformState(termState)

    colorAx = colorMap.get_axes()
    colorAx.text(y + .5, x + .5, 'G', ha="center", va="center", color='black', size=100, fontweight='bold')
    #ax.annotate('G',xy=(y + .2, x + 0.2),color='black',size=85, fontweight='bold')
    
    for state in range(nStates):
        
        x, y = transformState(state)
        
        #Fill The wall elements
        if maze[y, x] == 1.:
            #ax.fill(x, y, "w")
            continue
        
        if (x * gridRows + y) == termState:
            continue
        
        action = policy[state]
        #action = np.random.choice(np.where(np.max(Q_table[state,:]) == Q_table[state,:])[0])
        nextState = next_state(state, action)
        xn, yn = transformState(nextState)
        ax.arrow(x + .5, y + .5, (xn - x) * 0.0001, (yn - y) * 0.0001, head_length=.3, head_width=.1, color='black')
        #ax.annotate(str(state), xy=(x + .2, y + .2), color='black',size=25)
    
    #ax.set_title('Policy')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.axis('off')

def ExpandMaze(maze):
    #This function is used to draw a border around the maze

    
    rightLeft = np.ones([maze.shape[0], 1])
    maze = np.hstack((maze, rightLeft))
    maze = np.hstack((rightLeft, maze))
    
    updown = np.ones([maze.shape[1]])
    maze = np.vstack((updown, maze))
    maze = np.vstack((maze, updown))

    return maze

def RemoveWalls(maze):
    #Function used to remove the walls from the given maze

    sh = maze.shape
    maze = maze[1:sh[0]-1, 1:sh[1]-1]

    return maze
    
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
                    print(stA)
                    return finishLearning

        print(stA)
        return 2000
    

if __name__ == '__main__':

    import h5py
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 28}

    matplotlib.rc('font', **font)

    if len(sys.argv) != 2:
        raise Exception('No HDF file given!')

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]) == True:

        print ('Testing mode')

        '''Test the individual solutions and compute the average fitness'''
        referenceWinsPerFile = []
        winsPerFile = []
        
        #Iterate over all given files and handle them
        fileNr = 0
        it = tqdm.tqdm(os.listdir(sys.argv[1]))
        for f in it:
            #Read the hdf5 file and collect the results
            hdf5File = h5py.File(sys.argv[1] + '/' + f, 'r')
            iterationsOnChip = hdf5File['iterationsOnChip'][0]
            multipleRuns = hdf5File['multipleRuns'][0]
            trialsPerIteration = hdf5File['trialsPerIteration'][0]
            maze = np.array(hdf5File['maze'][:])
            goalStateX = hdf5File['goalX'][0]
            goalStateY = hdf5File['goalY'][0]
            gridRows, gridCols = maze.shape

            steps = iterationsOnChip * trialsPerIteration

            if hdf5File['platform'][0] == 'SW': 
                goalState = goalStateX * gridCols + goalStateY
                platform = 0
            else:
                maze = ExpandMaze(maze)
                goalStateX += 1
                goalStateY += 1
                goalState = goalStateX * gridCols + goalStateY
                platform = 1
            
            #Get the last QTable for each run
            Qtables = []
            policies = []
            wins = []
            for run in range(multipleRuns):
                Qtables.append(np.array(hdf5File['Run_' + str(run)]['Qtable' + str(iterationsOnChip - 1)][:]))
                policies.append(np.array(hdf5File['Run_' + str(run)]['policy' + str(iterationsOnChip - 1)][:]))
                
                #Sum over wins
                tempWins = 0
                for i in range(iterationsOnChip):
                    tempWins += np.array(hdf5File['Run_' + str(run)]['Wins' + str(i)][:])

                wins.append(tempWins)
            
            hdf5File.close()

            wins = np.array(wins)

            #Average over the wins per run to get the wins per file
            winsPerFile.append(wins.mean())

            #fixedPolicy=np.array([0., 0., 0., 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.])
            fixedPolicy=[]
            nStates, nActions = Qtables[0].shape
            differences, winsReference = ComparePolicy(maze, (goalStateX, goalStateY), nStates, nActions, toComparePolicy = policies[-1], platform=platform, plot=False, playSteps=steps)#, fixedPolicy=fixedPolicy))
            referenceWinsPerFile.append( winsReference)

        winsPerFile = np.array(winsPerFile)
        referenceWinsPerFile = np.array(referenceWinsPerFile)
        #print ('Differences per Files: ' + str(differencePerFile))
        #print ('Successful trials: ' + str(len(differencePerFile) - np.count_nonzero(differencePerFile)) + ' of ' + str(len(differencePerFile)))
        #print ('Average difference: ' + str(differencePerFile.mean()))
        print ('Simulation wins: ' + str(winsPerFile.mean()) + '+-' + str(np.std(winsPerFile)))
        print ('Reference wins: ' + str(referenceWinsPerFile.mean()) + '+-' + str(np.std(referenceWinsPerFile)))
        
        exit()

    #Read the hdf5 file and collect the results
    hdf5File = h5py.File(sys.argv[1], 'r')
    iterationsOnChip = hdf5File['iterationsOnChip'][0]
    multipleRuns = hdf5File['multipleRuns'][0]
    maze = np.array(hdf5File['maze'][:])
    goalStateX = hdf5File['goalX'][0]
    goalStateY = hdf5File['goalY'][0]
    gridRows, gridCols = maze.shape

    if hdf5File['platform'][0] == 'SW': 
        goalState = goalStateX * gridCols + goalStateY
        platform = 0
    else:
        maze = ExpandMaze(maze)
        goalStateX += 1
        goalStateY += 1
        goalState = goalStateX * (gridCols + 2) + goalStateY
        platform = 1
    
    #Get the last QTable for each run
    Qtables = []
    policies = []
    wins = []
    for run in range(multipleRuns):
        Qtables.append(np.array(hdf5File['Run_' + str(run)]['Qtable' + str(iterationsOnChip - 1)][:]))
        policies.append(np.array(hdf5File['Run_' + str(run)]['policy' + str(iterationsOnChip - 1)][:]))
        #Sum over wins
        tempWins = 0
        #for i in range(iterationsOnChip):
        #    tempWins += np.array(hdf5File['Run_' + str(run)]['Wins' + str(i)][:])
        #wins.append(tempWins)

    hdf5File.close()

    #wins = np.array(wins)
    #print ('Average wins: ' + str(wins.mean()))

    #policies[-1] = np.array([2, 3, 3, 2, 0, 0, 0, 0, 0])
    #print (policies[-1])
    #fixedPolicy=np.array([0., 0., 0., 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.])
    #nStates, nActions = Qtables[0].shape
    #print ('Difference: ' + str(ComparePolicy(maze, (goalStateX, goalStateY), nStates, nActions, toComparePolicy = policies[-1], platform=platform, fixedPolicy=fixedPolicy)))
    #exit()

    #The difference for the maze is that the SW uses the walls as state while the HW implementation doesn't
    #SW uses library implementation of the environment and the HW uses an own implementation for memory reasons
    if platform == 0:
        fig, ax = plt.subplots()
        PlotPolicy2(ax, policies[-1], Qtables[0], maze, gridRows, gridCols, goalState)
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.show()
    else:
        fig, ax = plt.subplots()
        PlotPolicy(ax, policies[-1], Qtables[0], maze, gridRows, gridCols, goalState)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
