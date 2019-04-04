#define PLAST 0

#include <s2pp.h>
#include <stdint.h>
#include "libnux/mailbox.h"
#include "libnux/dls_v2.h"
#include "libnux/random.h"
#include "spikes.h"
#include "MDP_Maze.h"
#include "Utils.h"
#include "libnux/fxdpt.h"
#include "libnux/exp.h"

//Temporary definition for inhibition wait cycles
#define N_CYCLES_FOR_INHIBITION 400
#define N_CYCLES_FOR_SPIKE 1000

//Address from which the test read command is performed
static uint32_t const signal_addr_offset = 0xff0;
static uint32_t const qOffset = 0xbf0;

//This is the number of already read bytes from the mailbox
extern uint32_t readBytes;
extern uint32_t iterationCounter;
extern uint8_t states[2];
extern int8_t rewards[2];
extern uint8_t actions[2];
extern uint8_t state;
extern uint32_t maxPValue;
extern uint32_t maxIteration;
extern uint32_t randomSeed;

int8_t gamma, lambda, eta;
//int8_t Q[32*32] = {1};
int8_t eligibilityTrace[32*32] = {1};

uint8_t weights[32 * 32] = {1};

uint32_t spikeCounts[MDP_ACTIONS] = {1};

enum {
	signal_wait = 0,
	signal_run = 1,
	signal_stop = 2,
};

//This function is used to read the action performed in the last state
uint8_t readOutAction()
{
	//If no spike occured, a random action will be taken, similarly if all neuron spiked
    uint32_t nonzero = 0;
    uint32_t randint;
    uint32_t num_nonzero = 0;
    uint32_t active_counter = 0;
    int i;

    reset_spike_counters(spikeCounts, MDP_ACTIONS);
    while (! nonzero)
    {
        nonzero = update_spike_counters(spikeCounts, MDP_ACTIONS, state + 1, 1 + MDP_STATES);
    }
    wait_cycles(N_CYCLES_FOR_INHIBITION);
    update_spike_counters(spikeCounts, MDP_ACTIONS, state + 1, 1 + MDP_STATES);
    
    for (i = 0; i < MDP_ACTIONS; i ++)
        if (spikeCounts[i])
            num_nonzero ++;
    
    // choose randomly among actions
    randint = (uint32_t) random_lcg(&randomSeed) >> 24;
    randint = randint % num_nonzero;
    for (i = 0; i < MDP_ACTIONS; i ++)
        if (spikeCounts[i] && active_counter++ == randint)
            return i;
    return 0;
}

uint8_t presentStateAndReadoutAction()
{
    uint32_t nonzero = 0;
    uint32_t randint;
    uint32_t num_nonzero = 0;
    uint32_t active_counter = 0;
    int i;

    //This function sends spikes to the action neuron and activates the correct one
    spike_t spike;

    //Send the spikes only to the first synapse driver
    spike.row_mask = 1;

    //Determine the address of the state neuron to activated
    //State neuron 0 has address 1
    spike.addr = state + 1;

    //Peform a loop and wait for the spikes of the action neuron
    while(!nonzero)
    {
        //Emits a single spike into the given neurons
        spikes_send(&spike);
        
        //Update the rate counters. The spikeCounts contains the counters only of the action neurons
        //However here only the action neurons are important and will be updated
        nonzero = update_spike_counters(spikeCounts, MDP_ACTIONS, state + 1, 1 + MDP_STATES);

        wait_cycles(N_CYCLES_FOR_SPIKE);
    }

    //If this point is reached, one neuron spiked
    //Wait for the inhibition to be active
    wait_cycles(N_CYCLES_FOR_INHIBITION);

    //Update the spike counter again and select and action from the neurons which spiked very close after each other
    /*update_spike_counters(spikeCounts, MDP_ACTIONS, 1 + MDP_STATES);
    
    for (i = 0; i < MDP_ACTIONS; i ++)
        if (spikeCounts[i])
            num_nonzero++;
    
    // choose randomly among actions
    randint = (uint32_t) random_lcg(&randomSeed) >> 24;
    randint = randint % num_nonzero;
    for (i = 0; i < MDP_ACTIONS; i ++)
        if (spikeCounts[i] && active_counter++ == randint)
            return i;

    //If this point is reached, pick a random action
    return ((uint32_t) random_lcg(&randomSeed) >> 24) % MDP_ACTIONS;*/
}

void loadPlastParameter()
{
    gamma = *((int8_t*) (&mailbox_base + readBytes));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    readBytes += sizeof(int8_t);
    lambda = *((int8_t*) (&mailbox_base + readBytes));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    readBytes += sizeof(int8_t);
    eta = *((int8_t*) (&mailbox_base + readBytes));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    readBytes += sizeof(int8_t);
    readBytes = 0;
}

void init()
{
    //Initialise the qtable and the eligibility traces
    for(uint32_t i = 0; i < (32 * 32); i++)
    {
        eligibilityTrace[i] = 0;
        //Q[i] = 0;
    }

    //Clear the spike counteres
    reset_spike_counters(spikeCounts, MDP_ACTIONS);
}

void saveQTable()
{
    //memcpy((int8_t*) (&mailbox_base + qOffset), (int8_t*)Q, sizeof(int8_t) * 32 * 32);
}

#if (PLAST == 0)
void weightUpdate()
{
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables
    uint32_t oldSynapseIndex = (1 + states[0]) * 32 + (actions[0] + 1 + MDP_STATES);
    uint32_t newSynapseIndex = (1 + states[1]) * 32 + (actions[1] + 1 + MDP_STATES);
    
    //Vector index for accessing the old weights    
    uint32_t oldWeightIndex = (1 + states[0]) * 2;
    if(actions[0] >= 16)
        oldWeightIndex++;

    uint32_t newWeightIndex = (1 + states[1]) * 2;
    if(actions[1] >= 16)
        newWeightIndex++;

    uint8_t oldActionAddress = (actions[0] + 1 + MDP_STATES) % 16;
    uint8_t newActionAddress = (actions[1] + 1 + MDP_STATES) % 16;

	// Declare temporary variables
	register vector int8_t oldWeightVect;
    register vector int8_t tempOldWeightVect;
	register vector int8_t newWeightVect;
    register vector int8_t newLearningRate;
    register vector int8_t dQ;

    //Create the vector for gamma and reward.
    vector int8_t gammaVect = vec_splat_s8(0);
	vector int8_t oldRewardVect = vec_splat_s8(0);
	vector int8_t learningVect = vec_splat_s8(0);
    vector int8_t learningVectDecay = vec_splat_s8(0);
    vector int8_t zeros = vec_splat_s8(0);

    gammaVect[oldActionAddress] = 0b01110000;//gamma;
    oldRewardVect[oldActionAddress] = rewards[0];//0b01000000;//oldReward;
    learningVect[oldActionAddress] = 0b00111000;//eta;

    //The decay is chosen such that the square root behavior is approximated. The decay is represented as a fractional
    learningVectDecay[oldActionAddress] = 0b01111111;
    
    //Caution, the newWeight and the old weight are not on the same position!!
    //Prepare the weight vector
    asm volatile (
            //Fetch the weight of the old and the new synapse
            "fxvinx %[oldWeightVect], %[dls_weight_base], %[oldWeightIndex]\n"
            "fxvshb %[oldWeightVect], %[oldWeightVect], 1\n"
            "fxvinx %[newWeightVect], %[dls_weight_base], %[newWeightIndex]\n"
            "fxvshb %[newWeightVect], %[newWeightVect], 1\n"
            : [oldWeightVect] "=&kv" (oldWeightVect),
              [newWeightVect] "=&kv" (newWeightVect)
            : [oldWeightIndex] "r" (oldWeightIndex),
              [newWeightIndex] "r" (newWeightIndex),
              [dls_weight_base] "b" (dls_weight_base)
            : );
    
    newWeightVect[oldActionAddress] = newWeightVect[newActionAddress];
    for(uint8_t i = 0; i < 16; i++)
        tempOldWeightVect[i] = oldWeightVect[i];
    
    //Write the generated vectors directly to the mailbox
    asm volatile (
            //Calculate delta = oldReward + gamma * newWeight - weight
            "fxvmulbfs %[dQ], %[gammaVect], %[newWeightVect]\n"
            "fxvaddbfs %[dQ], %[oldRewardVect], %[dQ]\n"
            "fxvsubbfs %[dQ], %[dQ], %[oldWeightVect]\n"
            //Calculate dQ = learningrate decayed * delta
            "fxvmulbfs %[dQ], %[dQ], %[learningVect]\n"
            "fxvmulbfs %[newLearningRate], %[learningVect], %[learningVectDecay]\n"
            //Update the weight w = w + dQ
            "fxvaddbfs %[oldWeightVect], %[oldWeightVect], %[dQ]\n"
            //Set negative weights to 0            
            "fxvcmpb %[oldWeightVect]\n"
		    "fxvsel %[oldWeightVect], %[oldWeightVect], %[zeros], 2\n"
            : [oldWeightVect] "=&kv" (oldWeightVect),
              [newWeightVect] "=&kv" (newWeightVect),
              [dQ] "=&kv" (dQ),
              [newLearningRate] "=&kv" (newLearningRate)
            : [oldWeightIndex] "r" (oldWeightIndex),
              [newWeightIndex] "r" (newWeightIndex),
              [gammaVect] "kv" (gammaVect),
              [oldRewardVect] "kv" (oldRewardVect),
              [learningVect] "kv" (learningVect),
              [learningVectDecay] "kv" (learningVectDecay),
              [dls_weight_base] "b" (dls_weight_base),
              [zeros] "kv" (zeros)
            : );
            
    //Restore the old weights and only modify the single one at oldActionAddress
    for(uint8_t i = 0; i < 16; i++)
    {
        if(i != oldActionAddress)
            oldWeightVect[i] = tempOldWeightVect[i];
        
        //Perform the shift right, since the weights are shifted to the left at the beginning
        oldWeightVect[i] = oldWeightVect[i] >> 1;
    }
    
    asm volatile (
            //Write the weights back to the chip
            "fxvoutx %[oldWeightVect], %[dls_weight_base], %[oldWeightIndex]\n"
            : 
            : [oldWeightVect] "kv" (oldWeightVect),
              [oldWeightIndex] "r" (oldWeightIndex),
              [dls_weight_base] "b" (dls_weight_base)
            : );
    
    //Recompute the new learning rate
    eta = newLearningRate[oldActionAddress];

    //Use directly the weights as the QValue, since the dQ vector is weired
    /*if((255 - (Q[oldSynapseIndex])) > (oldWeightVect[oldActionAddress] - weights[oldSynapseIndex]))
        Q[oldSynapseIndex] += oldWeightVect[oldActionAddress] - weights[oldSynapseIndex];//(int8_t)dQ[oldActionAddress];
    else
        Q[oldSynapseIndex] = 255;*/
    
    //Update the internal weight set
    weights[oldSynapseIndex] = oldWeightVect[oldActionAddress];
}

#else
void weightUpdate()
{
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables

    uint32_t oldSynapseIndex = (1 + states[0]) * 32 + (actions[0] + 1 + MDP_STATES);
    uint32_t newSynapseIndex = (1 + states[1]) * 32 + (actions[1] + 1 + MDP_STATES);
    
    //Vector index for accessing the old weights    
    uint32_t oldWeightIndex = (1 + states[0]) * 2;
    if(actions[0] >= 16)
        oldWeightIndex++;

    uint32_t newWeightIndex = (1 + states[1]) * 2;
    if(actions[1] >= 16)
        newWeightIndex++;

    uint8_t oldActionAddress = (actions[0] + 1 + MDP_STATES) % 16;
    uint8_t newActionAddress = (actions[1] + 1 + MDP_STATES) % 16;

    //Increase the eligibility trace of the old synapse. The eligibility traces are stored in RAM
    if(63 - eligibilityTrace[oldSynapseIndex] > 5)
        eligibilityTrace[oldSynapseIndex] += 5;//++;
    else
        eligibilityTrace[oldSynapseIndex] = 63;

	// Declare temporary variables
	register vector int8_t oldWeightVect;
	register vector int8_t newWeightVect;
    register vector int8_t d = vec_splat_s8(0);
    register vector int8_t dQ;
    register vector int8_t tempEligibility;
    register vector int8_t temp;
    register vector int8_t newLearningRate;

    //Create the vector for gamma and reward.
    vector int8_t gammaVect = vec_splat_s8(0);//gamma);
    vector int8_t lambdaVect = vec_splat_s8(0b01111100);//lambda);
	vector int8_t oldRewardVect = vec_splat_s8(0);//oldReward);
    vector int8_t learningVect = vec_splat_s8(0b00111000);//eta);
    vector int8_t learningVectDecay = vec_splat_s8(0);
    vector int8_t zeros = vec_splat_s8(0);

    gammaVect[oldActionAddress] = 0b01110000;//gamma;
    oldRewardVect[oldActionAddress] = rewards[0];//0b01000000;//oldReward;
    //learningVect[oldActionAddress] = 0b00111000;//eta;

    //The decay is chosen such that the square root behavior is approximated. The decay is represented as a fractional
    learningVectDecay[oldActionAddress] = 0b01111111;

    //Caution, the newWeight and the old weight are not on the same position!!
    //Prepare the weight vector
    asm volatile (
            //Fetch the weight of the old and the new synapse
            "fxvinx %[oldWeightVect], %[dls_weight_base], %[oldWeightIndex]\n"
            "fxvshb %[oldWeightVect], %[oldWeightVect], 1\n"
            "fxvinx %[newWeightVect], %[dls_weight_base], %[newWeightIndex]\n"
            "fxvshb %[newWeightVect], %[newWeightVect], 1\n"
            : [oldWeightVect] "=&kv" (oldWeightVect),
              [newWeightVect] "=&kv" (newWeightVect)
            : [oldWeightIndex] "r" (oldWeightIndex),
              [newWeightIndex] "r" (newWeightIndex),
              [dls_weight_base] "b" (dls_weight_base)
            : );
    
    //Prepare the new weight vector such that the old and new weight are aligned
    newWeightVect[oldActionAddress] = newWeightVect[newActionAddress];

    //Write the generated vectors directly to the mailbox
    asm volatile (
            //Fetch the weight of the old and the new synapse
            //Calculate delta = oldReward + gamma * newWeight - weight
            "fxvmulbfs %[d], %[gammaVect], %[newWeightVect]\n"
            "fxvaddbfs %[d], %[oldRewardVect], %[d]\n"
            "fxvsubbfs %[d], %[d], %[oldWeightVect]\n"
            : [d] "=&kv" (d)
            : [gammaVect] "kv" (gammaVect),
              [oldRewardVect] "kv" (oldRewardVect),
              [oldWeightVect] "kv" (oldWeightVect),
              [newWeightVect] "kv" (newWeightVect)
            : );

    int8_t dVal = d[oldActionAddress];
    for(uint8_t i = 0; i < 16; i++)
        d[i] = dVal;

    //The vector d contains the result of
    //d = oldReward + gamma * newWeight - weight
    //The result is already shifted to the left << 1
    
    //Loop over all state-action pairs and update the synapses accordingly
    //Note: The struture of the network must be kept. e.g. it should not happen,
    //That new connections appear due to the update rule. The first row can be skipped, since there are only connections
    //from the FPGA to the state neurons
    for (uint32_t vectorIndex = 2; vectorIndex < (2 + MDP_STATES * 2)/*dls_num_synapse_vectors*/; vectorIndex++)
    {
        uint32_t eligibilityIndex = vectorIndex * 16;
        
        asm volatile (
            //Fetch the weight of the current vector index
            "fxvinx %[oldWeightVect], %[dls_weight_base], %[vectorIndex]\n"
            "fxvshb %[oldWeightVect], %[oldWeightVect], 1\n"
            //Calculate dQ = learningrate * delta * self.eligibilityTraces[state, action]
            //The weights and the eligibility trace are aligned
            //The learning rate will not be decayed
            "fxvmulbfs %[dQ], %[d], %[learningVect]\n"
            "fxvlax %[tempEligibility], %[eligibilityTrace], %[eligibilityIndex]\n"
            "fxvmulbfs %[dQ], %[dQ], %[tempEligibility]\n"
            //Update the weight
            "fxvaddbfs %[oldWeightVect], %[oldWeightVect], %[dQ]\n"
            //Set negative weights to 0
            "fxvcmpb %[oldWeightVect]\n"
		    "fxvsel %[oldWeightVect], %[oldWeightVect], %[zeros], 2\n"
            //Write the weights back to the chip
            "fxvshb %[oldWeightVect], %[oldWeightVect], -1\n"
            "fxvoutx %[oldWeightVect], %[dls_weight_base], %[vectorIndex]\n"
            //Update the eligibility traces: gamma * lam * self.eligibilityTraces[state, action]
            "fxvmulbfs %[temp], %[gammaVect], %[lambdaVect]\n"
            "fxvmulbfs %[tempEligibility], %[temp], %[tempEligibility]\n"
            : [oldWeightVect] "=&kv" (oldWeightVect),
              [dQ] "=&kv" (dQ),
              [tempEligibility] "=&kv" (tempEligibility),
              [temp] "=&kv" (temp)
            : [vectorIndex] "r" (vectorIndex),
              [eligibilityIndex] "r" (eligibilityIndex),
              [gammaVect] "kv" (gammaVect),
              [lambdaVect] "kv" (lambdaVect),
              [oldRewardVect] "kv" (oldRewardVect),
              [dls_weight_base] "b" (dls_weight_base),
              [learningVect] "kv" (learningVect),
              [eligibilityTrace] "r" (eligibilityTrace),
              [zeros] "kv" (zeros),
              [d] "kv" (d)
            : );

        //Consider that the eligibility trace of the currently selected action must not be changed here
        //Set this lamddaVect to 1 and gamma also to 1
        if(oldWeightIndex == vectorIndex)
        {
            tempEligibility[oldActionAddress] = eligibilityTrace[oldSynapseIndex];
        }

        asm volatile (
            //Store the temporary eligibility trace
            "fxvstax %[tempEligibility], %[eligibilityTrace], %[eligibilityIndex]\n"
            : 
            : [tempEligibility] "kv" (tempEligibility),
              [eligibilityIndex] "r" (eligibilityIndex),
              [eligibilityTrace] "r" (eligibilityTrace)
            : );
            
        //Update the Q table
        for(uint32_t i = 0; i < 16; i++)
        {
            /*if((255 - Q[eligibilityIndex + i]) > dQ[i])
                Q[eligibilityIndex + i] += dQ[i];
            else
                Q[eligibilityIndex + i] = 255;*/

            weights[eligibilityIndex + i] = oldWeightVect[i];
        }
        
    }

    //Decay the learning rate
    asm volatile (
            //Decay the learning rate
            "fxvmulbfs %[newLearningRate], %[learningVect], %[learningVectDecay]\n"
            : [newLearningRate] "=&kv" (newLearningRate)
            : [learningVect] "kv" (learningVect),
              [learningVectDecay] "kv" (learningVectDecay)
            : );
            
    //Update the learning rate
    eta = newLearningRate[0];
}
#endif

void runSimulation()
{
    //Read the weights once
    readWeights(weights);
    
    //Simulate an entire run and always take the action with the highest weights
    for(uint32_t i = 0; i < maxIteration + 1; i++)
    {
        //Present the state to the network and wait for a spike
        //presentStateAndReadoutAction();

        //Take the highest weight for the current state
        uint32_t synapseStartIndex = (1 + state) * 32;
        uint8_t maxWeightIdx = (1 + MDP_STATES);
        uint8_t maxWeight = 0;
        
        //Skip the state neurons
        for(uint8_t j = (1 + MDP_STATES); j < 32; j++)
        {
            if(weights[synapseStartIndex + j] > maxWeight)
            {
                maxWeight = weights[synapseStartIndex + j];
                maxWeightIdx = j;
            }
        }
        
        //Get the action based on the max weight
        uint8_t selectedAction = maxWeightIdx - (1 + MDP_STATES);
        
        //Perform the action
        performAction(selectedAction);
        
        //Perform the weight update
        if(i > 0)
            weightUpdate();
    }
    
    //The Q table does not need to be stored, since the weights are directly accessed by the python program
    //Store the Q table in mailbox
    //saveQTable();
}


//This is the main PPU program
void start(void)
{
    //Initialize the MDP problem
    initEnvironment();
    
    //Initialise the local variables
    init();

    //Load the additional plasticity parameters
    loadPlastParameter();

    //Run the simulation of the process
    //libnux_mailbox_write_string("Run\n");
    runSimulation();

    //Perform the weight update once
    //weightUpdateLambda();

    //plot();

    //Just test simple action selection and check the mailbox
    //readWeights(weights);
    //performAction(0);
    //weightUpdate();
    //performAction(0);
    //weightUpdate();
    //performAction(0);
    //weightUpdate();
    //performAction(0);
    //weightUpdate();
    
    /*performAction(1);
    performAction(1);
    performAction(2);
    performAction(0);*/

    //libnux_mailbox_write_string("\nProgram exited!\n");

    uint32_t a = 0;
    uint8_t signal = signal_wait;
    while(1)
    {
        a++;
        if (signal == signal_run)
            break;
    }

    //loop for N training samples
	
	do 
    {
		signal = libnux_mailbox_read_u8(signal_addr_offset);
		if (signal == signal_run) 
        {
			libnux_mailbox_write_u8(signal_addr_offset, signal_wait);

			//measure_correlation(ca_hist, ac_hist, ca_offsets, ac_offsets);
            
            //Perform the steps after the network selected an action
            if(iterationCounter > 0)
            {
                //libnux_mailbox_write_int(iterationCnt);
                //Readout action and get rewards
            	uint8_t action = readOutAction();

            	//Perform action on MDP / Maze
                

                //Perform synapse update based on reward


                //Store reward somewhere
            }

            //Present pattern
            uint32_t rnd = 0;
            
            //Send a test spike to one input neuron
            spike_t spike;
            spike.row_mask = 0; //To indicate that the first row is used. This is where all state neurons listen to
            spike.addr = 2; //This is the address of the 2nd state neuron

            //Wait for simulation to finish
            
		}
	} while (signal != signal_stop);
    

    //After training print learnt policy into mailbox
    //mailbox_write(ac_addr_offset, (uint8_t*)(ca_hist), num_bins * sizeof(uint32_t));
	//mailbox_write(ca_addr_offset, (uint8_t*)(ac_hist), num_bins * sizeof(uint32_t));
	libnux_mailbox_write_string("\nProgram exited!\n");
    
}
