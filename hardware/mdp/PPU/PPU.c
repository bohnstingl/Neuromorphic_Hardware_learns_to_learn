//PLAST 0 ... TD(1) 8-bit
//PLAST 1 ... TD(Lam) 8-bit
//PLAST 2 ... TD(1) 8-bit simplified code
//PLAST 3 ... TD(Lam) 8-bit simplified code
//PLAST 4 ... TD(1) 32-bit parameters

#define PLAST 2
#define RATE_SELECT 0
#define SHIFT_WEIGHTS 1
#define USE_SPIKES 1

#include <s2pp.h>
#include <stdint.h>
#include "libnux/mailbox.h"
#include "libnux/dls_v2.h"
#include "libnux/random.h"
#include "spikes.h"
#include "MDP_Maze.h"
#include "Utils.h"

//Temporary definition for inhibition wait cycles
#define N_CYCLES_FOR_INHIBITION 100
#define N_CYCLES_FOR_SPIKE 2000

//Address from which the test read command is performed
static uint32_t const signal_addr_offset = 0xff0;

//This is the number of already read bytes from the mailbox
extern uint32_t readBytes;
extern uint32_t iterationCounter;
extern uint8_t states[2];
extern uint8_t rewards[2];
extern uint8_t actions[2];
extern uint8_t state;
extern uint32_t maxPValue;
extern uint32_t maxIteration;
extern uint32_t randomSeed;

int8_t gamma, lambda, eta;
uint8_t weights[MDP_STATES * MDP_ACTIONS] = {1};
int8_t eligibilityTrace[MDP_STATES * MDP_ACTIONS] = {1};
uint8_t weightLower = 0;
uint8_t weightUpper = 0;
uint32_t weightRescaleFreq = 0;
uint32_t spikeCounts[MDP_ACTIONS] = {1};

uint8_t presentStateAndReadoutAction()
{
    uint32_t nonzero = 0;
    uint32_t randint;
    uint32_t num_nonzero = 0;
    uint32_t active_counter = 0;
    int i, z;
    uint8_t actionNr;
    uint8_t maxSpikeCnt = 0;
    uint8_t amountMaxSpike = 0;

    //This function sends spikes to the action neuron and activates the correct one
    spike_t spike;

    //Send the spikes to the desired state neuron driver. Watch out, use the off-diagonal elements
    spike.row_mask = 1 << ((state + 1) % MDP_STATES);

    //Use the recurrent address of the state neuron
    spike.addr = state;

    update_spike_counters(spikeCounts, MDP_ACTIONS, MDP_STATES);
    reset_spike_counters(spikeCounts, MDP_ACTIONS);

    //Send a single spike. The state neuron is connected recurrently.
    //If an action neuron spikes, this will stop the recurrency
    
    for(z = 0; z < 3; z++)
    {
        spikes_send(&spike);
        wait_cycles(100);
    }
    
    wait_cycles(N_CYCLES_FOR_SPIKE);
    
    //Update the spike counter and get
    num_nonzero = update_spike_counters(spikeCounts, MDP_ACTIONS, MDP_STATES);
    
#if (RATE_SELECT == 1)
    // choose the action with the highest spike counter. If multiple of them have the same counter,
    // take a random one

    for (i = 0; i < MDP_ACTIONS; i ++)
    {
        if (spikeCounts[i] > maxSpikeCnt)
        {
            maxSpikeCnt = spikeCounts[i];
            amountMaxSpike = 1;
        }
        else if (spikeCounts[i] == maxSpikeCnt)
        {
            amountMaxSpike++;
        }
    }

    randint = (uint32_t) random_lcg(&randomSeed) >> 24;
    randint = randint % amountMaxSpike;

    //Choose the random action among the 
    for (i = 0; i < MDP_ACTIONS; i ++)
    {
        if (spikeCounts[i] == maxSpikeCnt && active_counter++ == randint)
        {
            actionNr = i;
            break;
        }
    }
    
#else

    // choose randomly among actions
    randint = (uint32_t) random_lcg(&randomSeed) >> 24;
    randint = randint % num_nonzero;
    //libnux_mailbox_write_string("Fired\n");
    //libnux_mailbox_write_int(num_nonzero);

    /*libnux_mailbox_write_string("Non zero ");
    libnux_mailbox_write_int(num_nonzero);
    libnux_mailbox_write_string("\n");
    
    for (i = 0; i < MDP_ACTIONS; i ++)
    {
        libnux_mailbox_write_int(spikeCounts[i]);
        libnux_mailbox_write_string(", ");
    }

    libnux_mailbox_write_string("\n");*/
    

    for (i = 0; i < MDP_ACTIONS; i ++)
    {
        if (spikeCounts[i] && active_counter++ == randint)
        {
            /*if(iterationCounter > 0)
            {
                libnux_mailbox_write_string("Activated ");
                libnux_mailbox_write_int(state);
                libnux_mailbox_write_string("\n Selected ");
                libnux_mailbox_write_int(i);
                libnux_mailbox_write_string("\n");
            }*/
            actionNr = i;
            break;
        } 
    }

#endif

    //libnux_mailbox_write_string("Rnd\n");

    //If no action neuron spiked, take an random action
    if(num_nonzero == 0)
    {
        //If this point is reached, pick a random action
        actionNr = ((uint32_t) random_lcg(&randomSeed) >> 24) % MDP_ACTIONS;
        //libnux_mailbox_write_string("\n Selected ");
        //libnux_mailbox_write_int(i);
    }

    //libnux_mailbox_write_string("\n\n");

    //Stop the recurrent spiking by using the inhibitory synapses of the action neurons.
    spike.row_mask = 0;
    for(z = 0; z < MDP_ACTIONS; z++)
    {
        spike.row_mask |= 1 << (MDP_STATES + z);
    }
    
    spike.addr = 50;
    for(z = 0; z < 10; z++)
    {
        spikes_send(&spike);
        wait_cycles(100);
    }

    update_spike_counters(spikeCounts, MDP_ACTIONS, MDP_STATES);
    reset_spike_counters(spikeCounts, MDP_ACTIONS);
    
    return actionNr;
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

#if (SHIFT_WEIGHTS == 1)
    //This parameter indicates the weight value, where the weights should be shifted to when enabled
    weightLower = *((uint8_t*) (&mailbox_base + readBytes));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    readBytes += sizeof(uint8_t);
    weightUpper = *((uint8_t*) (&mailbox_base + readBytes));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    readBytes += sizeof(uint8_t);

    //Read the rescale frequency
    memcpy((uint8_t*)&weightRescaleFreq, (uint8_t*) (&mailbox_base + readBytes), sizeof(uint32_t));
    *((int8_t*) (&mailbox_base + readBytes)) = 0;
    *((int8_t*) (&mailbox_base + readBytes + 1)) = 0;
    *((int8_t*) (&mailbox_base + readBytes + 2)) = 0;
    *((int8_t*) (&mailbox_base + readBytes + 3)) = 0;
    readBytes += sizeof(uint32_t);
#endif
    readBytes = 0;

    /*libnux_mailbox_write_string("Weight shifting ");
    libnux_mailbox_write_int(weightLower);
    libnux_mailbox_write_string("\n\n");
    libnux_mailbox_write_string("Weight shifting ");
    libnux_mailbox_write_int(weightUpper);
    libnux_mailbox_write_string("\n\n");*/
}

void init()
{
#if (PLAST == 1 || PLAST == 3)
    //Initialise the qtable and the eligibility traces
    for(uint32_t i = 0; i < (MDP_STATES * MDP_ACTIONS); i++)
    {
        eligibilityTrace[i] = 0;
    }
#endif

    //Clear the spike counteres
    reset_spike_counters(spikeCounts, MDP_ACTIONS);
}

#if (PLAST == 0)
void weightUpdate()
{
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables
    uint32_t oldSynapseIndex = states[0] * 32 + actions[0] + MDP_STATES;
    uint32_t newSynapseIndex = states[1] * 32 + actions[1] + MDP_STATES;
    
    //Vector index for accessing the old weights    
    uint32_t oldWeightIndex = states[0] * 2;
    if((actions[0] + MDP_STATES) >= 16)
        oldWeightIndex++;

    uint32_t newWeightIndex = states[1] * 2;
    if((actions[1] + MDP_STATES) >= 16)
        newWeightIndex++;

    uint8_t oldActionAddress = (actions[0] + MDP_STATES) % 16;
    uint8_t newActionAddress = (actions[1] + MDP_STATES) % 16;

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

    gammaVect[oldActionAddress] = gamma;//gamma;//0b01111000;
    oldRewardVect[oldActionAddress] = rewards[0];//0b01000000;//oldReward;
    learningVect[oldActionAddress] = eta;//eta;//0b00110000;

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

    /*libnux_mailbox_write_string("Start: \n");
    libnux_mailbox_write_int(rewards[0]);
    libnux_mailbox_write_string("\n");
    libnux_mailbox_write_int(weights[oldSynapseIndex]);
    libnux_mailbox_write_string("\n");
    libnux_mailbox_write_int(oldWeightVect[oldActionAddress]);
    libnux_mailbox_write_string("\n\n");*/
    
    //Recompute the new learning rate
    eta = newLearningRate[oldActionAddress];
    
    //Update the internal weight set
    weights[oldSynapseIndex] = oldWeightVect[oldActionAddress];
}

#elif (PLAST == 1)
void weightUpdate()
{
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables

    uint32_t oldSynapseIndex = states[0] * 32 + (actions[0] + MDP_STATES);
    uint32_t newSynapseIndex = states[1] * 32 + (actions[1] + MDP_STATES);
    
    //Vector index for accessing the old weights    
    uint32_t oldWeightIndex = states[0] * 2;
    if((actions[0] + MDP_STATES) >= 16)
        oldWeightIndex++;

    uint32_t newWeightIndex = states[1] * 2;
    if((actions[1] + MDP_STATES) >= 16)
        newWeightIndex++;

    uint8_t oldActionAddress = (actions[0] + MDP_STATES) % 16;
    uint8_t newActionAddress = (actions[1] + MDP_STATES) % 16;

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
    vector int8_t lambdaVect = vec_splat_s8(lambda);//lambda);
	vector int8_t oldRewardVect = vec_splat_s8(0);//oldReward);
    vector int8_t learningVect = vec_splat_s8(eta);//eta);
    vector int8_t learningVectDecay = vec_splat_s8(0);
    vector int8_t zeros = vec_splat_s8(0);

    gammaVect[oldActionAddress] = gamma;//gamma;
    oldRewardVect[oldActionAddress] = rewards[0];//0b01000000;//oldReward;
    //learningVect[oldActionAddress] = eta;//eta;

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
    
    //update the eligibility trace
    uint8_t stateIdx, actionIdx;
    for(stateIdx = 0; stateIdx < MDP_STATES; stateIdx++)
    {
        for(actionIdx = 0; actionIdx < MDP_ACTIONS; actionIdx++)
        {
            uint8_t currIdx = stateIdx * 32 + actionIdx;

            //Handle the eligibility traces
            if(stateIdx == states[0] && actionIdx == actions[0])
            {
                eligibilityTrace[currIdx] = ((gamma * lambda * eligibilityTrace[currIdx]) >> 14) + 5;

                if(eligibilityTrace[currIdx] > 127)
                    eligibilityTrace[currIdx] = 127;
            }
            else
            {
                eligibilityTrace[currIdx] = (gamma * eligibilityTrace[currIdx]) >> 14;
            }
        }        
    }

    for (uint32_t vectorIndex = 2; vectorIndex < (2 + MDP_STATES * 2)/*dls_num_synapse_vectors*/; vectorIndex++)
    {
        uint32_t eligibilityIndex = vectorIndex * 16;
        
        asm volatile (
            //Fetch the weight of the current vector index
            "fxvinx %[oldWeightVect], %[dls_weight_base], %[vectorIndex]\n"
            "fxvshb %[oldWeightVect], %[oldWeightVect], 1\n"
            //Update the eligibility traces: gamma * lam * self.eligibilityTraces[state, action]
            //"fxvmulbfs %[temp], %[gammaVect], %[lambdaVect]\n"
            //"fxvmulbfs %[tempEligibility], %[temp], %[tempEligibility]\n"
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
            
        //Update the Q table
        for(uint32_t i = 0; i < 16; i++)
        {
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
    //eta = newLearningRate[0];
}

#elif (PLAST == 2)
void weightUpdate()
{
    //Update rule TD(1) with simplified code

    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables
    uint8_t oldSynapseIndex = states[0] * MDP_ACTIONS + actions[0];
    uint8_t newSynapseIndex = states[1] * MDP_ACTIONS + actions[1];
    
    uint8_t oldWeight = weights[oldSynapseIndex];
    uint8_t newWeight = weights[newSynapseIndex];

    //The fixed point numbers can be used as numbers which are multiplied by 2**8.
    //Interpreting them as normal numbers results in
    //delta << 7 = oldReward + gamma * newWeight - weight << 7
    //int64_t delta = rewards[0] + gamma * newWeight - (oldWeight << 7);

    //Values are in the range of 2**6 * 2**7 = 2**13
    //For the maze case check if the reward is negative
    int64_t delta = 0;
    delta = gamma * newWeight - (oldWeight << 7);
    if(((uint8_t)rewards[0]) > 127)
    {
        delta = delta - (rewards[0] << 6);
    }
    else
    {
        delta = delta + (rewards[0] << 6);
    }
    
    delta = delta * eta;
    uint8_t updatedWeight = (uint8_t)((delta + (oldWeight << 14)) >> 14);
        
    //Set the weight
    set_weight(states[0], actions[0] + MDP_STATES, updatedWeight);

    /*libnux_mailbox_write_string("Start: \n");
    libnux_mailbox_write_int(rewards[0]);
    libnux_mailbox_write_string("\n");
    libnux_mailbox_write_int(oldWeight);
    libnux_mailbox_write_string("\n");
    libnux_mailbox_write_int(newWeight);
    libnux_mailbox_write_string("\n");
    libnux_mailbox_write_int(updatedWeight);
    libnux_mailbox_write_string("\n\n");*/

    //Update the internal weight set
    weights[oldSynapseIndex] = updatedWeight;

    //Recompute the new learning rate
    //eta = (int8_t)((eta * 0b01111111) >> 7);
}

#elif (PLAST == 3)
void weightUpdate()
{
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables

    uint8_t oldSynapseIndex = states[0] * MDP_ACTIONS + actions[0];
    uint8_t newSynapseIndex = states[1] * MDP_ACTIONS + actions[1];
    
    uint8_t oldWeight = weights[oldSynapseIndex];
    uint8_t newWeight = weights[newSynapseIndex];

    //The fixed point numbers can be used as numbers which are multiplied by 2**8.
    //Interpreting them as normal numbers results in
    //delta << 7 = oldReward + gamma * newWeight - weight << 7
    //int64_t delta = (rewards[0] << 6) + gamma * newWeight - (oldWeight << 7);
    int64_t delta = 0;
    delta = gamma * newWeight - (oldWeight << 7);
    if(((uint8_t)rewards[0]) > 127)
    {
        delta = delta - (rewards[0] << 6);
    }
    else
    {
        delta = delta + (rewards[0] << 6);
    }

    //Shift back, but take into account, that negative numbers must be converted into positive ones before.
    delta = delta * eta;

    //Iterate over all state-action pairs
    uint8_t stateIdx, actionIdx;
    for(stateIdx = 0; stateIdx < MDP_STATES; stateIdx++)
    {
        for(actionIdx = 0; actionIdx < MDP_ACTIONS; actionIdx++)
        {
            uint8_t currIdx = stateIdx * MDP_ACTIONS + actionIdx;

            //Handle the eligibility traces
            if(stateIdx == states[0] && actionIdx == actions[0])
            {
                eligibilityTrace[currIdx] += 1;

                if(eligibilityTrace[currIdx] > 127)
                    eligibilityTrace[currIdx] = 127;
            }
            
            if(((int64_t)(delta * eligibilityTrace[currIdx]) + (int64_t)((weights[currIdx] << 21))) < 0)
                weights[currIdx] = 0;
            else          
                weights[currIdx] = ((weights[currIdx] << 21) + delta * eligibilityTrace[currIdx]) >> 21;
                
            eligibilityTrace[currIdx] = ((gamma * lambda * eligibilityTrace[currIdx]) >> 14);

            set_weight(stateIdx, MDP_STATES + actionIdx, weights[currIdx]);
        }        
    }
            
    //Update the learning rate
    //eta = newLearningRate[0];
}

#elif (PLAST == 4)
void weightUpdate()
{
    //State update function using TD(1) with full precision weights and parameters in memory
        
    //The old state, new state, old action, new action and the reward are generated in the environment and can be accessed via external variables
    uint32_t oldSynapseIndex = states[0] * MDP_ACTIONS + actions[0];
    uint32_t newSynapseIndex = states[1] * MDP_ACTIONS + actions[1];

    //Vector index for accessing the old weights    
    uint32_t oldWeightIndex = states[0] * 2;
    if((actions[0] + MDP_STATES) >= 16)
        oldWeightIndex++;

    uint32_t newWeightIndex = states[1] * 2;
    if((actions[1] + MDP_STATES) >= 16)
        newWeightIndex++;

    uint8_t oldActionAddress = (actions[0] + MDP_STATES) % 16;
    uint8_t newActionAddress = (actions[1] + MDP_STATES) % 16;

	// Declare temporary variables
	register vector int8_t oldWeightVect;
	register vector int8_t newWeightVect;

    //Caution, the newWeight and the old weight are not on the same position!!
    //Prepare the weight vector
    asm volatile (
            //Fetch the weight of the old and the new synapse
            "fxvinx %[oldWeightVect], %[dls_weight_base], %[oldWeightIndex]\n"
            //"fxvshb %[oldWeightVect], %[oldWeightVect], 1\n"
            "fxvinx %[newWeightVect], %[dls_weight_base], %[newWeightIndex]\n"
            //"fxvshb %[newWeightVect], %[newWeightVect], 1\n"
            : [oldWeightVect] "=&kv" (oldWeightVect),
              [newWeightVect] "=&kv" (newWeightVect)
            : [oldWeightIndex] "r" (oldWeightIndex),
              [newWeightIndex] "r" (newWeightIndex),
              [dls_weight_base] "b" (dls_weight_base)
            : );

    //double delta = rewards[0] + gamma * weights[newSynapseIndex] - weights[oldSynapseIndex];
    //uint32_t w = (uint32_t)(weights[oldSynapseIndex] + delta);
    uint32_t w = 0;
    
    //Update the value of the old weight
    oldWeightVect[oldActionAddress] = w;
    
    asm volatile (
            //Write the weights back to the chip
            "fxvoutx %[oldWeightVect], %[dls_weight_base], %[oldWeightIndex]\n"
            : 
            : [oldWeightVect] "kv" (oldWeightVect),
              [oldWeightIndex] "r" (oldWeightIndex),
              [dls_weight_base] "b" (dls_weight_base)
            : );
    
    //Recompute the new learning rate
    //eta = newLearningRate[oldActionAddress];
    
    //Update the internal weight set
    weights[oldSynapseIndex] = w;
}
#endif

void runSimulation()
{
    //Read the weights once
    readWeightsSelective(weights, MDP_STATES, MDP_ACTIONS);

    //libnux_mailbox_write_string("I am here\n");
    
    //Simulate an entire run and always take the action with the highest weights
    for(uint32_t i = 0; i < maxIteration + 1; i++)
    {
#if (USE_SPIKES == 0)
        //Take the highest weight for the current state
        uint8_t synapseStartIndex = state * MDP_ACTIONS;
        uint8_t maxWeightIdx = 0;
        uint32_t maxWeight = 0;
        
        //Skip the state neurons
        for(uint8_t j = 0; j < MDP_ACTIONS; j++)
        {
            if(weights[synapseStartIndex + j] > maxWeight)
            {
                maxWeight = weights[synapseStartIndex + j];
                maxWeightIdx = j;
            }
        }
        
        //Get the action based on the max weight
        uint8_t selectedAction = maxWeightIdx;
#else
        //Present the state to the network and wait for a spike
        uint8_t selectedAction = presentStateAndReadoutAction();
#endif
        
        //Perform the action
        performAction(selectedAction);

        /*libnux_mailbox_write_string("Max ");
        libnux_mailbox_write_int(maxWeight);
        libnux_mailbox_write_string("\n");
        libnux_mailbox_write_string("Performed ");
        libnux_mailbox_write_int(selectedAction);
        libnux_mailbox_write_string("\n\n");*/
        
        //Perform the weight update
        if(i > 0)
            weightUpdate();

#if (SHIFT_WEIGHTS == 1)
        //At a cerain period the weights will be shifted to a certain value range.
        //This is done since the resolution is better in this particular region
        if ((i % weightRescaleFreq) == 0 && i > 0)
        {
            readWeightsSelective(weights, MDP_STATES, MDP_ACTIONS);

            for(uint8_t si = 0; si < MDP_STATES; si++)
            {
                uint8_t weightsPerState[MDP_ACTIONS] = {1};
                uint8_t maxWeightVal = 0, minWeightVal = 100;
                
                for(uint8_t ai = 0; ai < MDP_ACTIONS; ai++)
                {
                    weightsPerState[ai] = weights[si * MDP_ACTIONS + ai];
                    
                    //Detect the min and the max of the current action weights
                    if (weightsPerState[ai] > maxWeightVal)
                    {
                        maxWeightVal = weightsPerState[ai];
                    }

                    if (weightsPerState[ai] < minWeightVal)
                    {
                        minWeightVal = weightsPerState[ai];
                    }
                }

                /*libnux_mailbox_write_string("W ");
                libnux_mailbox_write_int(maxWeightVal);
                libnux_mailbox_write_string(", ");
                libnux_mailbox_write_int(minWeightVal);
                libnux_mailbox_write_string("\n\n");*/

                
                //k * x + d
                //k = (newUpper - newLower) / (oldUpper - oldLower)
                //d = newUpper - k * oldUpper
                //calculate k1 and d1 from it
                //k1 = 1 / k
                //d1 = oldUpper - k * newUpper
               
                //Check if the max and the min Value are all the same and the difference is 0.
                //If this is the case, set all to the new maximum value
                if (minWeightVal == maxWeightVal)
                {
                    for(uint8_t ai = 0; ai < MDP_ACTIONS; ai++)
                    {
                        weightsPerState[ai] = weightUpper;
                    }
                }                
                else
                {
                    //Compute k1 * 1024
                    //uint32_t k1 = udiv32((maxWeightVal - minWeightVal) << 10, weightUpper - weightLower);
                    uint32_t k = udiv32((weightUpper - weightLower) << 10, maxWeightVal - minWeightVal);

                    //Compute d1 * 1024; this value might be negative
                    //int32_t d1 = (maxWeightVal << 10) - k1 * weightUpper;
                    int32_t d = (weightUpper << 10) - k * maxWeightVal;

                    //Scale the weights
                    //If all weights are the same, they will be scaled to the upper given weight
                    for(uint8_t ai = 0; ai < MDP_ACTIONS; ai++)
                    {
                        if(weightsPerState[ai] == maxWeightVal)
                        {
                            weightsPerState[ai] = weightUpper;
                            continue;
                        }
                        else if (weightsPerState[ai] == minWeightVal)
                        {
                            weightsPerState[ai] = weightLower;
                            continue;
                        }

                        //Scale the weight inbetween
                        uint8_t newWeight = (uint8_t)((k * weightsPerState[ai] + d) >> 10);

                        /*libnux_mailbox_write_string("From19863 ");
                        libnux_mailbox_write_int(weightsPerState[ai]);
                        libnux_mailbox_write_string(" to ");
                        libnux_mailbox_write_int(newWeight);
                        libnux_mailbox_write_string("\n\n");*/

                        weightsPerState[ai] = newWeight;
                    }
                }

                //Actually set the weights in the synram
                set_weights(si, MDP_STATES, weightsPerState, MDP_ACTIONS);
            }
        }
#endif

        wait_cycles(2000);
    }
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
    runSimulation();
}
