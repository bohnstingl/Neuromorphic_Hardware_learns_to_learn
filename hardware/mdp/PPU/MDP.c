/*
 ============================================================================
 Name        : MDP.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
#include <stdint.h>
#include "MDP_Maze.h"

#if (CHIP == 1)
#include <s2pp.h>
#include "libnux/random.h"
#include "libnux/mailbox.h"
#include "Utils.h"
#else
#include "time.h"
#include "stdlib.h"
#endif

static const uint32_t stateOffset = 0x000;
static const uint32_t actionOffset = 0x800;
static const uint32_t iterationCounterOffset = 0xffc;

//The MDP is initialize with int values which are then transformed to rational numbers
#if (CHIP == 1)
uint32_t P[MDP_ACTIONS][MDP_STATES][MDP_STATES] = {1};
uint8_t R[MDP_ACTIONS][MDP_STATES][MDP_STATES] = {1};
#else
unsigned int P[MDP_ACTIONS][MDP_STATES][MDP_STATES] = {{{MAX_PVAL, 0}, {0, MAX_PVAL}},
											           {{MAX_PVAL, 0}, {0, MAX_PVAL}},
											           {{0, MAX_PVAL}, {0, MAX_PVAL}},
											           {{MAX_PVAL, 0}, {MAX_PVAL, 0}}};
int R[MDP_ACTIONS][MDP_STATES][MDP_STATES] = {{{0, 0}, {0, 0}},
											  {{0, 0}, {0, 0}},
											  {{0, 1}, {0, 0}},
											  {{0, 0}, {1, 0}}};
#endif

//Those variables are needed to be accessible from outside
uint32_t iterationCounter = 0;
uint32_t maxIteration = 0;
uint8_t states[2] = {0, 0};
uint8_t rewards[2] = {0, 0};
uint8_t actions[2] = {0, 0};
uint8_t state = 0;

uint32_t randomSeed;
uint32_t maxPValue = 0;
uint32_t readBytes = 0;

//This function returns a position for the next random starting point
void randomStart(uint8_t *s)
{
	//Get a random number between 0 and amountStartingPos
#if (CHIP == 1)
	//Read the random number from the mailbox. Appended at the end of the maze
    //*s = (uint8_t)((random_lcg(&randomSeed) >> 24) % MDP_STATES);
    *s = 0;
#else
    *s = 0;
#endif
}

//This function reads the maze configuration from the mailbox
void read()
{
#if (CHIP == 1)
    //Copy the matrices P and R directly into memory. Those are located after the random seed
    memcpy((uint32_t*)P, (uint32_t*) (&mailbox_base + readBytes), sizeof(uint32_t) * MDP_MATRIX_SIZE);
    readBytes += sizeof(uint32_t) * MDP_MATRIX_SIZE;
    memcpy((int8_t*)R, (int8_t*) (&mailbox_base + readBytes), sizeof(int8_t) * MDP_MATRIX_SIZE);
    readBytes += sizeof(int8_t) * MDP_MATRIX_SIZE;

    //Read the values for transforming the transition and reward prob. The values are located after the matrices
    maxPValue = *((uint32_t*) (&mailbox_base + readBytes));
    readBytes += sizeof(uint32_t);

    maxIteration = *((uint32_t*) (&mailbox_base + readBytes));
    readBytes += sizeof(uint32_t);
#else
    //Read the values for transforming the transition and reward prob
    maxPValue = 1;
    maxIteration = 10;
#endif
}

void saveToMailbox(n)
{
#if (CHIP == 1)
    *((uint8_t*)(&mailbox_base + (stateOffset + iterationCounter) * sizeof(uint8_t))) = states[0];
    *((uint8_t*)(&mailbox_base + (actionOffset + iterationCounter) * sizeof(uint8_t))) = actions[0];
    *((uint32_t*)(&mailbox_base + iterationCounterOffset)) = iterationCounter - 1;
#else
    printf("Reward: %d\n", rewards[0]);
    printf("State: %d\n", states[0]);
    printf("Action: %d\n", actions[0]);
#endif
}

void initEnvironment()
{
#if (CHIP == 1)
    //Update the random seed from the mailbox
    randomSeed = *((uint32_t*) (&mailbox_base));
    readBytes += sizeof(uint32_t);
#else
    srand(time(NULL));
    randomSeed = 4;
#endif

    //libnux_mailbox_write_string("Seed collected\n");

	//Read the maze from the mailbox
	read();

	//Init the maze at an random starting position
	randomStart(&state);

    //Cleanup read bytes
    uint8_t z = 0x00;

    for(uint32_t l = 0; l < readBytes; l++)
        memcpy((uint8_t*) (&mailbox_base + sizeof(uint8_t) * l), &z, sizeof(uint8_t));

    //Save the initial configurations
    saveToMailbox();
    iterationCounter++;

    //libnux_mailbox_write_string("maxPValue ");
    //libnux_mailbox_write_int(maxPValue);
    //libnux_mailbox_write_string("\n\n");
}

void plot()
{
#if (CHIP == 1)
    //libnux_mailbox_write_int(1);
    //libnux_mailbox_write_int(2);
#else
	printf("Observation is: %d\n", state);
	printf("Reward: %d\n", reward[1]);
#endif
}

//This function resets the maze to a random starting position
void reset()
{
	randomStart(&state);
	rewards[0] = 0;
    rewards[1] = 0;
    actions[0] = 0;
    actions[1] = 0;
    states[0] = 0;
    states[1] = 0;
}

//This function selects the next state based on the transition probabilities
uint8_t selectNextState(uint8_t st, uint8_t action)
{
#if (CHIP == 1)
    //Generates random numbers of type int
	uint32_t rnd = ((uint32_t)random_lcg(&randomSeed)) % maxPValue;
#else
	uint32_t rnd = (((double)rand()) / RAND_MAX) * (maxPValue);
#endif

	//The random number is transformed to a uint value and then compared
	uint32_t sum = 0;
	for(uint8_t nState = 0; nState < MDP_STATES; nState++)
	{
		sum += P[action][st][nState];
		if(rnd <= sum)
		    return nState;
	}

	return MDP_STATES - 1;
}

//This function performs an action for the current state
void performAction(uint8_t act)
{
	//The transition matrix P as well as the reward matrix R are arranged as [old State][action][new State] and give an integer value
	uint8_t newState = selectNextState(state, act);

    /*if(state >= MDP_STATES)
    {
        libnux_mailbox_write_u8(0xff0, 255);
        libnux_mailbox_write_u8(0xff1, state);
        while(1)
            ;
    }

    if(newState >= MDP_STATES)
    {
        libnux_mailbox_write_u8(0xff0, 254);
        libnux_mailbox_write_u8(0xff1, newState);
        while(1)
            ;
    }*/
    
    //Update the old values
    states[0] = states[1];
    rewards[0] = rewards[1];
    actions[0] = actions[1];

	//Get the reward. The reward is given in the fractional representation between 1 and 0.
    rewards[1] = R[act][state][newState];
    actions[1] = act;
    states[1] = state;        
    
    state = newState;
    
    saveToMailbox();
    iterationCounter++;
    
}
