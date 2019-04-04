/*
 ============================================================================
 Name        : Maze.c
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

//The rows are the height and the columns are the width
//x specifies the rows and y specifies the columns
#if (CHIP == 1)
uint8_t maze[MAX_MAZE_HEIGHT][MAX_MAZE_WIDTH] = {1};
#else
uint8_t maze[MAX_MAZE_HEIGHT][MAX_MAZE_WIDTH] = {{0, 0, 0, 1, 0},
		                                         {0, 1, 0, 1, 0},
												 {0, 1, 0, 1, 0},
												 {0, 1, 0, 1, 0},
												 {0, 1, 0, 0, 0}};
#endif

static const uint32_t stateOffset = 0x000;
static const uint32_t actionOffset = 0x0800;
static const uint32_t rewardOffset = 0xf00;
static const uint32_t winOffset = 0xff0;
static const uint32_t iterationCounterOffset = 0xffc;

//Those variables are needed to be accessible from outside
uint32_t iterationCounter = 0;
uint32_t episodeLength = 0;
uint32_t maxIteration = 0;
uint8_t states[2] = {0, 0};
uint8_t rewards[2] = {0, 0};
uint8_t actions[2] = {0, 0};
uint8_t state = 0;

uint8_t goalPosition = 0;
uint8_t potentialStartingPos[MAX_MAZE_STATES] = {1};
uint32_t amountStartingPos = 0;
uint32_t randomSeed = 0;
uint32_t maxPValue = 0;
uint32_t readBytes = 0;
uint32_t wins = 0;

//This function returns a position for the next random starting point
void randomStart(uint8_t *s)
{
	//Get a random number between 0 and amountStartingPos
#if (CHIP == 1)
	//Read the random number from the mailbox. Appended at the end of the maze
    uint8_t randomIdx = (uint8_t)((random_lcg(&randomSeed) >> 24) % amountStartingPos);
#else
    uint8_t randomIdx = 6;
#endif
    *s = potentialStartingPos[randomIdx];
    //*s = 0;
}

//This function reads the maze configuration from the mailbox
void read()
{
#if (CHIP == 1)

    //Copy the maze directly into memory. The maze is located after the random seed
    memcpy((uint8_t*)maze, (uint8_t*) (&mailbox_base + readBytes), sizeof(uint8_t) * MAX_MAZE_STATES);
    readBytes += sizeof(uint8_t) * MAX_MAZE_STATES;

    //The goal position is located after the maze
    uint8_t goalPositionX = *((uint8_t*) (&mailbox_base + readBytes));
    readBytes += sizeof(uint8_t);

    uint8_t goalPositionY = *((uint8_t*) (&mailbox_base + readBytes));
    readBytes += sizeof(uint8_t);

    //Read the amount of max iterations
    memcpy((uint8_t*)&maxIteration, (uint8_t*) (&mailbox_base + readBytes), sizeof(uint32_t));
    readBytes += sizeof(uint32_t);
#else
    uint8_t goalPositionX = 4;
    uint8_t goalPositionY = 0;
#endif

    goalPosition = goalPositionX * MAX_MAZE_WIDTH + goalPositionY;
}

void saveToMailbox(n)
{
#if (CHIP == 1)
    *((uint8_t*)(&mailbox_base + (stateOffset + iterationCounter) * sizeof(uint8_t))) = states[0];
    *((uint8_t*)(&mailbox_base + (actionOffset + iterationCounter) * sizeof(uint8_t))) = actions[0];
    //*((uint8_t*)(&mailbox_base + (rewardOffset + iterationCounter) * sizeof(uint8_t))) = rewards[0];
    *((uint32_t*)(&mailbox_base + iterationCounterOffset)) = iterationCounter - 1;
#else
    printf("Reward: %d\n", rewards[0]);
    printf("State: %d\n", states[0]);
    printf("Action: %d\n", actions[0]);
#endif
}

void initEnvironment()
{
    readBytes = 0;
#if (CHIP == 1)
    //Update the random seed from the mailbox
    randomSeed = *((uint32_t*) (&mailbox_base));
    readBytes += sizeof(uint32_t);
#else
    srand(time(NULL));
    randomSeed = 4;
#endif

	//Read the maze from the mailbox
	read();

	amountStartingPos = 0;
	for(uint8_t i = 0; i < MAX_MAZE_HEIGHT; i++)
	{
		for(uint8_t j = 0; j < MAX_MAZE_WIDTH; j++)
		{
			if((i * MAX_MAZE_WIDTH + j) != goalPosition && maze[i][j] == FIELD_FREE)
			{
				potentialStartingPos[amountStartingPos] = i * MAX_MAZE_WIDTH + j;
				amountStartingPos++;
			}
		}
	}

	//Init the maze at an random starting position
	randomStart(&state);

    //Cleanup read bytes
    uint8_t z = 0x00;

    for(uint32_t l = 0; l < readBytes; l++)
        memcpy((uint8_t*) (&mailbox_base + sizeof(uint8_t) * l), &z, sizeof(uint8_t));

    //Save the initial configurations
    saveToMailbox();
    iterationCounter++;
}

void plot()
{
    uint8_t stateX, stateY;
    uint8_t goalX, goalY;
    transformState(state, &stateX, &stateY);
    transformState(goalPosition, &goalX, &goalY);
    
    for(uint8_t i = 0; i < MAX_MAZE_HEIGHT; i++)
	{
		for(uint8_t j = 0; j < MAX_MAZE_WIDTH; j++)
		{
			if (stateX == i && stateY == j)
			{
				libnux_mailbox_write_int(2);
			}
            else if(goalX == i && goalY == j)
			{
				libnux_mailbox_write_int(5);
			}
			else if(maze[i][j] == FIELD_FREE)
			{
				libnux_mailbox_write_int(0);
			}
            else
			{
				libnux_mailbox_write_int(1);
			}
            libnux_mailbox_write_string(", ");
		}
		libnux_mailbox_write_string("\n");
	}

    libnux_mailbox_write_string("\n\n");
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

void transformState(uint8_t st, uint8_t *stateX, uint8_t *stateY)
{
	//This function is used to transform the state into x and y coordinates
	*stateX = st / MAX_MAZE_WIDTH;
	*stateY = st % MAX_MAZE_WIDTH;
}

//This function selects the next state based on the transition probabilities
void selectNextState(uint8_t st, uint8_t action, uint8_t *nextSt, uint8_t *reward)
{
	//action 0 -> move to North
	//action 1 -> move to East
	//action 2 -> move to South
	//action 3 -> move to West

	//Get the transformed state
	uint8_t stateX, stateY;
	transformState(st, &stateX, &stateY);
    

	//Check if the borders are hit
	if(action == 0 && stateX == 0)
	{
		*reward = REWARD_BORDER;
		*nextSt = st;
		return;
	}

	if(action == 1 && stateY == (MAX_MAZE_WIDTH - 1))
	{
		*reward = REWARD_BORDER;
		*nextSt = st;
		return;
	}

	if(action == 2 && stateX == (MAX_MAZE_HEIGHT - 1))
	{
		*reward = REWARD_BORDER;
		*nextSt = st;
		return;
	}

	if(action == 3 && stateY == 0)
	{
		*reward = REWARD_BORDER;
		*nextSt = st;
		return;
	}

	//If this line is rewached, then there was no collision with the game borders
	//Now check the walls itself
	if(action == 0 && (maze[stateX - 1][stateY] == FIELD_WALL))
	{
		*reward = REWARD_WALL;
		*nextSt = st;
		return;
	}

	if(action == 1 && (maze[stateX][stateY + 1] == FIELD_WALL))
	{
		*reward = REWARD_WALL;
		*nextSt = st;
		return;
	}

	if(action == 2 && (maze[stateX + 1][stateY] == FIELD_WALL))
	{
		*reward = REWARD_WALL;
		*nextSt = st;
		return;
	}

	if(action == 3 && (maze[stateX][stateY - 1] == FIELD_WALL))
	{
		*reward = REWARD_WALL;
		*nextSt = st;
		return;
	}


	//If this line is reached, the action is valid and will be performed
	if(action == 0)
		stateX--;

	if(action == 1)
		stateY++;

	if(action == 2)
		stateX++;

	if(action == 3)
		stateY--;

    uint8_t goalPositionX, goalPositionY;
    transformState(goalPosition, &goalPositionX, &goalPositionY);
    
	if(stateX == goalPositionX && stateY == goalPositionY)
	{
        //The goal was hit, reset the environment
		randomStart(nextSt);
		*reward = REWARD_GOAL;
        episodeLength = 0;
	    wins++;
        *((uint32_t*)(&mailbox_base + winOffset)) = wins;
        //libnux_mailbox_write_u8(0xff0, iterationCounter);        
        //while(1)
        //    ;
	}
	else
	{
		*reward = NO_REWARD;
		*nextSt = stateX * MAX_MAZE_WIDTH + stateY;
	}
}

//This function performs an action for the current state
void performAction(uint8_t action)
{
    //Increase the episode length
    episodeLength++;

	//Update the old values
    states[0] = states[1];
    rewards[0] = rewards[1];
    actions[0] = actions[1];
	
	uint8_t newState;
	uint8_t reward;
	selectNextState(state, action, &newState, &reward);
	
	//Store the new reward
	rewards[1] = reward;
	actions[1] = action;
    states[1] = state;

	state = newState;
    
	saveToMailbox();
	iterationCounter++;

    //If a certain amount of trials are not sucessful, reset the agent
    if((episodeLength % MAX_MAZE_EPISODE_LENGTH) == 0)
    {
        randomStart(&state);
        episodeLength = 0;
    }
}
