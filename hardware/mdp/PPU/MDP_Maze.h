/*
 ============================================================================
 Name        : MDPMaze.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#if (MDP == 1)
#define MDP_STATES 6
#define MDP_ACTIONS 8
#define MDP_MATRIX_SIZE (MDP_ACTIONS * MDP_STATES * MDP_STATES)
#else
#define MAX_MAZE_WIDTH 3
#define MAX_MAZE_HEIGHT 3
#define MAX_MAZE_STATES (MAX_MAZE_WIDTH * MAX_MAZE_HEIGHT)
#define MAX_MAZE_EPISODE_LENGTH 100
#define REWARD_BORDER 255
#define REWARD_WALL 255
#define REWARD_GOAL 127
#define NO_REWARD 0
#define FIELD_FREE 0
#define FIELD_WALL 1
#define MDP_STATES MAX_MAZE_STATES
#define MDP_ACTIONS 4
#endif

//Defines used by both environments
#define CHIP 1
#define MAX_PVAL 65536

void randomStartMDP(uint8_t* s);
void read();
void saveToMailbox();
void initEnvironment();
void plot();
void reset();
void performAction(uint8_t act);
