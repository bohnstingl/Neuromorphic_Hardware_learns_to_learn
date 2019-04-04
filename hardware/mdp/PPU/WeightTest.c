#include <s2pp.h>
#include <stdint.h>
#include "libnux/mailbox.h"
#include "libnux/dls_v2.h"
#include "spikes.h"
#include "Utils.h"
#include "libnux/fxdpt.h"

//Address from which the test read command is performed
static uint32_t const signal_addr_offset = 0xff0;

enum {
	signal_wait = 0,
	signal_run = 1,
	signal_stop = 2,
};

uint8_t weights[32 * 32];

//This is the main PPU program
void start(void)
{
    //Read all weights and write them into the mailbox
    //readWeights(weights);

    //libnux_mailbox_write_string("Before\n");
    
    /*for (uint8_t i = 0; i < 32; i ++) {
        for (uint8_t j = 0; j < 16; j ++) {
            weights[i * 16 + j] = 1;
        }
    }

    libnux_mailbox_write_string("Modified weights\n");

    writeWeights(weights);*/

    int32_t b = FP(0.001);

    setWeight(0, 2, 0);
    
    readWeights(weights);

    //libnux_mailbox_write_string("Start\n");

    //Endless loop to block further execution
    uint32_t a = 0;
    uint8_t signal = signal_wait;
    while(1)
    {
        a++;
        if (signal == signal_run)
            break;
    }
}
