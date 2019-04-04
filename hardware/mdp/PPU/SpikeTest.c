#include <s2pp.h>
#include <stdint.h>
#include "libnux/mailbox.h"
#include "libnux/dls_v2.h"
#include "spikes.h"

//Address from which the test read command is performed
static uint32_t const signal_addr_offset = 0xff0;

enum {
	signal_wait = 0,
	signal_run = 1,
	signal_stop = 2,
};

//This is the main PPU program
void start(void)
{
    //Emitt a spike
    libnux_mailbox_write_string("Started\n");

    spike_t spike;
    spike.row_mask = 1;
    spike.addr = 20;

    //Emits a single spike into the given neurons
    int i = 0;
    for(i = 0; i < 1; i++)
    {
        spikes_send(&spike);
        wait_cycles(100);
    }

    libnux_mailbox_write_string("Spike sent\n");
}
