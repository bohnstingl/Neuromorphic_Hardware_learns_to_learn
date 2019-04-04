#include "spikes.h"

void send_uniform_spiketrain(spike_t* single_spike, uint32_t number, uint32_t isi_usec)
{
	uint32_t i;
	for (i = number; i > 0; i--) {
		send_spike(single_spike);
		sleep_cycles(isi_usec * USEC);
	}
}
