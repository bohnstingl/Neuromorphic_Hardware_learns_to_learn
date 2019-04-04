#pragma once

#include <stdint.h>
#include "dls.h"
#include "time.h"

typedef struct
{
	uint32_t row_mask;
	uint8_t addr;
} spike_t;

// FIXME: expose DLSv3 bug where MSB of addresses is fixed to 0/1 for even/odd rows
static inline void send_spike(spike_t* sp)
{
	volatile uint32_t* ptr = (uint32_t*) (dls_spike_base + (sp->addr));
	*ptr = sp->row_mask;
}

void send_uniform_spiketrain(spike_t* single_spike, uint32_t number, uint32_t isi_usec);
