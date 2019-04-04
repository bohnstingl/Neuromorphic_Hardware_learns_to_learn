#include "counter.h"
#include "dls.h"


#if (LIBNUX_DLS_VERSION == 2)
#include "counter_v2.c"
#endif

#if (LIBNUX_DLS_VERSION == 3)
#include "counter_v3.c"
#endif

void reset_all_neuron_counters()
{
	uint8_t neuron;
	for (neuron = 0; neuron < dls_num_columns; neuron++) {
		reset_neuron_counter(neuron);
	}
}
