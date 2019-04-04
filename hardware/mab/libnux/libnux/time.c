#include "time.h"
#include "spr.h"

/*
	Idle for (approx.) `cycles` cycles.
*/
void __attribute__((optimize("O2"))) sleep_cycles(uint32_t cycles)
{
	static const uint8_t offset = 9;
	time_base_t start;
	start = get_time_base();
	while ((uint32_t)(get_time_base() - start) <= (cycles - offset));
}
