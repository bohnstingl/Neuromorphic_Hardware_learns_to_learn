#pragma once

#include <s2pp.h>
#include <stdint.h>
#include "dls.h"

// Get weights of synapse row `row` and save in vectors `first_half`, `second_half`.
inline void get_weights(vector uint8_t* first_half, vector uint8_t* second_half, uint8_t row)
{
	asm volatile(
		// clang-format off
		"fxvinx %[first], %[base], %[first_index]\n"
		"fxvinx %[second], %[base], %[second_index]"
		: [first] "=kv" (*first_half),
		  [second] "=kv" (*second_half)
		: [base] "b" (dls_weight_base),
		  [first_index] "r" (row*2),
		  [second_index] "r" (row*2+1)
		: /* no clobber */
		// clang-format on
	);
}

inline void set_weights(vector uint8_t* first_half, vector uint8_t* second_half, uint8_t row)
{
	asm volatile(
		// clang-format off
		"fxvoutx %[first], %[base], %[first_index]\n"
		"fxvoutx %[second], %[base], %[second_index]"
		: /* no outputs */
		: [first] "kv" (*first_half),
		  [second] "kv" (*second_half),
		  [base] "b" (dls_weight_base),
		  [first_index] "r" (row*2),
		  [second_index] "r" (row*2+1)
		: /* no clobber */
		// clang-format on
	);
}
