#include <s2pp.h>
#include "libnux/dls.h"
#include "libnux/unittest.h"
#include "libnux/random.h"


void set_synram_random(uint32_t const base_address, uint32_t const mask, uint32_t seed) {
	/* Initialize data in the synram */
	vector uint8_t data[dls_num_synapse_vectors];
	for (uint32_t index = 0; index < dls_num_synapse_vectors; index++) {
		for (uint32_t component = 0; component < sizeof(vector uint8_t); component++) {
			data[index][component] = xorshift32(&seed) & mask;
		}
	}

	/* Memory barrier to ensure all write operations finished: no instruction,
	 * no output and no input, but memory clobbered. */
	asm volatile ("" : : : "memory");

	/* Set the vector in the synram */
	vector uint8_t* d_it = data;
	for (uint32_t index = 0; index < dls_num_synapse_vectors; index++, d_it++) {
		asm volatile (
				"fxvoutx %[vec], %[base], %[index]"
				: /* no output */
				: [vec] "kv" (*d_it),
				  [base] "b" (base_address),
				  [index] "r" (index)
				: /* no clobbers */);
	}

	/* Wait for all vector instructions and r/w operations to finish */
	asm volatile ("sync");
}


void test_weight_read(uint32_t seed) {
	libnux_testcase_begin("test_weight_read");

	for (uint32_t index = 0; index < dls_num_synapse_vectors; index++) {
		/* Explicitely load the weights, store to memory and synchronize */
		vector uint8_t data;
		register vector uint8_t temp;
		asm volatile (
				"fxvinx %[temp], %[base], %[index]\n"
				"fxvstax %[temp], 0, %[addr]\n"
				"sync"
				: [temp] "=&kv" (temp)
				: [base] "b" (dls_weight_base),
				  [index] "r" (index),
				  [addr] "r" (&data)
				: /* no clobbers */);
		for (uint32_t j = 0; j < sizeof(vector uint8_t); j++) {
			libnux_test_equal(data[j], xorshift32(&seed) & dls_weight_mask);
		}
	}

	libnux_testcase_end();
}


void test_decoder_read(uint32_t seed) {
	libnux_testcase_begin("test_decoder_read");

	for (uint32_t index = 0; index < dls_num_synapse_vectors; index++) {
		/* Explicitely load the weights, store to memory and synchronize */
		vector uint8_t data;
		register vector uint8_t temp;
		asm volatile (
				"fxvinx %[temp], %[base], %[index]\n"
				"fxvstax %[temp], 0, %[addr]\n"
				"sync"
				: [temp] "=&kv" (temp)
				: [base] "b" (dls_decoder_base),
				  [index] "r" (index),
				  [addr] "r" (&data)
				: /* no clobbers */);
		for (uint32_t j = 0; j < sizeof(vector uint8_t); j++) {
			libnux_test_equal(data[j], xorshift32(&seed) & dls_decoder_mask);
		}
	}

	libnux_testcase_end();
}


void start(void) {
	libnux_test_init();
	set_synram_random(dls_weight_base, dls_weight_mask, 42);
	set_synram_random(dls_decoder_base, dls_decoder_mask, 1);
	test_weight_read(42);
	test_decoder_read(1);
	libnux_test_summary();
	libnux_test_shutdown();
}
