#include <s2pp.h>

#include "libnux/mailbox.h"
#include "libnux/unittest.h"

#define NUM_VECTOR_REGISTERS 32
#define NUM_VECTORS_TO_USE 10 * NUM_VECTOR_REGISTERS

// Have this function non-optimized, such that it would not be inlined in
// test_many_vectors. This generates an easier to read assembly.
void __attribute__((optimize("O0"))) test_equal(uint32_t a, uint32_t b)
{
	libnux_test_equal(a, b);
}

/*
 * Allocate more vectors than vector registers exist.
 * The compiler is therefore forced to do implicit load/stores between
 * registers and memory at some point.
 * From the generated assembly it looks like currently (2018-05-17) only one
 * vector register is used. Spilling in the sense of relocating vectors into
 * the main memory due to missing hardware resources is therefore not tested.
 */
void test_many_vectors()
{
	libnux_testcase_begin(__func__);

	vector uint8_t vectors[NUM_VECTORS_TO_USE];

	for (uint32_t i = 0; i < NUM_VECTORS_TO_USE; i++) {
		vectors[i] = vec_splat_u8(i);
	}

	for (uint32_t i = 0; i < NUM_VECTORS_TO_USE; i++) {
		for (uint8_t j = 0; j < 16; j++) {
			// Make sure all vectors are correct, crop to 8bit
			test_equal(vectors[i][j], i % 256);
		}
	}

	libnux_testcase_end();
}

int start()
{
	libnux_test_init();
	test_many_vectors();
	libnux_test_summary();
	libnux_test_shutdown();

	return 0;
}
