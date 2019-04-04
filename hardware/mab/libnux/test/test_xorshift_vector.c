#include <libnux/unittest.h>
#include "libnux/random.h"

void test_xorshift_vector()
{
	libnux_testcase_begin("xorshift_vector");

	// only set one bit to 1 in first and last word to ease calculation
	// of expected result
	vector uint8_t seed = {0, 0, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 1};
	xorshift_vector(&seed);

	vector uint8_t expected = {4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 1, 0, 0, 8, 8};
	for (uint32_t index = 0; index < sizeof(vector uint8_t); index++) {
		libnux_test_equal(seed[index], expected[index]);
	}

	libnux_testcase_end();
}

void start(void)
{
	libnux_test_init();
	test_xorshift_vector();
	libnux_test_summary();
	libnux_test_shutdown();
}
