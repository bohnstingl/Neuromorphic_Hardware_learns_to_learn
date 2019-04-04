#include "random.h"

uint32_t xorshift32(uint32_t* seed)
{
	*seed ^= *seed << 13;
	*seed ^= *seed >> 17;
	*seed ^= *seed << 5;
	return *seed;
}

/*
	Calculate cutoff:
	cutoff = target_freq * dt * 0xffffffff
 */
uint32_t draw_poisson(uint32_t* seed, uint32_t cutoff, uint32_t dt)
{
	uint32_t t = 0;
	xorshift32(seed);
	while (*seed > cutoff) {
		xorshift32(seed);
		t += dt;
	}
	return t;
}

// ~3 times faster than calling xorshift32 four times.
void xorshift_vector(const vector uint8_t* seed)
{
	// Check if vector is 16B long. Otherwise xorshift_vector won't work
	// because it assumes exactly 4 uint32_t to fit in a vector.
	_Static_assert(
		sizeof(vector uint8_t) == 16, "vector size is not 16B, xorshift_vector will not work.");

	// xorshift128
	uint32_t* x = (uint32_t*) seed;
	uint32_t t = *x ^ (*x << 11);
	*x = *(x + 1);
	*(x + 1) = *(x + 2);
	*(x + 2) = *(x + 3);
	*(x + 3) ^= (*(x + 3) >> 19) ^ t ^ (t >> 8);
}

uint32_t random_lcg(uint32_t *seed) {
  // constants from Numerical Recipes via Wikipedia
  uint32_t rv = 1664525 * (*seed) + 1013904223;
  *seed = rv;
  return rv;
}
