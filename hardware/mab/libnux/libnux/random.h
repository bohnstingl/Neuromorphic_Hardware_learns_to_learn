#pragma once
#include <limits.h>
#include <s2pp.h>
#include <stdint.h>

uint32_t xorshift32(uint32_t* seed);
uint32_t draw_poisson(uint32_t* seed, uint32_t cutoff, uint32_t dt);
void xorshift_vector(const vector uint8_t* seed);
uint32_t random_lcg(uint32_t *seed);
