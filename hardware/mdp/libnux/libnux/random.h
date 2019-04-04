#pragma once

static int random_lcg(int *seed) {
	// constants from Numerical Recipes via Wikipedia
	int rv = 1664525 * (*seed) + 1013904223;
	*seed = rv;
	return rv;
}

