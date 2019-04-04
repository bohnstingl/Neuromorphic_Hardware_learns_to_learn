#include <stdint.h>

#include "libnux/fxv.h"
#include "libnux/constants.h"
#include "libnux/sync.h"

void fxv_zero_vrf() {
	int i;
	volatile uint8_t zeros[NUM_BYTES_PER_ARRAY];
	for(i=0; i<NUM_BYTES_PER_ARRAY; i++) {
		zeros[i] = 0;
	}
	sync();

	asm volatile(
		"fxvlax 0, 0, %[z]\n"
		"fxvlax 1, 0, %[z]\n"
		"fxvlax 2, 0, %[z]\n"
		"fxvlax 3, 0, %[z]\n"
		"fxvlax 4, 0, %[z]\n"
		"fxvlax 5, 0, %[z]\n"
		"fxvlax 6, 0, %[z]\n"
		"fxvlax 7, 0, %[z]\n"
		"fxvlax 8, 0, %[z]\n"
		"fxvlax 9, 0, %[z]\n"
		"fxvlax 10, 0, %[z]\n"
		"fxvlax 11, 0, %[z]\n"
		"fxvlax 12, 0, %[z]\n"
		"fxvlax 13, 0, %[z]\n"
		"fxvlax 14, 0, %[z]\n"
		"fxvlax 15, 0, %[z]\n"
		"fxvlax 16, 0, %[z]\n"
		"fxvlax 17, 0, %[z]\n"
		"fxvlax 18, 0, %[z]\n"
		"fxvlax 19, 0, %[z]\n"
		"fxvlax 20, 0, %[z]\n"
		"fxvlax 21, 0, %[z]\n"
		"fxvlax 22, 0, %[z]\n"
		"fxvlax 23, 0, %[z]\n"
		"fxvlax 24, 0, %[z]\n"
		"fxvlax 25, 0, %[z]\n"
		"fxvlax 26, 0, %[z]\n"
		"fxvlax 27, 0, %[z]\n"
		"fxvlax 28, 0, %[z]\n"
		"fxvlax 29, 0, %[z]\n"
		"fxvlax 30, 0, %[z]\n"
		"fxvlax 31, 0, %[z]\n"
		:	/* no output */
		:	[z] "r" (&zeros)
	);
	sync();
}
