#include <stdint.h>

uint32_t __attribute__((optimize("O0"))) grow_stack(uint32_t const depth, uint32_t const ret) {
	if (depth) {
		return grow_stack(depth - 1, ret);
	}
	return ret;
}

uint32_t start() {
	return grow_stack(8, 0xdeadbeef);
}
