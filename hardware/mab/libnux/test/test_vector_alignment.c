#include <s2pp.h>

#include "libnux/mailbox.h"
#include "libnux/unittest.h"


vector uint8_t global_first = {
	0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
	0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
uint8_t global_small = 0xff;
vector uint8_t global_second = {
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
	0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};


void dump_vector(vector uint8_t* ptr) {
	libnux_mailbox_write_string("vector at ");
	libnux_mailbox_write_int((uint32_t)ptr);
	libnux_mailbox_write_string(": (");
	for (uint32_t i = 0; i < 16; i++) {
		if (i > 0) {
			libnux_mailbox_write_string(", ");
		}
		libnux_mailbox_write_int((*ptr)[i]);
	}
	libnux_mailbox_write_string(")\n");
}


void test_vector_alignment() {
	libnux_testcase_begin(__func__);

	vector uint8_t local_first = {
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
		0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};
	uint8_t local_small = 0xff;
	vector uint8_t local_second = {
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
		0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f};

	libnux_test_equal(global_first[0], 0);
	libnux_test_equal(global_small, 0xff);
	libnux_test_equal(global_second[0], 0x10);
	libnux_test_equal(local_first[0], 0x20);
	libnux_test_equal(local_small, 0xff);
	libnux_test_equal(local_second[0], 0x30);

	libnux_test_equal(((uint32_t)&global_first) % 16, 0);
	libnux_test_equal(((uint32_t)&global_second) % 16, 0);
	libnux_test_equal(((uint32_t)&local_first) % 16, 0);
	libnux_test_equal(((uint32_t)&local_second) % 16, 0);

	dump_vector(&global_first);
	dump_vector(&global_second);
	dump_vector(&local_first);
	dump_vector(&local_second);

	libnux_testcase_end();
}


int start(void) {
	libnux_test_init();
	test_vector_alignment();
	libnux_test_summary();
	libnux_test_shutdown();

	return 0;
}
