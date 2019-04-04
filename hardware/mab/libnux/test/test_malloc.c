#include "libnux/unittest.h"
#include "libnux/mailbox.h"
#include "libnux/malloc.h"

void start(void) {
	libnux_test_init();

	libnux_testcase_begin("test malloc");
	int* i = malloc(sizeof(int));
	int* j = malloc(sizeof(int));
	libnux_test_not_null(i);
	libnux_test_not_null(j);
	libnux_test_true((((intptr_t)i + (intptr_t)sizeof(int)) == (intptr_t)j));
	libnux_testcase_end();

	libnux_test_summary();
	libnux_test_shutdown();
}
