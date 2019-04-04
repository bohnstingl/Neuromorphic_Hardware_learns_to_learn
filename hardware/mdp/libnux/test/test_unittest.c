#include <libnux/unittest.h>

void basic_test(void) {
	libnux_testcase_begin("basic_checks");

	char* s = (void*)(0);
	int a = 0;
	int b = 0;
	int c = 0;

	libnux_test_equal(0, a);
	libnux_test_equal(0, b);
	libnux_test_equal(0, c);

	libnux_test_equal(0, 1-1);
	libnux_test_equal(0, -1+1);
	libnux_test_equal(-1, -2+1);

	libnux_test_null(s);
	libnux_test_true(1);

	libnux_testcase_end();
}

void failing_test(void) {
	libnux_testcase_begin("failing_test");

	libnux_test_equal(0, 1+1);
	libnux_test_equal(0, 1+2);

	libnux_testcase_end();
}

void start(void) {
	libnux_test_init();
	basic_test();
	failing_test();
	libnux_test_summary();
	libnux_test_shutdown();
}
