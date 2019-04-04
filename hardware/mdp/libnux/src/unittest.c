#include "libnux/system.h"
#include "libnux/unittest.h"

static uint32_t num_passed = 0;
static uint32_t num_failed = 0;

static uint32_t num_testcases_passed= 0;
static uint32_t num_testcases_failed = 0;

static uint32_t testcase_passed_offset = 0;
static uint32_t testcase_failed_offset = 0;

static libnux_test_action_type test_action = libnux_test_action_warning;


void libnux_test_init(void) {
}

void libnux_test_shutdown(void) {
	stop();
}

uint32_t libnux_test_get_passed(void) {
	return num_passed;
}

uint32_t libnux_test_get_failed(void) {
	return num_failed;
}

uint32_t libnux_testcase_get_passed(void) {
	return num_testcases_passed;
}

uint32_t libnux_testcase_get_failed(void) {
	return num_testcases_failed;
}

void libnux_test_set_action(libnux_test_action_type const action) {
	test_action = action;
}

libnux_test_action_type libnux_test_get_action(void) {
	return test_action;
}

void libnux_test_inc_passed(void) {
	num_passed++;
}

void libnux_test_inc_failed(void) {
	num_failed++;
}

uint32_t get_num_passed_in_testcase(void) {
	return num_passed - testcase_passed_offset;
}

uint32_t get_num_failed_in_testcase(void) {
	return num_failed - testcase_failed_offset;
}

void libnux_testcase_begin(char const * name) {
	libnux_test_write_string("[ Run      ] ");
	libnux_test_write_string(name);
	libnux_test_write_string("\n");
	testcase_passed_offset = num_passed;
	testcase_failed_offset = num_failed;
}

void libnux_testcase_end(void) {
	if(get_num_failed_in_testcase() == 0) {
		libnux_test_write_string("[       Ok ] Passed ");
		libnux_test_write_int(get_num_passed_in_testcase());
		libnux_test_write_string(" tests\n");
		num_testcases_passed++;
	} else {
		libnux_test_write_string("[  Failed  ]\n");
		num_testcases_failed++;
	}
}

void libnux_test_summary(void) {
	libnux_test_write_string("\n[==========]\n");
	libnux_test_write_string("[  Passed  ] ");
	libnux_test_write_int(num_testcases_passed);
	libnux_test_write_string(" test cases\n");
	libnux_test_write_string("[  Failed  ] ");
	libnux_test_write_int(num_testcases_failed);
	libnux_test_write_string(" test cases\n");

	libnux_test_write_string("\n[==========]\n");
	libnux_test_write_string("[  Passed  ] ");
	libnux_test_write_int(num_passed);
	libnux_test_write_string(" tests\n");
	libnux_test_write_string("[  Failed  ] ");
	libnux_test_write_int(num_failed);
	libnux_test_write_string(" tests\n");
}
