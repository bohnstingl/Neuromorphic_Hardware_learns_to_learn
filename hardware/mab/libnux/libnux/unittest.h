#pragma once

#include <stdint.h>

uint32_t libnux_test_write_string(char const * str);
uint32_t libnux_test_write_int(uint32_t const n);

void libnux_test_init(void);
void libnux_test_shutdown(void);

typedef enum {
	libnux_test_action_warning,
	libnux_test_action_shutdown
} libnux_test_action_type;

uint32_t libnux_test_get_passed();
uint32_t libnux_test_get_failed();
uint32_t libnux_testcase_get_passed();
uint32_t libnux_testcase_get_failed();

void libnux_test_set_action(libnux_test_action_type const);
libnux_test_action_type libnux_test_get_action();

void libnux_test_inc_passed(void);
void libnux_test_inc_failed(void);

/* Print macros */

#define LIBNUX_TEST_STRINGIFY(x) #x
#define LIBNUX_TEST_TO_STRING(x) LIBNUX_TEST_STRINGIFY(x)

#ifdef LIBNUX_TEST_MODE_VERBOSE
#define libnux_test_write_passed(msg, args) \
	do { \
		libnux_test_write_string(__FILE__); \
		libnux_test_write_string(":"); \
		libnux_test_write_string(LIBNUX_TEST_TO_STRING(__LINE__)); \
		libnux_test_write_string(": passed: "); \
		libnux_test_write_string(msg); \
		libnux_test_write_string("("); \
		libnux_test_write_string(args); \
		libnux_test_write_string(")\n"); \
	} while (0)
#else
#define libnux_test_write_passed(msg, args)
#endif /* write_passed */

#define libnux_test_write_failed(msg, args) \
	do { \
		libnux_test_write_string(__FILE__); \
		libnux_test_write_string(":"); \
		libnux_test_write_string(LIBNUX_TEST_TO_STRING(__LINE__)); \
		libnux_test_write_string(": failed: "); \
		libnux_test_write_string(msg); \
		libnux_test_write_string("("); \
		libnux_test_write_string(args); \
		libnux_test_write_string(")\n"); \
	} while (0)

#define libnux_test_failed(msg, args) \
	do { \
		libnux_test_write_failed(msg, args); \
		libnux_test_inc_failed(); \
		if (libnux_test_get_action() == libnux_test_action_shutdown) { \
			libnux_test_shutdown(); \
		} \
	} while (0)

#define libnux_test_passed(msg, args) \
	do { \
		libnux_test_write_passed(msg, args); \
		libnux_test_inc_passed(); \
	} while (0)

/* Checks */

#define libnux_test(cond, msg, args) \
	do { \
		if ( (cond) ) { \
			libnux_test_passed(msg, args); \
		} else { \
			libnux_test_failed(msg, args); \
		} \
	} while (0)

#define libnux_test_equal(fst, snd) \
	libnux_test( (fst) == (snd), "equal", #fst ", " #snd)

#define libnux_test_true(cond) \
	libnux_test( (cond), "true", #cond)

#define libnux_test_null(ptr) \
	libnux_test( (ptr) == (void*)0, "null", #ptr)

#define libnux_test_not_null(ptr) \
	libnux_test( (ptr) != (void*)0, "not_null", #ptr)

/* Testcases */

void libnux_testcase_begin(char const * name);
void libnux_testcase_end(void);

/* Test summary */

void libnux_test_summary(void);
