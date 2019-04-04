#include <s2pp.h>
#include "libnux/unittest.h"
#include "libnux/mailbox.h"

void test_fxvsel_inline_asm() {
	libnux_testcase_begin(__func__);
	// Create a vector with 0, +1, -1, 0, +1, -1, ... . The -1 is explicitely
	// initialized as its two's complement, which is 0xff
	vector uint8_t cmp = {
		0x00, 0x01, 0x00, 0xff, 0x00, 0x01, 0x00, 0xff,
		0x00, 0x01, 0x00, 0xff, 0x00, 0x01, 0x00, 0xff};
	vector uint8_t lhs = vec_splat_u8(2);
	vector uint8_t rhs = vec_splat_u8(1);
	/* Condition always */
	{
		vector uint8_t res;
		asm volatile (
				// Use vector cmp for component wise comparison
				"fxvcmpb %[cmp]\n"
				// Select third argument if condition "always" matches, e.g.
				// always select rhs
				"fxvsel 1, %[lhs], %[rhs], 0\n"
				// Store at the memory location of vector res
				"fxvstax 1, %[zero], %[res_ptr]\n"
				// Call sync to wait for store operation to finish
				"sync"
				: /* no output */
				: [cmp] "kv" (cmp),
				  [lhs] "kv" (lhs),
				  [rhs] "kv" (rhs),
				  [res_ptr] "r" (&res),
				  [zero] "r" (0)
				: "kv1", "memory");
		for (uint32_t i = 0; i < sizeof(vector uint8_t); i++) {
			libnux_test_equal(res[i], rhs[i]);
		}
	}
	/* Condition gt */
	{
		vector uint8_t res;
		asm volatile (
				"fxvcmpb %[cmp]\n"
				"fxvsel 1, %[lhs], %[rhs], 1\n"
				"fxvstax 1, %[zero], %[res_ptr]\n"
				"sync"
				: /* no output */
				: [cmp] "kv" (cmp),
				  [lhs] "kv" (lhs),
				  [rhs] "kv" (rhs),
				  [res_ptr] "r" (&res),
				  [zero] "r" (0)
				: "kv1", "memory");
		for (uint32_t i = 0; i < sizeof(vector uint8_t); i++) {
			if ((int8_t)(cmp[i]) > 0) {
				libnux_test_equal(res[i], rhs[i]);
			} else {
				libnux_test_equal(res[i], lhs[i]);
			}
		}
	}
	/* Condition lt */
	{
		vector uint8_t res;
		asm volatile (
				"fxvcmpb %[cmp]\n"
				"fxvsel 1, %[lhs], %[rhs], 2\n"
				"fxvstax 1, %[zero], %[res_ptr]\n"
				"sync"
				: /* no output */
				: [cmp] "kv" (cmp),
				  [lhs] "kv" (lhs),
				  [rhs] "kv" (rhs),
				  [res_ptr] "r" (&res),
				  [zero] "r" (0)
				: "kv1", "memory");
		for (uint32_t i = 0; i < sizeof(vector uint8_t); i++) {
			if ((int8_t)(cmp[i]) < 0) {
				libnux_test_equal(res[i], rhs[i]);
			} else {
				libnux_test_equal(res[i], lhs[i]);
			}
		}
	}
	/* Condition eq */
	{
		vector uint8_t res;
		asm volatile (
				"fxvcmpb %[cmp]\n"
				"fxvsel 1, %[lhs], %[rhs], 3\n"
				"fxvstax 1, %[zero], %[res_ptr]\n"
				"sync"
				: /* no output */
				: [cmp] "kv" (cmp),
				  [lhs] "kv" (lhs),
				  [rhs] "kv" (rhs),
				  [res_ptr] "r" (&res),
				  [zero] "r" (0)
				: "kv1", "memory");
		for (uint32_t i = 0; i < sizeof(vector uint8_t); i++) {
			if ((int8_t)(cmp[i]) == 0) {
				libnux_test_equal(res[i], rhs[i]);
			} else {
				libnux_test_equal(res[i], lhs[i]);
			}
		}
	}

	libnux_testcase_end();
}

void start(void) {
	libnux_test_init();
	test_fxvsel_inline_asm();
	libnux_test_summary();
	libnux_test_shutdown();
}
