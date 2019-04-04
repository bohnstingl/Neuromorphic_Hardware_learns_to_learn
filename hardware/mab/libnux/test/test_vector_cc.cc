extern "C" {
#include <s2pp.h>
#include "libnux/unittest.h"
#include "libnux/mailbox.h"
}
#include "libnux/new.hpp"

template<typename T>
struct CRTP {
	CRTP() {
		libnux_testcase_begin(T::test_name);
	}

	~CRTP() {
		libnux_testcase_end();
	}
};

struct VectorTestCRTP : public CRTP<VectorTestCRTP> {
	VectorTestCRTP() {
		vector uint8_t lhs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		vector uint8_t rhs = vec_splat_u8(1);
		vector uint8_t res = vec_add(lhs, rhs);
		for (uint32_t index = 0; index < 16; index++) {
			libnux_test_equal(res[index], index + 1);
		}
	}
	constexpr static const char* test_name = "vector_add_CRTP";
};

struct INHE {
	INHE(const char* test_name) {
		libnux_testcase_begin(test_name);
	}

	~INHE() {
		libnux_testcase_end();
	}
};

struct VectorTestINHE : public INHE {
	VectorTestINHE() :
		INHE("vector_add_INHE")
	{
		vector uint8_t lhs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		vector uint8_t rhs = vec_splat_u8(1);
		vector uint8_t res = vec_add(lhs, rhs);
		for (uint32_t index = 0; index < 16; index++) {
			libnux_test_equal(res[index], index + 1);
		}
	}
};

struct VIRT {
	VIRT() = default;

	void operator()() {
		libnux_testcase_begin(get_test_name());
		test();
	}

	virtual ~VIRT() {
		libnux_testcase_end();
	}

	virtual void test() = 0;
	virtual const char* get_test_name() = 0;
};

struct VectorTestVIRT : public VIRT {
	void test() override {
		vector uint8_t lhs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		vector uint8_t rhs = vec_splat_u8(1);
		vector uint8_t res = vec_add(lhs, rhs);
		for (uint32_t index = 0; index < 16; index++) {
			libnux_test_equal(res[index], index + 1);
		}
	}

	const char* get_test_name() override {
		return "vector_add_VIRT";
	}
};

struct VectorTestGLOB {
	void test() {
		libnux_testcase_begin(test_name);
		vector uint8_t lhs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		vector uint8_t rhs = vec_splat_u8(1);
		vector uint8_t res = vec_add(lhs, rhs);
		for (uint32_t index = 0; index < 16; index++) {
			libnux_test_equal(res[index], index + 1);
		}
	}

	VectorTestGLOB() = default;

	~VectorTestGLOB() {
		libnux_testcase_end();
	}

	constexpr static const char* test_name = "vector_add_GLOB";
};

VectorTestGLOB test4;

struct VectorTestNEW {
	VectorTestNEW(char const* /*test_name*/) {
		libnux_mailbox_write_string(__PRETTY_FUNCTION__);
		libnux_mailbox_write_string("\n");
	}

	VectorTestNEW() : VectorTestNEW("VectorTestNEW_default")
	{
		libnux_mailbox_write_string(__PRETTY_FUNCTION__);
		libnux_mailbox_write_string("\n");
	}

	~VectorTestNEW() {
		libnux_mailbox_write_string(__PRETTY_FUNCTION__);
		libnux_mailbox_write_string("\n");
	}
};

struct TestInit {
	TestInit() {
		libnux_test_init();
	}

	~TestInit() {
		libnux_test_summary();
		libnux_test_shutdown();
	}
};

void print_memory_layout() {
	void* p = 0;
	libnux_mailbox_write_string("MEMORY LAYOUT:");
	libnux_mailbox_write_string("\n\theap_base = ");
	libnux_mailbox_write_int(reinterpret_cast<intptr_t>(&heap_base));
	libnux_mailbox_write_string("\n\theap_end = ");
	libnux_mailbox_write_int(reinterpret_cast<intptr_t>(&heap_end));
	libnux_mailbox_write_string("\n\tcurrent stack ptr = ");
	libnux_mailbox_write_int(reinterpret_cast<intptr_t>(&p));
	libnux_mailbox_write_string("\n\tmailbox_base = ");
	libnux_mailbox_write_int(reinterpret_cast<intptr_t>(&mailbox_base));
	libnux_mailbox_write_string("\n\tmailbox_end = ");
	libnux_mailbox_write_int(reinterpret_cast<intptr_t>(&mailbox_end));
	libnux_mailbox_write_string("\n");
}

extern "C" {
void start(void) {
	VectorTestNEW* test5 = new VectorTestNEW("VectorTestNEW");
	delete(test5);

	VectorTestNEW* test6 = new VectorTestNEW[2];
	delete[](test6);

	char* buf = new char[sizeof(VectorTestNEW)];
	VectorTestNEW* test7 = new (buf) VectorTestNEW("VectorTestNEWagain");
	delete(test7);

	print_memory_layout();

	TestInit t;
	{
		VectorTestCRTP test0;
	}
	{
		VectorTestINHE test2;
	}
	{
		VectorTestVIRT test3;
		test3();
	}


	test4.test(); // non-working output (mailbox ordering...)
}
}
