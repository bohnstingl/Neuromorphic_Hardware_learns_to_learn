#pragma once

#include <stdint.h>

#include "libnux/constants.h"


//--------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------

typedef union {
	int32_t words[NUM_WORDS_PER_ARRAY];
	int16_t halfwords[NUM_HALFWORDS_PER_ARRAY];
	int8_t bytes[NUM_BYTES_PER_ARRAY];
} fxv_array_t;

typedef enum {
	COND_ALWAYS = 0x0,
	COND_GT     = 0x1,
	COND_LT     = 0x2,
	COND_EQ     = 0x3
} fxv_cond_t;

typedef enum {
	REORDER_TYPE_HALFWORD = 1,
	REORDER_TYPE_BYTE     = 0
} fxv_reorder_t;


//--------------------------------------------------------------------------------
// Macros
//--------------------------------------------------------------------------------

#define _fxv_store_array(v, reg) \
	asm volatile ( \
		"fxvstax " #reg ", 0, %[o]\n" \
		:: [o] "r" (v) \
	)

#define fxv_store_array(a, b) _fxv_store_array(a, b)

#define _fxv_load_array(reg, v) \
	asm volatile ( \
		"fxvlax " #reg ", 0, %[o]\n" \
		:: [o] "r" (v) \
	)

#define fxv_load_array(a, b) _fxv_load_array(a, b)

#define _fxv_pack_halfwords(upper, lower, ra, rb, size) \
	asm volatile ( \
		"fxvpckbu " #upper ", " #ra ", " #rb ", " #size "\n" \
		"fxvpckbl " #lower ", " #ra ", " #rb ", " #size "\n" \
	);

#define fxv_pack_halfwords(a, b, c, d) _fxv_pack_halfwords(a, b, c, d, 0)
#define fxv_pack_halfwords_padded(a, b, c, d, e) _fxv_pack_halfwords(a, b, c, d, e)

#define _fxv_unpack_to_halfwords(ra, rb, upper, lower, size) \
	asm volatile ( \
		"fxvupckbl " #ra ", " #upper ", " #lower ", " #size "\n" \
		"fxvupckbr " #rb ", " #upper ", " #lower ", " #size "\n" \
	)

#define fxv_unpack_to_halfwords(a, b, c, d) _fxv_unpack_to_halfwords(a, b, c, d, 0)
#define fxv_unpack_to_halfwords_padded(a, b, c, d, e) _fxv_unpack_to_halfwords(a, b, c, d, e)


//--------------------------------------------------------------------------------
// Arithmetic vector instructions
//--------------------------------------------------------------------------------

#define _fxv_insn3(op, rt, ra, rb) asm volatile(op " " #rt ", " #ra ", " #rb)
#define _fxv_insn3_cond(op, rt, ra, rb, cond) asm volatile(op " " #rt ", " #ra ", " #rb ", " #cond)
#define _fxv_insn2(op, rt, ra) asm volatile(op " " #rt ", " #ra)
#define _fxv_insn_gpr1(op, rt, gpra) asm volatile(op " " #rt ", %[a]" :: [a] "r"(gpra))
#define _fxv_insn1(op, rt) asm volatile(op " " #rt)

// modulo halfword arithmetic
#define fxv_addhm(x, y, z) _fxv_insn3("fxvaddhm", x, y, z)
#define fxv_addhm_gt(x, y, z) _fxv_insn3_cond("fxvaddhm", x, y, z, 1)
#define fxv_addhm_lt(x, y, z) _fxv_insn3_cond("fxvaddhm", x, y, z, 2)
#define fxv_addhm_eq(x, y, z) _fxv_insn3_cond("fxvaddhm", x, y, z, 3)
#define fxv_subhm(x, y, z) _fxv_insn3("fxvsubhm", x, y, z)
#define fxv_cmph(x) _fxv_insn1("fxvcmph", x)
#define fxv_mulhm(x, y, z) _fxv_insn3("fxvmulhm", x, y, z)
#define fxv_mtach(x) _fxv_insn1("fxvmtach", x)
#define fxv_mahm(x, y, z) _fxv_insn3("fxvmahm", x, y, z)
#define fxv_mahm_gt(x, y, z) _fxv_insn3_cond("fxvmahm", x, y, z, 1)
#define fxv_mahm_lt(x, y, z) _fxv_insn3_cond("fxvmahm", x, y, z, 2)

// fractional halfword arithmetic
#define fxv_addhfs(x, y, z) _fxv_insn3("fxvaddhfs", x, y, z)
#define fxv_addhfs_lt(x, y, z) _fxv_insn3_cond("fxvaddhfs", x, y, z, 2)
#define fxv_subhfs(x, y, z) _fxv_insn3("fxvsubhfs", x, y, z)
#define fxv_mulhfs(x, y, z) _fxv_insn3("fxvmulhfs", x, y, z)
#define fxv_mahfs(x, y, z) _fxv_insn3("fxvmahfs", x, y, z)
#define fxv_mahfs_gt(x, y, z) _fxv_insn3_cond("fxvmahfs", x, y, z, 1)
#define fxv_mahfs_lt(x, y, z) _fxv_insn3_cond("fxvmahfs", x, y, z, 2)
#define fxv_matachfs(x, y) _fxv_insn2("fxvmatachfs", x, y)
#define fxv_mtachf(x) _fxv_insn1("fxvmtachf", x)

// fractional halfword arithmetic
#define fxv_addbm(x, y, z) _fxv_insn3("fxvaddbm", x, y, z)
#define fxv_addbm_gt(x, y, z) _fxv_insn3_cond("fxvaddbm", x, y, z, 1)
#define fxv_addbm_lt(x, y, z) _fxv_insn3_cond("fxvaddbm", x, y, z, 2)
#define fxv_addbm_eq(x, y, z) _fxv_insn3_cond("fxvaddbm", x, y, z, 3)
#define fxv_subbm(x, y, z) _fxv_insn3("fxvsubbm", x, y, z)
#define fxv_cmpb(x) _fxv_insn1("fxvcmpb", x)
#define fxv_mulbm(x, y, z) _fxv_insn3("fxvmulbm", x, y, z)
#define fxv_mtacb(x) _fxv_insn1("fxvmtacb", x)
#define fxv_mabm(x, y, z) _fxv_insn3("fxvmabm", x, y, z)
#define fxv_mabm_gt(x, y, z) _fxv_insn3_cond("fxvmabm", x, y, z, 1)
#define fxv_mabm_lt(x, y, z) _fxv_insn3_cond("fxvmabm", x, y, z, 2)

// fractional byte arithmetic
#define fxv_addbfs(x, y, z) _fxv_insn3("fxvaddbfs", x, y, z)
#define fxv_addbfs_lt(x, y, z) _fxv_insn3_cond("fxvaddbfs", x, y, z, 2)
#define fxv_addtacb(x, y) _fxv_insn2("fxvaddtacb", x, y)
#define fxv_subbfs(x, y, z) _fxv_insn3("fxvsubbfs", x, y, z)
#define fxv_mulbfs(x, y, z) _fxv_insn3("fxvmulbfs", x, y, z)
#define fxv_mabfs(x, y, z) _fxv_insn3("fxvmabfs", x, y, z)
#define fxv_mabfs_gt(x, y, z) _fxv_insn3_cond("fxvmabfs", x, y, z, 1)
#define fxv_mabfs_lt(x, y, z) _fxv_insn3_cond("fxvmabfs", x, y, z, 2)
#define fxv_mtacbf(x) _fxv_insn1("fxvmtacbf", x)

// special operations
#define fxv_splath(x, y) _fxv_insn_gpr1("fxvsplath", x, y)

/** Bugfix for fxvsplatb in HICANN-DLS v1 !
 * This instruction has incorrect hazard detection for the general-purpose register
 * argument. E.g.:
 *   lis r9, 15
 *   fxvsplatb 1, r9
 * will use the old value of register r9.
 *
 * The fix uses the fxvsplath instruction, which does hazard detection correctly. */
//#define fxv_splatb(x, y) _fxv_insn_gpr1("fxvsplatb", x, y)
#define _fxv_splatb(rt, gpra) \
  asm volatile("fxvsplath " #rt ", %[a]" \
      :: [a] "r"( (( (gpra) & 0xff) << 8) | ( (gpra) & 0xff) ))
#define fxv_splatb(x, y) _fxv_splatb(x, y)
/* End of bugfix for fxvsplatb */

// shift operations
#define fxv_shh(x, y, z) _fxv_insn3("fxvshh", x, y, z)
#define fxv_shb(x, y, z) _fxv_insn3("fxvshb", x, y, z)

// select operation
#define fxv_sel_gt(x, y, z) _fxv_insn3_cond("fxvsel", x, y, z, 1)
#define fxv_sel_lt(x, y, z) _fxv_insn3_cond("fxvsel", x, y, z, 2)
#define fxv_sel_eq(x, y, z) _fxv_insn3_cond("fxvsel", x, y, z, 3)
#define fxv_mov(x, y) _fxv_insn3("fxvsel", x, 0, y)


//--------------------------------------------------------------------------------
// Other operations
//--------------------------------------------------------------------------------

#define fxv_zero_set(x) fxv_splath(x, 0)

extern void fxv_zero_vrf();

