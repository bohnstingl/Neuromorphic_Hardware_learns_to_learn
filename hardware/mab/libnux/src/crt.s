/*
 * This program runs all the programs generated with the C-compiler
 * */

.extern start 
.extern reset
.extern _isr_undefined
.extern isr_einput
.extern isr_alignemnt
.extern isr_program
.extern isr_doorbell
.extern isr_fit
.extern isr_dec

.extern stack_ptr_init

# External callable functions
.globl exit
.type exit, @function

# Code section
.section .text.crt

reset:
	b __init

	# interrupt jump table
	int_mcheck:    b _isr_undefined
	int_cinput:    b _isr_undefined
	int_dstorage:  b _isr_undefined
	int_istorage:  b _isr_undefined
	int_einput:    b isr_einput
	int_alignment: b isr_alignment 
	int_program:   b isr_program
	int_syscall:   b _isr_undefined
	int_doorbell:  b isr_doorbell
	int_cdoorbell: b _isr_undefined
	int_fit:       b isr_fit
	int_dec:       b isr_dec

__init:
	# set the stack pointer
	lis 1, stack_ptr_init@h
	addi 1, 1, stack_ptr_init@l
	# construct global things
	bl __call_constructors
	# start actual program
	bl start
	# destruct global things
	bl __call_destructors

exit:
	# load stack base into r11 and save the current stack pointer to the
	# stack base
	lis 11, stack_ptr_init@h
	addi 11, 11, stack_ptr_init@l
	stw 1, 0(11)
	# push return value from r3 to the SP + 12 to have the lowest stack
	# frame look like:
	# | SP + 0      | SP + 4      | SP + 8      | SP + 12     |
	# +-------------+-------------+-------------+-------------+
	# | aa aa aa aa | bb bb bb bb | 00 00 00 00 | cc cc cc cc |
	# where "a" is the saved stack pointer "b" is the saved link register
	# and "c" the return code
	stw 3, 12(11)

stop:
end_loop:
	wait
	b end_loop
