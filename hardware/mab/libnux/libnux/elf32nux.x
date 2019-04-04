/* ENTRY(start); (or first in .text or at 0) is implicit */
MEMORY {
	ram(rwx) : ORIGIN = 0, LENGTH = 16K
}

/* ECM(2017-12-04): We should provide this numbers from waf configure step FIXME */
ram_base = ORIGIN(ram);
ram_end = ORIGIN(ram) + LENGTH(ram);
mailbox_size = 4096;
mailbox_end = 0x4000;
mailbox_base = mailbox_end - mailbox_size;

/* In the PowerPC EABI calling convention the link register is saved in the
calling function's stack frame at SP+4. Therefore, the base pointer needs to be
shifted by at least 8 bytes from the mailbox to capture the saved link
register. For debugging purposes -- in particular easier examination of the
stack frames -- the stack base is shifted by 16 bytes for an improved
alignment. */
stack_ptr_init = mailbox_base - 16;

SECTIONS {
	/* . = 0x0; location counter starts implicitly at 0 */
	.text : {
		_isr_undefined = .;

		/* put the C runtime first */
		*crt.s.[0-9].o(.text)
		*(.text)
		KEEP(*(.text.crt*));
		KEEP(*(.text.__cxa_pure_virtual*));
		KEEP(*(.text.start*));

		/* map all undefined isr_* to isr_undefined */
		PROVIDE(isr_einput = _isr_undefined);
		PROVIDE(isr_alignment = _isr_undefined);
		PROVIDE(isr_program = _isr_undefined);
		PROVIDE(isr_doorbell = _isr_undefined);
		PROVIDE(isr_fit = _isr_undefined);
		PROVIDE(isr_dec = _isr_undefined);
	} > ram

	.init_array : {
		PROVIDE_HIDDEN (__init_array_start = .);
		KEEP(*(.init_array*))
		KEEP (*(SORT_BY_INIT_PRIORITY(.ctors*)))
		PROVIDE_HIDDEN (__init_array_end = .);
	} > ram

	.fini_array :
	{
		PROVIDE_HIDDEN (__fini_array_start = .);
		KEEP(*(.fini_array*))
		KEEP (*(SORT_BY_INIT_PRIORITY(.dtors*)))
		PROVIDE_HIDDEN (__fini_array_end = .);
	} > ram

	.data : {
		*(.data)
		*(.rodata)
	} > ram

	.bss : {
		*(.bss)
		*(.sbss)
	} > ram

	/* drop some stuff */
	/DISCARD/ : {
		*(.eh_frame) /* exception handling tables */
		*(.comment) /* whatever... */
		*(.debug_*) /* live fast die young */
	}

	/* global symbol marking the end */
	_end = .;
}

ASSERT( . < mailbox_base, "Filled mailbox ram region, aborting.");

heap_base = .;
heap_end = mailbox_base - 8;
