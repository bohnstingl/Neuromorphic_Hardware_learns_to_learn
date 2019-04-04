MEMORY {
	ram(rwx) : ORIGIN = 0, LENGTH = 16K
}


/*mailbox_size = 256;*/
mailbox_size = 4096;
mailbox_end = 0x4000;
mailbox_base = mailbox_end - mailbox_size;
stack_ptr_init = mailbox_base - 8;

SECTIONS {
	.text : {
        _isr_undefined = .;

        *crt.s.[0-9].o(.text)
        *(.text)

        PROVIDE(isr_einput = _isr_undefined);
        PROVIDE(isr_alignment = _isr_undefined);
        PROVIDE(isr_program = _isr_undefined);
        PROVIDE(isr_doorbell = _isr_undefined);
        PROVIDE(isr_fit = _isr_undefined);
        PROVIDE(isr_dec = _isr_undefined);
	} > ram

	.data : {
		*(.data)
		*(.rodata)
	} > ram

	.bss : {
		*(.bss)
		*(.sbss)
	} > ram

  /DISCARD/ : {
    *(.eh_frame)
  }
}


