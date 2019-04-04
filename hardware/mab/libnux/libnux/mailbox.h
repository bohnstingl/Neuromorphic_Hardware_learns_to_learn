#pragma once

#include <stdint.h>

extern uint8_t mailbox_base;
extern uint8_t mailbox_end;

uint32_t mailbox_write(uint32_t const offset, uint8_t const * src, uint32_t const size);
uint32_t mailbox_read(uint8_t * dest, uint32_t const offset, uint32_t const size);

uint8_t libnux_mailbox_read_u8(uint32_t const offset);
void libnux_mailbox_write_u8(uint32_t const offset, uint8_t byte);

uint32_t libnux_mailbox_write_string(char const * str);
uint32_t libnux_mailbox_write_int(uint32_t const n);
