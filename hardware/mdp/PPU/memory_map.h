#pragma once

#include <stdint.h>

static uint32_t const FXVIO_WEIGHT_BASE   = 0x0000;
static uint32_t const FXVIO_DECODER_BASE  = 0x4000;
static uint32_t const FXVIO_CAUSAL_BASE   = 0x8000;
static uint32_t const FXVIO_VRESET_C_BASE = 0x9000;
static uint32_t const FXVIO_ACAUSAL_BASE  = 0xc000;
static uint32_t const FXVIO_VRESET_A_BASE = 0xd000;

static uint32_t const FXVIO_LOCATION_MASK      = 0x000fff;
static uint32_t const FXVIO_TEST_MASK          = 0x100000;
static uint32_t const FXVIO_BUFFER_ENABLE_MASK = 0x200000;

static uint32_t const SPIKE_BASE_ADDR = (0x3c000040 << 2);
static uint32_t const DECODER_BASE_ADDR = (0x20014000 << 2);

static uint32_t const COUNTER_BASE_ADDR = (0x3e000000 << 2);
