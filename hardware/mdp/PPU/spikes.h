#pragma once

#ifndef SPIKES_H__
#define SPIKES_H__

uint32_t i;
static uint32_t const SPIKE_BASE_ADDR = (0x3c000040 << 2);
extern uint32_t wait_cycles(uint32_t time_out);

typedef struct {
    uint32_t row_mask;
    uint8_t addr;
} spike_t;


static void spikes_send(spike_t* sp) {
    volatile uint32_t* ptr = (volatile uint32_t*)(SPIKE_BASE_ADDR + (sp->addr << 2));
    *ptr = sp->row_mask;
}

static void send_uniform_spiketrain(spike_t* single_spike, uint32_t number, uint32_t usec) {

    for (i = number; i > 0; i--){
        spikes_send(single_spike);
        wait_cycles(usec * 100);
    }
}

#endif
