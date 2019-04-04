#include <s2pp.h>
#include "Utils.h"

extern uint32_t const dls_num_synapse_vectors;
extern uint32_t const dls_weight_base;

void memset ( void * ptr, int value, uint32_t num )
{
    while (num--)
    {
        *((int32_t*)ptr) = value;
        ptr++;
    }
}

void memcpy ( void * destination, const void * source, uint32_t num )
{
    while (num--) 
    {
        *((int8_t*)destination) = *((int8_t*)source);
        destination++;
        source++;
    }
}

void readWeights(uint8_t *weights)
{
    uint8_t *weight_base = weights;
    for (uint32_t index = 0; index < dls_num_synapse_vectors; index ++) {
        uint32_t w_ind = index * 16;
        asm volatile (
            "fxvinx 1, %[dls_weight_base], %[index]\n"
            "fxvstax 1, %[weight_base], %[w_ind]\n"
            : 
            : [index] "r" (index),
              [dls_weight_base] "b" (dls_weight_base),
              [w_ind] "r" (w_ind),
              [weight_base] "r" (weight_base)
            : "kv1");
    }
}

void wait100() {
    uint64_t a = rdtsc();
    a = a - 0;
    a = rdtsc();
    a = a - 0;
}

uint32_t wait_cycles(uint32_t time_out) {
    uint64_t begin = rdtsc();
    uint64_t current = begin;
    while ((current - begin) < time_out) {
        current = rdtsc();
    }
    return (uint32_t) (current - begin);
}
