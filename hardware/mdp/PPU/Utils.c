#include <s2pp.h>
#include "Utils.h"


extern uint32_t const dls_num_synapse_vectors;
extern uint32_t const dls_weight_base;
extern uint32_t* const dls_rates_base;

void * memset ( void * ptr, int value, uint32_t num )
{
    unsigned char *ptr1 = (unsigned char *) ptr;
    unsigned char *ptr2 = ptr1;
    while (ptr2 + num > ptr1) 
    {
        *ptr1 = (unsigned char) value;
        ptr1 ++;
    }
    return (void *) ptr2;
}

void * memcpy ( void * destination, const void * source, uint32_t num )
{
    while (num--) 
    {
        *((int8_t*)destination) = *((int8_t*)source);
        destination++;
        source++;
    }
}

void readWeightsSelective(uint8_t *weights, uint8_t states, uint8_t actions)
{
    uint8_t weight_base[32] = {1};
    for (uint32_t index = 0; index < dls_num_synapse_vectors; index ++) {
        uint8_t w_ind = (index % 2) * 16;
        asm volatile (
            "fxvinx 1, %[dls_weight_base], %[index]\n"
            "fxvstax 1, %[weight_base], %[w_ind]\n"
            : 
            : [index] "r" (index),
              [dls_weight_base] "b" (dls_weight_base),
              [w_ind] "r" (w_ind),
              [weight_base] "r" (weight_base)
            : "kv1");

        if(index > 0 && ((index + 1) % 2) == 0 && (index / 2) < states)
        {
            //Copy the relevant weights
            for(uint8_t i = 0; i < actions; i++)
            {
                weights[(index / 2) * actions + i] = weight_base[states + i];
            }
        }
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
    
    /*for (uint8_t i = 0; i < 32; i ++) {
        for (uint8_t j = 0; j < 16; j ++) {
            libnux_mailbox_write_u8(i * 16 + j, weights[i * 16 + j]);
        }
    }*/
}

void writeWeights(uint8_t *weights)
{
    uint8_t *weight_base = weights;
    for (uint32_t index = 0; index < dls_num_synapse_vectors; index ++) {
        uint32_t w_ind = index * 16;
        asm volatile (
            "fxvlax 1, %[weight_base], %[w_ind]\n"
            "fxvoutx 1, %[dls_weight_base], %[index]\n"
            : 
            : [index] "r" (index),
              [dls_weight_base] "b" (dls_weight_base),
              [w_ind] "r" (w_ind),
              [weight_base] "r" (weight_base)
            : "kv1");
    }
}

void setWeight(uint8_t rowIndex, uint8_t colIndex, uint8_t weight)
{
    //Read only the corresponding synapse driver and update it
    uint8_t weight_base[16];
    uint32_t w_ind = 0;
    
    //Index to read from the weight RAM
    uint32_t index = rowIndex * 2;

    if(colIndex >= 16)
        index ++;

    asm volatile (
        "fxvinx 1, %[dls_weight_base], %[index]\n"
        "fxvstax 1, %[weight_base], %[w_ind]\n"
        : 
        : [index] "r" (index),
          [dls_weight_base] "b" (dls_weight_base),
          [w_ind] "r" (w_ind),
          [weight_base] "r" (weight_base)
        : "kv1");

    weight_base[colIndex % 16] = weight;

    asm volatile (
        "fxvlax 1, %[weight_base], %[w_ind]\n"
        "fxvoutx 1, %[dls_weight_base], %[index]\n"
        : 
        : [index] "r" (index),
          [dls_weight_base] "b" (dls_weight_base),
          [w_ind] "r" (w_ind),
          [weight_base] "r" (weight_base)
        : "kv1");
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

// assume clear on read of spike counters
// returns != 0 if one spike counter is nonzero
int update_spike_counters(uint32_t *spikeCounts, uint8_t numActions, uint8_t numStates) {
    int i;
    int nonzero = 0;
    for (i = 0; i < numActions; i++) {
        spikeCounts[i] += dls_rates_base[numStates + i];
        if (spikeCounts[i])
            nonzero += 1;
    }
    return nonzero;
}

void reset_spike_counters(uint32_t *spikeCounts, uint8_t numActions) {
    memset(spikeCounts, 0, numActions * sizeof(uint32_t));
}

uint32_t udiv32(uint32_t n, uint32_t d) {
    uint32_t q = 0;
    while (n >= d) {
        uint32_t i = 0, d_t = d;
        while (n >= (d_t << 1) && ++i)
            d_t <<= 1;
        q |= 1 << i;
        n -= d_t;
    }
    return q;
}

uint64_t udiv64(uint64_t n, uint64_t d) {
    uint64_t q = 0;
    while (n >= d) {
        uint32_t i = 0;
        uint64_t d_t = d;
        while (n >= (d_t << 1) && ++i)
            d_t <<= 1;
        q |= 1 << i;
        n -= d_t;
    }
    return q;
}

void set_weight(uint8_t row, uint8_t column, uint8_t weight) {
    uint32_t weight_offset = row * 2 + column / 16;
    register vector uint8_t vec_weights;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    //Clip the weight to the useful range
    if(((uint8_t)weight) > 63)
        weight = 63;
    
    vec_weights[column % 16] = weight;
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}

void set_weights(uint8_t row, uint8_t startColumn, uint8_t weights[], uint8_t amount) {
    uint32_t weight_offset = row * 2 + startColumn / 16;
    register vector uint8_t vec_weights;
    uint8_t z = 0;
    
    asm volatile (
        "fxvinx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        : [vec_weights] "=&kv" (vec_weights)
        : [weight_offset] "r" (weight_offset),
          [dls_weight_base] "b" (dls_weight_base)
        : );
    
    for(z = 0; z < amount; z++)
    {
        //Clip the weight to the useful range
        if(((uint8_t)weights[z]) > 63)
            weights[z] = 63;

        vec_weights[(startColumn % 16) + z] = weights[z];
    }
    
    asm volatile (
        "fxvoutx %[vec_weights], %[dls_weight_base], %[weight_offset]\n"
        :
        : [weight_offset] "r" (weight_offset),
          [vec_weights] "kv" (vec_weights),
          [dls_weight_base] "b" (dls_weight_base)
        : );
}
