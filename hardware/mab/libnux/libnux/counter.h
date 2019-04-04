#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
	bool fire_interrupt;
	bool clear_on_read;
} neuron_counter_config;

uint32_t get_neuron_counter(uint8_t neuron);
void reset_neuron_counter(uint8_t neuron);
void reset_all_neuron_counters();

#if (LIBNUX_DLS_VERSION == 2)
void enable_neuron_counters(uint32_t enable_mask);
uint32_t get_enabled_neuron_counters();
void configure_neuron_counter(neuron_counter_config config);
neuron_counter_config get_neuron_counter_configuration();
void clear_neuron_counters_on_read(bool value);
void fire_interrupt(bool value);
#endif
