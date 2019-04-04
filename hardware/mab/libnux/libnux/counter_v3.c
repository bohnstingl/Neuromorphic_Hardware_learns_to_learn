inline uint32_t get_neuron_counter(uint8_t neuron)
{
	volatile uint32_t* ptr = (uint32_t*) (dls_rates_base + neuron*4);
	return *ptr;
}

inline void reset_neuron_counter(uint8_t neuron)
{
	volatile uint32_t* ptr = (uint32_t*) (dls_rates_reset + neuron*4);
	*ptr = 0; // arbitrary
}