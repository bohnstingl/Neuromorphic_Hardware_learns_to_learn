void * memset(void *ptr, int value, uint32_t num);
void * memcpy(void *destination, const void * source, uint32_t num);
void readWeights(uint8_t *weights);
void readWeightsSelective(uint8_t *weights, uint8_t states, uint8_t actions);
void writeWeights(uint8_t *weights);
void setWeight(uint8_t rowIndex, uint8_t colIndex, uint8_t weight);
uint32_t wait_cycles(uint32_t time_out);
void wait100();
int update_spike_counters(uint32_t *spikeCounts, uint8_t numActions, uint8_t numStates);
void reset_spike_counters(uint32_t *spikeCounts, uint8_t numActions);
uint32_t udiv32(uint32_t n, uint32_t d);
uint64_t udiv64(uint64_t n, uint64_t d);
void set_weight(uint8_t row, uint8_t column, uint8_t weight);
void set_weights(uint8_t row, uint8_t startColumn, uint8_t weights[], uint8_t amount);

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}
