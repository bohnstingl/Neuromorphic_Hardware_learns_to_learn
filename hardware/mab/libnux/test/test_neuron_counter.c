#include "libnux/unittest.h"
#include "libnux/dls.h"
#include "libnux/counter.h"


void test_counter_reset()
{
    uint8_t neuron, tmp_counter;
    libnux_testcase_begin("test_counter_reset");
    reset_all_neuron_counters();
    for (neuron = 0; neuron < dls_num_columns; neuron++)
    {
        tmp_counter = get_neuron_counter(neuron);
        libnux_test_equal(tmp_counter, 0);
    }
    libnux_testcase_end();
}

#if (LIBNUX_DLS_VERSION == 2)
void test_disable_counters()
{
    uint32_t disable_mask = 0;
    uint32_t enabled_counters;
    libnux_testcase_begin("test_disable_counters");
    enable_neuron_counters(disable_mask);
    enabled_counters = get_enabled_neuron_counters();
    libnux_test_equal(enabled_counters, disable_mask);
    libnux_testcase_end();
}

void test_enable_counters()
{
    uint32_t enable_mask = 0xffffffff;
    uint32_t enabled_counters;
    libnux_testcase_begin("test_enable_counters");
    enable_neuron_counters(enable_mask);
    enabled_counters = get_enabled_neuron_counters();
    libnux_test_equal(enabled_counters, enable_mask);
    libnux_testcase_end();
}

void test_counter_config()
{
    libnux_testcase_begin("test_counter_config");
    neuron_counter_config write_config, read_config;
    write_config.clear_on_read = 0;
    write_config.fire_interrupt = 0;
    configure_neuron_counter(write_config);
    read_config = get_neuron_counter_configuration();
    libnux_test_equal(read_config.clear_on_read, write_config.clear_on_read);
    libnux_test_equal(read_config.fire_interrupt, write_config.fire_interrupt);
    write_config.clear_on_read = 1;
    write_config.fire_interrupt = 1;
    configure_neuron_counter(write_config);
    read_config = get_neuron_counter_configuration();
    libnux_test_equal(read_config.fire_interrupt, write_config.fire_interrupt);
    libnux_test_equal(read_config.clear_on_read, write_config.clear_on_read);
    libnux_testcase_end();
}
#endif

void start()
{
    libnux_test_init();
    test_counter_reset();
#if (LIBNUX_DLS_VERSION == 2)
    test_disable_counters();
    test_enable_counters();
    test_counter_config();
#endif
    libnux_test_shutdown();
}
