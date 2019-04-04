import pydls as dls
import json
import helpers as hp
import pylogging
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze weights of neurons")
    parser.add_argument('--cap_mem', type=str, help="This is the cap mem config to use")
    parser.add_argument('--dac_conf', type=str, help="This is the dac config to use")
    args = parser.parse_args()

    # activate logging
    pylogging.reset()
    pylogging.default_config(
        level=pylogging.LogLevel.INFO,
        fname="",
        print_location=False,
        color=True,
        date_format='RELATIVE')
    log = pylogging.get("main")

    #Open the cap_mem and dac config
    with open(args.cap_mem, 'r') as f:
        capmem_defaults = json.load(f)

    with open(args.dac_conf, 'r') as f:
        dac_default = json.load(f)

    chip = dls.Chip()
    hp.set_cap_mem_values(chip.cap_mem, capmem_defaults)

    spikeSenders = []
    spikeMeanTime = []
    spikeAmount = []    

    #Iterate over the entire weight space:
    for weight in range(64):

        #Measurement setup
        measurementInterval = 100000
        maxSpikeAmount = 1
        
        #connect all neurons to a single input line
        for inputID in range(32):
            syn = chip.synram.get(dls.Synapse_row(inputID), dls.Synapse_column(inputID))
            syn.weight(weight)  # 6bit
            syn.address(20)  # address
            chip.synram.set(dls.Synapse_row(inputID), dls.Synapse_column(inputID), syn)

            neuron = chip.neurons.get(dls.Neuron_index(inputID))
            neuron.enable_out(True)
            neuron.mux_readout_mode(dls.Neuron.vmem)
            chip.neurons.set(dls.Neuron_index(inputID), neuron)

        #Setup of capmem
        builder = dls.Dls_program_builder()  # capmem einpendeln
        builder.set_time(0)
        builder.set_chip(chip)
        builder.wait_for(100000)
        builder.halt()

        #Prepare measurement
        spikes_builder = dls.Dls_program_builder()
        spikes_builder.set_time(0)
        spikes_builder.wait_for(1000)

        #Fire successively spike more spikes into the neuron
        for nspikes in range(maxSpikeAmount):
            for spk in range(nspikes + 1):
                #Keep the 45 cycles delay for the spikes to not mess up the chip
                spikes_builder.fire(2**32 - 1, 20)  # row (32bit) as mask, address
                spikes_builder.wait_for(50)
            
            #After all neurons are prepared with the spikes, wait for a long time to avoid interference
            spikes_builder.wait_for(measurementInterval)

        spikes_builder.wait_for(measurementInterval)
        spikes_builder.halt()

        #Load program to DLS
        with dls.connect(dls.get_allocated_board_ids()[0]) as c:
            dls.soft_reset(c)
            hp.setup_dac(c, dac_default)  # soft reset ? resets dac?

            builder.transfer(c, 0)  # connection, 0 is program address
            builder.execute(c, 0)  # triggers execution on FPGA
            builder.fetch(c)
            
            spikes_builder.transfer(c, 0)
            spikes_builder.execute(c, 0)
            spikes_builder.fetch(c)
        
        #Evaluate the spikes        
        spike_train = spikes_builder.get_spikes()
        categorizedSpikes = []
        spikeList = []

        for spike in spike_train:
            spikeList.append(spike)

        #Prefilter spike train
        #for spike in spikeList:
        #    if spike.time > 100000:
        #        spikeList.pop(0)

        for spike in spikeList:
            print spike

        for intervalID in range(maxSpikeAmount + 1):
            currentSpikeInterval = []
            for spike in spikeList:
                if spike.time > (intervalID) * measurementInterval and spike.time < (intervalID + 1) * measurementInterval:
                    currentSpikeInterval.append(spike)
            categorizedSpikes.append(currentSpikeInterval)

        #Analze the spikes
        intervalMeanTimeTillFirst = []
        intervalSpikes = []
        intervalSenders = []
        for interval in range(maxSpikeAmount):
            tempSenders = []
            tempTimes = []
            for spike in categorizedSpikes[interval]:
                if spike.address not in tempSenders:
                    tempTimes.append(spike.time - interval * measurementInterval - 50 * interval)
                tempSenders.append(spike.address)

            intervalSenders.append(len(list(set(tempSenders))))
            intervalMeanTimeTillFirst.append(np.mean(tempTimes))
            intervalSpikes.append(len(categorizedSpikes[interval]))

        spikeSenders.append(intervalSenders)
        spikeMeanTime.append(intervalMeanTimeTillFirst)
        spikeAmount.append(intervalSpikes)

    print spikeAmount
