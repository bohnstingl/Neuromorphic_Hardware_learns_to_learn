#pragma once

#define deprecated(X)

#include "../src/frickel-dls/dacs.h"
#include "../src/frickel-dls/adcamp.h"
#include "../src/frickel-dls/registers.h"
#include "../src/frickel-dls/com.h"
#include "../src/frickel-dls/execute.h"
#include "../src/frickel-dls/omnibus.h"
#include "../src/frickel-dls/synram.h"
#include "../src/frickel-dls/capmem.h"
#include "../src/frickel-dls/neuron.h"
#include "../src/frickel-dls/spikes.h"
#include "../src/frickel-dls/ppu.h"
#include "../src/frickel-dls/correlation.h"
#include "../src/frickel-dls/printing.h"
#include "../src/frickel-dls/dls_program_builder.h"
#include "../src/frickel-dls/spike_router.h"

namespace {
    static inline void instantiate() {
        size_t a;
        static_cast<void>(a);
        a = sizeof( frickel_dls::Raw_image );
        a = sizeof( frickel_dls::Spiketrain );
    }
}                                        
