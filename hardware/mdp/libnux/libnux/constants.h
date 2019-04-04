#pragma once


#if defined(SYSTEM_HICANN_DLS_MINI)

#	define NUM_SLICES           (1)
#	define NUM_BYTES_PER_SLICE (16)

#elif defined(SYSTEM_HICANN_DLS)

#	define NUM_SLICES           (8)
#	define NUM_BYTES_PER_SLICE (16)

#else
#	error "You must define the target system!"
#endif


#define NUM_BYTES_PER_ARRAY (NUM_BYTES_PER_SLICE * NUM_SLICES)
#define NUM_HALFWORDS_PER_ARRAY (NUM_BYTES_PER_ARRAY/2)
#define NUM_WORDS_PER_ARRAY (NUM_HALFWORDS_PER_ARRAY/2)

