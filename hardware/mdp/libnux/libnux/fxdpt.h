#ifndef FXDPT_H__
#define FXDPT_H__

#include <stdint.h>

#include "libnux/attrib.h"

/** INV_SCALE 0x10000 is the 16.16 fixedpoint representation
 * MAX:       65536
 * MIN:      -65536
 * precision:    15.26e-6
 * */
#define INV_SCALE  0x10000
#define INV_SCALE_16 0x10
//#define INV_SCALE 16777216/128
#define SCALE (1.0/(double)INV_SCALE)
#define SCALE_16 (1.0/(double)INV_SCALE)

#define FP(x) ((int32_t)(INV_SCALE * (x)))
#define FP16(x) ((int16_t)(INV_SCALE_16 * (x)))
#define INV_FP(x) ((double)x*SCALE)
#define FP_MUL(x,y) (x*y / INV_SCALE)

ATTRIB_UNUSED static int16_t fxdpt_32_to_16_bits(const int32_t x)
{
	if( INV_SCALE_16 < INV_SCALE )
		return (int16_t)( x / (INV_SCALE / INV_SCALE_16) );
	else
		return (int16_t)( x * (INV_SCALE_16 / INV_SCALE) );
}


#define F16_MAX (0x7fff)
#define F16_MIN (0x8000)

// convert floating-point to 16 bit fractional fixed-point
#define F16(x) ((int16_t)( 0x8000 * (x) ))

#define F8_MAX (0x7f)
#define F8_MIN (0x80)

// convert floating-point to 8 bit fractional fixed-point
#define F8(x) ((int8_t)( 0x80 * (x) ))

#endif

