#include "correlation.h"


void reset_all_correlations()
{
	uint8_t row;
	for (row = 0; row < dls_num_rows; row++) {
		reset_correlation(row);
	}
}
