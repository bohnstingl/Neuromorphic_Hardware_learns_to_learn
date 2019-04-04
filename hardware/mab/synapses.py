from __future__ import print_function

import numpy as np

def setup_synram(weight_matrix, address_matrix, chip, use_pydlsnew=False):
    # check validity of weight matrix and address matrix
    if weight_matrix.dtype != np.int or weight_matrix.shape != (32, 32):
        raise ValueError('Check weight matrix shape/dtype')
    if address_matrix.dtype != np.int or address_matrix.shape != (32, 32):
        raise ValueError('Check address matrix shape/dtype')
    value_error = False
    for i in range(32):
        if not (np.all(weight_matrix[i, :] <= 0) or np.all(weight_matrix[i, :] >= 0)):
            value_error = True
    if np.any(weight_matrix < -63) or np.any(weight_matrix > 63):
        value_error = True
    if value_error:
        raise ValueError('Check weight matrix values')
    if np.any(address_matrix < 0) or np.any(address_matrix > 63):
        raise ValueError('Check address matrix values')
    
    # compatibility code
    if use_pydlsnew:
        import pydlsnew.coords as coords
    else:
        import pydls as coords
    
    excitatory_drivers = (np.sum(weight_matrix, axis=1) >= 0).astype(np.bool)
    for row in range(32):
        chip.syndrv_config.senx(coords.Synapse_row(row), excitatory_drivers[row])
        chip.syndrv_config.seni(coords.Synapse_row(row), not excitatory_drivers[row])
        for col in range(32):
            syn = chip.synram.get(coords.Synapse_row(row), coords.Synapse_column(col))
            syn.weight(weight_matrix[row, col] * (1 if excitatory_drivers[row] else -1))
            syn.address(address_matrix[row, col])
            chip.synram.set(coords.Synapse_row(row), coords.Synapse_column(col), syn)

