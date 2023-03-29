# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida.engine import calcfunction
from aiida import orm
"""
Utility functions for the potential alignment workchain
"""

class AllValuesMaskedError(ValueError):
    """
    Error raised when no values are left after the masking procedure.
    If one proceeds to compute averages using an array in which all values
    are masked, the resulting object is an instance of 'numpy.ma.core.MaskedConstant'
    which cannot be stored by AiiDA and is, in any case, meaningless.
    """
    pass

@calcfunction
def get_alignment(potential_difference, charge_density, tolerance):
    """
    Compute the density-weighted potential alignment
    """
    # Unpack
    tol = tolerance.value
    v_diff = potential_difference.get_array(
        potential_difference.get_arraynames()[0])
    charge_density = charge_density.get_array(
        charge_density.get_arraynames()[0])

    # Get array mask based on elements' charge exceeding the tolerance.
    mask = np.ma.greater(np.abs(charge_density), tol)

    # Apply this mask to the diff array
    v_diff_masked = np.ma.masked_array(v_diff, mask=mask)

    # Check if any values are left after masking
    if v_diff_masked.count() == 0:
        raise AllValuesMaskedError

    # Compute average alignment
    alignment = np.average(np.abs(v_diff_masked))

    return orm.Float(alignment)