# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from qe_tools import CONSTANTS

from aiida.engine import calcfunction
from aiida import orm

from aiida_defects.utils import get_cell_matrix, get_grid

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
def convert_Hat_to_Ryd(potential):
    v_model = orm.ArrayData()
    v_model.set_array('data', potential.get_array(potential.get_arraynames()[0])*-2.0)

    return v_model

@calcfunction
def get_alignment(potential_difference, defect_site, cutoff_radius=lambda: orm.Float(0.5)):
    """
    Compute the mean-absolute error potential alignment

    Parameters
    ----------
        potential_difference - numpy array
            The difference in the electrostatic potentials to be aligned
        defect_site - length 3 list, tuple or array
            defect postion in crystal coordinates
        cutoff_radius - float
            distance cutoff from defect site in crystal coordinates. Coordinates
            less than this distance are considered to be influenced by the defect
            and are excluded from the alignment
    """
    # Unpack ArrayData object
    v_diff = potential_difference.get_array(
        potential_difference.get_arraynames()[0])

    # Generate a crystal grid of the same dimension as the data
    ijk_array = get_grid(v_diff.shape, endpoint=False)
    # Compute the distance from the defect site to every other.
    distance_vectors = np.array(defect_site.get_list()).reshape(3,1) - ijk_array
    # Apply minimum image
    min_image_vectors = (distance_vectors - np.rint(distance_vectors))
    # Compute distances and reshape to match input data
    distances = np.linalg.norm(min_image_vectors, axis=0).reshape(v_diff.shape)

    # In crystal coordinates, the maximum separation between interacting
    # images is d=1 so look for coordinates at a distance of less than d=0.5.
    # These are the coordinates within the shphere of interaction of the defect.
    # Mask these and only compute the alignment the remaining, most distance points.
    mask = np.ma.less(distances, cutoff_radius.value)
    v_diff_masked = np.ma.masked_array(v_diff, mask=mask)
    values_remaining = (v_diff_masked.count()/np.prod(v_diff.shape))*100.0
    print('{:.2f}% of values remain'.format(values_remaining))

    # Check if any values are left after masking
    if v_diff_masked.count() == 0:
        raise AllValuesMaskedError

    fit_result = fit_potential(v_diff_masked)
    alignment = -1.*fit_result.x*CONSTANTS.ry_to_ev

    return orm.Float(alignment)


def fit_potential(v_diff):
    """
    Find the offset between two potentials, delta_z, that minimises the summed absolute error.
    """
    from scipy.optimize import minimize

    def obj(delta_z):
        """
        Objective function. Delta_z is the alignment of potentials
        """
        return np.sum(np.abs(v_diff-delta_z))

    initial_guess = 1.0
    result = minimize(obj, initial_guess)
    return result
