# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import
from aiida.engine import calcfunction
from aiida import orm
"""
Utility functions for the potential alignment workchain
"""


@calcfunction
def get_potential_difference(first_potential, second_potential):
    """
    Calculate the difference of two potentials

    Parameters
    ----------
    first_potential: ArrayData
        The first potential array
    second_potential: ArrayData
        The second potential, to be subtracted from the first

    Returns
    -------
    difference_potential
        The calculated difference of the two potentials
    """

    first_array = first_potential.get_array(first_potential.get_arraynames()[0])
    second_array = second_potential.get_array(second_potential.get_arraynames()[0])

    difference_array = first_array - second_array
    difference_potential = orm.ArrayData()
    difference_potential.set_array('difference_potential', difference_array)
    
    return difference_potential