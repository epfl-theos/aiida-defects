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


@calcfunction
def get_potential_difference(first_potential, second_potential):
    """
    Calculate the difference of two potentials that have the same size

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

    first_array = first_potential.get_array(
        first_potential.get_arraynames()[0])
    second_array = second_potential.get_array(
        second_potential.get_arraynames()[0])

#    if first_array.shape != second_array.shape:
#        target_shape = orm.List(list=np.max(np.vstack((first_array.shape, second_array.shape)), axis=0).tolist())
#        first_array = get_interpolation(first_potential, target_shape).get_array('interpolated_array')
#        second_array = get_interpolation(second_potential, target_shape).get_array('interpolated_array')

    difference_array = first_array - second_array
    difference_potential = orm.ArrayData()
    difference_potential.set_array('difference_potential', difference_array)

    return difference_potential

@calcfunction
def get_interpolation(input_array, target_shape):
    """
    Interpolate an array into a larger array of size, `target_size`

    Parameters
    ----------
    array: orm.ArrayData
        Array to interpolate
    target_shape: orm.List
        The target shape to interpolate the array to

    Returns
    -------
    interpolated_array
        The calculated difference of the two potentials
    """

    from scipy.ndimage.interpolation import map_coordinates

    # Unpack
    array = input_array.get_array(input_array.get_arraynames()[0])
    target_shape = target_shape.get_list()

    # map_coordinates takes two parameters - an input array and a coordinates
    # array. It then evaluates what the interpolation of the input array should be
    # at the target coordinates.
    # Generate the target grid.
    i = np.linspace(0, array.shape[0]-1, target_shape[0])
    j = np.linspace(0, array.shape[1]-1, target_shape[1])
    k = np.linspace(0, array.shape[2]-1, target_shape[2])
    ii,jj,kk = np.meshgrid(i,j,k, indexing='ij')
    target_coords = np.array([ii,jj,kk])
    # Do the interpolation
    interp_array = map_coordinates(input=np.real(array), coordinates=target_coords)

    interpolated_array = orm.ArrayData()
    interpolated_array.set_array('interpolated_array', interp_array)

    return interpolated_array
