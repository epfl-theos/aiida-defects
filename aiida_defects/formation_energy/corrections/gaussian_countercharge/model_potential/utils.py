# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit

from aiida import orm
from aiida.engine import calcfunction
from qe_tools.constants import hartree_to_ev



@calcfunction
def create_model_structure(base_structure, scale_factor):
    """
    Prepare a model structure with a give scale factor, based on a base_structure
    """
    base_cell = np.array(base_structure.cell)
    model_cell = base_cell * scale_factor
    model_structure = orm.StructureData(cell=model_cell)

    return model_structure


def get_cell_matrix(structure):
    """
    Get the cell matrix (in bohr) from an AiiDA StructureData object

    Parameters
    ----------
    structure: AiiDA StructureData
        The structure object of interest

    Returns
    -------
    cell_matrix
        3x3 cell matrix array in units of Bohr

    """
    from qe_tools.constants import bohr_to_ang
    cell_matrix = np.array(structure.cell) / bohr_to_ang  # Angstrom to Bohr
    return cell_matrix


def get_reciprocal_cell(cell_matrix):
    """
    For a given cell_matrix, compute the reciprocal cell matrix

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell

    Returns
    -------
    reciprocal_cell
        3x3 cell matrix array in reciprocal units
    """
    from numpy.linalg import inv
    #reciprocal_cell = (inv(cell_matrix)).transpose()
    reciprocal_cell = (2 * np.pi * inv(cell_matrix)
                       ).transpose()  # Alternative definition (2pi)

    return reciprocal_cell


def get_reciprocal_grid(cell_matrix, cutoff):
    """
    Prepare a reciprocal space grid to achieve a given planewave energy cutoff
    cutoff (Ry)

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the reciprocal-space cell
    cutoff: float
        Desired kinetic energy cutoff in Rydberg

    Returns
    -------
    grid_max
        A numpy vector of grid dimensions for the given cutoff

    """

    # Radius of reciprocal space sphere containing planewaves with a given kinetic energy
    G_max = 2.0 * np.sqrt(cutoff)  # Ry

    # Get the number of G-vectors needed along each cell vector
    # Note, casting to int alway rounds down so we add one
    grid_max = (G_max / np.linalg.norm(cell_matrix, axis=1)).astype(int) + 1

    # For convenience later, ensure the grid is odd valued
    for axis in range(3):
        if grid_max[axis] % 2 == 0:
            grid_max[axis] += 1

    return orm.List(list=grid_max.tolist())


@calcfunction
def get_charge_model(limits,
                     dimensions,
                     defect_position,
                     sigma=orm.Float(1.0),
                     charge=orm.Float(-1.0)):
    """
    For a given system charge, create a model charge distribution.
    The charge model for now is a Gaussian.
    Grid = coord grid
    TODO: Change of basis
    """
    limits = limits.get_list()
    dimensions = dimensions.get_list()
    defect_position = defect_position.get_list()
    sigma = sigma.value
    charge = charge.value

    print("DEBUG: Dimensions = {}".format(dimensions))
    print("DEBUG: Limits = {}".format(limits))

    i = np.linspace(0, limits[0], dimensions[0])
    j = np.linspace(0, limits[1], dimensions[1])
    k = np.linspace(0, limits[2], dimensions[2])
    grid = np.meshgrid(i, j, k)

    # Get the gaussian at the defect position
    g = get_gaussian_3d(grid, defect_position, sigma)

    # Get the offsets
    offsets = np.zeros(3)
    for axis in range(3):
        # Capture the offset needed for an image
        if defect_position[axis] > limits[axis] / 2.0:
            offsets[axis] = -limits[axis]
        else:
            offsets[axis] = +limits[axis]

    # Apply periodic boundary conditions
    g = 0
    for dim0 in range(2):
        for dim1 in range(2):
            for dim2 in range(2):
                image_offset = [dim0, dim1, dim2] * offsets
                g = g + get_gaussian_3d(
                    grid, defect_position + image_offset, sigma=sigma)

    # Scale the charge density to the desired charge
    #int_charge_density = np.trapz(np.trapz(np.trapz(g, i), j), k)
    int_charge_density = get_integral(g, dimensions, limits)
    print(
        "DEBUG: Integrated charge density (g) = {}".format(int_charge_density))

    g = g / (int_charge_density / charge)

    # Compensating jellium background
    print("DEBUG: Integrated charge density (scaled_g) = {}".format(
        get_integral(g, dimensions, limits)))

    #scaled_g = scaled_g - np.sum(scaled_g)/np.prod(scaled_g.shape)
    print("DEBUG: Integrated charge density (jellium) = {}".format(
        get_integral(g, dimensions, limits)))

    # Pack the array
    model_charge_array = orm.ArrayData()
    model_charge_array.set_array('model_charge', g)

    return model_charge_array


def get_gaussian_3d(grid, position, sigma):
    """
    Calculate 3D Gaussian on grid
    NOTE: Minus sign at front give negative values of charge density throughout the cell
    """
    x = grid[0] - position[0]
    y = grid[1] - position[1]
    z = grid[2] - position[2]

    gaussian = -np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2)) / (
        (2.0 * np.pi)**1.5 * np.sqrt(sigma))

    return gaussian


def get_integral(data, dimensions, limits):
    """
    Get the integral of a uniformly spaced 3D data array
    """
    limits = np.array(limits)
    dimensions = np.array(dimensions)
    volume_element = np.prod(limits / dimensions)
    return np.sum(data) * volume_element


def get_fft(grid):
    """
    Get the FFT of a grid
    """
    return np.fft.fftshift(np.fft.fftn(grid))


def get_inverse_fft(fft_grid):
    """
    Get the inverse FFT of an FFT grid
    """
    return np.fft.ifftn(np.fft.ifftshift(fft_grid))


@calcfunction
def get_model_potential(cell_matrix, dimensions, charge_density, epsilon):
    """
    3D possion solver

    Parameters
    ----------
    cell_matrix: 3x3 array
        The reciprocal space cell matrix
    dimensions: vector-like
        The dimensions required for the reciprocal space grid
    charge_density:  array
        The calculated model charge density on a 3-dimensional real space grid
    epsilon: float
        The value of the dielectric constant

    Returns
    -------
    V_model_r
        The calculated model potential in real space array
    """

    dimensions = np.array(dimensions)
    cell_matrix = cell_matrix.get_array('cell_matrix')
    charge_density = charge_density.get_array('model_charge')
    epsilon = epsilon.value

    # Set up a reciprocal space grid for the potential
    # Prepare coordinates in a 3D array of ijk vectors
    # (Note: Indexing is column major order, but the 4th dimension vectors remain ijk)
    dimensions = dimensions // 2  #floor division

    ijk_array = np.mgrid[-dimensions[0]:dimensions[0] +
                         1, -dimensions[1]:dimensions[1] +
                         1, -dimensions[2]:dimensions[2] + 1].T

    # Get G vectors
    G_array = np.dot(
        ijk_array,
        (cell_matrix.T
         ))  # To do: check why we use a grid that goes way past the recip cell

    # Calculate the square modulus
    G_sqmod_array = np.linalg.norm(G_array, axis=3)**2

    # Get the reciprocal space charge density
    charge_density_g = get_fft(charge_density)

    # Compute the model potential
    v_model = np.divide(
        charge_density_g, G_sqmod_array, where=G_sqmod_array != 0.0)
    V_model_g = v_model * 4. * np.pi / epsilon

    V_model_g[dimensions + 1, dimensions + 1, dimensions + 1] = 0.0

    # Get the model potential in real space
    V_model_r = get_inverse_fft(V_model_g)

    # Pack up the array
    V_model_array = orm.ArrayData()
    V_model_array.set_array('model_potential', V_model_r)

    return V_model_array


def get_energy(potential, charge_density, dimensions, limits):
    """
    Calculate the total energy for a given model potential
    """

    potential = potential.get_array('model_potential')
    charge_density = charge_density.get_array('model_charge')

    ii = np.linspace(0., limits[0], dimensions[0])
    jj = np.linspace(0., limits[1], dimensions[1])
    kk = np.linspace(0., limits[2], dimensions[2])

    energy = np.real(0.5 * np.trapz(
        np.trapz(np.trapz(charge_density * potential, ii), jj), kk)) * hartree_to_ev
    return orm.Float(energy)
