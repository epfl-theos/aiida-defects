# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal

from aiida import orm
from aiida.engine import calcfunction
from qe_tools import CONSTANTS


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
    cell_matrix = np.array(structure.cell) / CONSTANTS.bohr_to_ang  # Angstrom to Bohr
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
    reciprocal_cell = (2 * np.pi * inv(cell_matrix)).transpose()  # Alternative definition (2pi)

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
    # Note, casting to int always rounds down so we add one
    grid_max = (G_max / np.linalg.norm(cell_matrix, axis=1)).astype(int) + 1

    # For convenience later, ensure the grid is odd valued
    for axis in range(3):
        if grid_max[axis] % 2 == 0:
            grid_max[axis] += 1

    return orm.List(list=grid_max.tolist())


def get_xyz_coords(cell_matrix, dimensions):
    """
    For a given array, generate an array of xyz coordinates in the cartesian basis
    """

    # Generate a grid of crystal coordinates
    i = np.linspace(0., 1., dimensions[0])
    j = np.linspace(0., 1., dimensions[1])
    k = np.linspace(0., 1., dimensions[2])
    # Generate NxN arrays of crystal coords
    iii, jjj, kkk = np.meshgrid(i, j, k, indexing='ij')
    # Flatten this to a 3xNN array
    ijk_array = np.array([iii.ravel(), jjj.ravel(), kkk.ravel()])
    # Change the crystal basis to a cartesian basis
    xyz_array = np.dot(cell_matrix.T, ijk_array)

    return xyz_array


def generate_charge_model(cell_matrix, peak_charge=None):
    """
    Return a function to compute a periodic gaussian on a grid.
    The returned function can be used for fitting.

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell
    peak_charge: float
        The peak charge density at the centre of the gaussian.
        Used for scaling the result.

    Returns
    -------
    compute_charge
        A function that will compute a periodic gaussian on a grid
        for a given cell and peak charge intensity
    """

    def compute_charge(
        xyz_real,
        x0, y0, z0,
        sigma_x, sigma_y, sigma_z,
        cov_xy, cov_xz, cov_yz):
        """
        For a given system charge, create a model charge distribution using
        an anisotropic periodic 3D gaussian.
        The charge model for now is a Gaussian.

        NOTE:
        The values for sigma and cov are not the values used in construction
        of the Gaussian. After the covariance matrix is constructed, its
        transpose is multiplied by itself (that is to construct a Gram matrix)
        to ensure that it is positive-semidefinite. It is this matrix which is
        the real covariance matrix. This transformation is to allow this
        function to be used directly by the fitting algorithm without a danger
        of crashing.

        Parameters
        ----------
        xyz_real: 3xN array
            Coordinates to compute the Gaussian for in cartesian coordinates.
        x0, y0, z0: float
            Center of the Gaussian in crystal coordinates.
        sigma_x, sigma_y, sigma_z: float
            Spread of the Gaussian (not the real values used, see note above).
        cov_xy, cov_xz, cov_yz: float
            Covariance values controlling the rotation of the Gaussian
            (not the real values used, see note above).

        Returns
        -------
        g
            Values of the Gaussian computed at all of the desired coordinates and
            scaled by the value of charge_integral.

        """

        # Construct the pseudo-covariance matrix
        V = np.array([[sigma_x, cov_xy, cov_xz],[cov_xy, sigma_y, cov_yz], [cov_xz, cov_yz, sigma_z]])
        # Construct the actual covariance matrix in a way that is always positive semi-definite
        covar = np.dot(V.T, V)

        gauss_position = np.array([x0, y0, z0])

        # Apply periodic boundary conditions
        g = 0
        for ii in [-1, 0, 1]:
            for jj in [-1, 0, 1]:
                for kk in [-1, 0, 1]:
                    # Compute the periodic origin in crystal coordinates
                    origin_crystal = (gauss_position + np.array([ii, jj, kk])).reshape(3,1)
                    # Convert this to cartesian coordinates
                    origin_real = np.dot(cell_matrix.T, origin_crystal)
                    # Compute the Gaussian centred at this position
                    g = g + get_gaussian_3d(xyz_real.T, origin_real, covar)

        print("DEBUG: Integrated charge density (unscaled) = {}".format(get_integral(g, cell_matrix)))

        print("DEBUG: g.max()  = {}".format(g.max()))
        # Scale the result to match the peak charge density
        if peak_charge:
            g = g * (peak_charge / g.max())
        print("DEBUG: Peak Charge target  = {}".format(peak_charge))
        print("DEBUG: Peak Charge scaled  = {}".format(g.max()))
        print("DEBUG: Integrated charge density (scaled) = {}".format(get_integral(g, cell_matrix)))

        return g

    return compute_charge


@calcfunction
def get_charge_model(cell_matrix, defect_charge, dimensions, gaussian_params, peak_charge=None):
    """
    For a given system charge, create a model charge distribution.

    Parameters
    ----------
    cell_matrix: 3x3 array
        Cell matrix of the real space cell.
    peak_charge : float
        The peak charge density at the centre of the gaussian.
    defect_charge : float
        Charge state of the defect
    dimensions: 3x1 array-like
        Dimensions of grid to compute charge on.
    gaussian_params: list (length 9)
        Parameters determining the distribution position and shape obtained
        by the fitting procedure.

    Returns
    -------
    model_charge_array
        The grid with the charge data as an AiiDA ArrayData object

    """

    cell_matrix = cell_matrix.get_array('cell_matrix')
    if peak_charge:
        peak_charge = peak_charge.value
    defect_charge = defect_charge.value
    dimensions = np.array(dimensions)
    gaussian_params = gaussian_params.get_list()

    xyz_coords = get_xyz_coords(cell_matrix, dimensions)

    get_model = generate_charge_model(cell_matrix, peak_charge)
    g = get_model(xyz_coords, *gaussian_params)

    # Unflatten the array
    g = g.reshape(dimensions)

    print("DEBUG: fit params: {}".format(gaussian_params))

    # Rescale to defect charge
    print("DEBUG: Integrated charge density target  = {}".format(defect_charge))
    g = g * (defect_charge / get_integral(g, cell_matrix))
    print("DEBUG: Integrated charge density (scaled) = {}".format(get_integral(g, cell_matrix)))

    # Compensating jellium background
    # g = g - np.sum(g)/np.prod(g.shape)
    # print("DEBUG: Integrated charge density (jellium) = {}".format(get_integral(g, cell_matrix)))

    # Pack the array
    model_charge_array = orm.ArrayData()
    model_charge_array.set_array('model_charge', g)

    return model_charge_array


# @calcfunction
# def get_charge_model_old(limits,
#                      dimensions,
#                      defect_position,
#                      sigma=orm.Float(1.0),
#                      charge=orm.Float(-1.0)):
#     """
#     For a given system charge, create a model charge distribution.
#     The charge model for now is a Gaussian.
#     Grid = coord grid
#     TODO: Change of basis
#     """
#     limits = limits.get_list()
#     dimensions = dimensions.get_list()
#     defect_position = defect_position.get_list()
#     sigma = sigma.value
#     charge = charge.value

#     print("DEBUG: Dimensions = {}".format(dimensions))
#     print("DEBUG: Limits = {}".format(limits))

#     i = np.linspace(0, limits[0], dimensions[0])
#     j = np.linspace(0, limits[1], dimensions[1])
#     k = np.linspace(0, limits[2], dimensions[2])
#     grid = np.meshgrid(i, j, k)

#     # Get the gaussian at the defect position
#     g = get_gaussian_3d(grid, defect_position, sigma)

#     # Get the offsets
#     offsets = np.zeros(3)
#     for axis in range(3):
#         # Capture the offset needed for an image
#         if defect_position[axis] > limits[axis] / 2.0:
#             offsets[axis] = -limits[axis]
#         else:
#             offsets[axis] = +limits[axis]

#     # Apply periodic boundary conditions
#     g = 0
#     for dim0 in range(2):
#         for dim1 in range(2):
#             for dim2 in range(2):
#                 image_offset = [dim0, dim1, dim2] * offsets
#                 g = g + get_gaussian_3d(
#                     grid, defect_position + image_offset, sigma=sigma)

#     # Scale the charge density to the desired charge
#     #int_charge_density = np.trapz(np.trapz(np.trapz(g, i), j), k)
#     int_charge_density = get_integral(g, dimensions, limits)
#     print(
#         "DEBUG: Integrated charge density (g) = {}".format(int_charge_density))

#     g = g / (int_charge_density / charge)

#     # Compensating jellium background
#     print("DEBUG: Integrated charge density (scaled_g) = {}".format(
#         get_integral(g, dimensions, limits)))

#     #scaled_g = scaled_g - np.sum(scaled_g)/np.prod(scaled_g.shape)
#     print("DEBUG: Integrated charge density (jellium) = {}".format(
#         get_integral(g, dimensions, limits)))

#     # Pack the array
#     model_charge_array = orm.ArrayData()
#     model_charge_array.set_array('model_charge', g)

#     return model_charge_array

def get_gaussian_3d(grid, origin, covar):
    """
    Compute anisotropic 3D Gaussian on grid

    Parameters
    ----------
    grid: array
        Array on which to compute gaussian
    origin: array
        Centre of gaussian
    covar: 3x3 array
        Covariance matrix of gaussian

    Returns
    -------
    gaussian
        anisotropic Gaussian on grid
    """

    origin = origin.ravel()
    gaussian = multivariate_normal.pdf(grid, origin, covar)

    return gaussian


# def get_gaussian_3d_old(grid, position, sigma):
#     """
#     Calculate 3D Gaussian on grid
#     NOTE: Minus sign at front give negative values of charge density throughout the cell
#     """
#     x = grid[0] - position[0]
#     y = grid[1] - position[1]
#     z = grid[2] - position[2]

#     gaussian = -np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2)) / (
#         (2.0 * np.pi)**1.5 * np.sqrt(sigma))

#     return gaussian


def get_integral(data, cell_matrix):
    """
    Get the integral of a uniformly spaced 3D data array by rectangular rule.
    Works better than trapezoidal or Simpson's rule for sharpely peaked coarse grids.
    """
    a = cell_matrix[0]
    b = cell_matrix[1]
    c = cell_matrix[2]
    cell_vol = np.dot(np.cross(a, b), c)
    element_volume = cell_vol / np.prod(data.shape)
    return np.sum(data) * element_volume


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
    epsilon: array
        3x3 dielectric tensor

    Returns
    -------
    V_model_r
        The calculated model potential in real space array
    """

    dimensions = np.array(dimensions)
    cell_matrix = cell_matrix.get_array('cell_matrix')
    charge_density = charge_density.get_array('model_charge')
    epsilon = epsilon.get_array('epsilon')

    # Set up a reciprocal space grid for the potential
    # Prepare coordinates in a 3D array of ijk vectors
    # (Note: Indexing is column major order, but the 4th dimension vectors remain ijk)
    dimensions = dimensions // 2  #floor division

    ijk_array = np.mgrid[
        -dimensions[0]:dimensions[0] + 1,
        -dimensions[1]:dimensions[1] + 1,
        -dimensions[2]:dimensions[2] + 1].T

    # Get G vectors
    G_array = np.dot(ijk_array, (cell_matrix.T))
    G_array_shape = G_array.shape[0:3] # Drop last axis - we know that each vector is len 3

    # Compute permittivity for all g-vectors
    dielectric = []
    for gvec in G_array.reshape(-1,3):
        dielectric.append(gvec@epsilon@gvec.T)
    dielectric = np.array(dielectric).reshape(G_array_shape).T

    # Get the reciprocal space charge density
    charge_density_g = get_fft(charge_density)

    # Compute the model potential
    v_model = np.divide(
        charge_density_g, dielectric, where=dielectric != 0.0)
    V_model_g = 4. * np.pi * v_model

    # Set the component G=0 to zero
    V_model_g[dimensions[0] + 1, dimensions[1] + 1, dimensions[2] + 1] = 0.0

    # Get the model potential in real space
    # V_model_r = get_inverse_fft(V_model_g) * CONSTANTS.hartree_to_ev
    V_model_r = get_inverse_fft(V_model_g)

    # Pack up the array
    V_model_array = orm.ArrayData()
    V_model_array.set_array('model_potential', V_model_r)

    return V_model_array


@calcfunction
def get_energy(potential, charge_density, cell_matrix):
    """
    Calculate the total energy for a given model potential
    """
    cell_matrix = cell_matrix.get_array('cell_matrix')
    # potential = potential.get_array('model_potential')
    potential = potential.get_array('model_potential') * CONSTANTS.hartree_to_ev
    charge_density = charge_density.get_array('model_charge')

    energy = np.real(0.5 * get_integral(charge_density*potential, cell_matrix))
    return orm.Float(energy)

