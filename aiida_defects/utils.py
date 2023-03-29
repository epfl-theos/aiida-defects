# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
import numpy as np

from qe_tools import CONSTANTS

# This a collection of common, generic methods for common tasks

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

def calc_pair_distance_xyz(cellmat,ri,rj):
    """"
    Calculate the distance between two atoms accross periodic boundary conditions
    starting from cartesian coords.
    Uses the general algorithm for the minimum image (Appendix B - Eq 9.) from:
    M. E. Tuckerman. Statistical Mechanics: Theory and Molecular Simulation.
    Oxford University Press, Oxford, UK, 2010.

    Parameters
    ----------
        cellmat_inv - 3x3 matrix
            The inverse of the 3x3 matrix describing the simulation cell
        ri,rj - 3x1 vector
            numpy vectors describing the position of atoms i and j

    Returns
    ---------
        dist - float
            The distance between the atoms i and j, according the minimum image
            convention.

    """
    si=np.dot(cellmat_inv,ri)
    sj=np.dot(cellmat_inv,rj)
    sij=si-sj
    sij=sij-np.rint(sij)
    rij=np.dot(cellmat,sij)

    # Get the magnitude of the vector
    dist=np.sqrt(np.dot(rij,rij))

    return dist

def get_grid(dimensions, endpoint=True):
    """
    Generate an array of coordinates
    """
    # Generate a grid of coordinates
    i = np.linspace(0., 1., dimensions[0], endpoint)
    j = np.linspace(0., 1., dimensions[1], endpoint)
    k = np.linspace(0., 1., dimensions[2], endpoint)
    # Generate NxN arrays of coords
    iii, jjj, kkk = np.meshgrid(i, j, k, indexing='ij')
    # Flatten this to a 3xNN array
    ijk_array = np.array([iii.ravel(), jjj.ravel(), kkk.ravel()])

    return ijk_array

def get_xyz_coords(cell_matrix, dimensions):
    """
    For a given array, generate an array of xyz coordinates in the cartesian basis
    """
    ijk_array = get_grid(dimensions)
    # Change the crystal basis to a cartesian basis
    xyz_array = np.dot(cell_matrix.T, ijk_array)

    return xyz_array