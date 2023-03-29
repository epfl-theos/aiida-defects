# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

"""
Utility functions for the gaussian countercharge workchain
"""

# def is_gaussian_isotrope(gaussian_params):
#     eps = 0.01
#     average_sigma = np.mean(gaussian_params[:3])
#     #check if the off-diagonal elements sigma_xy, sigma_xz and simga_yz are all close to zero
#     test_1 = all(np.array(gaussian_params[3:])/average_sigma < eps)
#     test_2 = all(abs((np.array(gaussian_params[:3])/average_sigma) - 1.0) < eps)
#     return test_1 and test_2

def is_isotrope(matrix, tol=0.01):
    '''
    check if a 3x3 matrix is isotropic i,e. it can be written as a product of a number and an identity matrix
    '''

    # check if all diagonal elements are the same
    diagonal = np.diagonal(matrix)
    test_1 = np.all(np.abs([diagonal[0]-diagonal[1], diagonal[1]-diagonal[2], diagonal[0]-diagonal[2]]) < tol)

    #check if all the off-diagonal are zero (within the tolerance, tol)
    test_2 = np.all(np.abs(matrix - np.diag(np.diagonal(matrix))) < tol)

    return test_1 and test_2

@calcfunction
def create_model_structure(base_structure, scale_factor):
    """
    Prepare a model structure with a give scale factor, based on a base_structure
    """
    base_cell = np.array(base_structure.cell)
    model_cell = base_cell * scale_factor
    model_structure = orm.StructureData(cell=model_cell)

    return model_structure


# @calcfunction
# def get_total_alignment(alignment_dft_model, alignment_q0_host, charge):
#     """
#     Calculate the total potential alignment

#     Parameters
#     ----------
#     alignment_dft_model: orm.Float
#         The correction energy derived from the alignment of the DFT difference
#         potential and the model potential
#     alignment_q0_host: orm.Float
#         The correction energy derived from the alignment of the defect
#         potential in the q=0 charge state and the potential of the pristine
#         host structure
#     charge: orm.Float
#         The charge state of the defect

#     Returns
#     -------
#     total_alignment
#         The calculated total potential alignment

#     """

#     # total_alignment = -1.0*(charge * alignment_dft_model) + (
#     #     charge * alignment_q0_host)

#     # The minus sign is incorrect. It is remove in the corrected formula below:
#     total_alignment = charge * alignment_dft_model + (
#         charge * alignment_q0_host)

#     return total_alignment

# @calcfunction
# def get_alignment(alignment_q_host_to_model, charge):
#     """
#     Calculate the total potential alignment

#     Parameters
#     ----------
#     alignment_q_host_to_model: orm.Float
#         The correction energy derived from the alignment of the DFT difference
#         potential of the charge defect and the host to the model potential
#     charge: orm.Float
#         The charge state of the defect

#     Returns
#     -------
#     total_alignment
#         The calculated total potential alignment
#     """

#     alignment = charge.value * alignment_q_host_to_model.value

#     return orm.Float(alignment)


@calcfunction
def get_total_correction(model_correction, potential_alignment, defect_charge):
    """
    Calculate the total correction, including the potential alignments

    Parameters
    ----------
    model_correction: orm.Float
        The correction energy derived from the electrostatic model
    potential_alignment: orm.Float
        The correction derived from the alignment of the difference of DFT
        potential of the charge defect and the pristine host structure to the model potential

    Returns
    -------
    total_correction
        The calculated correction, including potential alignment

    """

    total_correction = model_correction - defect_charge * potential_alignment

    return total_correction


@calcfunction
def fit_energies(dimensions_dict, energies_dict):
    """
    Fit the model energies

    Parameters
    ----------
    energies_dict : Dict (orm.StructureData : orm.Float)
        AiiDA dictionary of the form: structure : energy
    """

    from scipy.optimize import curve_fit

    def fitting_func(x, a, b, c):
        """
        Function to fit:
        E = a*Omega^(-3) + b*Omega^(-1) + c
        Where:
            Omega is the volume of the cell
            a,b,c are parameters to be fitted

        Parameters
        ----------
        x: Float
            Linear cell dimension
        a,b,c: Float
            Parameters to fit
        """
        return a * x + b * x**3 + c

    dimensions_dict = dimensions_dict.get_dict()
    energies_dict = energies_dict.get_dict()

    # Sort these scale factors so that they are in ascending order
    #keys_list = dimensions_dict.keys()
    #keys_list.sort()

    linear_dim_list = []
    energy_list = []
    # Unpack the dictionaries:
    for key in sorted(dimensions_dict.keys()):
        linear_dim_list.append(dimensions_dict[key])
        energy_list.append(energies_dict[key])

    # Fit the data to the function

    fit_params = curve_fit(fitting_func, np.array(linear_dim_list),
                           np.array(energy_list))[0]

    # Get the isolated model energy at linear dimension = 0.0
    isolated_energy = fitting_func(0.0, *fit_params)

    return orm.Float(isolated_energy)


@calcfunction
def calc_correction(isolated_energy, model_energy):
    """
    Get the energy correction for each model size

    Parameters
    ----------
    isolated_energy: orm.Float
        The estimated energy of the isolated model
    *model_energies: orm.Float objects
        Any number of calculated model energies
    """
    correction_energy = isolated_energy - model_energy

    return orm.Float(correction_energy)


@calcfunction
def get_charge_model_fit(rho_host, rho_defect_q, host_structure):
    """
    Fit the charge model to the defect data

    Parameters
    ----------
    model_correction: orm.Float
        The correction energy derived from the electrostatic model
    total_alignment: orm.Float
        The correction energy derived from the alignment of the DFT difference
        potential and the model potential, and alignment of the defect potential
        in the q=0 charge state and the potential of the pristine host structure

    Returns
    -------
    total_correction
        The calculated correction, including potential alignment

    """

    from scipy.optimize import curve_fit
    from .model_potential.utils import generate_charge_model, get_xyz_coords, get_cell_matrix

    # Get the cell matrix
    cell_matrix = get_cell_matrix(host_structure)

    # Compute the difference in charge density between the host and defect systems
    rho_defect_q_data = rho_defect_q.get_array(rho_defect_q.get_arraynames()[0])
    rho_host_data = rho_host.get_array(rho_host.get_arraynames()[0])

    # Charge density from QE is in e/cubic-bohr, so convert if necessary
    # TODO: Check if the CUBE file format is strictly Bohr or if this is a QE thing
    #rho_diff = (rho_host_data - rho_defect_q_data)/(bohr_to_ang**3)
    rho_diff = rho_host_data - rho_defect_q_data

    # Detect the centre of the charge in the data
    max_pos_mat = np.array(np.unravel_index(rho_diff.argmax(), rho_diff.shape)) # matrix coords
    max_pos_ijk = (max_pos_mat*1.)/(np.array(rho_diff.shape)-1) # Compute crystal coords
    max_i = max_pos_ijk[0]
    max_j = max_pos_ijk[1]
    max_k = max_pos_ijk[2]

    # Generate cartesian coordinates for a grid of the same size as the charge data
    xyz_coords = get_xyz_coords(cell_matrix, rho_diff.shape)

    # Set up some safe parameters for the fitting
    guesses = [max_i, max_j, max_k, 1., 1., 1., 0., 0., 0.]
    bounds = (
        [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
        [1., 1., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    peak_charge = rho_diff.max()

    # Do the fitting
    fit, covar_fit = curve_fit(
        generate_charge_model(cell_matrix, peak_charge),
        xyz_coords,
        rho_diff.ravel(),
        p0=guesses,
        bounds=bounds)

    # Compute the one standard deviation errors from the 9x9 covariance array
    fit_error = np.sqrt(np.diag(covar_fit))

    fitting_results = {}

    fitting_results['fit'] = fit.tolist()
    fitting_results['peak_charge'] = peak_charge
    fitting_results['error'] = fit_error.tolist()

    return orm.Dict(dict=fitting_results)


