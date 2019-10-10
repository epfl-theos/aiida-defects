# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
"""
Utility functions for the gaussian countercharge workchain
"""


@calcfunction
def create_model_structure(base_structure, scale_factor):
    """
    Prepare a model structure with a give scale factor, based on a base_structure
    """
    base_cell = np.array(base_structure.cell)
    model_cell = base_cell * scale_factor
    model_structure = orm.StructureData(cell=model_cell)

    return model_structure


@calcfunction
def get_total_alignment(alignment_dft_model, alignment_q0_host, charge):
    """
    Calculate the total potential alignment

    Parameters
    ----------
    alignment_dft_model: orm.Float
        The correction energy derived from the alignment of the DFT difference 
        potential and the model potential
    alignment_q0_host: orm.Float
        The correction energy derived from the alignment of the defect 
        potential in the q=0 charge state and the potential of the pristine 
        host structure
    charge: orm.Float
        The charge state of the defect

    Returns
    -------
    total_alignment
        The calculated total potential alignment
    
    """

    total_alignment = (charge * alignment_dft_model) + (
        charge * alignment_q0_host)

    return total_alignment


@calcfunction
def get_total_correction(model_correction, total_alignment):
    """
    Calculate the total correction, including the potential alignments

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

    total_correction = model_correction - total_alignment

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
    keys_list = dimensions_dict.keys()
    keys_list.sort()

    linear_dim_list = []
    energy_list = []
    # Unpack the dictionaries:
    for key in keys_list:
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
