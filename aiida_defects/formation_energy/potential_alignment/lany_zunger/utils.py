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
Utility functions for the Lany-Zunger potential alignment workchain
"""


@calcfunction
def get_spherical_potential(site, structure):
    """
    Computes the spherically averaged electrostatic potential for a
    given site, in a given structure

    Parameters
    ----------
    site: AiiDA Site object
        The atom site for which the spherical average should be computed
    structure: AiiDA StructureData object
        The structure for which the average should be computed



    """
    pass

    # def avg_potential_at_core(func_at_core, symbols):
    #     """
    #     Computes the average potential per type of atom in the structure
    #     :param func_at_core: dictionary with potential at each core extracted from the interpolation of the FFT grid
    #     :patam symbols: the list of symbols in the structure
    #     :result avg_atom_pot: average electrostatic potential for type of atom
    #     """

    #     #potential = func['func_at_core']
    #     #symbols = func['symbols']

    #     species = list(set(symbols))
    #     avg_pot_at_core = {}
    #     pot_at_core = []
    #     for specie in species:
    #         for atom, pot in six.iteritems(func_at_core):
    #             if atom.split('_')[0] == specie:
    #                 pot_at_core.append(pot)

    #         avg_pot_at_core[str(specie)] = np.mean(
    #             np.array(pot_at_core).astype(np.float))
    #         pot_at_core = []

    #     return avg_pot_at_core

    def interpolate_grid():
        """
        Interpolate grided data
        """
        from scipy.interpolate import griddata
        input_z = np.linspace(0, 6.648582943808 * (ii + 2),
                              model_potential_avg.shape[0])
        target_z = np.linspace(0, 6.648582943808 * (ii + 2),
                               dft_potential_avg.shape[0])
        interp_data = griddata(input_z, model_potential_avg, target_z)
