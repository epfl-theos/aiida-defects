# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pkg_resources
import pytest
import six
from six.moves import range


class TestPoissonSolver(object):
    """
    Tests for the PoissonSolver module.
    """

    def test_poisson_solver(self, test_structures):
        """
        Test poisson solver
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.poisson_solver import poisson_solver
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import (
            get_cell_matrix, get_reciprocal_cell, get_charge_model)

        structure = test_structures['halite_unitcell']
        cell_matrix = get_cell_matrix(structure)
        reciprocal_cell_matrix = get_reciprocal_cell(cell_matrix)

        epsilon = 1.0
        dimensions = np.array([3, 3, 3])
        limits = structure.cell_lengths
        defect_position = [0., 0., 0.]
        rho_r = get_charge_model(
            limits, dimensions, defect_position, sigma=0.5)

        V_r_calculated = poisson_solver(reciprocal_cell_matrix, dimensions,
                                        rho_r, epsilon)
        V_r_reference = np.array(
            [[[
                -0.51066218 - 5.20183366e-01j, 0.01916153 - 2.33370250e-01j,
                -0.58731887 - 1.76723564e-01j
            ],
              [
                  0.01916153 - 2.33370250e-01j, 0.28862014 - 8.66086636e-02j,
                  -0.00475136 + 1.85037171e-17j
              ],
              [
                  -0.58731887 - 1.76723564e-01j, -0.00475136 - 1.85037171e-17j,
                  -0.58731887 + 1.76723564e-01j
              ]],
             [[
                 0.01916153 - 2.33370250e-01j, 0.28862014 - 8.66086636e-02j,
                 -0.00475136 + 1.85037171e-17j
             ],
              [
                  0.28862014 - 8.66086636e-02j, 0.47069923 - 5.55111512e-17j,
                  0.28862014 + 8.66086636e-02j
              ],
              [
                  -0.00475136 + 5.55111512e-17j, 0.28862014 + 8.66086636e-02j,
                  0.01916153 + 2.33370250e-01j
              ]],
             [[
                 -0.58731887 - 1.76723564e-01j, -0.00475136 - 9.25185854e-18j,
                 -0.58731887 + 1.76723564e-01j
             ],
              [
                  -0.00475136 - 2.77555756e-17j, 0.28862014 + 8.66086636e-02j,
                  0.01916153 + 2.33370250e-01j
              ],
              [
                  -0.58731887 + 1.76723564e-01j, 0.01916153 + 2.33370250e-01j,
                  -0.51066218 + 5.20183366e-01j
              ]]])
        print(V_r_calculated)
        print("Diff:")
        print(V_r_calculated - V_r_reference)

        np.testing.assert_array_almost_equal(
            V_r_calculated, V_r_reference, decimal=8)
