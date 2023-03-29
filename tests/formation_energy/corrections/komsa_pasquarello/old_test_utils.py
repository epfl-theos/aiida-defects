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


class TestUtils(object):
    """
    Tests for the utils module.
    """

    def test_get_cell_matrix(self, test_structures):
        """
        Test get_cell_matrix
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_cell_matrix
        unitcell_structure = test_structures['halite_unitcell']
        calculated_cell_matrix = get_cell_matrix(unitcell_structure)
        reference_cell_matrix = np.array([[10.305243520465277, 0., 0.],
                                          [0., 10.305243520465277, 0.],
                                          [0., 0., 10.305243520465277]])
        np.testing.assert_array_almost_equal(
            calculated_cell_matrix, reference_cell_matrix, decimal=15)

    def test_get_reciprocal_cell(self):
        """
        Test get_reciprocal_cell.
        Expected result - Creates correct reciprocal cell matrix
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_reciprocal_cell

        # Cubic cell
        real_space_cell = np.array([[10.0, 0., 0.], [0., 10., 0.],
                                    [0., 0., 10.]])
        reference_cell = np.array([[2 * np.pi / 10, 0., 0.],
                                   [0., 2 * np.pi / 10, 0.],
                                   [0., 0., 2 * np.pi / 10]])
        calculated_reciprocal_cell = get_reciprocal_cell(real_space_cell)
        np.testing.assert_array_equal(calculated_reciprocal_cell,
                                      reference_cell)

        # Triclinic cell
        # 8.5784 12.9600 7.2112 90.30 116.03 89.125
        real_space_cell = np.array(
            [[8.57840000e+00, 0.00000000e+00, 0.00000000e+00],
             [1.97912266e-01, 1.29584881e+01, 0.00000000e+00],
             [-3.16457176e+00, 1.05685888e-02, 6.47972018e+00]])
        reference_cell = np.array(
            [[7.32442566e-01, -1.11865953e-02, 3.57729344e-01],
             [0.00000000e+00, 4.84870387e-01, -7.91037971e-04],
             [0.00000000e+00, 0.00000000e+00, 9.69669726e-01]])
        calculated_reciprocal_cell = get_reciprocal_cell(real_space_cell)
        np.testing.assert_array_almost_equal(
            calculated_reciprocal_cell, reference_cell, decimal=6)

    def test_get_reciprocal_grid(self):
        """
        Test get_reciprocal_grid
        Expected result
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_reciprocal_grid

        reciprocal_cell = np.array([[2 * np.pi / 10, 0., 0.],
                                    [0., 2 * np.pi / 10, 0.],
                                    [0., 0., 2 * np.pi / 10]])

    def test_get_uniform_grid(self):
        """
        Get a uniform grid of coordinates of a given dimension
        """

    def test_get_charge_model(self):
        """
        Test get_charge_model
        Expected result
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_charge_model

        limits = [10., 10., 10.]
        dimensions = [50, 50, 50]
        position = [5., 5., 5.]

        coords = np.linspace(0., limits[0], dimensions[0])

        # Charge = -1
        g = get_charge_model(
            limits, dimensions, position, sigma=1.0, charge=-1.0)
        calculated_charge = np.trapz(
            np.trapz(np.trapz(g, coords), coords), coords)
        reference_charge = -1.0
        np.testing.assert_array_almost_equal(
            calculated_charge, reference_charge, decimal=15)

        # Charge = -3
        g = get_charge_model(
            limits, dimensions, position, sigma=1.0, charge=-3.0)
        calculated_charge = np.trapz(
            np.trapz(np.trapz(g, coords), coords), coords)
        reference_charge = -3.0
        np.testing.assert_array_almost_equal(
            calculated_charge, reference_charge, decimal=15)

        # Charge = -3 , sigma = 3.0
        g = get_charge_model(
            limits, dimensions, position, sigma=3.0, charge=-3.0)
        calculated_charge = np.trapz(
            np.trapz(np.trapz(g, coords), coords), coords)
        reference_charge = -3.0
        np.testing.assert_array_almost_equal(
            calculated_charge, reference_charge, decimal=15)

    def test_get_gaussian_3d(self):
        """
        Test for get_gaussian_3d
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_gaussian_3d

        i = np.linspace(0, 10., 100)
        j = np.linspace(0, 10., 100)
        k = np.linspace(0, 10., 100)
        position = [0., 0., 0.]
        grid = np.meshgrid(i, j, k)

        calculated_g = get_gaussian_3d(grid, position, 1.0)
        integral = np.trapz(np.trapz(np.trapz(calculated_g, i), j), k)
        np.testing.assert_almost_equal(integral, -0.125, decimal=15)

    def test_get_fft(self):
        """
        Test for get_gaussian_3d
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_fft

        x = np.linspace(-5., 5., 10)
        g = np.exp(-x**2 / 2) / (np.sqrt(2.0 * np.pi))
        g_k = get_fft(g)

        g_k_ref = np.array([
            0.89999979 + 0.j, -0.72947318 - 0.2370202j,
            0.38407615 + 0.27904766j, -0.12525384 - 0.17239712j,
            0.02065841 + 0.06358004j, 0. + 0.j, 0.02065841 - 0.06358004j,
            -0.12525384 + 0.17239712j, 0.38407615 - 0.27904766j,
            -0.72947318 + 0.2370202j
        ])

        np.testing.assert_array_almost_equal(g_k, g_k_ref)

    def test_get_inverse_fft(self):
        """
        Test for get_gaussian_3d
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_inverse_fft

        n = np.linspace(0., 1., 10)

    def test_get_cell_volume(self):
        """
        Test for get_cell_volume
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_cell_volume

        cell = np.array([[10.0, 0., 0.], [0., 10., 0.], [0., 0., 10.]])
        calculated_volume = get_cell_volume(cell)
        reference_volume = 1000.0
        np.testing.assert_equal(calculated_volume, reference_volume)

    def test_get_energy(self):
        """
        Test for get_energy
        """
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.poisson_solver import poisson_solver
        from aiida_defects.formation_energy.corrections.komsa_pasquarello.utils import get_energy, get_charge_model

        limits = [10., 10., 10.]
        dimensions = [100, 100, 100]
        position = [5., 5., 5.]
        cell = np.array([[10.0, 0., 0.], [0., 10., 0.], [0., 0., 10.]])

        coords = np.linspace(0., limits[0], dimensions[0])

        rho_r = get_charge_model(
            limits, dimensions, position, sigma=1.0, charge=-1.0)
        v_r = poisson_solver(cell, dimensions, rho_r, epsilon=1.0)

        energy_calculated = (get_energy(cell, v_r, rho_r, dimensions, limits))
        energy_reference = 0.0019787356882235594
        np.testing.assert_equal(energy_calculated, energy_reference)
