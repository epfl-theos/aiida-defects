# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
"""Tests for the `GaussianCounterChargeWorkchain` class."""
import pytest
from aiida.common import AttributeDict
from aiida.orm import Float, List, Dict, Int, Str
from aiida_defects.formation_energy.corrections.gaussian_countercharge.gaussian_countercharge import GaussianCounterChargeWorkchain

@pytest.fixture
def generate_inputs_gaussian_countercharge(generate_structure, generate_array_data):
    """Generate default inputs for `GaussianCounterChargeWorkchain`"""

    def _generate_inputs_gaussian_countercharge():
        """Generate default inputs for `GaussianCounterChargeWorkchain`"""

        mock_array = generate_array_data(3)

        inputs = {
            'host_structure' : generate_structure(),
            'defect_charge' : Float(-2.),
            'defect_site' : List(list=[0.5,0.5,0.5]),
            'epsilon' : mock_array,
            'v_host' : mock_array,
            'v_defect_q0' : mock_array,
            'v_defect_q' : mock_array,
            'rho_host' : mock_array,
            'rho_defect_q' : mock_array,
            'charge_model': {
                'model_type': Str('fitted'),
                'fitted': {}
            }
        }

        return inputs

    return _generate_inputs_gaussian_countercharge



@pytest.fixture
def generate_workchain_gaussian_countercharge(generate_workchain, generate_inputs_gaussian_countercharge):
    """Generate an instance of a `GaussianCounterChargeWorkchain`."""

    def _generate_workchain_gaussian_countercharge(exit_code=None):
        entry_point = 'defects.formation_energy.corrections.gaussian_countercharge'
        inputs = generate_inputs_gaussian_countercharge()
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_gaussian_countercharge

def test_setup(aiida_profile, generate_workchain_gaussian_countercharge):
    """
    Test `GaussianCounterChargeWorkchain.setup`.
    This checks that we can create the workchain successfully, and that it is initialised in to the correct state.
    """
    process = generate_workchain_gaussian_countercharge()
    process.setup()

    # assert process.ctx.restart_calc is None
    assert process.ctx.model_iteration.value == 0
    assert process.ctx.model_energies == {}
    assert process.ctx.model_structures == {}
    assert process.ctx.model_correction_energies == {}