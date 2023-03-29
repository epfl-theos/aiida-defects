# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
"""Tests for the `ModelPotentialWorkchain` class."""
import pytest
import numpy as np

from aiida.common import AttributeDict
from aiida.orm import Float, List, ArrayData, Dict, Int, StructureData
from aiida_defects.formation_energy.corrections.gaussian_countercharge.model_potential.model_potential import ModelPotentialWorkchain

@pytest.fixture
def generate_inputs_model_potential(generate_structure, generate_array_data):
    """Generate default inputs for `ModelPotentialWorkchain`"""

    def _generate_inputs_model_potential():
        """Generate default inputs for `ModelPotentialWorkchain`"""

        mock_array = generate_array_data(3)

        inputs = {
            'peak_charge': Float(1.0),
            'defect_charge': Float(1.0),
            'scale_factor': Int(2),
            'host_structure': generate_structure(),
            'defect_site': List(list=[0.5,0.5,0.5]),
            'epsilon': mock_array,
            'gaussian_params': List(list=[1.,1.,1.,1.,1.,1.,1.,1.,1.])
        }

        return inputs

    return _generate_inputs_model_potential



@pytest.fixture
def generate_workchain_model_potential(generate_workchain, generate_inputs_model_potential):
    """Generate an instance of a `ModelPotentialWorkchain`."""

    def _generate_workchain_model_potential(exit_code=None):
        entry_point = 'defects.formation_energy.corrections.gaussian_countercharge.model_potential'
        inputs = generate_inputs_model_potential()
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_model_potential

def test_get_model_structure(aiida_profile, generate_workchain_model_potential):
    """
    Test `ModelPotentialWorkchain.get_model_structure`.
    This checks that we can create the workchain successfully, and that model structure
    is created correctly.
    """
    from numpy import ndarray

    process = generate_workchain_model_potential()
    process.get_model_structure()

    assert isinstance(process.ctx.model_structure, StructureData)
    assert isinstance(process.ctx.real_cell, ndarray)
    assert isinstance(process.ctx.reciprocal_cell, ndarray)
    assert isinstance(process.ctx.limits, List)