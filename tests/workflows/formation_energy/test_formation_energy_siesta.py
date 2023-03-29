# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
"""Tests for the `FormationEnergyWorkchainSiesta` class."""
import pytest
from aiida.common import AttributeDict
from aiida.orm import Float, List, Dict, Int, Str
from aiida_defects.formation_energy.formation_energy_siesta import FormationEnergyWorkchainSiesta

@pytest.fixture
def generate_inputs_formation_energy_siesta(fixture_code, generate_structure, generate_kpoints_mesh, generate_array_data, generate_upf_data):
    """Generate default inputs for `FormationEnergyWorkchainSiesta`"""

    def _generate_inputs_formation_energy_siesta():
        """Generate default inputs for `FormationEnergyWorkchainSiesta`"""

        mock_structure = generate_structure()

        inputs = {
            "host_structure": mock_structure,
            "defect_structure": mock_structure,
            "host_unitcell": mock_structure,
            'defect_charge': Float(1.0),
            'defect_site': List(list=[0.5,0.5,0.5]),
            "fermi_level":  Float(1.0),
            "add_or_remove": Str('remove'),
            "formation_energy_dict": Dict(dict={}),
            "compound": Str("SiO2"),
            "dependent_element": Str("O"),
            "correction_scheme": Str('gaussian'),
            "run_dfpt": Bool(True),
            'run_pw_host': Bool(True),
            'run_pw_defect_q0': Bool(True),
            'run_pw_defect_q': Bool(True),
            "siesta": {
            }
        }

        return inputs

    return _generate_inputs_formation_energy_siesta



@pytest.fixture
def generate_workchain_formation_energy_siesta(generate_workchain, generate_inputs_formation_energy_siesta):
    """Generate an instance of a `FormationEnergyWorkchainSiesta` workchain."""

    def _generate_workchain_formation_energy_siesta(exit_code=None):
        entry_point = 'defects.formation_energy.siesta'
        inputs = generate_inputs_formation_energy_siesta()
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_formation_energy_siesta

@pytest.mark.skip(reason="Siesta version of workchain not implemented")
def test_setup(aiida_profile, generate_workchain_formation_energy_siesta):
    """
    Test `FormationEnergyWorkchainSiesta.setup`.
    This checks that we can create the workchain successfully, and that it is initialised into the correct state.
    """
    process = generate_workchain_formation_energy_siesta()
    process.setup()