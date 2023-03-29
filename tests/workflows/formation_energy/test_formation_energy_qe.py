# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
"""Tests for the `FormationEnergyWorkchainQE` class."""
import pytest
from aiida.common import AttributeDict
from aiida.orm import Float, List, Dict, Int, Str, Bool
from aiida_defects.formation_energy.formation_energy_qe import FormationEnergyWorkchainQE

@pytest.fixture
def generate_inputs_formation_energy_qe(fixture_code, generate_structure, generate_kpoints_mesh, generate_array_data, generate_upf_data):
    """Generate default inputs for `FormationEnergyWorkchainQE`"""

    def _generate_inputs_formation_energy_qe():
        """Generate default inputs for `FormationEnergyWorkchainQE`"""

        mock_array = generate_array_data(3)
        mock_structure = generate_structure()
        mock_parameters = Dict(dict={})
        mock_kpoints = generate_kpoints_mesh(2)
        mock_pseudos = {'Si': generate_upf_data('Si')}

        inputs = {
            'run_pw_host': Bool(True),
            'run_pw_defect_q0': Bool(True),
            'run_pw_defect_q': Bool(True),
            'run_v_host': Bool(True),
            'run_v_defect_q0': Bool(True),
            'run_v_defect_q': Bool(True),
            'run_rho_host': Bool(True),
            'run_rho_defect_q0': Bool(True),
            'run_rho_defect_q': Bool(True),
            'run_dfpt': Bool(True),
            "host_structure": mock_structure,
            "defect_structure": mock_structure,
            "host_unitcell": mock_structure,
            'defect_charge': Float(1.0),
            'defect_species': Str('Si'),
            'defect_site': List(list=[0.5,0.5,0.5]),
            "fermi_level":  Float(1.0),
            "chempot_sign": Dict(dict={}),
            "formation_energy_dict": Dict(dict={}),
            "compound": Str("SiO2"),
            "dependent_element": Str("O"),
            "correction_scheme": Str('gaussian'),
            "run_dfpt": Bool(True),
            'epsilon' : mock_array,
            'run_pw_host': Bool(True),
            'run_pw_defect_q0': Bool(True),
            'run_pw_defect_q': Bool(True),
            "qe": {
                "dft": {
                    "supercell": {
                        "code": fixture_code('quantumespresso.pw'),
                        "parameters": mock_parameters,
                        "scheduler_options": mock_parameters,
                        "pseudopotential_family": Str("sssp"),
                        "settings": mock_parameters,
                    },
                    "unitcell": {
                        "code": fixture_code('quantumespresso.pw'),
                        "parameters": mock_parameters,
                        "scheduler_options": mock_parameters,
                        "pseudopotential_family": Str("sssp"),
                        "settings": mock_parameters,
                    },
                },
                "pp":{
                    "code": fixture_code('quantumespresso.pp'),
                    "scheduler_options": mock_parameters,
                },
                "dfpt":{
                    "code": fixture_code('quantumespresso.ph'),
                    "scheduler_options": mock_parameters,
                }
            }
        }

        return inputs

    return _generate_inputs_formation_energy_qe



@pytest.fixture
def generate_workchain_formation_energy_qe(generate_workchain, generate_inputs_formation_energy_qe):
    """Generate an instance of a `FormationEnergyWorkchainQE` workchain."""

    def _generate_workchain_formation_energy_qe(exit_code=None):
        entry_point = 'defects.formation_energy.qe'
        inputs = generate_inputs_formation_energy_qe()
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_formation_energy_qe

def test_setup(aiida_profile, generate_workchain_formation_energy_qe):
    """
    Test `FormationEnergyWorkchainQE.setup`.
    This checks that we can create the workchain successfully, and that it is initialised into the correct state.
    """
    process = generate_workchain_formation_energy_qe()
    process.setup()
