# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain


class PotentialAlignmentWorkchain(WorkChain):
    """
    Align two electrostatic potentials according to a specified method.
    """

    @classmethod
    def define(cls, spec):
        super(PotentialAlignmentLanyZunger, cls).define(spec)
        spec.input("host_structure", valid_type=StructureData)
        spec.input("defect_structure", valid_type=StructureData)
        spec.input("run_pw_host", valid_type=Bool, required=True)
        #spec.input_group('host_fftgrid', required=False)
        spec.input(
            "host_parent_folder",
            valid_type=(FolderData, RemoteData),
            required=False)
        spec.input(
            "host_parent_calculation",
            valid_type=PwCalculation,
            required=False)
        spec.input("run_pw_defect", valid_type=Bool, required=True)
        spec.input(
            "defect_parent_folder", valid_type=RemoteData, required=False)
        spec.input(
            "defect_parent_calculation",
            valid_type=PwCalculation,
            required=False)
        spec.input("code_pp", valid_type=Str)
        spec.input("code_pw", valid_type=Str, required=False)
        spec.input("pseudo_family", valid_type=Str, required=False)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData, required=False)
        spec.input(
            'parameters_pw_host', valid_type=ParameterData, required=False)
        spec.input(
            'parameters_pw_defect', valid_type=ParameterData, required=False)
        spec.input('parameters_pp', valid_type=ParameterData)
        spec.input(
            'alignment_type',
            valid_type=Str,
            required=False,
            default=Str('lany_zunger'))
        spec.outline(
            cls.validate_inputs,
            cls.run_host,
            cls.run_atomic_sphere_pot_host,
            cls.extract_values_host,
            cls.run_defect,
            cls.run_atomic_sphere_pot_defect,
            cls.extract_values_defect,
            cls.compute_alignment,
        )
        spec.dynamic_output()
