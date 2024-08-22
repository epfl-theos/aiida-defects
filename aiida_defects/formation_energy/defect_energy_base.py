# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, if_

# from .corrections.gaussian_countercharge.gaussian_countercharge import GaussianCounterChargeWorkchain
from .chemical_potential.chemical_potential import ChemicalPotentialWorkchain
# from .utils import (
#     get_raw_formation_energy,
#     get_corrected_formation_energy,
#     get_corrected_aligned_formation_energy)



class DefectFormationEnergyWorkchainBase(WorkChain):
    """
    The base class to compute the formation energy for a given defect, containing the
    generic, code-agnostic methods, error codes, etc.

    Any computational code can be used to calculate the required energies and relative permittivity.
    However, different codes must be setup in specific ways, and so separate classes are used to implement these
    possibilities. This is an abstract class and should not be used directly, but rather the
    concrete code-specific classes should be used instead.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Structures
        spec.input(
            "host_structure",
            valid_type=orm.StructureData,
            help="Pristine structure"
        )
        # Chemical potential
        # spec.input('run_chem_pot_wc', valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.expose_inputs(ChemicalPotentialWorkchain, namespace='chemicalpotential')
        # spec.input('formation_energy_dict', required=False, valid_type=orm.Dict)
        # spec.input('compound', required=False, valid_type=orm.Str)
        # spec.input('dependent_element', required=False, valid_type=orm.Str)
        # spec.input("dopant_elements", valid_type=orm.List, default=lambda: orm.List(list=[]))
        # spec.input("ref_energy", valid_type=orm.Dict, required=False, help="The reference chemical potential of elements in the structure")
        # spec.input('tolerance', valid_type=orm.Float, default=lambda: orm.Float(1E-4))
        # spec.input(
        #      "chemical_potential",
        #      valid_type=orm.Dict, required=False,
        #      help="The chemical potential of the given defect type. The convention is that removing an atom is positive")
        
        # spec.outline(
        #     cls.run_chemical_potential_workchain,
        #     cls.check_chemical_potential_workchain
        # )
        # Outputs
        spec.expose_outputs(ChemicalPotentialWorkchain)
        # spec.output(
        #     "formation_energy_uncorrected", valid_type=orm.Float, required=True)
        # spec.output(
        #     "formation_energy_corrected", valid_type=orm.Float, required=True)
        # spec.output(
        #     "formation_energy_corrected_aligned", valid_type=orm.Float, required=True)

        # Error codes
        spec.exit_code(201, "ERROR_INVALID_CORRECTION",
            message="The requested correction scheme is not recognised",)
        spec.exit_code(202, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly",)
        spec.exit_code(301, "ERROR_CORRECTION_WORKCHAIN_FAILED",
            message="The correction scheme sub-workchain failed",)
        spec.exit_code(302, "ERROR_DFT_CALCULATION_FAILED",
            message="DFT calculation failed",)
        spec.exit_code(303, "ERROR_PP_CALCULATION_FAILED",
            message="A post-processing calculation failed",)
        spec.exit_code(304, "ERROR_DFPT_CALCULATION_FAILED",
            message="DFPT calculation failed")
        spec.exit_code(406, "ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED",
            message="The chemical potential calculation failed")
        spec.exit_code(500, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly")
        spec.exit_code(999, "ERROR_NOT_IMPLEMENTED",
            message="The requested method is not yet implemented")
        # fmt: on

    # def setup(self):
    #     """
    #     Setup the workchain
    #     """

    #     # Check if correction scheme is valid:
    #     correction_schemes_available = ["gaussian", "point"]
    #     if self.inputs.correction_scheme is not None:
    #         if self.inputs.correction_scheme not in correction_schemes_available:
    #             return self.exit_codes.ERROR_INVALID_CORRECTION

    def run_chemical_potential_workchain(self):
        """Submit the chemical potential workchain"""
        
        self.report('Submitting the chemical potential workchain')
        inputs = {
            "formation_energy_dict": self.inputs.chemicalpotential.formation_energy_dict,
            "compound": self.inputs.chemicalpotential.compound,
            "dependent_element": self.inputs.chemicalpotential.dependent_element,
            "dopant_elements": self.inputs.chemicalpotential.dopant_elements,
            "ref_energy": self.inputs.chemicalpotential.ref_energy,
            "tolerance": self.inputs.chemicalpotential.tolerance,
        }
        workchain_future = self.submit(ChemicalPotentialWorkchain, **inputs)
        label = "chemical_potential_workchain"
        self.to_context(**{label: workchain_future})

    def check_chemical_potential_workchain(self):
        """
        Check if the chemical potential workchain have finished correctly.
        If yes, assign the output to context
        """
        chem_potential_wc = self.ctx["chemical_potential_workchain"]
        if not chem_potential_wc.is_finished_ok:
            self.report(
                f'Chemical potential workchain failed with status {chem_potential_wc.exit_status}'
            )
            return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED
            #return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CORRECTION
        else:
            self.ctx.chemical_potential = chem_potential_wc.outputs.chemical_potential
            # self.out('chemical_potential', chem_potential_wc.outputs.chemical_potential)
            self.out_many(self.exposed_outputs(chem_potential_wc, ChemicalPotentialWorkchain))

    def raise_not_implemented(self):
        """
        Raise a not-implemented error
        """
        return self.exit_codes.ERROR_NOT_IMPLEMENTED
