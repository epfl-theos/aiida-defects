# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit

from .utils import (
    get_raw_formation_energy,
    get_corrected_formation_energy,
    get_corrected_aligned_formation_energy,
)

class FormationEnergyWorkchainBase(WorkChain):
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
        super(FormationEnergyWorkchainBase, cls).define(spec)
        # fmt: off
        # Structures
        spec.input(
            "host_structure",
            valid_type=orm.StructureData,
            help="Pristine structure"
        )
        spec.input(
            "defect_structure",
            valid_type=orm.StructureData,
            help="Defective structure"
        )
        spec.input(
            "host_unitcell",
            valid_type=orm.StructureData,
            help="Pristine structure to use in the calculation of permittivity",
            required=False,
        )

        # Defect details
        spec.input(
             "defect_charge",
             valid_type=orm.Float,
             help="Defect charge state")
        spec.input(
            "defect_specie",
            valid_type=orm.Str)
        spec.input(
            "defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates" )
        #spec.input('ref_energy',valid_type=orm.Float)
        spec.input(
             "fermi_level",
             valid_type=orm.Float,
             default=lambda: orm.Float(0.0),
             help="Fermi level position with respect to the valence band maximum")
        # spec.input(
        #     "chemical_potential",
        #     valid_type=orm.Float,
        #     help="The chemical potential of the given defect type. The convention is that removing an atom is positive",
        # )
        spec.input("add_or_remove", valid_type=orm.Str,
                help="To determine the sign of the chemical potential. The convention is that removing an atom is negative")

        # Chemical potential
        spec.input('formation_energy_dict', valid_type=orm.Dict)
        spec.input('compound', valid_type=orm.Str)
        spec.input('dependent_element', valid_type=orm.Str)
        spec.input('tolerance', valid_type=orm.Float, default=lambda: orm.Float(1E-4))


        spec.input("run_dfpt", valid_type=orm.Bool)

        # Methodology
        spec.input(
            "correction_scheme",
            valid_type=orm.Str,
            help="The correction scheme to apply",
        )

        # Outputs
        spec.output(
            "formation_energy_uncorrected", valid_type=orm.Float, required=True
        )
        spec.output(
            "formation_energy_corrected", valid_type=orm.Float, required=True
        )
        spec.output(
            "formation_energy_corrected_aligned", valid_type=orm.Float, required=True
        )

        # Error codes
        spec.exit_code(201, "ERROR_INVALID_CORRECTION",
            message="The requested correction scheme is not recognised",
        )
        spec.exit_code(202, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly",
        )
        spec.exit_code(301, "ERROR_CORRECTION_WORKCHAIN_FAILED",
            message="The correction scheme sub-workchain failed",
        )
        spec.exit_code(302, "ERROR_DFT_CALCULATION_FAILED",
            message="DFT calculation failed",
        )
        spec.exit_code(303, "ERROR_PP_CALCULATION_FAILED",
            message="A post-processing calculation failed",
        )
        spec.exit_code(304, "ERROR_DFPT_CALCULATION_FAILED",
            message="DFPT calculation failed"
        )
        spec.exit_code(406, "ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED",
            message="The chemical potential calculation failed"
        )
        spec.exit_code(500, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly"
        )
        spec.exit_code(999, "ERROR_NOT_IMPLEMENTED",
            message="The requested method is not yet implemented",
        )
        # fmt: on

    def setup(self):
        """
        Setup the workchain
        """

        # Check if correction scheme is valid:
        correction_schemes_available = ["gaussian", "point"]
        if self.inputs.correction_scheme is not None:
            if self.inputs.correction_scheme not in correction_schemes_available:
                return self.exit_codes.ERROR_INVALID_CORRECTION

    def if_run_dfpt(self):
        return self.inputs.run_dfpt

    def correction_required(self):
        """
        Check if correction is requested
        """
        if self.inputs.correction_scheme is not None:
            return True
        else:
            return False

    def is_gaussian_scheme(self):
        """
        Check if Gaussian countercharge correction scheme is being used
        """
        return self.inputs.correction_scheme == "gaussian"

    def is_point_scheme(self):
        """
        Check if Point countercharge correction scheme is being used
        """
        return self.inputs.correction_scheme == "point"

    def host_unitcell_provided(self):
        """
        Check if a cell other than the host supercell is provided, such as a unitcell.
        An additional DFT calculation is required in this instance
        """
        if self.inputs.host_unitcell:
            return True
        else:
            return False

    def run_gaussian_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian_countercharge.gaussian_countercharge import (
            GaussianCounterChargeWorkchain,
        )

        self.report("Computing correction via the Gaussian Countercharge scheme")

        inputs = {
            "v_host": self.ctx.v_host,
            "v_defect_q0": self.ctx.v_defect_q0,
            "v_defect_q": self.ctx.v_defect_q,
            "rho_host": self.ctx.rho_host,
            "rho_defect_q": self.ctx.rho_defect_q,
            "defect_charge": self.inputs.defect_charge,
            "defect_site": self.inputs.defect_site,
            "host_structure": self.inputs.host_structure,
            "epsilon": self.ctx.epsilon,
        }

        workchain_future = self.submit(GaussianCounterChargeWorkchain, **inputs)
        label = "correction_workchain"
        self.to_context(**{label: workchain_future})

    def prepare_point_correction_workchain(self):
        """
        Get the required inputs for the Point Countercharge correction workchain

        TODO: Finish implementing this interface
        """
        return

    def run_point_correction_workchain(self):
        """
        Run the workchain for the Point Countercharge correction

        TODO: Finish implementing this interface
        """
        from .corrections.point_countercharge.point_countercharge import (
            PointCounterChargeWorkchain,
        )

        self.report("Computing correction via the Point Countercharge scheme")

        inputs = {}

        workchain_future = self.submit(PointCounterChargeWorkchain, **inputs)
        label = "correction_workchain"
        self.to_context(**{label: workchain_future})

    def check_correction_workchain(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        correction_wc = self.ctx["correction_workchain"]
        if not correction_wc.is_finished_ok:
            self.report(
                "Correction workchain failed with status {}".format(
                    correction_wc.exit_status
                )
            )
            return self.exit_codes.ERROR_CORRECTION_WORKCHAIN_FAILED
        else:
            self.ctx.total_correction = correction_wc.outputs.total_correction
            self.ctx.electrostatic_correction = (
                correction_wc.outputs.electrostatic_correction
            )
            self.ctx.total_alignment = correction_wc.outputs.total_alignment

    def run_chemical_potential_workchain(self):
        from .chemical_potential.chemical_potential import (
                ChemicalPotentialWorkchain, )

        self.report('Computing the chemical potential of {}'.format(self.inputs.defect_specie.value))
        inputs = {
            "formation_energy_dict": self.inputs.formation_energy_dict,
            "compound": self.inputs.compound,
            "dependent_element": self.inputs.dependent_element,
            "defect_specie": self.inputs.defect_specie,
            #"ref_energy": self.inputs.ref_energy,
            "tolerance": self.inputs.tolerance,
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
                "Chemical potential workchain failed with status {}".format(
                    chem_potential_wc.exit_status
                )
            )
            return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED
            #return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CORRECTION
        else:
            self.ctx.chemical_potential = chem_potential_wc.outputs.chemical_potential

    def compute_formation_energy(self):
        """
        Compute the formation energy
        """
        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_energy,
            self.ctx.host_energy,
            self.inputs.add_or_remove,
            self.ctx.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm
            )
        self.report(
            "The computed uncorrected formation energy is {} eV".format(
                self.ctx.e_f_uncorrected.value
            )
        )
        self.out("formation_energy_uncorrected", self.ctx.e_f_uncorrected)

        # Corrected formation energy
        self.ctx.e_f_corrected = get_corrected_formation_energy(
            self.ctx.e_f_uncorrected, self.ctx.electrostatic_correction
        )
        self.report(
            "The computed corrected formation energy is {} eV".format(
                self.ctx.e_f_corrected.value
            )
        )
        self.out("formation_energy_corrected", self.ctx.e_f_corrected)

        # Corrected formation energy with potential alignment
        self.ctx.e_f_corrected_aligned = get_corrected_aligned_formation_energy(
            self.ctx.e_f_corrected, self.ctx.total_alignment
        )
        self.report(
            "The computed corrected formation energy, including potential alignments, is {} eV".format(
                self.ctx.e_f_corrected_aligned.value
            )
        )
        self.out("formation_energy_corrected_aligned", self.ctx.e_f_corrected_aligned)

    def raise_not_implemented(self):
        """
        Raise a not-implemented error
        """
        return self.exit_codes.ERROR_NOT_IMPLEMENTED
