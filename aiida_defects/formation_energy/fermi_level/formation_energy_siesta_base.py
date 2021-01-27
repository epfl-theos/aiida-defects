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

       # aakhtar 
       # spec.input(
       #     "host_unitcell",
       #     valid_type=orm.StructureData,
       #     help="Pristine structure to use in the calculation of permittivity",
       #     required=False,
       # )

        # Defect details
        spec.input(
            "defect_charge", 
            valid_type=orm.Float, 
            help="Defect charge state")
        spec.input(
            "defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates",
        )
        spec.input(
            "fermi_level",
            valid_type=orm.Float,
            default=orm.Float(0.0),
            help="Fermi level position with respect to the valence band maximum",
        )
        spec.input(
            "chemical_potential",
            valid_type=orm.Float,
            help="The chemical potential of the given defect type. The convention is that removing an atom is positive",
        )

        # Methodology
        spec.input(
            "correction_scheme",
            valid_type=orm.Str,
            help="The correction scheme to apply",
        )
        #aakhtar
        spec.input(
                "gaussian_sigma",
                valid_type=orm.Float,
                help="the width fo gaussian sigma",
        )

        spec.input(
                "epsilon",
                valid_type=orm.Float,
                help="the width fo gaussian sigma",
        )
        #aakhtar

        
        # Outputs
        spec.output("neutral_formation_energy_uncorrected", 
                valid_type=orm.Float, 
                required=False
        )
        spec.output("charged_formation_energy_uncorrected", valid_type=orm.Float, required=False
        )
        
        spec.output(
            "formation_energy_uncorrected", valid_type=orm.Float, required=False
        )
        spec.output(
            "formation_energy_corrected", valid_type=orm.Float, required=False
        )
        spec.output(
            "formation_energy_corrected_aligned", valid_type=orm.Float, required=False
        )

        # Error codes
        spec.exit_code( 401, "ERROR_INVALID_CORRECTION",
            message="The requested correction scheme is not recognised",
        )
        spec.exit_code(402, "ERROR_CORRECTION_WORKCHAIN_FAILED",
            message="The correction scheme sub-workchain failed",
        )
        spec.exit_code(403, "ERROR_DFT_CALCULATION_FAILED", 
            message="DFT calculation failed",
        )
        spec.exit_code(404, "ERROR_PP_CALCULATION_FAILED",
            message="A post-processing calculation failed",
        )
        spec.exit_code(500, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly",
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
        self.report("Checking Formation Scheme")
        correction_schemes_available = ["gaussian","point","none"]
        if self.inputs.correction_scheme is not None:
            if self.inputs.correction_scheme not in correction_schemes_available:
                return self.exit_codes.ERROR_INVALID_CORRECTION
               

    def correction_required(self):
        """
        Check if correction is requested
        """
        #if self.inputs.correction_scheme is not None:
        if self.inputs.correction_scheme=="gaussian" or self.inputs.correction_scheme=="point": 
            self.report("Ther will be Corrections applied")
            return True
        else:
            self.report("Ther will be no Corrections applied")
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

    def is_none_scheme(self):
        """
        Check if there is none scheme for correction
        """
        return self.inputs.correction_scheme == "none"
  
    def run_gaussian_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian.gaussian_countercharge import (
            GaussianCounterChargeWorkchain,
        )

        self.report("Computing correction via the Gaussian Countercharge scheme")

        inputs = {
            "v_host": self.ctx.v_host,
            "v_defect_q0": self.ctx.v_defect_q0,
            "v_defect_q": self.ctx.v_defect_q,
            "defect_charge": self.inputs.defect_charge,
            "defect_site": self.inputs.defect_site,
            "host_structure": self.inputs.host_structure,
            "epsilon": self.input.epsilon
            #            "epsilon": self.ctx.epsilon,
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
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CORRECTION
        else:
            self.ctx.total_correction = correction_wc.outputs.total_correction
            self.ctx.electrostatic_correction = (
                correction_wc.outputs.electrostatic_correction
            )
            self.ctx.total_alignment = correction_wc.outputs.total_alignment

    
    
    
    #============================================
    # Compute Charged Formation Energy without Corrections
    #============================================    
    def compute_neutral_formation_energy(self):
        """ 
        Compute the formation energy without Correction
        """
        # Raw formation energy
        self.ctx.n_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_q0_energy,
            self.ctx.host_energy,
            self.inputs.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm
        )
        self.report(
            "The computed neutral formation energy without correction is {} eV".format(
                self.ctx.n_f_uncorrected.value
            )
        )
        self.out("neutral_formation_energy_uncorrected", self.ctx.n_f_uncorrected)

    #==============================================
    # Compute Charged Formation Energy without Corrections
    #=============================================  
    def compute_charged_formation_energy_no_corre(self):
        """ 
        Compute the formation energy without Correction
        """
        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_q_energy,
            self.ctx.host_energy,
            self.inputs.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm
        )
        self.report(
            "The computed charge {} e  formation energy without correction is {} eV".format(
                self.inputs.defect_charge.value ,self.ctx.e_f_uncorrected.value
            )
        )
        
        self.report(
               "The Grid Units is {}".format(
                   self.ctx.host_VT.grid_unit
                   )
         )

        
        self.out("charged_formation_energy_uncorrected", self.ctx.e_f_uncorrected)

    
    
    
    
    
    #============================================
    # Compute Formation Energy with Corrections
    #============================================    
    def compute_corrected_formation_energy(self):
        """ 
        Compute the formation energy
        """

        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_energy,
            self.ctx.host_energy,
            self.inputs.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm,
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
