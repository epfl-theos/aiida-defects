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
        spec.input("host_structure",
                    valid_type = orm.StructureData,
                    help = "Pristine structure")

        spec.input("defect_structure",
                    valid_type = orm.StructureData,
                    help = "Defective structure")

        spec.input("host_unitcell",
                    valid_type = orm.StructureData,
                    help = "Pristine structure to use in the calculation of permittivity",
                    required = False)

        # Defect details
        spec.input("defect_charge",
                    valid_type = orm.Float,
                    help = "Defect charge state")

        spec.input("defect_specie",
                    valid_type = orm.Str,
                    required = False)

        spec.input("defect_site",
                    valid_type = orm.List,
                    help = "Defect site position in crystal coordinates" )

        spec.input("sigma",
                    valid_type = orm.Float,
                    required = False,
                    default = lambda: orm.Float(2.614),
                    help = "Gaussian Sigam")
        
        spec.input("epsilon", 
                    valid_type = orm.Float, 
                    help = "Dielectric constant of the host", 
                    required = False)
  
        #spec.input('ref_energy',valid_type=orm.Float)
        spec.input("fermi_level",
                    valid_type = orm.Float,
                    default = lambda: orm.Float(0.0),
                    help = "Fermi level position with respect to the valence band maximum")

        spec.input("chemical_potential",
                    valid_type = orm.Float,
                    help = "The chemical potential of the given defect type. The convention is that removing an atom is positive",
                    required = True)

        spec.input("add_or_remove", 
                   valid_type = orm.Str,
                   help = "To determine the sign of the chemical potential. The convention is that removing an atom is negative",
                   required = False)

        # Chemical potential
        spec.input('formation_energy_dict', valid_type=orm.Dict,required=False)
        spec.input('compound', valid_type=orm.Str,required=False)
        spec.input('dependent_element', valid_type=orm.Str,required=False )
        spec.input('tolerance', valid_type=orm.Float, default=lambda: orm.Float(1E-4),required=False)


        spec.input("run_dfpt", valid_type=orm.Bool,required=False)

        # Methodology
        spec.input("correction_scheme",
                   valid_type = orm.Str,
                   default = lambda: orm.Str('none'),
                   help = "The correction scheme to apply")

        spec.input("use_siesta_mesh_cutoff",
                    valid_type = orm.Bool,
                    default =lambda: orm.Bool(False),
                    required = False,
                    help = "Whether use Siesta Mesh size to Generate the Model Potential or Not ")
        spec.input("siesta_grid",
                    valid_type = orm.ArrayData,
                    required = False)
        # Outputs
        spec.output("formation_energy_uncorrected", valid_type=orm.Float, required=True)
        spec.output("formation_energy_corrected", valid_type=orm.Float, required=False  )
        spec.output("formation_energy_corrected_aligned", valid_type=orm.Float, required=False)

        # Error codes
        spec.exit_code(201, "ERROR_INVALID_CORRECTION",
            message="The requested correction scheme is not recognised",
        )
        spec.exit_code(202, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly",
        )
        spec.exit_code(203, "ERROR_INPUT_SIGMA_PARAMETER_NEEDED",
            message="Gaussaian Input parameter needed",
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
        self.report("Check if correction scheme is valid!")
        # Check if correction scheme is valid
        correction_schemes_available = ["gaussian-model","gaussian-rho", "point","rho","none"]
        if self.inputs.correction_scheme is not None:
            if self.inputs.correction_scheme not in correction_schemes_available:
                return self.exit_codes.ERROR_INVALID_CORRECTION

       
    def if_run_dfpt(self):
        """
        """
        return self.inputs.run_dfpt

    def correction_required(self):
        """
        Check if correction is requested
        """
        #if self.inputs.correction_scheme is not None:
        if self.inputs.correction_scheme=="gaussian-model" or self.inputs.correction_scheme=="point" or self.inputs.correction_scheme=="gaussian-rho":
            self.report(f"The {self.inputs.correction_scheme.value)} Corrections will be applied!")
            return True
        if self.inputs.correction_scheme=="none":
            self.report("Ther will be no Corrections applied")
            return False


    def is_charged_system(self):
        """
        """
        if self.inputs.defect_charge.value > 0 or self.inputs.defect_charge.value < 0 :
            self.report(f"System IS CHARGED with q={self.inputs.defect_charge.value}")
            return True
        else:
            self.report(f"System IS NOT CHARGED")
            return False


    def is_none_scheme(self):
        """
        Check if None correction scheme is being used
        """
        return self.inputs.correction_scheme == "none"

    #def is_sigma_in_inputs(self):
    #    """
    #
    #    """
    #    if "sigma" in self.inputs:
    #        return True
    #
    #    else:
    #        return self.exit_codes.ERROR_INPUT_SIGMA_PARAMETER_NEEDED


    def is_gaussian_model_scheme(self):
        """
        Check if Gaussian model countercharge correction scheme is being used
        """
        return self.inputs.correction_scheme == "gaussian-model"

    def is_gaussian_rho_scheme(self):
        """
        Check if Gaussian Rho countercharge correction scheme is being used
        """
        return self.inputs.correction_scheme == "gaussian-rho"

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

    def run_gaussian_model_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian_countercharge.gaussian_countercharge_model import (
            GaussianCounterChargeWorkchain,
        )

        self.report("Computing correction via the Gaussian Model Countercharge scheme")

        inputs = {
            "v_host": self.ctx.v_host,
            "v_defect_q0": self.ctx.v_defect_q0,
            "v_defect_q": self.ctx.v_defect_q,
            "siesta_grid":self.ctx.v_host_grid,
            "defect_charge": self.inputs.defect_charge,
            "defect_site": self.inputs.defect_site,
            "host_structure": self.inputs.host_structure,
            "epsilon": self.inputs.epsilon,
            "sigma": self.inputs.sigma,
            "use_siesta_mesh_cutoff": self.inputs.use_siesta_mesh_cutoff
        }

        workchain_future = self.submit(GaussianCounterChargeWorkchain, **inputs)
        label = "correction_workchain"
        self.to_context(**{label: workchain_future})

    def run_gaussian_rho_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian_countercharge.gaussian_countercharge_rho import (
            GaussianCounterChargeWorkchain,
        )

        self.report("Computing correction via the Gaussian Rho Countercharge scheme")

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

    def check_gaussian_model_correction_workchain(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        self.report("Checking Gaussian Model Correction Workchain ...")
        correction_wc = self.ctx["correction_workchain"]
        if not correction_wc.is_finished_ok:
            self.report(f"Correction workchain failed with status {correction_wc.exit_status}" )
            return self.exit_codes.ERROR_CORRECTION_WORKCHAIN_FAILED
        else:
            self.ctx.total_correction = correction_wc.outputs.total_correction
            self.ctx.electrostatic_correction = (correction_wc.outputs.electrostatic_correction)
            self.ctx.total_alignment = correction_wc.outputs.total_alignment

    def check_correction_workchain(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        correction_wc = self.ctx["correction_workchain"]
        if not correction_wc.is_finished_ok:
            self.report(f"Correction workchain failed with status {correction_wc.exit_status}")
            return self.exit_codes.ERROR_CORRECTION_WORKCHAIN_FAILED
        else:
            self.ctx.total_correction = correction_wc.outputs.total_correction
            self.ctx.electrostatic_correction = (correction_wc.outputs.electrostatic_correction)
            self.ctx.total_alignment = correction_wc.outputs.total_alignment

    def run_chemical_potential_workchain(self):
        from .chemical_potential.chemical_potential import ChemicalPotentialWorkchain 

        self.report(f'Computing the chemical potential of {self.inputs.defect_specie.value)}')
        
        inputs = {"formation_energy_dict": self.inputs.formation_energy_dict,
                  "compound": self.inputs.compound,
                  "dependent_element": self.inputs.dependent_element,
                  "defect_specie": self.inputs.defect_specie,
                  #"ref_energy": self.inputs.ref_energy,
                  "tolerance": self.inputs.tolerance,}

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
            self.report(f"Chemical potential workchain failed with status {chem_potential_wc.exit_status}")
            return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED
            #return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CORRECTION
        else:
            self.ctx.chemical_potential = chem_potential_wc.outputs.chemical_potential

    def compute_formation_energy_gaussian_model(self):
        """
        Compute the formation energy
        """
        self.report("Computing Formation Energy (Gaussian Model Correction) ...")
        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_energy,
            self.ctx.host_energy,
            self.inputs.add_or_remove,
            #self.ctx.chemical_potential,
            self.inputs.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm
            )
        self.report(f"The computed uncorrected formation energy is {self.ctx.e_f_uncorrected.value} eV")
        self.out("formation_energy_uncorrected", self.ctx.e_f_uncorrected)

        # Corrected formation energy
        self.ctx.e_f_corrected = get_corrected_formation_energy(self.ctx.e_f_uncorrected, 
                                                                self.ctx.electrostatic_correction)
        self.report(f"The computed corrected formation energy is {self.ctx.e_f_corrected.value} eV")
        self.out("formation_energy_corrected", self.ctx.e_f_corrected)

        # Corrected formation energy with potential alignment
        self.ctx.e_f_corrected_aligned = get_corrected_aligned_formation_energy(self.ctx.e_f_corrected, 
                                                                                self.ctx.total_alignment)
        self.report(f"The computed corrected formation energy, including potential alignments, is {self.ctx.e_f_corrected_aligned.value} eV")
        self.out("formation_energy_corrected_aligned", self.ctx.e_f_corrected_aligned)
 

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
        self.report(f"The computed uncorrected formation energy is {self.ctx.e_f_uncorrected.value}  eV")
        self.out("formation_energy_uncorrected", self.ctx.e_f_uncorrected)

        # Corrected formation energy
        self.ctx.e_f_corrected = get_corrected_formation_energy(self.ctx.e_f_uncorrected, 
                                                                self.ctx.electrostatic_correction)

        self.report(f"The computed corrected formation energy is {self.ctx.e_f_corrected.value} eV")
        self.out("formation_energy_corrected", self.ctx.e_f_corrected)

        # Corrected formation energy with potential alignment
        self.ctx.e_f_corrected_aligned = get_corrected_aligned_formation_energy(self.ctx.e_f_corrected, 
                                                                                self.ctx.total_alignment)
        self.report(f"The computed corrected formation energy, including potential alignments, is {self.ctx.e_f_corrected_aligned.value} eV")
        self.out("formation_energy_corrected_aligned", self.ctx.e_f_corrected_aligned)
    
    def compute_no_corrected_formation_energy(self):
        """

        """

        self.report("Working On..")
        self.report("Computing Non Corrected Formation Energy")
        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.defect_energy,
            self.ctx.host_energy,
            self.inputs.add_or_remove,
            self.inputs.chemical_potential,
            self.inputs.defect_charge,
            self.inputs.fermi_level,
            self.ctx.host_vbm )
        self.report(f"The computed uncorrected formation energy is {self.ctx.e_f_uncorrected.value} eV")
        self.out("formation_energy_uncorrected", self.ctx.e_f_uncorrected)
    
    def raise_not_implemented(self):
        """
        Raise a not-implemented error
        """
        self.report("DEBUG: Working On Implementation")
        return self.exit_codes.ERROR_NOT_IMPLEMENTED
