# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida import orm

from aiida.common.constants import hartree_to_ev
from aiida_defects.formation_energy.utils import run_pw_calculation

from .utils import get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy


class FormationEnergyWorkchain(WorkChain):
    """
    Compute the formation energy for a given defect.
    """

    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchain, cls).define(spec)
        spec.input(
            "host_structure",
            valid_type=orm.StructureData,
            help="Pristine structure")
        spec.input(
            "defect_structure",
            valid_type=orm.StructureData,
            help="Defective structure")
        spec.input(
            "defect_charge", valid_type=orm.Float, help="Defect charge state")
        spec.input(
            "defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates")
        spec.input(
            "correction_scheme",
            valid_type=orm.Str,
            help="The correction scheme to apply")
        spec.input(
            "pw_code",
            valid_type=orm.Code,
            help="The pw.x code to use for the calculations")
        spec.input(
            "kpoints",
            valid_type=orm.KpointsData,
            help="The k-point grid to use for the calculations")
        spec.input("pw_parameters", valid_type=orm.Dict, help="")
        spec.input("pw_scheduler_options", valid_type=orm.Dict, help="")
        spec.input_namespace(
            "pseudopotentials",
            valid_type=orm.UpfData,
            dynamic=True,
            help="The pseudopotential family for use with the code, if required"
        )
        spec.input(
            "pp_code",
            valid_type=orm.Code,
            help="The pp.x code to use for the calculations")
        spec.input(
            "pp_scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the pp.x calculations")

        spec.outline(
            cls.setup,
            if_(cls.correction_required)(
                if_(cls.is_gaussian_scheme)(
                    cls.prep_calcs_gaussian_correction,
                    cls.check_calcs_gaussian_correction,
                    cls.get_dft_potentials_gaussian_correction,
                    cls.check_dft_potentials_gaussian_correction,
                    cls.run_gaussian_correction_workchain),
                if_(cls.is_point_scheme)(
                    cls.prepare_point_correction_workchain,
                    cls.run_point_correction_workchain),
                cls.check_correction_workchain), cls.compute_formation_energy)

        spec.output(
            'formation_energy_uncorrected',
            valid_type=orm.Float,
            required=True)
        spec.output(
            'formation_energy_corrected', valid_type=orm.Float, required=True)
        spec.output(
            'formation_energy_corrected_aligned',
            valid_type=orm.Float,
            required=True)

        spec.exit_code(
            401,
            'ERROR_INVALID_CORRECTION',
            message='The requested correction scheme is not recognised')
        spec.exit_code(
            402,
            'ERROR_CORRECTION_WORKCHAIN_FAILED',
            message='The correction scheme sub-workchain failed')
        spec.exit_code(
            403,
            'ERROR_DFT_CALCULATION_FAILED',
            message='DFT calculation failed')
        spec.exit_code(
            404,
            'ERROR_PP_CALCULATION_FAILED',
            message='A post-processing calculation failed')


    def setup(self):
        """ 
        Setup the workchain
        """

        # Check if correction scheme is valid:
        correction_schemes_available = ['gaussian', 'point']
        if self.inputs.correction_scheme is not None:
            if self.inputs.correction_scheme not in correction_schemes_available:
                return self.exit_codes.ERROR_INVALID_CORRECTION

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
        return (self.inputs.correction_scheme == 'gaussian')

    def is_point_scheme(self):
        """
        Check if Point countercharge correction scheme is being used
        """
        return (self.inputs.correction_scheme == 'point')

    def prep_calcs_gaussian_correction(self):
        """
        Get the required inputs for the Gaussian Countercharge correction workchain.
        This method runs the required calculations to generate the energies and potentials 
        for the gaussian scheme.
        """

        from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

        self.report("Setting up the Gaussian Countercharge correction workchain")

        pw_inputs = self.inputs.pw_code.get_builder()
        pw_inputs.pseudos = self.inputs.pseudopotentials
        pw_inputs.kpoints = self.inputs.kpoints
        pw_inputs.metadata = self.inputs.pw_scheduler_options.get_dict()

        parameters = self.inputs.pw_parameters.get_dict()

        # Host structure
        pw_inputs.structure = self.inputs.host_structure
        parameters['SYSTEM']['tot_charge'] = orm.Float(0.)
        pw_inputs.parameters = orm.Dict(dict=parameters)

        future = self.submit(pw_inputs)
        self.report(
            'Launching PWSCF for host structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.host_structure.pk, "0.0", future.pk))
        self.to_context(**{'calc_host': future})

        # Defect structure; neutral charge state
        pw_inputs.structure = self.inputs.defect_structure
        parameters['SYSTEM']['tot_charge'] = orm.Float(0.)
        pw_inputs.parameters = orm.Dict(dict=parameters)

        future = self.submit(pw_inputs)
        self.report(
            'Launching PWSCF for defect structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.defect_structure.pk, "0.0", future.pk))
        self.to_context(**{'calc_defect_q0': future})

        # Defect structure; target charge state
        pw_inputs.structure = self.inputs.defect_structure
        parameters['SYSTEM']['tot_charge'] = self.inputs.defect_charge
        pw_inputs.parameters = orm.Dict(dict=parameters)

        future = self.submit(pw_inputs)
        self.report(
            'Launching PWSCF for defect structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.defect_structure.pk,
                    self.inputs.defect_charge.value, future.pk))
        self.to_context(**{'calc_defect_q': future})


    def check_calcs_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        host_calc = self.ctx['calc_host']
        if not host_calc.is_finished_ok:
            self.report(
                'PWSCF for the host structure has failed with status {}'.
                format(host_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

        # Defect (q=0)
        defect_q0_calc = self.ctx['calc_defect_q0']
        if not defect_q0_calc.is_finished_ok:
            self.report(
                'PWSCF for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

        # Defect (q=q)
        defect_q_calc = self.ctx['calc_defect_q']
        if not defect_q_calc.is_finished_ok:
            self.report(
                'PWSCF for the defect structure (with charge {}) has failed with status {}'
                .format(self.inputs.defect_charge.value,
                        defect_q_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

    def get_dft_potentials_gaussian_correction(self):
        """
        Obtain the electrostatic potentials from the PWSCF calculations. 
        """

        # Run a PP calc
        pp_inputs = self.inputs.pp_code.get_builder()
        pp_inputs.metadata = self.inputs.pp_scheduler_options.get_dict()
        pp_inputs.plot_number = orm.Int(11) # Elctrostatic potential
        pp_inputs.plot_dimension = orm.Int(3) # 3D

        pp_inputs.parent_folder = self.ctx['calc_host'].outputs.remote_folder    
        future = self.submit(pp_inputs)
        self.report(
            'Launching PP.x for host structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.host_structure.pk, "0.0", future.pk))
        self.to_context(**{'pp_host': future})

        pp_inputs.parent_folder = self.ctx['calc_defect_q0'].outputs.remote_folder    
        future = self.submit(pp_inputs)
        self.report(
            'Launching PP.x for defect structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.defect_structure.pk, "0.0", future.pk))
        self.to_context(**{'pp_defect_q0': future})

        pp_inputs.parent_folder = self.ctx['calc_defect_q'].outputs.remote_folder    
        future = self.submit(pp_inputs)
        self.report(
            'Launching PP.x for defect structure (PK={}) with charge {} (PK={})'
            .format(self.inputs.defect_structure.pk,
                    self.inputs.defect_charge.value, future.pk))
        self.to_context(**{'pp_defect_q': future})


    def check_dft_potentials_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        host_pp = self.ctx['pp_host']
        if not host_pp.is_finished_ok:
            self.report(
                'Post processing for the host structure has failed with status {}'.
                format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_host = v_data

        # Defect (q=0)
        defect_q0_pp = self.ctx['pp_defect_q0']
        if not defect_q0_pp.is_finished_ok:
            self.report(
                'Post processing for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_defect_q0 = v_data
            

        # Defect (q=q)
        defect_q_pp = self.ctx['pp_defect_q']
        if not defect_q_pp.is_finished_ok:
            self.report(
                'Post processing for the defect structure (with charge {}) has failed with status {}'
                .format(self.inputs.defect_charge.value,
                        defect_q_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_defect_q = v_data


    def run_gaussian_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian_countercharge.gaussian_countercharge import GaussianCounterChargeWorkchain

        self.report(
            "Computing correction via the Gaussian Countercharge scheme")

        inputs = {
            'v_host': self.ctx.v_host,
            'v_defect_q0': self.ctx.v_defect_q0,
            'v_defect_q': self.ctx.v_defect_q,
            'defect_charge': self.inputs.defect_charge,
            'defect_site': self.inputs.defect_site,
            'host_structure': self.inputs.host_structure,
            'epsilon': self.ctx.epsilon,
        }

        workchain_future = self.submit(GaussianCounterChargeWorkchain,
                                       **inputs)
        label = 'correction_workchain'
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
        from .corrections.point_countercharge.point_countercharge import PointCounterChargeWorkchain

        self.report("Computing correction via the Point Countercharge scheme")

        inputs = {}

        workchain_future = self.submit(PointCounterChargeWorkchain, **inputs)
        label = 'correction_workchain'
        self.to_context(**{label: workchain_future})


    def check_correction_workchain(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        correction_wc = self.ctx['correction_workchain']
        if not correction_wc.is_finished_ok:
            self.report('Correction workchain failed with status {}'.format(
                correction_wc.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_CORRECTION
        else:
            self.ctx.total_correction = correction_wc.outputs.total_correction
            self.ctx.electrostatic_correction = correction_wc.outputs.electrostatic_correction
            self.ctx.calculated_alignment = correction_wc.output.total_alignment


    def compute_formation_energy(self):
        """ 
        Compute the formation energy
        """

        # Raw formation energy
        self.ctx.e_f_uncorrected = get_raw_formation_energy(
            self.ctx.host_energy, self.ctx.defect_energy,
            self.ctx.chemical_potential, self.inputs.defect_charge,
            self.ctx.fermi_level)
        self.report(
            'The computed uncorrected formation energy is {} eV'.format(
                self.ctx.e_f_uncorrected.value * hartree_to_ev))
        self.out('formation_energy_uncorrected', self.ctx.e_f_uncorrected)

        # Corrected formation energy
        self.e_f_corrected = get_corrected_formation_energy(
            self.ctx.e_f_uncorrected, self.ctx.electrostatic_correction)
        self.report('The computed corrected formation energy is {} eV'.format(
            self.ctx.e_f_corrected.value * hartree_to_ev))
        self.out('formation_energy_corrected', self.ctx.e_f_corrected)

        # Corrected formation energy with potential alignment
        self.ctx.e_f_corrected_aligned = get_corrected_aligned_formation_energy(
            self.ctx.e_f_corrected, self.ctx.total_alignment)
        self.report(
            'The computed corrected formation energy, including potential alignments, is {} eV'
            .format(self.ctx.e_f_corrected_aligned.value * hartree_to_ev))
        self.out('formation_energy_corrected_aligned',
                 self.ctx.e_f_corrected_aligned)
