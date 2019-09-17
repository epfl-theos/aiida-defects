# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import WorkChain, calcfunction, ToContext, while_
from aiida import orm
from aiida.common.constants import hartree_to_ev
from aiida_defects.formation_energy.potential_alignment.potential_alignment import PotentialAlignmentWorkchain
from .model_potential.model_potential import ModelPotentialWorkchain
from aiida_defects.formation_energy.common import run_pw_calculation

from aiida_defects.formation_energy.potential_alignment.utils import get_potential_difference
from aiida_defects.formation_energy.corrections.gaussian_countercharge.utils import get_correction_energy

from .utils import fit_energies, calc_correction


class GaussianCounterChargeWorkchain(WorkChain):
    """
    Compute the electrostatic correction for charged defects according to the 
    Guassian counter-charge method.
    Here we implement the Komsa-Pasquarello method (https://doi.org/10.1103/PhysRevLett.110.095505), 
    which is itself based on the Freysoldt method 
    (https://doi.org/10.1103/PhysRevLett.102.016402).
    """

    @classmethod
    def define(cls, spec):
        super(GaussianCounterChargeWorkchain, cls).define(spec)
        spec.input("v_host", valid_type=orm.ArrayData)
        spec.input("v_defect_q0", valid_type=orm.ArrayData)
        spec.input("v_defect_q", valid_type=orm.ArrayData)
        spec.input("defect_charge", valid_type=orm.Float)
        spec.input(
            "defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates")
        spec.input("host_structure", valid_type=orm.StructureData)
        spec.input(
            "model_iterations_required",
            valid_type=orm.Int,
            default=orm.Int(3))
        spec.input(
            "epsilon",
            valid_type=orm.Float,
            help="Dielectric constant for the host material")
        spec.input(
            "cutoff",
            valid_type=orm.Float,
            default=orm.Float(40.),
            help="Plane wave cutoff for electrostatic model")

        # spec.input("bulk_structure",valid_type=StructureData)
        # spec.input("code_pw",valid_type=Str)
        # spec.input("pseudo_family",valid_type=Str)
        # spec.input('options', valid_type=ParameterData)
        # spec.input("settings", valid_type=ParameterData)
        # spec.input("kpoints", valid_type=KpointsData)
        # spec.input('parameters', valid_type=ParameterData)
        # spec.input('epsilon_r', valid_type=Float)
        # spec.input('defect_charge', valid_type=Float)
        #spec.input('defect_position', valid_type=ArrayData)
        spec.outline(
            cls.setup,
            while_(cls.should_run_model)(cls.compute_model_potential, ),
            cls.check_model_potential_workchains,
            cls.compute_dft_difference_potential,
            cls.submit_alignment_workchains,
            cls.check_alignment_workchains,
            cls.get_isolated_energy,
            cls.get_model_corrections,
            cls.compute_correction,
        )
        spec.output('v_dft_difference', valid_type=orm.ArrayData)
        spec.output('alignment_q0_to_host', valid_type=orm.Float)
        spec.output('alignment_dft_to_model', valid_type=orm.Float)
        spec.output('correction_energy', valid_type=orm.Float)
        spec.output('isolated_energy', valid_type=orm.Float, required=True)
        spec.output(
            'model_correction_energies', valid_type=orm.List, required=True)
        spec.exit_code(
            401,
            'ERROR_INVALID_INPUT_ARRAY',
            message='the input ArrayData object can only contain one array')
        spec.exit_code(
            409,
            'ERROR_SUB_PROCESS_FAILED_ALIGNMENT',
            message='the electrostatic potentials could not be aligned')
        spec.exit_code(
            413,
            'ERROR_SUB_PROCESS_FAILED_MODEL_POTENTIAL',
            message='The model electrostatic potential could not be computed')
        spec.exit_code(
            410,
            'ERROR_SUB_PROCESS_FAILED_FINAL_SCF',
            message='the final scf PwBaseWorkChain sub process failed')
        spec.exit_code(
            411,
            'ERROR_BAD_INPUT_ITERATIONS_REQUIRED',
            message='The required number of iterations must be at least 3')

        # def run_scf_defect_neutral(self):
        # """
        # Run an SCF calculation on the defect supercell with no overall charge
        # """

        # pw_inputs={
        #     'code' : Code.get_from_string(str(self.inputs.code_pw)),
        #     'pseudo_family' : Str(self.inputs.pseudo_family),
        #     'parameters' : self.inputs.parameters,
        #     'settings' : self.inputs.settings,
        #     'options' : self.inputs.options,
        #     'structure' : self.inputs.defect_structure
        # }

        # run_pw_calculation(pw_inputs, 0.0, )

    def setup(self):
        """
        Setup the calculation
        """

        ## Verification
        #if self.inputs.model_iterations_required < 3:
        #    self.report('The requested number of iterations, {}, is too low. At least 3 are required to achieve an #adequate data fit'.format(self.inputs.model_iterations_required.value))
        #    return self.exit_codes.ERROR_BAD_INPUT_ITERATIONS_REQUIRED

        # Track iteration number
        self.ctx.model_iteration = orm.Int(0)

        # Check that the input ArrayData objects contain only one array
        for arraydata in [
                self.inputs.v_host, self.inputs.v_defect_q0,
                self.inputs.v_defect_q
        ]:
            if len(arraydata.get_arraynames()) != 1:
                self.report('Input array is invalid')
                return self.exit_codes.ERROR_INVALID_INPUT_ARRAY

        v_defect_q0 = self.inputs.v_defect_q0
        self.ctx.v_defect_q0_array = v_defect_q0.get_array(
            v_defect_q0.get_arraynames()[0])

        v_defect_q = self.inputs.v_defect_q
        self.ctx.v_defect_q_array = v_defect_q.get_array(
            v_defect_q.get_arraynames()[0])

        # Dict to store model energies
        self.ctx.model_energies = {}

        # Dict to store model structures
        self.ctx.model_structures = {}

        # Dict to store correction energies
        self.ctx.model_correction_energies = {}

        return

    def should_run_model(self):
        """
        Return whether a model workchain should be run, which is dependant on the number of model energies computed 
        with respect to to the total number of model energies needed.
        """
        return self.ctx.model_iteration < self.inputs.model_iterations_required

    def compute_model_potential(self):
        """
        Compute the potential for the system using a model charge distribution
        """
        self.ctx.model_iteration += 1
        scale_factor = self.ctx.model_iteration

        self.report("Computing model potential for scale factor {}".format(
            scale_factor.value))

        inputs = {
            'defect_charge': self.inputs.defect_charge,
            'scale_factor': scale_factor,
            'host_structure': self.inputs.host_structure,
            'defect_site': self.inputs.defect_site,
            'cutoff': self.inputs.cutoff,
            'epsilon': self.inputs.epsilon,
        }
        workchain_future = self.submit(ModelPotentialWorkchain, **inputs)
        label = 'model_potential_scale_factor_{}'.format(scale_factor.value)
        self.to_context(**{label: workchain_future})

    def check_model_potential_workchains(self):
        """
        Check if the model potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        for ii in range(self.inputs.model_iterations_required):
            scale_factor = ii + 1
            label = 'model_potential_scale_factor_{}'.format(scale_factor)
            model_workchain = self.ctx[label]
            if not model_workchain.is_finished_ok:
                self.report(
                    'Model potential workchain for scale factor {} failed with status {}'
                    .format(model_workchain.scale_factor,
                            model_workchain.exit_status))
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MODEL_POTENTIAL
            else:
                if scale_factor == 1:
                    self.ctx.v_model = model_workchain.outputs.model_potential
                self.ctx.model_energies[str(
                    scale_factor)] = model_workchain.outputs.model_energy
                self.ctx.model_structures[str(
                    scale_factor)] = model_workchain.outputs.model_structure

    def compute_dft_difference_potential(self):
        """
        Compute the difference in the DFT potentials for the cases of q=q and q=0
        """
        self.ctx.v_defect_q_q0 = get_potential_difference(
            self.inputs.v_defect_q, self.inputs.v_defect_q0)
        self.out('v_dft_difference', self.ctx.v_defect_q_q0)

    def submit_alignment_workchains(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """

        # Compute the alignment between the defect, in q=0, and the host
        inputs = {
            "first_potential": self.inputs.v_defect_q0,
            "second_potential": self.inputs.v_host
        }
        workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        label = 'workchain_alignment_q0_to_host'
        self.to_context(**{label: workchain_future})

        # Compute the alignment between the defect DFT difference potential, and the model
        inputs = {
            "first_potential": self.ctx.v_defect_q_q0,
            "second_potential": self.ctx.v_model,
            "interpolate":
            orm.Bool(True)  # This will more or less always be required
        }
        workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        label = 'workchain_alignment_dft_to_model'
        self.to_context(**{label: workchain_future})

    def check_alignment_workchains(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        # q0 to host
        alignment_wc = self.ctx['workchain_alignment_q0_to_host']
        if not alignment_wc.is_finished_ok:
            self.report(
                'Potential alignment workchain (defect q=0 to host) failed with status {}'
                .format(alignment_wc.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        else:
            self.ctx.alignment_q0_to_host = alignment_wc.outputs.alignment_required

        # DFT diff to model
        alignment_wc = self.ctx['workchain_alignment_dft_to_model']
        if not alignment_wc.is_finished_ok:
            self.report(
                'Potential alignment workchain (DFT diff to model) failed with status {}'
                .format(alignment_wc.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        else:
            self.ctx.alignment_dft_to_model = alignment_wc.outputs.alignment_required

    def get_isolated_energy(self):
        """
        Fit the calculated model energies and obtain an estimate for the isolated model energy
        """

        # Get the linear dimensions of the structures
        linear_dimensions = {}

        for scale, structure in self.ctx.model_structures.items():
            volume = structure.get_cell_volume()
            linear_dimensions[scale] = 1 / (volume**(1 / 3.))

        self.report(
            "Fitting the model energies to obtain the model energy for the isolated case"
        )
        self.ctx.isolated_energy = fit_energies(
            orm.Dict(dict=linear_dimensions),
            orm.Dict(dict=self.ctx.model_energies))
        self.report("The isolated model energy is {} eV".format(
            self.ctx.isolated_energy.value * hartree_to_ev))

    def get_model_corrections(self):
        """
        Get the energy corrections for each model size
        """
        self.report("Computing the required correction for each model size")

        for scale_factor, model_energy in self.ctx.model_energies.items():
            self.ctx.model_correction_energies[scale_factor] = calc_correction(
                self.ctx.isolated_energy, model_energy)

    def compute_correction(self):
        """
	    Compute the Gaussian Countercharge correction
        """

        total_correction = get_correction_energy(
            self.ctx.model_correction_energies['1'],
            self.ctx.alignment_dft_to_model, self.ctx.alignment_q0_to_host,
            self.inputs.defect_charge)

        self.report(
            'The computed correction, including potential alignments, is {} eV'
            .format(total_correction.value * hartree_to_ev))
        self.out('correction_energy', total_correction)
        self.report('Gaussian Countercharge workchain completed successfully')