# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import WorkChain, calcfunction, ToContext, while_, if_
from aiida import orm

from aiida_defects.formation_energy.potential_alignment.potential_alignment import PotentialAlignmentWorkchain
from .model_potential.model_potential import ModelPotentialWorkchain
from aiida_defects.formation_energy.potential_alignment.utils import get_potential_difference, get_interpolation
from .utils import get_total_correction, get_charge_model_fit, fit_energies, calc_correction, is_isotrope
from qe_tools import CONSTANTS
import numpy as np

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

        spec.input("host_structure",
            valid_type=orm.StructureData,
            help="The structure of the host system.")
        spec.input("defect_charge",
            valid_type=orm.Float,
            help="The target defect charge state.")
        spec.input("defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates.")
        spec.input("epsilon",
            valid_type=orm.ArrayData,
            help="Dielectric tensor (3x3) for the host material. The name of the array in epsilon")
        spec.input("model_iterations_required",
            valid_type=orm.Int,
            default=lambda: orm.Int(3),
            help="The number of model charge systems to compute. More may improve convergence.")
        spec.input("cutoff",
            valid_type=orm.Float,
            default=lambda: orm.Float(40.),
            help="Plane wave cutoff for electrostatic model.")
        spec.input("v_host",
            valid_type=orm.ArrayData,
            help="The electrostatic potential of the host system (in eV).")
        spec.input("v_defect_q0",
            valid_type=orm.ArrayData,
            help="The electrostatic potential of the defect system in the 0 charge state (in eV).")
        spec.input("v_defect_q",
            valid_type=orm.ArrayData,
            help="The electrostatic potential of the defect system in the target charge state (in eV).")
        spec.input("rho_host",
            valid_type=orm.ArrayData,
            help="The charge density of the host system.")
        spec.input("rho_defect_q",
            valid_type=orm.ArrayData,
            help="The charge density of the defect system in the target charge state.")

        # Charge Model Settings
        spec.input_namespace('charge_model',
            help="Namespace for settings related to different charge models")
        spec.input("charge_model.model_type",
            valid_type=orm.Str,
            help="Charge model type: 'fixed' or 'fitted'",
            default=lambda: orm.Str('fixed'))
        # Fixed
        spec.input_namespace('charge_model.fixed', required=False, populate_defaults=False,
            help="Inputs for a fixed charge model using a user-specified multivariate gaussian")
        # spec.input("charge_model.fixed.gaussian_params",
        #     valid_type=orm.List,
        #     help="A length 9 list of parameters needed to construct the "
        #     "gaussian charge distribution. The format required is "
        #     "[x0, y0, z0, sigma_x, sigma_y, sigma_z, cov_xy, cov_xz, cov_yz]")
        spec.input("charge_model.fixed.covariance_matrix",
            valid_type=orm.ArrayData,
            help="The covariance matrix used to construct the gaussian charge distribution.")
            # "gaussian charge distribution. The format required is "
            # "[x0, y0, z0, sigma_x, sigma_y, sigma_z, cov_xy, cov_xz, cov_yz]")

        # Fitted
        spec.input_namespace('charge_model.fitted', required=False, populate_defaults=False,
            help="Inputs for a fitted charge model using a multivariate anisotropic gaussian.")
        spec.input("charge_model.fitted.tolerance",
            valid_type=orm.Float,
            help="Permissable error for any fitted charge model parameter.",
            default=lambda: orm.Float(1.0e-3))
        spec.input("charge_model.fitted.strict_fit",
            valid_type=orm.Bool,
            help="When true, exit the workchain if a fitting parameter is outside the specified tolerance.",
            default=lambda: orm.Bool(True))

        spec.outline(
            cls.setup,
            if_(cls.should_fit_charge)(
                cls.fit_charge_model,
            ),
            while_(cls.should_run_model)(
                cls.compute_model_potential,
            ),
            cls.check_model_potential_workchains,
            cls.get_isolated_energy,
            cls.compute_dft_difference_potential,
            cls.submit_alignment_workchains,
            cls.check_alignment_workchains,
            #cls.get_isolated_energy,
            cls.get_model_corrections,
            cls.compute_correction,
        )
        spec.output('gaussian_parameters', valid_type=orm.Dict, required=False)
#        spec.output('v_dft_difference', valid_type=orm.ArrayData)
#        spec.output('alignment_q0_to_host', valid_type=orm.Float)
#        spec.output('alignment_diff_q_q0_to_model', valid_type=orm.Float)
#        spec.output('alignment_diff_q_host_to_model', valid_type=orm.Float)
        spec.output('potential_alignment', valid_type=orm.Float, required=True)
        spec.output('total_correction', valid_type=orm.Float)
        spec.output('electrostatic_correction', valid_type=orm.Float)
        # spec.output('isolated_energy', valid_type=orm.Float, required=True) # Not sure if anyone would use this
        # spec.output('model_correction_energies', valid_type=orm.Dict, required=True)

        spec.exit_code(201,
            'ERROR_INVALID_INPUT_ARRAY',
            message='the input ArrayData object can only contain one array')
        spec.exit_code(202,
            'ERROR_BAD_INPUT_ITERATIONS_REQUIRED',
            message='The required number of iterations must be at least 3')
        spec.exit_code(203,
            'ERROR_INVALID_CHARGE_MODEL',
            message='the charge model type is not known')
        spec.exit_code(204,
            'ERROR_BAD_INPUT_CHARGE_MODEL_PARAMETERS',
            message='Only the parameters relating to the chosen charge model should be specified')
        spec.exit_code(301,
            'ERROR_SUB_PROCESS_FAILED_ALIGNMENT',
            message='the electrostatic potentials could not be aligned')
        spec.exit_code(302,
            'ERROR_SUB_PROCESS_FAILED_MODEL_POTENTIAL',
            message='The model electrostatic potential could not be computed')
        spec.exit_code(303,
            'ERROR_SUB_PROCESS_FAILED_FINAL_SCF',
            message='the final scf PwBaseWorkChain sub process failed')
        spec.exit_code(304,
            'ERROR_BAD_CHARGE_FIT',
            message='the mode fit to charge density is exceeds tolerances')


    def setup(self):
        """
        Setup the calculation
        """

        ## Verification
        # Minimum number of iterations required.
        # TODO: Replace this with an input ports validator
        if self.inputs.charge_model.model_type == 'fitted' and self.inputs.model_iterations_required < 3:
           self.report('The requested number of iterations, {}, is too low. At least 3 are required to achieve an adequate data fit'.format(self.inputs.model_iterations_required.value))
           return self.exit_codes.ERROR_BAD_INPUT_ITERATIONS_REQUIRED

        # Check if charge model scheme is valid:
        model_schemes_available = ["fixed", "fitted"]
        self.ctx.charge_model = self.inputs.charge_model.model_type
        if self.ctx.charge_model not in model_schemes_available:
            return self.exit_codes.ERROR_INVALID_CHARGE_MODEL

        # Check if required charge model namespace is specified
        # TODO: Replace with input ports validator
        if self.ctx.charge_model == 'fitted':
            if 'fitted' not in self.inputs.charge_model: #Wanted fitted, but no params given
                return self.exit_codes.ERROR_BAD_INPUT_CHARGE_MODEL_PARAMETERS
            elif 'fixed' in self.inputs.charge_model: #Wanted fitted, but gave fixed params
                return self.exit_codes.ERROR_BAD_INPUT_CHARGE_MODEL_PARAMETERS
        elif self.ctx.charge_model == 'fixed':
            self.ctx.is_model_isotrope = False
            # check if the gaussian parameters correspond to an isotropic gaussian
            if is_isotrope(self.inputs.charge_model.fixed.covariance_matrix.get_array('sigma')) and is_isotrope(self.inputs.epsilon.get_array('epsilon')):
                self.report('DEBUG: the given gaussian charge distribution and dielectric constant are isotropic')
                self.inputs.model_iterations_required = orm.Int(1)
                self.ctx.is_model_isotrope = True
                self.ctx.sigma = np.mean(np.diag(self.inputs.charge_model.fixed.covariance_matrix.get_array('sigma')))
            if 'fixed' not in self.inputs.charge_model: #Wanted fixed, but no params given
                return self.exit_codes.ERROR_BAD_INPUT_CHARGE_MODEL_PARAMETERS
            elif 'fitted' in self.inputs.charge_model: #Wanted fixed, but gave fitted params
                return self.exit_codes.ERROR_BAD_INPUT_CHARGE_MODEL_PARAMETERS

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


    def should_fit_charge(self):
        """
        Return whether the charge model should be fitted
        """
        return (self.ctx.charge_model == 'fitted')


    def fit_charge_model(self):
        """
        Fit an anisotropic gaussian to the charge state electron density
        """

        fit = get_charge_model_fit(
            self.inputs.rho_host,
            self.inputs.rho_defect_q,
            self.inputs.host_structure)

        self.ctx.fitted_params = orm.List(list=fit['fit'])
        self.ctx.peak_charge = orm.Float(fit['peak_charge'])
        self.out('gaussian_parameters', fit)
        self.report('DEBUG: the gaussian parameters obtained from fitting are: {}'.format(fit['fit']))

        for parameter in fit['error']:
            if parameter > self.inputs.charge_model.fitted.tolerance:
                self.logger.warning("Charge fitting parameter worse than allowed tolerance")
                if self.inputs.charge_model.fitted.strict_fit:
                    return self.exit_codes.ERROR_BAD_CHARGE_FIT

        if is_gaussian_isotrope(self.ctx.fitted_params.get_list()[3:]) and is_isotrope(self.inputs.epsilon.get_array('epsilon')):
            self.report('The fitted gaussian and the dielectric constant are isotropic. The isolated model energy will be computed analytically')
            self.inputs.model_iterations_required = orm.Int(1)
            self.ctx.is_model_isotrope = True
            self.ctx.sigma = np.mean(self.ctx.fitted_params.get_list()[3:6])
        else:
            self.ctx.is_model_isotrope = False

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

        if self.ctx.charge_model == 'fitted':
            gaussian_params = self.ctx.fitted_params
            peak_charge = self.ctx.peak_charge
        else:
            # gaussian_params = self.inputs.charge_model.fixed.gaussian_params
            cov_mat = self.inputs.charge_model.fixed.covariance_matrix.get_array('sigma')
            params = self.inputs.defect_site.get_list()+list(np.diag(cov_mat))+list(cov_mat[0, 1:])+[cov_mat[1,2]]
            gaussian_params = orm.List(list=params)
            peak_charge = orm.Float(0.)

        inputs = {
            'peak_charge': peak_charge,
            'defect_charge': self.inputs.defect_charge,
            'scale_factor': scale_factor,
            'host_structure': self.inputs.host_structure,
            'defect_site': self.inputs.defect_site,
            'cutoff': self.inputs.cutoff,
            'epsilon': self.inputs.epsilon,
            'gaussian_params' : gaussian_params
        }
        workchain_future = self.submit(ModelPotentialWorkchain, **inputs)
        label = 'model_potential_scale_factor_{}'.format(scale_factor.value)
        self.to_context(**{label: workchain_future})


    def check_model_potential_workchains(self):
        """
        Check if the model potential workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        for ii in range(self.inputs.model_iterations_required.value):
            scale_factor = ii + 1
            label = 'model_potential_scale_factor_{}'.format(scale_factor)
            model_workchain = self.ctx[label]
            if not model_workchain.is_finished_ok:
                self.report(
                    'Model potential workchain for scale factor {} failed with status {}'
                    .format(scale_factor,
                            model_workchain.exit_status))
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_MODEL_POTENTIAL
            else:
                if scale_factor == 1:
                    self.ctx.v_model = model_workchain.outputs.model_potential
                    self.ctx.charge_model = model_workchain.outputs.model_charge
                self.ctx.model_energies[str(scale_factor)] = model_workchain.outputs.model_energy
                self.ctx.model_structures[str(scale_factor)] = model_workchain.outputs.model_structure


    def compute_dft_difference_potential(self):
        """
        Compute the difference in the DFT potentials for the cases of q=q and q=0
        """
        #self.ctx.v_defect_q_q0 = get_potential_difference(
        #    self.inputs.v_defect_q, self.inputs.v_defect_q0)
        #self.out('v_dft_difference', self.ctx.v_defect_q_q0)

        first_array_shape = self.inputs.v_defect_q.get_array('data').shape
        second_array_shape = self.inputs.v_host.get_array('data').shape
        # second_array_shape = self.inputs.v_defect_q0.get_array('data').shape
        if first_array_shape != second_array_shape:
            target_shape = orm.List(list=np.max(np.vstack((first_array_shape, second_array_shape)), axis=0).tolist())
            first_array = get_interpolation(self.inputs.v_defect_q, target_shape)#.get_array('interpolated_array')
            second_array = get_interpolation(self.inputs.v_host, target_shape)#.get_array('interpolated_array')
            # second_array = get_interpolation(self.inputs.v_defect_q0, target_shape)
            self.ctx.v_defect_q_host = get_potential_difference(first_array, second_array)
        else:
            self.ctx.v_defect_q_host = get_potential_difference(
                   self.inputs.v_defect_q, self.inputs.v_host)
            # self.ctx.v_defect_q_host = get_potential_difference(
            #         self.inputs.v_defect_q, self.inputs.v_defect_q0)
            #self.out('v_dft_difference', self.ctx.v_defect_q_q0)


    def submit_alignment_workchains(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """

        # # Compute the alignment between the defect, in q=0, and the host
        # inputs = {
        #     "allow_interpolation": orm.Bool(True),
        #     "mae":{
        #         "first_potential": self.inputs.v_defect_q0,
        #         "second_potential": self.inputs.v_host,
        #         "defect_site": self.inputs.defect_site
        #     },
        # }

        # workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        # label = 'workchain_alignment_q0_to_host'
        # self.to_context(**{label: workchain_future})

        # Convert units from from eV in model potential to Ryd unit as in DFT potential, and also change sign
        # The potential alignment has to be converted back to eV. It is done in the mae/utils.py. Not pretty, has to be cleaned
        # TODO: Check if this breaks provenance graph
        # v_model = orm.ArrayData()
        # # v_model.set_array('data',
        # #     self.ctx.v_model.get_array(self.ctx.v_model.get_arraynames()[0])/(-1.0*CONSTANTS.ry_to_ev)) # eV to Ry unit of potential - This is dirty - need to harmonise units
        # v_model.set_array('data',
        #     self.ctx.v_model.get_array(self.ctx.v_model.get_arraynames()[0])*-2.0) # Hartree to Ry unit of potential - This is dirty - need to harmonise units

        # # Compute the alignment between the difference of DFT potentials v_q and v_q0, and the model
        # inputs = {

        #     "allow_interpolation": orm.Bool(True),
        #     "mae":{
        #         "first_potential": self.ctx.v_defect_q_q0,
        #         # "second_potential": v_model,
        #         "second_potential": self.ctx.v_model,
        #         "defect_site": self.inputs.defect_site
        #     },
        # }
        # workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        # label = 'workchain_alignment_q-q0_to_model'
        # self.to_context(**{label: workchain_future})

        # Compute the alignment between the difference of DFT potentials v_q and v_host, and the model
        inputs = {

            "allow_interpolation": orm.Bool(True),
            "mae":{
                "first_potential": self.ctx.v_defect_q_host,
                "second_potential": self.ctx.v_model,
                "defect_site": self.inputs.defect_site
            },
        }
        workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        label = 'workchain_alignment_q-host_to_model'
        self.to_context(**{label: workchain_future})

    def check_alignment_workchains(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        # # q0 to host
        # alignment_wc = self.ctx['workchain_alignment_q0_to_host']
        # if not alignment_wc.is_finished_ok:
        #     self.report(
        #         'Potential alignment workchain (defect q=0 to host) failed with status {}'
        #         .format(alignment_wc.exit_status))
        #     return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        # else:
        #     self.ctx.alignment_q0_to_host = alignment_wc.outputs.alignment_required

        # # DFT q-q0 to model
        # alignment_wc = self.ctx['workchain_alignment_q-q0_to_model']
        # if not alignment_wc.is_finished_ok:
        #     self.report(
        #         'Potential alignment workchain (DFT q-q0 to model) failed with status {}'
        #         .format(alignment_wc.exit_status))
        #     return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        # else:
        #     self.ctx['alignment_q-q0_to_model'] = alignment_wc.outputs.alignment_required

        # DFT q-host to model
        alignment_wc = self.ctx['workchain_alignment_q-host_to_model']
        if not alignment_wc.is_finished_ok:
            self.report(
                'Potential alignment workchain (DFT q-host to model) failed with status {}'
                .format(alignment_wc.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        else:
            self.ctx['alignment_q-host_to_model'] = alignment_wc.outputs.alignment_required

    def get_isolated_energy(self):
        """
        Fit the calculated model energies and obtain an estimate for the isolated model energy
        """

        if not self.ctx.is_model_isotrope:
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
        else:
            sigma = self.ctx.sigma
            defect_charge = self.inputs.defect_charge.value
            # # Epsilon is now expected to be a tensor, and so to get a scalar here we diagonalise.
            # epsilon_tensor = self.inputs.epsilon.get_array('epsilon')
            epsilon = np.mean(np.diag(self.inputs.epsilon.get_array('epsilon')))
            self.report(
                    "Computing the energy of the isolated gaussian analytically"
            )
            self.ctx.isolated_energy = orm.Float(defect_charge**2/(2*epsilon*sigma*np.sqrt(np.pi))*CONSTANTS.hartree_to_ev)

        self.report("The isolated model energy is {} eV".format(
            self.ctx.isolated_energy.value))


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

        electrostatic_correction = self.ctx.model_correction_energies['1']
        potential_alignment = self.ctx['alignment_q-host_to_model']

        # total_alignment = get_total_alignment(self.ctx['alignment_q-q0_to_model'],
        #                                       self.ctx['alignment_q0_to_host'],
        #                                       self.inputs.defect_charge)
        # total_alignment = get_alignment(self.ctx['alignment_q-host_to_model'], self.inputs.defect_charge)

        total_correction = get_total_correction(electrostatic_correction,
                                                potential_alignment,
                                                self.inputs.defect_charge)

        self.report('The computed potential alignment is {} eV'.format(
            potential_alignment.value))
        self.out('potential_alignment', potential_alignment)

        self.report('The computed electrostatic correction is {} eV'.format(
            electrostatic_correction.value))
        self.out('electrostatic_correction', electrostatic_correction)

        self.report(
            'The computed total correction, including potential alignments, is {} eV'
            .format(total_correction.value))
        self.out('total_correction', total_correction)

        # Store additional outputs
        # self.out('alignment_q0_to_host', self.ctx.alignment_q0_to_host)
        # self.out('alignment_diff_q_q0_to_model', self.ctx['alignment_q-q0_to_model'])
        # self.out('alignment_diff_q_host_to_model', self.ctx['alignment_q-host_to_model'])

        self.report('Gaussian Countercharge workchain completed successfully')
