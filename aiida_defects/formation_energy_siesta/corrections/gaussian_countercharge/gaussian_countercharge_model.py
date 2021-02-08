# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import WorkChain, calcfunction, ToContext, while_ ,if_
from aiida import orm

from aiida_defects.formation_energy_siesta.potential_alignment.potential_alignment import PotentialAlignmentWorkchain
from .model_potential.model_potential_model import ModelPotentialWorkchain
from aiida_defects.formation_energy_siesta.potential_alignment.utils import get_potential_difference
from aiida_defects.formation_energy_siesta.corrections.gaussian_countercharge.utils import get_total_correction, get_total_alignment

from .utils import fit_energies, calc_correction
from .model_potential.utils import (get_charge_model_with_siesta_mesh,create_model_structure,get_cell_matrix,get_reciprocal_cell)
import numpy as np
#from qe_tools.constants import bohr_to_ang
from qe_tools import CONSTANTS #bohr_to_ang

bohr_to_ang = CONSTANTS.bohr_to_ang


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
        spec.input("v_host", 
                    valid_type = orm.ArrayData)
        spec.input("v_defect_q0", 
                    valid_type = orm.ArrayData)
        spec.input("v_defect_q", 
                    valid_type = orm.ArrayData)
        spec.input("defect_charge", 
                    valid_type = orm.Float)
        spec.input("defect_site",
                   valid_type = orm.List,
                   help = "Defect site position in crystal coordinates")
        spec.input("host_structure", 
                    valid_type = orm.StructureData)
        spec.input("epsilon",
                    valid_type = orm.Float,
                    help = "Dielectric constant for the host material")
        spec.input("model_iterations_required",
                    valid_type = orm.Int,
                    default = orm.Int(3))
        spec.input("cutoff",
                    valid_type = orm.Float,
                    default = orm.Float(40.),
                    help = "Plane wave cutoff for electrostatic model.For Siesta Read by Built-In Mesh",
                    required = False )
        spec.input("use_siesta_mesh_cutoff",
                    valid_type = orm.Bool,
                    required = True,
                    help = "Whether use Siesta Mesh size to Generate the Model Potential or Not ")
        spec.input("sigma",
                    valid_type = orm.Float,
                    required = True,
                    help = "Gaussian Sigma")       
        spec.input("siesta_grid",
                    valid_type = orm.ArrayData,
                    required = False,
                    help = "siesta Mesh grid for model potential generation") 
        
        
        
        spec.outline(
            cls.setup,
            if_(cls.is_use_siesta_mesh_cutoff)(while_(cls.should_run_model)(
                                                      cls.compute_model_potential_with_siesta_mesh,
                                                      ),
                                               cls.check_model_potential_workchains,
                                               cls.get_isolated_energy,
                                               cls.compute_dft_difference_potential,
                                               cls.get_model_corrections,
                                               cls.submit_alignment_workchains,
                                               cls.check_alignment_workchains,
                                               cls.get_model_corrections,
                                               cls.compute_correction
                                               )
            #cls.check_model_potential_workchains,  
            #cls.compute_dft_difference_potential,
            #cls.submit_alignment_workchains,
            #cls.check_alignment_workchains,
            #cls.get_isolated_energy,
            ##cls.get_model_corrections,
            ##cls.compute_correction,
                     )
        spec.output('v_dft_difference', valid_type=orm.ArrayData)
        spec.output('alignment_q0_to_host', valid_type=orm.Float)
        spec.output('alignment_dft_to_model', valid_type=orm.Float)
        spec.output('total_alignment', valid_type=orm.Float, required=True)
        spec.output('total_correction', valid_type=orm.Float)
        spec.output('electrostatic_correction', valid_type=orm.Float)
        # spec.output('isolated_energy', valid_type=orm.Float, required=True) # Not sure if anyone would use this
        # spec.output('model_correction_energies', valid_type=orm.Dict, required=True) # Again, not sure if useful
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


    def is_use_siesta_mesh_cutoff(self):
        """

        """
        if self.inputs.use_siesta_mesh_cutoff :
            self.report("Using Siesta Mesh grid Size To Generate The Model Potential")
            return True
        else:
            self.report("Using {} To Generate The Model Potential".format(self.inputs.cutoff.value))
            return False
    
        
    def setup(self):
        """
        Setup the calculation
        """
        #self.report ("DEBUG:"+str(self.inputs.sigma.value) )
        self.report("Setting up the Gaussian-model Correction With Sigma ({}) ...".format(self.inputs.sigma.value))
        ## Verification
        if self.inputs.model_iterations_required < 3:
           self.report('The requested number of iterations, {}, is too low. At least 3 are required to achieve an #adequate data fit'.format(self.inputs.model_iterations_required.value))
           return self.exit_codes.ERROR_BAD_INPUT_ITERATIONS_REQUIRED

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
        
        ## Dict to store model charges
        #self.ctx.model_charges = {}
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

        self.report("Computing For Scale Factor {} ...".format(
            scale_factor.value))

        inputs = {
            'defect_charge': self.inputs.defect_charge,
            'scale_factor': scale_factor,
            'host_structure': self.inputs.host_structure,
            'defect_site': self.inputs.defect_site,
            'cutoff': self.inputs.cutoff,
            'epsilon': self.inputs.epsilon,
            'sigma':self.input.sigma
            }
        workchain_future = self.submit(ModelPotentialWorkchain, **inputs)
        label = 'model_potential_scale_factor_{}'.format(scale_factor.value)
        self.to_context(**{label: workchain_future})

    def compute_model_potential_with_siesta_mesh(self):
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
            'siesta_grid': self.inputs.siesta_grid,
            'epsilon': self.inputs.epsilon,
            'sigma': self.inputs.sigma,
            'use_siesta_mesh_cutoff': self.inputs.use_siesta_mesh_cutoff
        }
        workchain_future = self.submit(ModelPotentialWorkchain, **inputs)
        label = 'model_potential_scale_factor_{}'.format(scale_factor.value)
        self.to_context(**{label: workchain_future})




    def check_model_potential_workchains(self):
        """
        Check if the model potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """
        self.report("Checking Model Potentials ...")
        for ii in range(self.inputs.model_iterations_required.value):
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
                    self.ctx.model_charge = model_workchain.outputs.model_charge
                self.ctx.model_energies[str(
                    scale_factor)] = model_workchain.outputs.model_energy
                self.ctx.model_structures[str(
                    scale_factor)] = model_workchain.outputs.model_structure
                #self.ctx.model_charges = [str(
                    #scale_factor)] = model_workchain.outputs.model_charges

    def compute_dft_difference_potential(self):
        """
        Compute the difference in the DFT potentials for the cases of q=q and q=0
        """
        self.report("Computing The Difference in The DFT Potentials For The Cases Of q=q and q=0 ...")
        self.ctx.v_defect_q_q0 = get_potential_difference(
            self.inputs.v_defect_q, self.inputs.v_defect_q0)

        self.out('v_dft_difference', self.ctx.v_defect_q_q0)

    def submit_alignment_workchains(self):
        """
        Align the electrostatic potential of the defective material in the q=0 charge
        state with the pristine host system
        """
        # Compute the alignment between the defect, in q=0, and the host
        self.report("Computing The Alignment Between The Defect, in q=0, and The Host ...")

        # DEBUG: Testing Allignment works?!
        #self.report("Generating model structure")
        #self.ctx.model_structure = create_model_structure(
        #self.inputs.host_structure, orm.Int(1))
        ## Get cell matricies
        #real_cell = get_cell_matrix(self.ctx.model_structure)
        #self.ctx.reciprocal_cell = get_reciprocal_cell(real_cell)
        #self.report("DEBUG: recip cell: {}".format(self.ctx.reciprocal_cell))
        #limits = np.array(self.ctx.model_structure.cell_lengths) / bohr_to_ang
        #self.ctx.limits = orm.List(list=limits.tolist())

        #self.ctx.grid_dimensions = orm.List(list = self.inputs.siesta_grid.get_array('grid').tolist())
        #self.ctx.charge_model = get_charge_model_with_siesta_mesh(charge = self.inputs.defect_charge,
        #                                                          dimensions = self.ctx.grid_dimensions,
        #                                                          limits = self.ctx.limits,
        #                                                          sigma = self.inputs.sigma ,
        #                                                          defect_position = self.inputs.defect_site)
        self.report("DEBUG:" + str(self.ctx.model_charge))
        #self.report("DEBUG:" + str(self.inputs.v_defect_q0))
        #inputs = {
        #    "lany_zunger":{
        #        "first_potential": self.inputs.v_defect_q0,
        #        "second_potential": self.inputs.v_host,
        #    },
        #    "allow_interpolation": orm.Bool(False)
        #}

        inputs = {
             "density_weighted":{
                 "first_potential": self.inputs.v_defect_q0,
                 "second_potential": self.inputs.v_host,
                 "charge_density": self.ctx.model_charge
             },
             #"first_potential": self.inputs.v_defect_q0,
             #"second_potential": self.inputs.v_host,
             "allow_interpolation": orm.Bool(False)
         }

        workchain_future = self.submit(PotentialAlignmentWorkchain, **inputs)
        label = 'workchain_alignment_q0_to_host'
        self.to_context(**{label: workchain_future})


        self.report("Computing The Alignment Between The Defect DFT difference potential and the model  ...")
        inputs = {
             "density_weighted":{
                 #"first_potential": self.inputs.v_defect_q_q0,
                 "first_potential": self.ctx.v_defect_q_q0,
                 "second_potential": self.ctx.v_model,
                 "charge_density": self.ctx.model_charge
             },
             #"first_potential": self.inputs.v_defect_q0,
             #"second_potential": self.inputs.v_host,
             "allow_interpolation": orm.Bool(False)
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
        self.report("Calculating isolated Charged Defect Energy ...")
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
            self.ctx.isolated_energy.value))

    def get_model_corrections(self):
        """
        Get the energy corrections for each model size
        """
        self.report("Computing the required correction for each model size ...")

        for scale_factor, model_energy in self.ctx.model_energies.items():
            self.ctx.model_correction_energies[scale_factor] = calc_correction(
                self.ctx.isolated_energy, model_energy)

    def compute_correction(self):
        """
	    Compute the Gaussian Countercharge correction
        """

        electrostatic_correction = self.ctx.model_correction_energies['1']

        total_alignment = get_total_alignment(self.ctx.alignment_dft_to_model,
                                              self.ctx.alignment_q0_to_host,
                                              self.inputs.defect_charge)

        total_correction = get_total_correction(electrostatic_correction,
                                                total_alignment)

        self.report('The computed total alignment is {} eV'.format(
            total_alignment.value))
        self.out('total_alignment', total_alignment)

        self.report('The computed electrostatic correction is {} eV'.format(
            electrostatic_correction.value))
        self.out('electrostatic_correction', electrostatic_correction)

        self.report(
            'The computed total correction, including potential alignments, is {} eV'
            .format(total_correction.value))
        self.out('total_correction', total_correction)

        # Store additional outputs
        self.out('alignment_q0_to_host', self.ctx.alignment_q0_to_host)
        self.out('alignment_dft_to_model', self.ctx.alignment_dft_to_model)

        self.report('Gaussian Countercharge workchain completed successfully')
