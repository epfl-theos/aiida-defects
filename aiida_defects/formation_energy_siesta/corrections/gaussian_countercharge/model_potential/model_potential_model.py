# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, calcfunction, if_ ,while_
#from qe_tools.constants import bohr_to_ang
from qe_tools import  CONSTANTS

from .utils import (create_model_structure, get_cell_matrix,
                    get_reciprocal_cell, get_reciprocal_grid, get_charge_model,
                    get_model_potential, get_energy)

from .utils import (get_charge_model_with_siesta_mesh,get_model_potential_with_siesta_mesh)

class ModelPotentialWorkchain(WorkChain):
    """
    Workchain to compute the model electrostatic potential
    """
    @classmethod
    def define(cls, spec):
        super(ModelPotentialWorkchain, cls).define(spec)
        spec.input('defect_charge', 
                    valid_type = orm.Float)
        spec.input('scale_factor', 
                    valid_type = orm.Int)
        spec.input('host_structure', 
                    valid_type = orm.StructureData)
        spec.input('defect_site',
                    valid_type = orm.List,
                    help = "Defect site position in crystal coordinates")
        spec.input('cutoff', 
                    valid_type = orm.Float, 
                    default = orm.Float(40.))
        spec.input("sigma", 
                    valid_type = orm.Float)
        spec.input('siesta_grid',
                    valid_type = orm.ArrayData,
                    required = False)
        spec.input("use_siesta_mesh_cutoff",
                    valid_type = orm.Bool,
                    required = True,
                    help = "Whether use Siesta Mesh size to Generate the Model Potential or Not ")
        spec.input('epsilon',
                    valid_type = orm.Float,
                    help = "Dielectric constant of the host material")
        
        
        spec.outline(
                if_(cls.is_use_siesta_mesh_cutoff)(
                                               cls.get_model_structure,
                                               cls.compute_model_charge_wtih_siesta_grid,
                                               cls.compute_model_potential_with_siesta_grid,
                                               cls.compute_energy_with_siesta_grid,
                                               cls.results
                ).else_(
                        cls.get_model_structure,
                        cls.compute_model_charge,
                        cls.compute_model_potential,
                        cls.compute_energy,
                        cls.results,
                        ))

        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('model_energy', 
                     valid_type = orm.Float, 
                     required = True)
        spec.output('model_potential', 
                     valid_type = orm.ArrayData, 
                     required = True)
        spec.output('model_structure',
                     valid_type = orm.StructureData,
                     required = True)
        spec.output("model_charge",
                    valid_type = orm.ArrayData , 
                    required = True)
        
        # Exit codes

    def is_use_siesta_mesh_cutoff(self):
        """

        """
        if self.inputs.use_siesta_mesh_cutoff :
            self.report("Using Siesta Mesh grid Size is = {} For Model Potential".format(self.inputs.siesta_grid.get_array('grid')))
            return True
        else:
            self.report("Using {} To Generate The Model Potential".format(self.inputs.cutoff.value))
            return False


    def setup(self):
        """
        Setup
        """
        pass

    def get_model_structure(self):
        """
        Generate the model structure by scaling the host structure
        """
        self.report("Generating model structure ...")
        self.ctx.model_structure = create_model_structure(
            self.inputs.host_structure, self.inputs.scale_factor)
        # Get cell matricies
        real_cell = get_cell_matrix(self.ctx.model_structure)
        self.ctx.reciprocal_cell = get_reciprocal_cell(real_cell)
        #self.report("DEBUG: recip cell: {}".format(self.ctx.reciprocal_cell))
        #limits = np.array(self.ctx.model_structure.cell_lengths) / bohr_to_ang
        limits = np.array(self.ctx.model_structure.cell_lengths) / CONSTANTS.bohr_to_ang

        self.ctx.limits = orm.List(list=limits.tolist())

    #----------------------------------------------------
    # If Using SIESTA Grid
    #---------------------------------------------------

    def compute_model_charge_wtih_siesta_grid(self):
        """
        Compute the model charge distribution
        """

        self.report("Computing Model Charge Distribution For Scale Factor {}".format(self.inputs.scale_factor.value))
        
        self.ctx.grid_dimensions = orm.List(list = self.inputs.siesta_grid.get_array('grid').tolist())
       
        #self.report("DEBUG: "+str(self.inputs.siesta_grid))
        #self.report("DEBUG: "+str(self.ctx.grid_dimensions.get_list()))
        
        self.ctx.charge_model = get_charge_model_with_siesta_mesh(charge=self.inputs.defect_charge,
                                                 dimensions = self.ctx.grid_dimensions,
                                                 limits = self.ctx.limits,
                                                 sigma = self.inputs.sigma ,  #TODO: Automated fitting/3-tuple of sigma values
                                                 defect_position = self.inputs.defect_site)

    def compute_model_potential_with_siesta_grid(self):
        """
        Compute the model potential according to the Gaussian Counter Charge scheme
        """
        self.report("Computing Model Potential For Scale Factor {}".format(
            self.inputs.scale_factor.value))
        #self.inputs.metadata.call_link_label = "Scale factor: {}".format(self.ctx.model_iteration+1)

        recip_cell = orm.ArrayData()
        recip_cell.set_array('cell_matrix', self.ctx.reciprocal_cell)

        self.ctx.model_potential = get_model_potential_with_siesta_mesh(cell_matrix = recip_cell,
                                                       dimensions = self.ctx.grid_dimensions,
                                                       charge_density = self.ctx.charge_model,
                                                       epsilon = self.inputs.epsilon)

    def compute_energy_with_siesta_grid(self):
        """
        Compute the model energy
        """
        self.report("Computing Model Energy For Scale Factor {}".format(
            self.inputs.scale_factor.value))

        #self.report("DEBUG: type {}".format(type(self.ctx.model_potential)))
        self.ctx.model_energy = get_energy(potential = self.ctx.model_potential,
                                           charge_density = self.ctx.charge_model,
                                           dimensions = self.ctx.grid_dimensions,
                                           limits = self.ctx.limits)
        self.report("Calculated model energy: {} eV".format(
            self.ctx.model_energy.value))

    #---------------------------------------------------------------
    # If Not Using SIESTA MESH (More Precise but need intepolation)
    #---------------------------------------------------------------


    def compute_model_charge(self):
        """
        Compute the model charge distribution
        """

        self.report("Computing Model Charge Distribution For Scale Factor {}".format(self.inputs.scale_factor.value))

        self.ctx.grid_dimensions = get_reciprocal_grid(
            self.ctx.reciprocal_cell, self.inputs.cutoff.value)
       
        #self.report("DEBUG: "+str(self.ctx.grid_dimensions))
        
        self.ctx.charge_model = get_charge_model(charge = self.inputs.defect_charge,
                                                 dimensions = self.ctx.grid_dimensions,
                                                 limits = self.ctx.limits,
                                                 sigma = self.inputs.sigma, #orm.Float(2.614),  #TODO: Automated fitting/3-tuple of sigma values
                                                 defect_position = self.inputs.defect_site)

    def compute_model_potential(self):
        """
        Compute the model potential according to the Gaussian Counter Charge scheme
        """
        self.report("Computing Model Potential For Scale Factor {}".format(
            self.inputs.scale_factor.value))
        #self.inputs.metadata.call_link_label = "Scale factor: {}".format(self.ctx.model_iteration+1)

        recip_cell = orm.ArrayData()
        recip_cell.set_array('cell_matrix', self.ctx.reciprocal_cell)

        self.ctx.model_potential = get_model_potential(
            cell_matrix=recip_cell,
            dimensions=self.ctx.grid_dimensions,
            charge_density=self.ctx.charge_model,
            epsilon=self.inputs.epsilon)

    def compute_energy(self):
        """
        Compute the model energy
        """
        self.report("Computing Model Energy For Scale Factor {}".format(
            self.inputs.scale_factor.value))

        #self.report("DEBUG: type {}".format(type(self.ctx.model_potential)))
        self.ctx.model_energy = get_energy(
            potential=self.ctx.model_potential,
            charge_density=self.ctx.charge_model,
            dimensions=self.ctx.grid_dimensions,
            limits=self.ctx.limits)

        self.report("Calculated model energy: {} eV".format(
            self.ctx.model_energy.value))

    def results(self):
        """
        Gather the results
        """
        # Return the model potential for the cell which corresponds to the host structure
        self.out('model_energy', self.ctx.model_energy)
        self.out('model_potential', self.ctx.model_potential)
        self.out('model_structure', self.ctx.model_structure)
        self.out('model_charge', self.ctx.charge_model)  # Needed For Alignment
        self.report("Finished Successfully")

