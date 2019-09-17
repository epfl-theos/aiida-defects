# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.common.constants import hartree_to_ev, bohr_to_ang
from aiida.engine import WorkChain, calcfunction, while_

from .utils import (
    create_model_structure, get_cell_matrix, get_reciprocal_cell, get_reciprocal_grid,
    get_charge_model, get_model_potential, get_energy
)

class ModelPotentialWorkchain(WorkChain):
    """
    Workchain to compute the model electrostatic potential
    """

    @classmethod
    def define(cls, spec):
        super(ModelPotentialWorkchain, cls).define(spec)
        spec.input('defect_charge', valid_type=orm.Float)
        spec.input('scale_factor', valid_type=orm.Int)
        spec.input('host_structure', valid_type=orm.StructureData)
        spec.input('defect_site', valid_type=orm.List,
            help="Defect site position in crystal coordinates")
        spec.input('cutoff', valid_type=orm.Float, default=orm.Float(40.))
        spec.input('epsilon', valid_type=orm.Float, help="Dielectric constant of the host material")
        spec.outline(
            cls.setup,
            cls.get_model_structure,
            cls.compute_model_charge,
            cls.compute_model_potential,
            cls.compute_energy,
            cls.results,           
        )
        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('model_energy', valid_type=orm.Float, required=True)
        spec.output('model_potential', valid_type=orm.ArrayData, required=True)
        spec.output('model_structure', valid_type=orm.StructureData, required=True)

        # Exit codes


    def setup(self):
        """
        Setup
        """
        pass

    def get_model_structure(self):
        """
        Generate the model structure by scaling the host structure
        """
        self.report("Generating model structure")
        self.ctx.model_structure = create_model_structure(
            self.inputs.host_structure,
            self.inputs.scale_factor
        )
        # Get cell matricies
        real_cell = get_cell_matrix(self.ctx.model_structure)
        self.ctx.reciprocal_cell = get_reciprocal_cell(real_cell)
        self.report("DEBUG: recip cell: {}".format(self.ctx.reciprocal_cell))
        limits = np.array(self.ctx.model_structure.cell_lengths)/bohr_to_ang
        self.ctx.limits = orm.List(list=limits.tolist())

    
    def compute_model_charge(self):
        """
        Compute the model charge distribution
        """

        self.report("Computing model charge distribution for scale factor {}".format(self.inputs.scale_factor.value))
  
        self.ctx.grid_dimensions = get_reciprocal_grid(self.ctx.reciprocal_cell, self.inputs.cutoff.value)
        
        self.ctx.charge_model = get_charge_model(
            charge=self.inputs.defect_charge,
            dimensions=self.ctx.grid_dimensions,
            limits=self.ctx.limits,
            sigma=orm.Float(2.614),    #TODO: Automated fitting/3-tuple of sigma values
            defect_position=self.inputs.defect_site
        )


    def compute_model_potential(self):
        """
        Compute the model potential according to the Gaussian Counter Charge scheme
        """
        self.report("Computing model potential for scale factor {}".format(self.inputs.scale_factor.value))
        #self.inputs.metadata.call_link_label = "Scale factor: {}".format(self.ctx.model_iteration+1)

        recip_cell = orm.ArrayData()
        recip_cell.set_array('cell_matrix', self.ctx.reciprocal_cell)

        self.ctx.model_potential = get_model_potential(
            cell_matrix=recip_cell,
            dimensions=self.ctx.grid_dimensions,
            charge_density=self.ctx.charge_model,
            epsilon=self.inputs.epsilon
        )
 

    def compute_energy(self):
        """
        Compute the model energy
        """
        self.report("Computing model energy for scale factor {}".format(self.inputs.scale_factor.value))

        self.report("DEBUG: type {}".format(type(self.ctx.model_potential)))
        self.ctx.model_energy = get_energy(
            potential = self.ctx.model_potential,
            charge_density = self.ctx.charge_model, 
            dimensions = self.ctx.grid_dimensions,
            limits = self.ctx.limits
        )

        self.report("Calculated model energy: {} eV".format(self.ctx.model_energy*hartree_to_ev))


    def results(self):
        """
        Gather the results
        """
        # Return the model potential for the cell which corresponds to the host structure 
        self.out('model_energy', self.ctx.model_energy)
        self.out('model_potential', self.ctx.model_potential)
        self.out('model_structure', self.ctx.model_structure)
        self.report("Finished successfully")

        