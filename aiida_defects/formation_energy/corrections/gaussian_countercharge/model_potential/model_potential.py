# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, calcfunction, while_
from qe_tools import CONSTANTS

from .utils import (create_model_structure, get_cell_matrix,
                    get_reciprocal_cell, get_reciprocal_grid, get_charge_model,
                    get_model_potential, get_energy)


class ModelPotentialWorkchain(WorkChain):
    """
    Workchain to compute the model electrostatic potential
    """
    @classmethod
    def define(cls, spec):
        super(ModelPotentialWorkchain, cls).define(spec)
        spec.input("defect_charge",
            valid_type=orm.Float,
            help="The target defect charge state")
        spec.input('scale_factor',
            valid_type=orm.Int,
            help="Scale factor to apply when constructing the model system")
        spec.input('host_structure',
            valid_type=orm.StructureData,
            help="The unscaled host structure")
        spec.input('defect_site',
            valid_type=orm.List,
            help="Defect site position in crystal coordinates")
        spec.input('cutoff',
            valid_type=orm.Float,
            default=lambda: orm.Float(40.),
            help="Energy cutoff for grids in Rydberg")
        spec.input('epsilon',
            valid_type=orm.ArrayData,
            help="Dielectric tensor (3x3) of the host material")
        spec.input('gaussian_params',
            valid_type=orm.List,
            help="A length 9 list of parameters needed to construct the "
            "gaussian charge distribution. The format required is "
            "[x0, y0, z0, sigma_x, sigma_y, sigma_z, cov_xy, cov_xz, cov_yz]")
        spec.input('peak_charge',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Peak charge of the defect charge density distribution. If set to zero, no scaling will be done.")

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
        spec.output('model_charge', valid_type=orm.ArrayData, required=True)
        spec.output('model_potential', valid_type=orm.ArrayData, required=True)
        spec.output('model_structure',
                    valid_type=orm.StructureData,
                    required=True)

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
            self.inputs.host_structure, self.inputs.scale_factor)
        # Get cell matrices
        self.ctx.real_cell = get_cell_matrix(self.ctx.model_structure)
        self.ctx.reciprocal_cell = get_reciprocal_cell(self.ctx.real_cell)
#        self.report("DEBUG: recip cell: {}".format(self.ctx.reciprocal_cell))
        limits = np.array(self.ctx.model_structure.cell_lengths) / CONSTANTS.bohr_to_ang
        self.ctx.limits = orm.List(list=limits.tolist())


    def compute_model_charge(self):
        """
        Compute the model charge distribution
        """

        self.report(
            "Computing model charge distribution for scale factor {}".format(
                self.inputs.scale_factor.value))

        self.ctx.grid_dimensions = get_reciprocal_grid(
            self.ctx.reciprocal_cell, self.inputs.cutoff.value)

        self.ctx.cell_matrix = orm.ArrayData()
        self.ctx.cell_matrix.set_array('cell_matrix', self.ctx.real_cell)

        if self.inputs.peak_charge == 0.0:
            peak_charge = None
        else:
            peak_charge = self.inputs.peak_charge

        self.ctx.charge_model = get_charge_model(
            cell_matrix = self.ctx.cell_matrix,
            peak_charge = peak_charge,
            defect_charge = self.inputs.defect_charge,
            dimensions = self.ctx.grid_dimensions,
            gaussian_params = self.inputs.gaussian_params
        )

    def compute_model_potential(self):
        """
        Compute the model potential according to the Gaussian Counter Charge scheme
        """
        self.report("Computing model potential for scale factor {}".format(
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
        self.report("Computing model energy for scale factor {}".format(
            self.inputs.scale_factor.value))

        self.ctx.model_energy = get_energy(
            potential=self.ctx.model_potential,
            charge_density=self.ctx.charge_model,
            cell_matrix=self.ctx.cell_matrix)

        self.report("Calculated model energy: {} eV".format(
            self.ctx.model_energy.value))

    def results(self):
        """
        Gather the results
        """
        # Return the model potential for the cell which corresponds to the host structure
        self.out('model_energy', self.ctx.model_energy)
        self.out('model_charge', self.ctx.charge_model)
        self.out('model_potential', self.ctx.model_potential)
        self.out('model_structure', self.ctx.model_structure)
        self.report("Finished successfully")
