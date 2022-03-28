# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import WorkChain, calcfunction, ToContext, while_
from aiida.orm import Float, Int, Str, List, Bool, Dict, ArrayData
import sys
import numpy as np
from pymatgen.core.composition import Composition
#from pymatgen import MPRester, Composition, Element
from itertools import combinations

from .utils import *

class ChemicalPotentialWorkchain(WorkChain):
    """
    Compute the range of chemical potential of different elements which are consistent with the stability
    of that compound.
    Here we implement method similar to Buckeridge et al., (https://doi.org/10.1016/j.cpc.2013.08.026),
    """

    @classmethod
    def define(cls, spec):
        super(ChemicalPotentialWorkchain, cls).define(spec)
        spec.input("formation_energy_dict", valid_type=Dict,
            help="The formation energies of all compounds in the phase diagram to which belong the material of interest")
        spec.input("compound", valid_type=Str,
            help="The name of the material of interest")
        spec.input("dependent_element", valid_type=Str,
            help="In a N-element phase diagram, the chemical potential of depedent_element is fixed by that of the other N-1 elements")
        spec.input("dopant_elements", valid_type=List, required=False, default=lambda: List(list=[]),
            help="The aliovalent dopants that might be introduce into the prestine material. Several dopants might be present in co-doping scenario.")
        spec.input("ref_energy", valid_type=Dict, 
            help="The reference chemical potential of elements in the structure")
        spec.input("tolerance", valid_type=Float, default=lambda: Float(1E-4),
            help="Use to determine if a point in the chemical potential space is a corner of the stability region or not")
        spec.input("grid_points", valid_type=Int, default=lambda: Int(25),
            help="The number of point on each axis to generate the grid of the stability region. This grid is needed to determine the centroid or to plot concentration or defect formation energy directly on top of the stability region")

        spec.outline(
            cls.setup,
            cls.generate_matrix_of_constraints,
            cls.solve_matrix_of_constraints,
            cls.get_chemical_potential,
        )
        spec.output('stability_vertices', valid_type=Dict)
        spec.output('matrix_of_constraints', valid_type=Dict)
        spec.output('chemical_potential', valid_type=Dict)

        spec.exit_code(601, "ERROR_CHEMICAL_POTENTIAL_FAILED",
            message="The stability region can't be determined. The compound is probably unstable"
        )
        spec.exit_code(602, "ERROR_INVALID_DEPENDENT_ELEMENT",
            message="In the case of aliovalent substitution, the dopant element has to be different from dependent element."
        )
    
    def setup(self):
        if self.inputs.dependent_element.value in self.inputs.dopant_elements.get_list():
            self.report('In the case of aliovalent substitution, the dopant element has to be different from dependent element. Please choose a different dependent element.')
            return self.exit_codes.ERROR_INVALID_DEPENDENT_ELEMENT
        
        composition = Composition(self.inputs.compound.value)
        element_list = [atom for atom in composition]

        if self.inputs.dopant_elements.get_list(): # check if the list empty
            element_list += [Element(atom) for atom in self.inputs.dopant_elements.get_list()]  # List concatenation
            N_species = len(composition) + len(self.inputs.dopant_elements.get_list())
        else:
            N_species = len(composition)
        
        self.ctx.element_list = element_list
        self.ctx.N_species = Int(N_species)
        formation_energy_dict = self.inputs.formation_energy_dict.get_dict()
        
        # check if the compound is stable or not. If not shift its energy down to put it on the convex hull and issue a warning.
        E_hull = get_e_above_hull(self.inputs.compound.value, element_list, formation_energy_dict)
        if E_hull > 0:
            self.report('WARNING! The compound {} is predicted to be unstable. For the purpose of determining the stability region, we shift its formation energy down so that it is on the convex hull. Use with care!'.format(self.inputs.compound.value))
            formation_energy_dict[self.inputs.compound.value] -= composition.num_atoms*(E_hull+0.005) # the factor 0.005 is added for numerical reason
        
        self.ctx.formation_energy_dict = Dict(dict=formation_energy_dict)

    def generate_matrix_of_constraints(self):


        ##############################################################################
        # Construct matrix containing all linear equations. The last column is the rhs 
        # of the system of equations
        ##############################################################################
        
        all_constraints_coefficients = get_full_matrix_of_constraints(
                                            self.ctx.formation_energy_dict,
                                            self.inputs.compound,
                                            self.inputs.dependent_element,
                                            self.inputs.dopant_elements,
                                            )

        self.ctx.master_eqn = get_master_equation(all_constraints_coefficients, self.inputs.compound)
        self.ctx.matrix_eqns = get_reduced_matrix_of_constraints(
                                    all_constraints_coefficients, 
                                    self.inputs.compound, 
                                    self.inputs.dependent_element,
                                    )
        self.out('matrix_of_constraints', self.ctx.matrix_eqns)

    def solve_matrix_of_constraints(self):
        self.ctx.stability_vertices = get_stability_vertices(
                                        self.ctx.master_eqn,
                                        self.ctx.matrix_eqns, 
                                        self.inputs.compound, 
                                        self.inputs.dependent_element,
                                        self.inputs.tolerance
                                        )
        #self.report('The stability vertices are : {}'.format(np.around(self.ctx.stability_vertices.get_dict()['data'], 3)))
        self.out("stability_vertices", self.ctx.stability_vertices)

    def get_chemical_potential(self):
        centroid = get_centroid_of_stability_region(
                        self.ctx.stability_vertices,
                        self.ctx.master_eqn,
                        self.ctx.matrix_eqns,
                        self.inputs.compound, 
                        self.inputs.dependent_element,
                        self.inputs.grid_points,
                        self.inputs.tolerance
                        )
        self.report('Centroid of the stability region is {}'.format(dict(zip(centroid.get_dict()['column'], centroid.get_dict()['data'][0]))))

        self.ctx.chemical_potential = get_absolute_chemical_potential(
                                            centroid, 
                                            self.inputs.ref_energy, 
                                            )
        self.out('chemical_potential', self.ctx.chemical_potential)
        self.report('The chemical potential is {}'.format(self.ctx.chemical_potential.get_dict()))

