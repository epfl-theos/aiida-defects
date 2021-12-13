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
from pymatgen import MPRester, Composition, Element
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
            help="The reference chemical potential of elements in the structure. Format of the dictionary: {'Element_symbol': energy, ...}")
        spec.input("tolerance", valid_type=Float, default=lambda: Float(1E-4),
            help="Use to determine if a point in the chemical potential space is a corner of the stability region or not")

        spec.outline(
            cls.setup,
            cls.generate_matrix_of_constraints,
            cls.solve_matrix_of_constraints,
            cls.get_chemical_potential,
        )
        spec.output('stability_corners', valid_type=ArrayData)
        spec.output('matrix_of_constraints', valid_type=ArrayData)
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
            formation_energy_dict[self.inputs.compound.value] -= composition.num_atoms*(E_hull+0.005) # the factor 0.005 is added for numerical precision to make sure that the compound is now on the convex hull

        self.ctx.formation_energy_dict = Dict(dict=formation_energy_dict)

    def generate_matrix_of_constraints(self):
        '''
        Construct the set of constraints given by each compounds in the phase diagram and which delineate the stability region.
        '''
        column_order = {} # To track which element corresponds to each column, the dependent element is always the last column
        i = 0
        for ele in self.ctx.element_list:
            if ele.symbol != self.inputs.dependent_element.value:
                column_order[ele.symbol] = i
                i += 1
        column_order[self.inputs.dependent_element.value] = self.ctx.N_species.value - 1
        self.ctx.column_order = Dict(dict=column_order)
        #self.report('Column order: {}'.format(column_order))

        # Construct matrix containing all linear equations. The last column is the rhs of the system of equations
        self.ctx.matrix_eqns = get_matrix_of_constraints(
                                    self.ctx.N_species,
                                    self.inputs.compound,
                                    self.inputs.dependent_element,
                                    self.ctx.column_order,
                                    self.ctx.formation_energy_dict
                                    )
        self.out('matrix_of_constraints', self.ctx.matrix_eqns)

    def solve_matrix_of_constraints(self):
        '''
        Solve the system of (linear) constraints to get the coordinates of the corners of polyhedra that delineate the stability region
        '''
        self.ctx.stability_corners = get_stability_corners(
                                        self.ctx.matrix_eqns,
                                        self.ctx.N_species,
                                        self.inputs.compound,
                                        self.inputs.tolerance
                                        )
        #self.report('The stability corner is : {}'.format(self.ctx.stability_corners.get_array('data')))
        self.out("stability_corners", self.ctx.stability_corners)

    def get_chemical_potential(self):
        '''
        Compute the centroid of the stability region
        '''
        centroid = get_center_of_stability(
                        self.inputs.compound,
                        self.inputs.dependent_element,
                        self.ctx.stability_corners,
                        self.ctx.N_species,
                        self.ctx.matrix_eqns
                        )
        self.report('Centroid of the stability region is {}'.format(centroid.get_array('data')))

        # Recover the absolute chemical potential by adding the energy of the reference elements to centroid
        self.ctx.chemical_potential = get_chemical_potential(
                                            centroid,
                                            self.inputs.ref_energy,
                                            self.ctx.column_order
                                            )
        self.out('chemical_potential', self.ctx.chemical_potential)
        self.report('The chemical potential is {}'.format(self.ctx.chemical_potential.get_dict()))

