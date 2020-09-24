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
#    ref_energy = {'Li':-195.51408, 'P':-191.03878, 'O':-557.49850, 'S':-326.67885, 'Cl':-451.66500, 'B':-86.50025, 'Zn':-6275.54609, 
#            'Mg':-445.18254, 'Ta':-1928.66788, 'Zr':-1348.75011, 'Sn':-2162.23795, 'Mo':-1865.95416, 'Ta':-1928.66325, 'Be':-382.31135,
#            'C':-246.491433, 'Si':-154.27445, 'Na': -1294.781, 'K': -1515.34028, 'Rb': -665.48096, 'Cs': -855.71637, 'Ca': -1018.30809, 
#            'Sr': -953.20309, 'Ba': -5846.81333}
    @classmethod
    def define(cls, spec):
        super(ChemicalPotentialWorkchain, cls).define(spec)
        spec.input("formation_energy_dict", valid_type=Dict)
        spec.input("compound", valid_type=Str)
        spec.input("dependent_element", valid_type=Str)
        spec.input("defect_specie", valid_type=Str)
        spec.input("ref_energy", valid_type=Dict, help="The reference chemical potential of elements in the structure")
        spec.input("tolerance", valid_type=Float)

        spec.outline(
            cls.set_matrix_of_constraint,
            cls.solve_matrix_of_constraint,
            cls.get_centroid,
            cls.chemical_potential,
        )
        #spec.output(stability_corners', valid_type=ArrayData)
        spec.output('matrix_of_constraints', valid_type=ArrayData)
        spec.output('chemical_potential', valid_type=Float)

        spec.exit_code(601, "ERROR_CHEMICAL_POTENTIAL_FAILED",
            message="The stability region can't be determined. The compound is probably unstable"
        )

    def set_matrix_of_constraint(self):
        compound_of_interest = Composition(self.inputs.compound.value)
        N_species = len(compound_of_interest)
        N_competing_phases = len(self.inputs.formation_energy_dict.get_dict()) - 1

        column_order = {} # To track which element corresponds to each column, the dependent element is always the last column
        i = 0
        for ele in compound_of_interest:
            if ele.symbol != self.inputs.dependent_element.value:
                column_order[ele.symbol] = i
                i += 1
        column_order[self.inputs.dependent_element.value] = N_species - 1
        self.ctx.column_order = Dict(dict=column_order)

        ##############################################################################
        # Construct matrix containing all linear equations. The last column is the rhs 
        # of the system of equations
        ##############################################################################

        # Construct the 1st equation corresponding to the compound of interest
        eqns = np.zeros(N_species+1)
        for ele in compound_of_interest:
                eqns[column_order[ele.symbol]] = -1.0*compound_of_interest[ele]
        eqns[N_species] = -1.0*self.inputs.formation_energy_dict.get_dict()[self.inputs.compound.value]
        self.ctx.first_eqn = eqns
        #print(eqns)

        # Now loop through all the competing phases
        for key in self.inputs.formation_energy_dict.keys():
            # if key != compound:
            if not same_composition(key, self.inputs.compound.value):
                tmp = np.zeros(N_species+1)
                temp_composition = Composition(key)
                for ele in temp_composition:
                    tmp[column_order[ele.symbol]] = temp_composition[ele]
                tmp[N_species] = self.inputs.formation_energy_dict.get_dict()[key]
                eqns = np.vstack((eqns, tmp))
        #print(eqns)

        # Add constraints corresponding to the stability with respect to decomposition into
        # elements and combine it with the constraint on the stability of compound of interest
        for ele in compound_of_interest:
            if ele.symbol != self.inputs.dependent_element.value:
                tmp = np.zeros(N_species+1)
                tmp[column_order[ele.symbol]] = 1.0
                eqns = np.vstack((eqns, tmp))
                tmp = np.zeros(N_species+1)
                tmp[column_order[ele.symbol]] = -1.0
                tmp[N_species] = -1.*self.inputs.formation_energy_dict.get_dict()[self.inputs.compound.value]/compound_of_interest[ele]
                eqns = np.vstack((eqns, tmp))
        #print(eqns)

        # Eliminate the dependent element (variable) from the equations
        mask = eqns[1:, N_species-1] != 0.0
        eqns_0 = eqns[1:,:][mask]
        common_factor = compound_of_interest[self.inputs.dependent_element.value]/eqns_0[:, N_species-1]
        eqns_0 = eqns_0*np.reshape(common_factor, (len(common_factor), 1)) # Use broadcasting
        eqns_0 = (eqns_0+eqns[0,:])/np.reshape(common_factor, (len(common_factor), 1)) # Use broadcasting
        eqns[1:,:][mask] = eqns_0
        #print(eqns)

        # Store the matrix of constraint (before removing the depedent-element row) in the database
        temp_data = ArrayData()
        temp_data.set_array('data', eqns)
        set_of_constraints = return_matrix_of_constraint(temp_data)
        #self.ctx.constraints = set_of_constraints
        self.out('matrix_of_constraints', set_of_constraints)

        matrix = np.delete(eqns, N_species-1, axis=1)
        matrix_data = ArrayData()
        matrix_data.set_array('set_of_constraints', matrix)
        self.ctx.matrix_eqns = matrix_data

    def solve_matrix_of_constraint(self):
        matrix_eqns = self.ctx.matrix_eqns.get_array('set_of_constraints')
        N_species = matrix_eqns.shape[1]

        ### Look at all combination of lines and find their intersections
        comb = combinations(np.arange(np.shape(matrix_eqns)[0]), N_species-1)
        intersecting_points = []
        for item in list(comb):
            try:
                point = np.linalg.solve(matrix_eqns[item,:-1], matrix_eqns[item,-1])    
                intersecting_points.append(point)
            except np.linalg.LinAlgError:
                ### Singular matrix: lines are parallels therefore don't have any intersection
                pass

        intersecting_points = np.array(intersecting_points)
        get_constraint = np.dot(intersecting_points, matrix_eqns[:,:-1].T)
        check_constraint = (get_constraint - np.reshape(matrix_eqns[:,-1] ,(1, matrix_eqns.shape[0]))) <= self.inputs.tolerance.value
        bool_mask = [not(False in x) for x in check_constraint]
        corners_of_stability_region = intersecting_points[bool_mask]
        corners_of_stability_region = remove_duplicate(corners_of_stability_region)
        
        if corners_of_stability_region.size == 0:
            self.report('The stability region cannot be determined. The compound {} is probably unstable'.format(self.inputs.compound.value)) 
            return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_FAILED

        stability_data = ArrayData()
        stability_data.set_array('stability_corners', corners_of_stability_region)
        self.ctx.stability_corners = stability_data
        self.report('The stability corner is : {}'.format(corners_of_stability_region))
        #self.out("stability_corners", self.ctx.stability_corners)

    def get_centroid(self):
        
        ### Use to determine centroid (as oppose to center). Applicable only in 2D chemical potential map (ternary systems)
        #stability_corners = Order_point_clockwise(self.ctx.stability_corners.get_array('stability_corners'))
        #P = Polygon(stability_corners)
        #centroid_of_stability = np.array([P.centroid.x, P.centroid.y])

        stability_corners = self.ctx.stability_corners.get_array('stability_corners')
        M = self.ctx.matrix_eqns.get_array('set_of_constraints')
        N_specie = M.shape[1]
        if N_specie == 2:
            ctr_stability = np.mean(stability_corners, axis=0) #without the dependent element
        else:
            #grid = get_grid(stability_corners[:,:-1], M)
            grid = get_grid(stability_corners, M)
            ctr_stability = get_centroid(grid) #without the dependent element
        
        ### Add the corresponding chemical potential of the dependent element
        with_dependent = (self.ctx.first_eqn[-1]-np.sum(ctr_stability*self.ctx.first_eqn[:-2]))/self.ctx.first_eqn[-2]
        centroid_of_stability = np.append(ctr_stability, with_dependent)
        self.report('center of stability is {}'.format(centroid_of_stability))

        ctrd = ArrayData()
        ctrd.set_array('data', centroid_of_stability)
        self.ctx.centroid = ctrd
        #self.out("centroid", self.ctx.centroid)

    def chemical_potential(self):
        index = self.ctx.column_order[self.inputs.defect_specie.value]
        #chemical_potential = get_chemical_potential(Float(self.ctx.centroid.get_array('data')[index]), self.inputs.ref_energy)
        chem_ref = self.inputs.ref_energy.get_dict()
        chemical_potential = get_chemical_potential(Float(self.ctx.centroid.get_array('data')[index]), Float(chem_ref[self.inputs.defect_specie.value]))
        self.ctx.chemical_potential = chemical_potential
        self.out('chemical_potential', chemical_potential)
        self.report('The chemical potential of {} is {}'.format(self.inputs.defect_specie.value, chemical_potential.value))
