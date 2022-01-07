# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import calcfunction
import numpy as np
from pymatgen.core.composition import Composition
from aiida.orm import ArrayData, Float, Dict
from pymatgen.core.periodic_table import Element
from itertools import combinations
from pymatgen.analysis.phase_diagram import *
from pymatgen.entries.computed_entries import ComputedEntry

@calcfunction
def get_matrix_of_constraints(N_species, compound, dependent_element, column_order, formation_energy_dict):
    N_species = N_species.value
    compound = compound.value
    dependent_element = dependent_element.value
    column_order = column_order.get_dict()
    formation_energy_dict = formation_energy_dict.get_dict()

    compound_of_interest = Composition(compound)

    # Construct the 1st equation corresponding to the compound of interest
    eqns = np.zeros(N_species+1)
    for ele in compound_of_interest:
            eqns[column_order[ele.symbol]] = -1.0*compound_of_interest[ele]
    eqns[N_species] = -1.0*formation_energy_dict[compound]
    #self.ctx.first_eqn = eqns

    # Now loop through all the competing phases
    for key in formation_energy_dict:
        # if key != compound:
        if not same_composition(key, compound):
            tmp = np.zeros(N_species+1)
            temp_composition = Composition(key)
            for ele in temp_composition:
                tmp[column_order[ele.symbol]] = temp_composition[ele]
            tmp[N_species] = formation_energy_dict[key]
            eqns = np.vstack((eqns, tmp))
    #print(eqns)

    # Add constraints corresponding to the stability with respect to decomposition into
    # elements and combine it with the constraint on the stability of compound of interest
    for ele in compound_of_interest:
        if ele.symbol != dependent_element:
            tmp = np.zeros(N_species+1)
            tmp[column_order[ele.symbol]] = 1.0
            eqns = np.vstack((eqns, tmp))
            tmp = np.zeros(N_species+1)
            tmp[column_order[ele.symbol]] = -1.0
            tmp[N_species] = -1.*formation_energy_dict[compound]/compound_of_interest[ele]
            eqns = np.vstack((eqns, tmp))
    #print(eqns)

    # Eliminate the dependent element (variable) from the equations
    mask = eqns[1:, N_species-1] != 0.0
    eqns_0 = eqns[1:,:][mask]
    common_factor = compound_of_interest[dependent_element]/eqns_0[:, N_species-1]
    eqns_0 = eqns_0*np.reshape(common_factor, (len(common_factor), 1)) # Use broadcasting
    eqns_0 = (eqns_0+eqns[0,:])/np.reshape(common_factor, (len(common_factor), 1)) # Use broadcasting
    eqns[1:,:][mask] = eqns_0
    #print(eqns)

    # Removing column corresponding to the dependent element from the set of equations correponding to the constraints
    # that delineate the stability region
    matrix = np.delete(eqns, N_species-1, axis=1)
    matrix_data = ArrayData()
    matrix_data.set_array('data', matrix)
    return matrix_data

@calcfunction
def get_stability_corners(matrix_eqns, N_species, compound, tolerance):
    matrix_eqns = matrix_eqns.get_array('data')
    N_species = N_species.value
    tolerance = tolerance.value
    compound = compound.value

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

    ### Determine the points that form the 'corners' of stability region. These are intersecting point that verify all the constraints.
    intersecting_points = np.array(intersecting_points)
    get_constraint = np.dot(intersecting_points, matrix_eqns[:,:-1].T)
    check_constraint = (get_constraint - np.reshape(matrix_eqns[:,-1] ,(1, matrix_eqns.shape[0]))) <= tolerance
    bool_mask = [not(False in x) for x in check_constraint]
    corners_of_stability_region = intersecting_points[bool_mask]
    ### In some cases, we may have several solutions corresponding to the same points. Hence, the remove_duplicate method
    corners_of_stability_region = remove_duplicate(corners_of_stability_region)

    if corners_of_stability_region.size == 0:
        self.report('The stability region cannot be determined. The compound {} is probably unstable'.format(compound))
        return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_FAILED

    ordered_stability_corner = ArrayData()
    ordered_stability_corner.set_array('data', Order_point_clockwise(corners_of_stability_region))
    #ordered_stability_corners = Order_point_clockwise(stability_data)
    #self.ctx.stability_corners = ordered_stability_corners

    #self.report('The stability corner is : {}'.format(ordered_stability_corners.get_array('data')))
    return ordered_stability_corner

@calcfunction
def get_center_of_stability(compound, dependent_element, stability_corners, N_species, matrix_eqns):
    '''
    Use to determine centroid (as oppose to center). The center is defined as the average coordinates of the corners 
    while a centroid is the average cooridinates of every point inside the polygone or polyhedron.
    For binary compounds, the stability region is a one-dimensional segment. The centroid coincides with the center.
    For ternary and quarternary compounds, the centroid is returned.
    For quinternary compound and hight, the center is returned.
    '''
    compound = compound.value
    dependent_element = dependent_element.value
    N_species = N_species.value
    stability_corners = stability_corners.get_array('data')
    matrix_eqns = matrix_eqns.get_array('data')

    if N_species == 2:
        ctr_stability = np.mean(stability_corners, axis=0) #without the dependent element
    else:
        grid = get_grid(stability_corners, matrix_eqns)
        ctr_stability = get_centroid(grid) #without the dependent element

    ### Add the corresponding chemical potential of the dependent element
    composition = Composition(compound)
    first_eqn = matrix_eqns[0]
    with_dependent = -1.0*(first_eqn[-1]-np.sum(ctr_stability*first_eqn[:-1]))/composition[dependent_element]
    centroid_of_stability = np.append(ctr_stability, with_dependent)
#    self.report('Centroid of the stability region is {}'.format(centroid_of_stability))

    ctrd = ArrayData()
    ctrd.set_array('data', centroid_of_stability)
    return ctrd


def get_e_above_hull(compound, element_list, formation_energy_dict):
    '''
    Get the energy above the convex hull. When the compound is unstable, e_hull > 0.
    '''
    composition = Composition(compound)
    mp_entries = []

    idx = 0
    for i, (material, Ef) in enumerate(formation_energy_dict.items()):
        if material == compound:
            idx = i
        mp_entries.append(ComputedEntry(Composition(material), Ef))
    for ref in element_list:
        mp_entries.append(ComputedEntry(Composition(ref.symbol), 0.0))
    #mp_entries.append(ComputedEntry(composition, E_formation))

    pd = PhaseDiagram(mp_entries)
    ehull = pd.get_e_above_hull(mp_entries[idx])

    return ehull

def same_composition(compound_1, compound_2):
        composition_1 = Composition(compound_1)
        composition_2 = Composition(compound_2)
        list_1 = [ele.symbol for ele in composition_1]
        list_2 = [ele.symbol for ele in composition_2]
        list_1.sort()
        list_2.sort()
        if list_1 != list_2:
                return False
        else:
                number_ele_1 = [composition_1[ele] for ele in list_1]
                number_ele_2 = [composition_2[ele] for ele in list_2]
                return number_ele_1 == number_ele_2

def is_point_in_array(ref_point, ref_array):
	for point in ref_array:
		if np.array_equal(ref_point, point):
			return True
	return False

def remove_duplicate(array):
        non_duplicate_array = []
        for point in array:
                if not is_point_in_array(point, non_duplicate_array):
                        non_duplicate_array.append(point)
        return np.array(non_duplicate_array)

def get_grid(stability_corners, matrix_eqns, N_point=50, tolerance=1E-4):
	xmin = np.amin(stability_corners[:,0])
	xmax = np.amax(stability_corners[:,0])
	ymin = np.amin(stability_corners[:,1])
	ymax = np.amax(stability_corners[:,1])
	x = np.linspace(xmin, xmax, N_point)
	y = np.linspace(ymin, ymax, N_point)

	dim = stability_corners.shape[1]
	if dim == 2:
		xx, yy = np.meshgrid(x, y)
		points = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)
	elif dim == 3:
		zmin = np.amin(stability_corners[:,2])
		zmax = np.amax(stability_corners[:,2])
		z = np.linspace(zmin, zmax, N_point)
		xx, yy, zz = np.meshgrid(x, y, z)
		points = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)
		points = np.append(points,zz.reshape(-1,1),axis=1)
	else:
		print('Not yet implemented for quinternary compounds and higher. Use center instead of centroid')
		return stability_corners

	get_constraint = np.dot(points, matrix_eqns[:,:-1].T)
	check_constraint = (get_constraint - np.reshape(matrix_eqns[:,-1] ,(1, matrix_eqns.shape[0]))) <= tolerance
	bool_mask = [not(False in x) for x in check_constraint]
	points_in_stable_region = points[bool_mask]

	return points_in_stable_region

def get_centroid(stability_region):
	return np.mean(stability_region, axis=0)

def Order_point_clockwise(points):
#    points = points.get_array('data')
    if len(points[0]) == 1:
        points_order = points
    else:
        center = np.mean(points, axis=0)
        # compute angle
        t = np.arctan2(points[:,0]-center[0], points[:,1]-center[1])
        sort_t = np.sort(t)
        t = list(t)
        u = [t.index(element) for element in sort_t]
        points_order = points[u]
#    ordered_points = ArrayData()
#    ordered_points.set_array('data', points_order)
#    return ordered_points
    return points_order

@calcfunction
def get_chemical_potential(centroid, ref_energy, column_order):
    centroid = centroid.get_array('data')
    ref_energy = ref_energy.get_dict()
    column_order = column_order.get_dict()
    chem_pot = {}
    for element in column_order.keys():
        chem_pot[element] = ref_energy[element]+centroid[column_order[element]]
    return Dict(dict=chem_pot)
