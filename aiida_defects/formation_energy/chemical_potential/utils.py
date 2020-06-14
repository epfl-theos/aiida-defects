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
from aiida.orm import ArrayData, Float
from pymatgen import Element
from shapely.geometry import Polygon
from itertools import combinations

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

def Order_point_clockwise(points):
        center = np.mean(points, axis=0)
        # compute angle 
        t = np.arctan2(points[:,0]-center[0], points[:,1]-center[1])
        sort_t = np.sort(t)
        t = list(t)
        u = [t.index(element) for element in sort_t]
        ordered_points = points[u]
        return ordered_points

def get_grid(stability_corners, matrix_eqns, N_point=100, tolerance=1E-4):
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

def PolygoneArea(stability_corners):
	# stability corners must be ordered clockwise or anticlockwise
	x = stability_corners[:,0]
	y = stability_corners[:,1]
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@calcfunction
def return_matrix_of_constraint(matrix_data):
    matrix = matrix_data.get_array('data')
    x = matrix + np.zeros_like(matrix)
    mat = ArrayData()
    mat.set_array('data', x)
    #matrix_data.set_array('data', eqn)
    return mat

@calcfunction
def get_chemical_potential(ref_energy, chem_pot):
    return Float(ref_energy.value + chem_pot.value)
