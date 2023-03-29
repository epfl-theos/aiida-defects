# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import calcfunction
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from aiida.orm import ArrayData, Float, Dict, List
from pymatgen.core.periodic_table import Element
from itertools import combinations
from pymatgen.analysis.phase_diagram import *
from pymatgen.entries.computed_entries import ComputedEntry

def pandas_df_to_Dict(df, index=False):
    '''
    Helper function to convert a pandas dataframe to AiiDA Dict.
    If index=False, the index of df won't be converted to (keys, values) pair in the Dict
    '''
    if index:
        return Dict({'column': df.columns, 'index': df.index, 'data': df.to_numpy()})
    else:
        return Dict({'column': df.columns, 'data': df.to_numpy()})

def Dict_to_pandas_df(py_dict):
    '''
    Helper function to convert a dict to a pandas dataframe
    '''
    if 'index' in py_dict.keys():
        return pd.DataFrame(np.array(py_dict['data']), index=py_dict['index'], columns=py_dict['column'])
    else:
        return pd.DataFrame(np.array(py_dict['data']), columns=py_dict['column'])

def get_full_matrix_of_constraints(formation_energy_dict, compound, dependent_element, dopant):
    '''
    The systems of linear constraints (before eliminating the dependent variable and the 'compound'), i.e. matrix of constraints is constructed as
    a pandas dataframe. Each columns corresponds to each element in the compounds and dopants ('Li', 'P', ...) while the last
    column is the formation energy (per fu) of each stable compounds in the phase diagram. The column before the last column is
    always reserved for the depedent element. Each row is indexed by the formula of each stable compound.
    When it is not possible to use pandas dataframe for ex. to pass as argument to a calcfuntion, the dataframe is 'unpacked' as
    a python dictionary in the form {'column': , 'index': , 'data': }
    '''

    formation_energy_dict = formation_energy_dict.get_dict()
    compound = compound.value
    dependent_element = dependent_element.value
    dopant = dopant.get_list()

    compound_of_interest = Composition(compound)

    # Setting up the matrix of constraints as pd dataframe and initialize it to zeros.
    stable_compounds, element_order = [], []
    for key in formation_energy_dict.keys():
        stable_compounds.append(key)
    for key in compound_of_interest:
        stable_compounds.append(key.symbol)

    element_order = [atom.symbol for atom in compound_of_interest if atom.symbol != dependent_element]
    if dopant == []:
        element_order.extend([dependent_element, 'Ef'])
        N_species = len(compound_of_interest)
    else:
        element_order.extend(dopant+[dependent_element, 'Ef'])
        N_species = len(compound_of_interest) + len(dopant)
        stable_compounds.extend(dopant)

    eqns = pd.DataFrame(np.zeros((len(stable_compounds), len(element_order))), index=stable_compounds, columns=element_order)

    # Setting the coefficients of the matrix of constraint
    # First, loop through all the competing phases
    for k, v in formation_energy_dict.items():
        composition = Composition(k)
        for element in composition:
            eqns.loc[k, element.symbol] = composition[element]
        eqns.loc[k, 'Ef'] = v
    # Then, loop over all elemental phases
    for element in compound_of_interest:
        eqns.loc[element.symbol, element.symbol] = 1.0
    if dopant:
        for element in dopant:
            eqns.loc[element, element] = 1.0

    return pandas_df_to_Dict(eqns, index=True)

# @calcfunction
# def get_master_equation(raw_constraint_coefficients, compound):
#     '''
#     The 'master' equation is simply the equality corresponding to the formation energy of the
#     compound under consideration. For ex. if we are studying the defect in Li3PO4, the master
#     equation is simply: 3*mu_Li + mu_P + 4*mu_O = Ef where Ef is the formation energy per fu
#     of Li3PO4. This equation is needed to replace the dependent chemical potential from the set
#     of other linear constraints and to recover the chemical potential of the dependent element
#     from the chemical potentials of independent elements
#     '''
#     all_coefficients = raw_constraint_coefficients.get_dict()
#     eqns = Dict_to_pandas_df(all_coefficients)
#     master_eqn = eqns.loc[[compound.value],:]

#     return pandas_df_to_Dict(master_eqn, index=True)

@calcfunction
def get_master_equation(formation_energy_dict, compound, dependent_element, dopant):
    '''
    The 'master' equation is simply the equality corresponding to the formation energy of the
    compound under consideration. For ex. if we are studying the defect in Li3PO4, the master
    equation is simply: 3*mu_Li + mu_P + 4*mu_O = Ef where Ef is the formation energy per fu
    of Li3PO4. This equation is needed to replace the dependent chemical potential from the set
    of other linear constraints and to recover the chemical potential of the dependent element
    from the chemical potentials of independent elements
    '''
    Ef_dict = formation_energy_dict.get_dict()
    compound = compound.value
    composition = Composition(compound)
    dependent_element = dependent_element.value
    dopant = dopant.get_list()

    element_order = [atom.symbol for atom in composition if atom.symbol != dependent_element]
    if dopant == []:
        element_order.extend([dependent_element, 'Ef'])
    else:
        element_order.extend(dopant+[dependent_element, 'Ef'])
    master_eqn = pd.DataFrame(np.zeros((1, len(element_order))), index=[compound], columns=element_order)

    for atom in composition:
        master_eqn.loc[compound, atom.symbol] = composition[atom]
    master_eqn.loc[compound, 'Ef'] = Ef_dict[compound]

    return pandas_df_to_Dict(master_eqn, index=True)


@calcfunction
def get_reduced_matrix_of_constraints(full_matrix_of_constraints, compound, dependent_element):
    '''
    The reduced matrix of constraints is obtained from the full matrix of constraint by eliminating
    the row corresponding to the master equation and the column associated with the dependent element
    after substituting the chemical potential of the dependent element by that of the independent
    elements using the master equation (which at this stage is the first row of the full matrix of
    constraints). Therefore, if the shape of the full matrix of constraint is NxM, then the shape
    of the reduced matrix of constraints is (N-1)x(M-1)
    '''
    compound = compound.value
    dependent_element = dependent_element.value
    all_coefficients = full_matrix_of_constraints.get_dict()
    eqns = Dict_to_pandas_df(all_coefficients)
    master_eqn = eqns.loc[[compound],:]
    M = master_eqn.loc[compound].to_numpy()
    M = np.reshape(M, (1,-1))

    # Substitute the dependent element (variable) from the equations
    tmp = np.reshape(eqns[dependent_element].to_numpy(), (-1,1))*M/master_eqn.loc[compound, dependent_element]
    eqns = pd.DataFrame(eqns.to_numpy()-tmp, index=eqns.index, columns=eqns.columns)
    # Remove master equation and the column corresponding to the dependent element from the full matrix of constraints
    eqns = eqns.drop(compound)
    eqns = eqns.drop(columns=dependent_element)
    # print(eqns)

    return pandas_df_to_Dict(eqns, index=True)

@calcfunction
def get_stability_vertices(master_eqn, matrix_eqns, compound, dependent_element, tolerance):
    '''
    Solving the (reduced) matrix of constraints to obtain the vertices of the stability region.
    The last column (or coordinate) corresponds to the dependent element.
    '''
    master_eqn = master_eqn.get_dict()
    matrix_eqns = matrix_eqns.get_dict()
    set_of_constraints = np.array(matrix_eqns['data'])
    compound = compound.value
    dependent_element = dependent_element.value
    tolerance = tolerance.value
    N_species = set_of_constraints.shape[1]

    ### Look at all combination of lines (or plans or hyperplans) and find their intersections
    comb = combinations(np.arange(np.shape(set_of_constraints)[0]), N_species-1)
    intersecting_points = []
    for item in list(comb):
        try:
            point = np.linalg.solve(set_of_constraints[item,:-1], set_of_constraints[item,-1])
            intersecting_points.append(point)
        except np.linalg.LinAlgError:
            ### Singular matrix: lines or (hyper)planes are parallels therefore don't have any intersection
            pass

    ### Determine the points that form the vertices of stability region. These are intersecting point that verify all the constraints.
    intersecting_points = np.array(intersecting_points)
    get_constraint = np.dot(intersecting_points, set_of_constraints[:,:-1].T)
    check_constraint = (get_constraint - np.reshape(set_of_constraints[:,-1] ,(1, set_of_constraints.shape[0]))) <= tolerance
    bool_mask = [not(False in x) for x in check_constraint]
    corners_of_stability_region = intersecting_points[bool_mask]
    ### In some cases, we may have several solutions corresponding to the same points. Hence, the remove_duplicate method
    corners_of_stability_region = remove_duplicate(corners_of_stability_region)

    # if corners_of_stability_region.size != 0:
    #     self.report('The stability region cannot be determined. The compound {} is probably unstable'.format(compound))
    #     return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_FAILED

    stability_corners = pd.DataFrame(corners_of_stability_region, columns=matrix_eqns['column'][:-1])
    master_eqn = Dict_to_pandas_df(master_eqn)
    # get the chemical potentials of the dependent element
    dependent_chempot = get_dependent_chempot(master_eqn, stability_corners.to_dict(orient='list'), compound, dependent_element)
    stability_corners = np.append(stability_corners, np.reshape(dependent_chempot, (-1,1)), axis =1)
    stability_vertices = Dict({'column': matrix_eqns['column'][:-1]+[dependent_element], 'data': stability_corners})

    return stability_vertices

def get_dependent_chempot(master_eqn, chempots, compound, dependent_element):
    '''
    Calculate the chemical potential of the 'dependent' elements from the chemical potentials of 'independent' elements
    '''
    tmp = 0
    for col in master_eqn.columns:
        if col != dependent_element and col != 'Ef':
            tmp += np.array(chempots[col])*master_eqn.loc[compound, col]
    return (master_eqn.loc[compound, 'Ef']-tmp)/master_eqn.loc[compound, dependent_element]


@calcfunction
def get_centroid_of_stability_region(stability_corners, master_eqn, matrix_eqns, compound, dependent_element, grid_points, tolerance):
    '''
    Use to determine centroid or in some cases the center of the stability region. The center is defined as the average
    coordinates of the vertices while a centroid is the average cooridinates of every point inside the polygone or polyhedron,
    i.e. its center of mass.
    For binary compounds, the stability region is a one-dimensional segment. The centroid coincides with the center.
    For ternary, quarternary and quinary compounds, the centroid is returned.
    For compound with more that 5 elements, the center is returned.
    '''
    stability_corners = np.array(stability_corners.get_dict()['data'])
    master_eqn = master_eqn.get_dict()
    matrix_eqns = matrix_eqns.get_dict()
    compound = compound.value
    dependent_element = dependent_element.value
    tolerance = tolerance.value
    grid_points = grid_points.value

    points_in_stability_region = get_points_in_stability_region(stability_corners[:,:-1], np.array(matrix_eqns['data']), grid_points, tolerance)
    ctr_stability = get_centroid(points_in_stability_region) #without the dependent element
    ctr_stability = pd.DataFrame(np.reshape(ctr_stability, (1, -1)), columns=matrix_eqns['column'][:-1])

    master_eqn = Dict_to_pandas_df(master_eqn)
    # Add the corresponding chemical potential of the dependent element
    dependent_chempot = get_dependent_chempot(master_eqn, ctr_stability.to_dict(orient='list'), compound, dependent_element)
    ctr_stability = np.append(ctr_stability, np.reshape(dependent_chempot, (-1,1)), axis=1)
    ctr_stability = Dict({'column': matrix_eqns['column'][:-1]+[dependent_element], 'data': ctr_stability})

    return ctr_stability

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
        mp_entries.append(ComputedEntry(Composition(ref), 0.0))
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

def get_points_in_stability_region(stability_corners, matrix_eqns, N_point, tolerance):
    dim = stability_corners.shape[1]
    if dim ==1:
        return stability_corners
    elif dim == 2:
        [xmin, ymin] = np.amin(stability_corners, axis=0)
        [xmax, ymax] = np.amax(stability_corners, axis=0)
        x = np.linspace(xmin, xmax, N_point)
        y = np.linspace(ymin, ymax, N_point)
        xx, yy = np.meshgrid(x, y)
        points = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)
    elif dim == 3:
        [xmin, ymin, zmin] = np.amin(stability_corners, axis=0)
        [xmax, ymax, zmax] = np.amax(stability_corners, axis=0)
        x = np.linspace(xmin, xmax, N_point)
        y = np.linspace(ymin, ymax, N_point)
        z = np.linspace(zmin, zmax, N_point)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.append(xx.reshape(-1,1), yy.reshape(-1,1), axis=1)
        points = np.append(points, zz.reshape(-1,1), axis=1)
    elif dim == 4:
        [xmin, ymin, zmin, umin] = np.amin(stability_corners, axis=0)
        [xmax, ymax, zmax, umax] = np.amax(stability_corners, axis=0)
        x = np.linspace(xmin, xmax, N_point)
        y = np.linspace(ymin, ymax, N_point)
        z = np.linspace(zmin, zmax, N_point)
        u = np.linspace(umin, umax, N_point)
        xx, yy, zz, uu = np.meshgrid(x, y, z, u)
        points = np.append(xx.reshape(-1,1), yy.reshape(-1,1), axis=1)
        points = np.append(points, zz.reshape(-1,1), axis=1)
        points = np.append(points, uu.reshape(-1,1), axis=1)
    else:
        print('Not yet implemented for systems having more than 5 elements. Use center instead of centroid')
        return stability_corners

    get_constraint = np.dot(points, matrix_eqns[:,:-1].T)
    check_constraint = (get_constraint - np.reshape(matrix_eqns[:,-1], (1, matrix_eqns.shape[0]))) <= tolerance
    bool_mask = [not(False in x) for x in check_constraint]
    points_in_stable_region = points[bool_mask]

    return points_in_stable_region

def get_centroid(stability_region):
	return np.mean(stability_region, axis=0)

@calcfunction
def substitute_chemical_potential(matrix_eqns, fixed_chempot):
    '''
    substitute chemical potentials in the matrix of constraints by some fixed values.
    Useful for ex. for determining the 'slice' of stability region.
    '''
    matrix_eqns = Dict_to_pandas_df(matrix_eqns.get_dict())
    fixed_chempot = fixed_chempot.get_dict()
    for spc in fixed_chempot.keys():
        matrix_eqns.loc[:, 'Ef'] -= matrix_eqns.loc[:, spc]*fixed_chempot[spc]
        matrix_eqns = matrix_eqns.drop(columns=spc)

    for spc in fixed_chempot.keys():
        if spc in matrix_eqns.index:
            # print('Found!')
            matrix_eqns = matrix_eqns.drop(spc)
    # print(matrix_eqns)
    return pandas_df_to_Dict(matrix_eqns, index=True)

def Order_point_clockwise(points):
    '''
    The vertices of the stability region has to be ordered clockwise or counter-clockwise for plotting.
    Work only in 2D stability region

    points: 2d numpy array

    '''
    if points.shape[1] == 3:
        # Excluding the column corresponding to the dependent element (last column)
        points = points[:,:-1]
        center = np.mean(points, axis=0)
        # compute angle
        t = np.arctan2(points[:,0]-center[0], points[:,1]-center[1])
        sort_t = np.sort(t)
        t = list(t)
        u = [t.index(element) for element in sort_t]
        ordered_points = points[u]
        return ordered_points
    else:
        raise ValueError('The argument has to be a Nx3 numpy array')


@calcfunction
def get_absolute_chemical_potential(relative_chemical_potential, ref_energy):
    ref_energy = ref_energy.get_dict()
    relative_chemical_potential = relative_chemical_potential.get_dict()
    relative_chempot = Dict_to_pandas_df(relative_chemical_potential)

    absolute_chemical_potential = {}
    for element in relative_chempot.columns:
        absolute_chemical_potential[element] = ref_energy[element] + np.array(relative_chempot[element])

    return Dict(absolute_chemical_potential)

@calcfunction
def get_StabilityData(matrix_eqns, stability_vertices, compound, dependent_element):

    # from aiida_defects.data.data import StabilityData
    from aiida.plugins import DataFactory

    M = matrix_eqns.get_dict()
    vertices = stability_vertices.get_dict()

    StabilityData = DataFactory('array.stability')
    stability_region = StabilityData()
    stability_region.set_data(np.array(M['data']), M['index'], M['column'], np.array(vertices['data']), compound.value, dependent_element.value)

    return stability_region
