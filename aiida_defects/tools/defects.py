# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import
import sys
import pymatgen
import numpy as np
from copy import deepcopy
from aiida.orm.nodes import Dict
from aiida.orm import StructureData
from aiida.orm.nodes import ArrayData
from aiida.engine import WorkFunction
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida.orm import load_node
import six
from six.moves import range
from six.moves import zip

#################################################################################################
#This module contains:								   #
# 1) defect_creator(host_structure, vacancies, substitutions, scale_sc, cluster)   #
# 2) explore_defect(host_structure, defective_structure, defect_type)		   #
# 3) distance_from_defect(defective_structure, defect_position)			   #
# 4) find_defect_index(defect_creator_output)                                      #
# 5) defect_creator_by_index(structure, find_defect_index_output)                  #
# 6) distance_from_defect_aiida(defective_structure, defect_position)              #
#################################################################################################


@workfunction
def defect_creator(host_structure, vacancies, substitutions, scale_sc,
                   cluster):
    """
    Workfunction that creates defects in a host structure on the basis of symmetry considerations.

    Starting from a given host structure, which can be a unit cell or a supercell, defects are 
    automatically created. Vacancy or substitutional defects can be created, as well as clusters.

    .. Todo:: 
        
        - Better labels, especially for substitutions and clusters.
        - Change input type test logic - currently it breaks if a list is reused.


    Parameters
    ----------
    host_structure : StructureData
        Corresponding to the host structure.
    vacancies : list
        List containing the specie of the host_structure for which we want to create vacancies.
        If clusters of the same vacancy type are to be formed the element should be repeated in
        the vacancies list.
    substitutions : dict 
        Dictionary like {"Mn" : ["Ti", "Co"]} meaning that Mn in the host_structure will be 
        substituted with either Ti or Co.                     
    scale_sc : list
        Scale parameter for the supercell creation. List of the form [a,b,c], which will result
        in a (a*b*c) supercell.
    cluster : boolean
        If true defective structures containing all the requested defects are created. 
        Otherwise defective structures, each containing only one of the requested defects, will be created.
    
    Returns
    -------    
    defective_structures : dict
            Dictionary of all the defective structures created, as StructureData objects. 
            The structure labeled `_0` is the host structure.
    """
    #Checking that you specified at leat one type of defects
    #if not list(vacancies) and not substitutions.get_dict():
    #    sys.exit("You did not specified a defect type. Please check your inputs")

    #Converting an AiiDA StructureData in pymatgen Structure
    host_mg = host_structure.get_pymatgen()

    #Check that the elements for which defects should be created are contained in host_structure
    elements_mg = host_mg.types_of_specie
    elements = []
    for element in elements_mg:
        elements.append(str(element))

    for specie in vacancies:
        if specie not in elements:
            sys.exit(
                "The  element {} for which the creation of vacancies was requested is not contained in the host structure. Check your input"
                .format(specie))

    substitutions = substitutions.get_dict()

    for specie in substitutions:
        if specie not in elements:
            sys.exit(
                "The  element {} for which the creation of substitutional/antisite defects was requested is not contained in the host structure. Check your input"
                .format(specie))

    #Check that more than one defect is specified when a cluster is requested
    n_vac = 0
    n_sub = 0
    for specie in vacancies:
        n_vac += 1
    for specie in substitutions:
        n_sub += 1

    if cluster and n_vac + n_sub <= 1:
        sys.exit(
            "You cannot create a cluster of defects by specifing only one defect"
        )

    from pymatgen.core import Structure
    from pymatgen.core.sites import PeriodicSite
    from pymatgen.analysis.defects.point_defects import Defect, Vacancy, ValenceIonicRadiusEvaluator

    def structure_analyzer(host_mg):
        """
        Perform symmetry analysis of defect sites

        Parameters
        ----------
        host_mg : 
            ?
        
        Returns
        -------
        vacancy : 
            ?

        """
        valence_evaluator = ValenceIonicRadiusEvaluator(host_mg)
        radii = valence_evaluator.radii
        valences = valence_evaluator.valences
        vacancy = Vacancy(host_mg, radii, valences)

        return vacancy

    #Setting up the defective supercell
    scaling_matrix = [[scale_sc[0], 0, 0], [0, scale_sc[1], 0],
                      [0, 0, scale_sc[2]]]
    limit_return_structures = None

    #Creating individual defects
    if not cluster:

        vacancy = structure_analyzer(host_mg)

        #Creating vacancies
        vac = {}
        vac_scs = []
        tmp_vac_scs_final = []

        for specie in vacancies:
            vac_scs = vacancy.make_supercells_with_defects(
                scaling_matrix, specie, limit_return_structures)
            tmp_vac_scs = deepcopy(vac_scs)
            host_final = tmp_vac_scs[0]
            del tmp_vac_scs[0]
            tmp_vac_scs_final += tmp_vac_scs

            for i in range(1, len(vac_scs)):
                vac_sc_site = list(
                    set(vac_scs[0].sites) - set(vac_scs[i].sites))

        if len(vacancies) > 0:
            vac_scs_final = [host_final]
        else:
            vac_scs_final = [host_mg]
        vac_scs_final += tmp_vac_scs_final

        #Creating substitutions & antisites

        sub = {}
        sub_scs = []
        sub_scs_final = []

        #substitutions=substitutions.get_dict()
        for specie, substitute in six.iteritems(substitutions):
            for element in substitute:
                sub_scs = vacancy.make_supercells_with_defects(
                    scaling_matrix, specie, limit_return_structures)

                for i in range(1, len(sub_scs)):
                    sub_sc_site = list(
                        set(sub_scs[0].sites) - set(sub_scs[i].sites))
                    sub_scs[i].append(element, sub_sc_site[0].frac_coords)

                tmp_sub_scs = deepcopy(sub_scs)
                host_final = tmp_sub_scs[0]
                del tmp_sub_scs[0]
                sub_scs_final += tmp_sub_scs

        sub_scs_1 = [host_final]
        sub_scs_1 += sub_scs_final

    #Creating clusters of defects
    else:

        #Creating vacancies
        vac = {}
        vac_scs = []
        num_vac = len(vacancies)
        num_defect = 0
        tmp_structure = [host_mg]

        while num_defect != num_vac:
            for structure in tmp_structure:
                vacancy = structure_analyzer(structure)
                vac_scs = vacancy.make_supercells_with_defects(
                    scaling_matrix, vacancies[num_defect],
                    limit_return_structures)

            num_defect += 1
            tmp_structure = vac_scs
            host_comp = tmp_structure[0]
            del tmp_structure[0]

        #Creating substitutions or antisites
        sub = {}
        sub_scs = []
        sub_scs_final = []
        sub_scs_final2 = []

        #substitutions=substitutions.get_dict()
        species = []
        for specie in substitutions:
            species.append(specie)

        if not vacancies:
            tmp_structure = [host_mg]
        else:
            tmp_structure = vac_scs

        num_defect = 0
        num_sub = len(species)

        if species:
            while num_defect != num_sub:
                for structure in tmp_structure:
                    for element in substitutions[str(species[num_defect])]:
                        vacancy = structure_analyzer(structure)
                        sub_scs = vacancy.make_supercells_with_defects(
                            scaling_matrix, str(species[num_defect]),
                            limit_return_structures)

                        for i in range(1, len(sub_scs)):
                            sub_sc_site = list(
                                set(sub_scs[0].sites) - set(sub_scs[i].sites))
                            sub_scs[i].append(element,
                                              sub_sc_site[0].frac_coords)

                        tmp_sub_scs = deepcopy(sub_scs)
                        #host_comp = tmp_sub_scs[0]
                        del tmp_sub_scs[0]
                        sub_scs_final += tmp_sub_scs

                tmp_sub_scs2 = deepcopy(sub_scs_final)
                sub_scs_final = []

                num_defect += 1
                tmp_structure = tmp_sub_scs2

        clusters = [host_comp]
        clusters += tmp_structure

    #Creating a dictionary containing all the defective structures, in the form of AiiDA StructureData objects
    defective_structures = {}

    if not cluster:
        if len(vac_scs_final) > 1:
            for n, supercell in enumerate(vac_scs_final):
                defective_structures["vacancy_" + str(n)] = StructureData(
                    pymatgen=supercell)
        if len(sub_scs_1) > 1:
            for n, supercell in enumerate(sub_scs_1):
                defective_structures["substitution_" + str(n)] = StructureData(
                    pymatgen=supercell)
    else:
        for n, supercell in enumerate(clusters):
            defective_structures["cluster_" + str(n)] = StructureData(
                pymatgen=supercell)

    return defective_structures


def explore_defect(host_structure, defective_structure, defect_type):
    """
    Finds the position and atom_type of a defect in a defective structure.

    Compares two supercells - one host supercell and one defective supercell, infering the
    positions and types of any defects present.

    Assumptions:

    1) Supercell shape, volume and atom coordinates are the same in host_structure and 
       defective_structure.
       If the defective structure was generated using the :meth:`defect_creator` workfunction, 
       use the host structure returned by the same function for host_structure.

    Parameters
    ----------
    host_structure : StructureData
        The host structure.
    defective_structure : StructureData
        The defective structure.
    defect_type: 
        Type of defect as given by the :meth:`defect_creator` workfunction (vacancy, substitution, cluster, unknown)
    
    Returns
    -------
    defect_info : dict
        A dictionary containing the following items:
            1) Numpy position vector of the defect (in cartesian coordinates)
            2) Atom type
            3) Name to classify the defect
    
    """
    defect_info = {}

    def fractional_coordinates(structure_mg):
        cell_a = structure_mg.lattice.a
        cell_b = structure_mg.lattice.b
        cell_c = structure_mg.lattice.c

        frac = []
        cart = []
        for i in structure_mg.sites:
            x = round(i.frac_coords[0], 5)
            y = round(i.frac_coords[1], 5)
            z = round(i.frac_coords[2], 5)

            while x < 0:
                x += 1.
            while y < 0:
                y += 1.
            while z < 0:
                z += 1.

            while x >= 1.:
                x -= 1.
            while y >= 1.:
                y -= 1.
            while z >= 1.:
                z -= 1.

            frac.append(
                str(i.specie) + "-" + str(x) + '_' + str(y) + '_' + str(z))
            cart.append(np.array([i.coords[0], i.coords[1], i.coords[2]]))
        return {'frac': frac, 'cart': cart}

    def explore_vacancy(host_mg, defect_mg):
        """
        ?

        Parameters
        ----------
        host_mg : 
            ?
        defect_mg :
            ?
        
        Returns
        -------
        defect_info : 
            ?

        """
        coord_host = fractional_coordinates(host_mg)
        coord_defect = fractional_coordinates(defect_mg)

        host_frac = coord_host['frac']
        def_frac = coord_defect['frac']

        defect_site = list(set(host_frac) - set(def_frac))

        n_cart = [
            n for n, site in enumerate(host_frac) if site in defect_site
        ][0]

        defect_info = {
            'atom_type': str(defect_site[0].split('_')[0].split('-')[0]),
            'defect_name':
            "V_" + str(defect_site[0].split('_')[0].split('-')[0]),
            'defect_position': coord_host['cart'][n_cart]
            #                        'defect_position' : list([str(defect_site[0].split('_')[0].split('-')[1]),
            #                                                  defect_site[0].split('_')[1],
            #                                                  defect_site[0].split('_')[2]]),
        }
        return defect_info

    def explore_substitution(host_mg, defect_mg):
        """
        ?

        Parameters
        ----------
        host_mg : 
            ?
        defect_mg :
            ?
        
        Returns
        -------
        defect_info : 
            ?

        """
        coord_host = fractional_coordinates(host_mg)
        coord_defect = fractional_coordinates(defect_mg)

        host_frac = coord_host['frac']
        def_frac = coord_defect['frac']

        defect_site = list(set(def_frac) - set(host_frac))
        defect_site_host = list(set(host_frac) - set(def_frac))

        n_cart = [
            n for n, site in enumerate(host_frac) if site in defect_site_host
        ][0]

        defect_info = {
            'atom_type':
            str(defect_site[0].split('_')[0].split('-')[0]),
            "defect_name":
            str(defect_site_host[0].split('_')[0].split('-')[0]) + "_" + str(
                defect_site[0].split('_')[0].split('-')[0]),
            'defect_position':
            coord_host['cart'][n_cart]
            #                       list([str(defect_site[0].split('_')[0].split('-')[1]),
            #                                                  defect_site[0].split('_')[1],
            #                                                  defect_site[0].split('_')[2]]),
        }
        return defect_info

    def explore_cluster(host_mg, defect_mg):
        """
        ?

        Parameters
        ----------
        host_mg : 
            ?
        defect_mg :
            ?
        
        Returns
        -------
        defect_info : 
            ?

        """
        elements_mg = host_mg.types_of_specie
        elements = []
        for element in elements_mg:
            elements.append(str(element))

        coord_host = fractional_coordinates(host_mg)
        coord_defect = fractional_coordinates(defect_mg)

        host_frac = coord_host['frac']
        def_frac = coord_defect['frac']

        defect_sites = list(set(host_frac) - set(def_frac))
        defect_sites_host = list(set(def_frac) - set(host_frac))

        for n, site in enumerate(defect_sites):
            check_vac = [
                site for j in def_frac if (str(site.split('-')[1])) in j
            ]

            if str(site.split('_')[0].split('-')[0]) in elements and bool(
                    check_vac) == False:
                n_cart = [k for k, sit in enumerate(host_frac)
                          if sit in site][0]
                defect_info = {
                    'atom_type' + "_v_" + str(n):
                    str(site.split('_')[0].split('-')[0]),
                    'defect_name' + "_v_" + str(n):
                    "V" + "_" + str(site.split('_')[0].split('-')[0]),
                    'defect_position' + "_v_" + str(n):
                    coord_host['cart'][n_cart]
                    #                                list([str(site.split('_')[0].split('-')[1]),
                    #                                                                          site.split('_')[1],
                    #                                                                          site.split('_')[2]])
                }

        for n, site in enumerate(defect_sites_host):
            for i in defect_sites:
                if site.split('_')[1] == i.split('_')[1]:
                    element = str(i.split('_')[0].split('-')[0])

            n_cart = [k for k, sit in enumerate(def_frac) if sit in site][0]

            defect_info['atom_type' + "_s_" + str(n)] = str(
                site.split('_')[0].split('-')[0])
            defect_info['defect_name' + "_s_" +
                        str(n)] = str(element) + "_" + str(
                            site.split('_')[0].split('-')[0])
            defect_info['defect_position' + "_s_" + str(n)] = list([
                str(site.split('_')[0].split('-')[1]),
                site.split('_')[1],
                site.split('_')[2]
            ])
        return defect_info

    host_mg = host_structure.get_pymatgen()
    defect_mg = defective_structure.get_pymatgen()

    if defect_type == "vacancy":
        defect_info = explore_vacancy(host_mg, defect_mg)
    elif defect_type == "substitution":
        defect_info = explore_substitution(host_mg, defect_mg)
    elif defect_type == "cluster":
        defect_info = explore_cluster(host_mg, defect_mg)
    elif defect_type == "unknown":

        n_atoms_host = 0
        for site in host_mg.sites:
            n_atoms_host += 1
        n_atoms_defect = 0
        for site in defect_mg.sites:
            n_atoms_defect += 1

        host_frac = fractional_coordinates(host_mg)['frac']
        def_frac = fractional_coordinates(defect_mg)['frac']
        n_defects = len(list(set(host_frac) - set(def_frac)))

        if n_defects == 1 and n_atoms_defect < n_atoms_host:
            defect_info = explore_vacancy(host_mg, defect_mg)
        elif n_defects == 1 and n_atoms_defect == n_atoms_host:
            defect_info = explore_substitution(host_mg, defect_mg)
        elif n_defects > 1:
            defect_info = explore_cluster(host_mg, defect_mg)
        else:
            sys.exit("Error: check your input structures.")
    else:
        sys.exit(
            "{} is not a valid value for the variable defect_type. Please insert one of the following: \
                \n vacancy, substitution, cluster, unknown".format(
                defect_type))

    return defect_info


def distance_from_defect(defective_structure, defect_position):
    """
    Computes the distance of each atom site from the defect site.

    Parameters
    ----------
    defective_structure : StructureData 
        The defective structure.
    defect_position : numpy.ndarray 
        Cartesian coordinates of the defect.

    Returns
    -------
    distances_from_defect : dict
        One entry per periodic site (as defined by pymatgen)
        corresponding to the distance of that site from the defect.
    """
    from math import sqrt
    from mpmath import nint

    cell_x = defective_structure.cell[0][0]
    cell_y = defective_structure.cell[1][1]
    cell_z = defective_structure.cell[2][2]

    defect_mg = defective_structure.get_pymatgen()
    distances = []
    #distances_from_defect = {}
    for site in defect_mg.sites:
        distance = site.distance_from_point(defect_position)
        distances.append(distance)

    distances_from_defect = list(zip(defect_mg.sites, distances))

    return distances_from_defect


def find_defect_index(defect_creator_output):
    """
    Identifies the index of the atom in the host structure that is removed/substituted
    in order to create the defect. 
    
    Parameters
    ----------
    defect_creator_output : dict
        Host and defective structures created using the :meth:`defect_creator` workfunction

    Returns
    -------
    Nested dictionary 
        containing one entry for every vacancy/substitution, which is a dictionary 
        containing:
           * the index
           * the defect_name
           * the defect_position 
           * the atom_type
        In the case of clusters, for every cluster created it contains one dictionary with the same 
        info as above for every defect created to obtain that cluster
        (e.g. cluster_1['defect_name_v_0] and cluster_1['defect_name_s_0] for a cluster made by a vacancy
        and a substitution)
    """

    def find_vacancy_index(defect_creator_output):
        vacancies = {}
        if 'vacancy_0' in defect_creator_output:
            ref = defect_creator_output['vacancy_0']
        else:
            pass
        for defect, structure in six.iteritems(defect_creator_output):
            if 'vacancy' in defect and str(defect) != 'vacancy_0':
                info = explore_defect(ref, structure, 'vacancy')
                vacancies[str(defect)] = {}
                for n, site in enumerate(ref.sites):
                    if np.array_equal(site.position, info['defect_position']):
                        vacancies[str(defect)]['index'] = n
                        vacancies[str(
                            defect)]['defect_name'] = info['defect_name']
                        vacancies[str(defect)]['defect_position'] = info[
                            'defect_position']
                        vacancies[str(defect)]['atom_type'] = info['atom_type']

        return vacancies

    def find_substitution_index(defect_creator_output):
        substitutions = {}
        if 'substitution_0' in defect_creator_output:
            ref = defect_creator_output['substitution_0']
        else:
            pass
        for defect, structure in six.iteritems(defect_creator_output):
            if 'substitution' in defect and str(defect) != 'substitution_0':
                info = explore_defect(ref, structure, 'substitution')
                substitutions[str(defect)] = {}
                for n, site in enumerate(ref.sites):
                    if np.array_equal(site.position, info['defect_position']):
                        substitutions[str(defect)]['index'] = n
                        substitutions[str(
                            defect)]['defect_name'] = info['defect_name']
                        substitutions[str(defect)]['defect_position'] = info[
                            'defect_position']
                        substitutions[str(
                            defect)]['atom_type'] = info['atom_type']

        return substitutions

    def find_cluster_index(defect_creator_output):
        clusters = {}
        if 'cluster_0' in defect_creator_output:
            ref = defect_creator_output['cluster_0']
        else:
            pass
        for defect, structure in six.iteritems(defect_creator_output):
            if 'cluster' in defect and str(defect) != 'cluster_0':
                info = explore_defect(ref, structure, 'cluster')
                clusters[str(defect)] = {}
                defect_name_list = [
                    key for key in info if 'defect_name' in key
                ]
                for element in defect_name_list:
                    clusters[str(defect)][str(element)] = {}
                    for n, site in enumerate(ref.sites):
                        if np.array_equal(
                                site.position,
                                info['defect_position_' + element.split('_')[2]
                                     + '_' + element.split('_')[3]]):
                            clusters[str(defect)][element]['index'] = n
                            clusters[str(defect)][element][
                                'defect_position_' + element.split('_')[2] +
                                '_' + element.split('_')[3]] = info[
                                    'defect_position_' + element.split('_')[2]
                                    + '_' + element.split('_')[3]]
                            clusters[str(defect)][element][
                                'atom_type_' + element.split('_')[2] + '_' +
                                element.split('_')[3]] = info[
                                    'atom_type_' + element.split('_')[2] + '_'
                                    + element.split('_')[3]]
        return clusters

    def merge_dicts(*dict_args):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    vac = find_vacancy_index(defect_creator_output)
    sub = find_substitution_index(defect_creator_output)
    clu = find_cluster_index(defect_creator_output)

    return merge_dicts(vac, sub, clu)


@workfunction
def defect_creator_by_index(structure, find_defect_index_output):
    """
    Workfunction to create defects on the basis of the index of the atoms in the structure.

    This function is meant to be used, for example, for a defect calculation in strained cells,
    where the symmetry can change compared to the unstrained structure. You apply the 
    :meth:`defect_creator` function on the unstrained structure, and then apply  
    :meth:`find_defect_index_output` on the resulting dictionary.
    Finally, you apply this function on the other strained cells.
    If you want to use it independently from the :meth:`defect_creator` and 
    :meth:`find_defect_index` functions, you should create a :meth:`find_defect_index_output` 
    ParameterData object in the following way:
    
        * For a vacancy:

            .. code-block:: python

                find_defect_index_output=ParameterData(
                    dict={
                        'vacancy_1': {'index': XX}
                    }
                )
    
        * For a substitution:
    
            .. code-block:: python
            
                find_defect_index_output=ParameterData(
                    dict={
                        'substitution_1': {'index': XX, 'atom_type' : YYY}
                    }
                )  
    
        * For a cluster:
        
            .. code-block:: python

                find_defect_index_output=ParameterData(
                    dict={
                        'cluster_1': {
                            'defect_name_v_0 ': {'index': XX},
                            'defect_name_s_0 ': {'index': XX, 'atom_type_s_0' : YYY}
                        }
                    }
                )

    Parameters
    ----------
    structure : StructureData 
        Host structure to create defects in.
    find_defect_index_output : ParameterData 
        Object returned by :meth:`find_defect_index` function.

    Returns
    -------
    defective_structures : dict
        Dictionarty containing StructureData objects with defects applied.
    
    """
    structure_mg = structure.get_pymatgen()
    find_defect_index_output = find_defect_index_output.get_dict()

    def create_vacancy(struct, index):
        struct2 = deepcopy(struct)
        del struct2[int(index)]
        return struct2

    def create_substitution(struct, index, dopant):
        struct2 = deepcopy(struct)
        sub_site = struct[int(index)]
        del struct2[int(index)]
        struct2.append(str(dopant), sub_site.frac_coords)
        return struct2

    defects_mg = {}

    for defect in find_defect_index_output:
        if 'vacancy_' in defect:
            defects_mg[defect] = create_vacancy(
                structure_mg, find_defect_index_output[defect]['index'])
        elif 'substitution_' in defect:
            defects_mg[defect] = create_substitution(
                structure_mg, find_defect_index_output[defect]['index'],
                find_defect_index_output[defect]['atom_type'])
        elif 'cluster_' in defect:
            for defect_name in find_defect_index_output[defect]:
                if 'defect_name_v' in defect_name:
                    #print  find_defect_index_output[defect][defect_name]
                    defects_mg[defect] = create_vacancy(
                        structure_mg,
                        find_defect_index_output[defect][defect_name]['index'])
            for defect_name in find_defect_index_output[defect]:
                if 'defect_name_s' in defect_name:
                    sub_site = structure_mg[int(
                        find_defect_index_output[defect][defect_name]
                        ['index'])]
                    for n, site in enumerate(defects_mg[defect].sites):
                        if np.array_equal(site.frac_coords,
                                          sub_site.frac_coords):
                            del defects_mg[defect][int(n)]
                            defects_mg[defect].append(
                                find_defect_index_output[defect][defect_name]
                                ['atom_type_' + defect_name.split('_')[2] + '_'
                                 + defect_name.split('_')[3]],
                                sub_site.frac_coords)

    defective_structures = {}
    for name, structs in six.iteritems(defects_mg):
        defective_structures[name] = StructureData(pymatgen=structs)

    for key in defects_mg.keys():
        if 'vacancy' in key:
            defective_structures['vacancy_0'] = structure
            break
    for key in defects_mg.keys():
        if 'substitution' in key:
            defective_structures['substitution_0'] = structure
            break
    for key in defects_mg.keys():
        if 'cluster' in key:
            defective_structures['cluster_0'] = structure
            break

    return defective_structures


def distance_from_defect_aiida(defective_structure, defect_position):
    """
    Computes the distance of each atom site from the defect site. 

    Equivalent to :meth:`distance_from_defect` but uses the coordinates in Ångstrom from the 
    AiiDA StructureData object without converting to the pymatgen format.
    
    .. warning::
        Only for use in orthogonal cells. For non-orthogonal cells use ...

    Parameters
    ----------
    defective_structure : StructureData 
        The defective structure.
    defect_position : numpy.ndarray
        The cartesian coordinates of the defect.

    Returns
    -------
    distances_from_defect : dict
        One entry per periodic site (pymatgen periodic site) corresponding to 
        the distance of that site from the defect.

    """
    from math import sqrt
    from mpmath import nint

    cell_x = defective_structure.cell[0][0]
    cell_y = defective_structure.cell[1][1]
    cell_z = defective_structure.cell[2][2]

    #defect_mg = defective_structure.get_pymatgen()
    distances = []
    #distances_from_defect = {}
    for site in defective_structure.sites:
        dist_x = site.position[0] - defect_position[0]
        dist_x = dist_x - nint(dist_x / cell_x) * cell_x
        dist_y = site.position[1] - defect_position[1]
        dist_y = dist_y - nint(dist_y / cell_y) * cell_y
        dist_z = site.position[2] - defect_position[2]
        dist_z = dist_z - nint(dist_z / cell_z) * cell_z

        distance = sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        distances.append(distance)

    distances_from_defect = list(zip(defective_structure.sites, distances))

    return distances_from_defect


def distance_from_defect_pymatgen(defective_structure, defect_position):
    """
    Computes the distance of each atom site from the defect site.

    .. warning::
        Only for use in orthogonal cells. For non-orthogonal cells use ...

    Parameters
    ----------
    defective_structure : StructureData 
        The defective structure.
    defect_position : numpy.ndarray 
        The cartesian coordinates of the defect

    Returns
    -------
    distances_from_defect : list of tuples
        One entry per periodic site (pymatgen periodic site)
        corresponding to the distance of the site from the defect
        The first item in the tuple is the pymatgen 'PeriodicSite'
        object, describing the site, and the second is a float 
        describing the distance of this site from the defect.
    """
    from math import sqrt
    from mpmath import nint

    cell_x = defective_structure.cell[0][0]
    cell_y = defective_structure.cell[1][1]
    cell_z = defective_structure.cell[2][2]

    defect_mg = defective_structure.get_pymatgen()
    distances = []
    #distances_from_defect = {}
    for site in defect_mg.sites:
        dist_x = site.coords[0] - defect_position[0]
        dist_x = dist_x - nint(dist_x / cell_x) * cell_x
        dist_y = site.coords[1] - defect_position[1]
        dist_y = dist_y - nint(dist_y / cell_y) * cell_y
        dist_z = site.coords[2] - defect_position[2]
        dist_z = dist_z - nint(dist_z / cell_z) * cell_z

        distance = sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        distances.append(distance)

    distances_from_defect = list(zip(defect_mg.sites, distances))

    return distances_from_defect
