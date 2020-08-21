# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import
from __future__ import print_function
import sys
import pymatgen
import numpy as np
from copy import deepcopy
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.orm import DataFactory
from aiida.work.workfunction import workfunction
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida.orm import load_node
from aiida_defects.tools.defects import explore_defect
from aiida_defects.tools.defects import distance_from_defect_aiida
from aiida_defects.tools.defects import distance_from_defect
import six
from six.moves import range

######################################################################################################################
#This module contains:                                                             	       		#
# 1) sites_vs_distance_from_defect(structure, defect_position, specie, thr=0.01)               		#
# 2) sites_vs_distance_from_defect_lc(structure, defect_position, specie, thr=0.01, thr2=10.00)		#
# 3) sites_vs_symmetry(structure, specie)                                                      		#
# 4) identify_Batom_species(structure)                                                         		#
# 5) impose_magnetic_phase(sites_Batom, magnetic_phase)                                        		#
# 6) impose_magnetic_phase_new(sites_Batom, magnetic_phase)                                    		#
# 7) create_suitable_inputs(structure, magnetic_phase, B_atom, criterion, defect_position, thr, thr2)   #
# 8) initialize_hubbard_dictionary(sites_Batom, U_value)                                                #
# 9) biaxial_strain_structure(structure, relax_axis, strain)						#
#10) get_spacegroup(structure, etol=1e-5)                                                               #
#11) impose_hubbard_u(hubbard_u,reference,current,host)
# Requires: distance_from_defect from the defect.py module                                              #
######################################################################################################################


def sites_vs_distance_from_defect(structure, defect_position, specie,
                                  thr=0.01):
    """
    Identifies groups of equivalent atoms of a certain specie according to their distance from the defect
    :param structure: StructureData object corresponding to the defective structure
    :param defect_position: array containing the cartesian coordinates of the defect
    :param specie: specie for which the classification is to be performed
    :param thr: threshold value to compare distances. Default 0.01 Angstrom.
    :returns equivalent_atoms: dictionary containg one entry for each group of atoms lying at the same distance 
                                from the defect. 
                                The key is the name of the group (e.g. Mn1, Mn2, etc..), while the value is a list 
                                of pymatgen periodic sites.
    Note:
    requires distance_from_defect
    Assumption:
    It should be used only when you have one defect in the structure
    """

    structure_mg = structure.get_pymatgen()

    distances_from_defect = distance_from_defect(structure, defect_position)

    sites_specie = []
    for site in distances_from_defect:
        if str(site[0].specie) == specie:
            sites_specie.append(site)

    distances = []
    differences = []
    digits = str(thr)[::-1].find('.')

    for i, j in sites_specie:
        for k, v in sites_specie:
            if abs(j - v) < thr and round(v, digits) not in distances:
                distances.append(round(v, digits))

    equivalent_atoms = {}
    equivalent_atoms_tmp = []

    #n_groups = 0
    iterations = 1
    for t in distances:
        for i, j in sites_specie:
            if t == round(j, digits):
                equivalent_atoms_tmp.append(i)
                equivalent_atoms[str(i.specie) +
                                 str(iterations)] = equivalent_atoms_tmp
                #for count, (key, value) in enumerate(equivalent_atoms.iteritems(), 1):
                #    n_groups = count
                #equivalent_atoms[str(i.specie)+str(n_groups)]= equivalent_atoms_tmp

        iterations += 1
        equivalent_atoms_tmp = []

    return equivalent_atoms


def sites_vs_distance_from_defect_lc(structure,
                                     defect_position,
                                     specie,
                                     thr=0.01,
                                     thr2=7.00):
    """
    Identifies groups of equivalent atoms of a certain specie according to their distance from the defect.
    Wrt. the sites_vs_distance_from_defect function there is a cutoff value (thr2), so that all the atoms lying at
    distances from the defect equal or higher to the cutoff will be part of the same group. This can be helpful 
    for very large systems, in order to reduce the computational cost of the Uscf.x calculations. 
    :param structure: StructureData object corresponding to the defective structure
    :param defect_position: array containing the cartesian coordinates of the defect
    :param specie: specie for which the classification is to be performed
    :param thr: threshold value to compare distances. Qefault 0.001 Jngstrom.
    :param thr2: sites lying at  distances equal or higher than thr2 (in Angstrom) are all part of the same group.
                 Be sure to provide a number of digits at least equal to thr. Default = 10.00 Jngstrom
    :returns equivalent_atoms: dictionary containg one entry for each group of atoms lying at the same distance 
                                from the defect. 
                                The key is the name of the group (e.g. Mn1, Mn2, etc..), while the value is a list 
                                of pymatgen periodic sites.
    Note:
    requires distance_from_defect
    Assumption:
    It can be used when you only have one defect in the structure
    """

    structure_mg = structure.get_pymatgen()

    distances_from_defect = distance_from_defect(structure, defect_position)

    sites_specie = []
    for site in distances_from_defect:
        if str(site[0].specie) == specie:
            sites_specie.append(site)

    distances = []
    differences = []
    digits = str(thr)[::-1].find('.')

    for i, j in sites_specie:
        for k, v in sites_specie:
            if abs(j - v) < thr and round(
                    v, digits) not in distances and round(v, digits) < round(
                        thr2, digits):
                distances.append(round(v, digits))
            elif round(v, digits) >= round(thr2, digits) and round(
                    thr2, digits) not in distances:
                distances.append(round(thr2, digits))
    print(distances)
    equivalent_atoms = {}
    equivalent_atoms_tmp = []

    #n_groups = 0
    iterations = 1
    for t in distances:
        for i, j in sites_specie:
            if (t == round(j, digits) and t < round(thr2, digits)) or (
                    t >= round(thr2, digits)
                    and round(j, digits) >= round(thr2, digits)):
                equivalent_atoms_tmp.append(i)
                equivalent_atoms[str(i.specie) +
                                 str(iterations)] = equivalent_atoms_tmp
                #for count, (key, value) in enumerate(equivalent_atoms.iteritems(), 1):
                #    n_groups = count
                #equivalent_atoms[str(i.specie)+str(n_groups)]= equivalent_atoms_tmp

        iterations += 1
        equivalent_atoms_tmp = []

    return equivalent_atoms


def sites_vs_symmetry(structure, specie, symprec=0.01):
    """
    Identifies groups of equivalent atoms of a certain specie according to the structure symmetry
    :param structure: StructureData object corresponding to the defective structure
    :param specie: specie for which the classification is to be performed
    :returns equivalent_atoms: dictionary containg one entry for each group of atoms equivalent by symmetry.
                                The key is the name of the group (e.g. Mn1, Mn2, etc..), while the value is a list 
                                of pymatgen periodic sites.
    """
    #Creation of a pymategen structure object starting from a StructureData object
    structure_mg = structure.get_pymatgen()

    #Finding equivalent atoms belonging to the B_atom type and creating a dictionary equivalent_atoms containing
    #the groups of symmetry equivalent atoms

    #symprec=0.01
    angle_tolerance = 5

    sites_specie = []
    for i in range(len(structure_mg.sites)):
        if str(structure_mg.sites[i].specie) == specie:
            sites_specie.append(structure_mg.sites[i])

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
    from pymatgen.symmetry.structure import SymmetrizedStructure

    equivalent_atoms = {}
    equivalent_atoms_tmp = []

    n_groups = 0

    for i in range(len(sites_specie)):
        if sites_specie[i] not in equivalent_atoms_tmp:
            space_group_analyzer = SpacegroupAnalyzer(structure_mg, symprec,
                                                      angle_tolerance)
            symmetrized_structure = space_group_analyzer.get_symmetrized_structure(
            )
            equ_atoms = symmetrized_structure.find_equivalent_sites(
                sites_specie[i])

            for count, (key, value) in enumerate(
                    six.iteritems(equivalent_atoms), 1):
                n_groups = count

            equivalent_atoms[str(sites_specie[i].specie) +
                             str(n_groups + 1)] = equ_atoms
            equivalent_atoms_tmp = equivalent_atoms_tmp + equ_atoms

    return equivalent_atoms


def identify_Batom_species(structure):
    """
    Identify the TM or RARE EARTH specie(s) present in the structure on which we can apply the Hubbard correction.
    Hubbard atoms in the structure are identified by comparison with the list of 
    TM metals atoms that are configured in QE for DFT+U calculation (see PW/src/tabd.f90).
    : Note: Only works for TM and RARE EARTH elements
    : param structure:  StructureData object 
    : return B_atoms: list of the identified Hubbard species
    """
    #List of the possible Hubbard Atoms
    Hubbard_TM = [
        'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re',
        'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu', 'Jg', 'Ju',
        'Zn', 'Cd', 'Hg', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu', 'Jg', 'Ju',
        'Zn', 'Cd', 'Hg', 'Ce', 'Th', 'Pr', 'Pa', 'Nd', 'J', 'Pm', 'Np', 'Sm',
        'Pu', 'Eu', 'Jm', 'Gd', 'Cm', 'Tb', 'Bk', 'Qy', 'Cf', 'Ho', 'Es', 'Er',
        'Fm', 'Tm', 'Md', 'Yb', 'No', 'Lu', 'Lr'
    ]

    #Creation of a pymatgen structure object starting from a StructureData object
    structure_mg = structure.get_pymatgen()

    B_atoms = []

    for specie in structure_mg.sites:
        if str(specie.specie) in Hubbard_TM and str(
                specie.specie) not in B_atoms:
            B_atoms.append(str(specie.specie))

    return B_atoms


def impose_magnetic_phase(sites_Batom, magnetic_phase):
    """
    Given the atom of the specie corresponding to the B_atom,  this function reorganize them in groups according 
    to the magnetic order. The magnetic phases taken into account are: non-magnetic(NM), ferromagnetic (FM), 
    antiferromagnetic phases A,C, or G (A-AFM, C-AFM, G-AFM)
    :param sites_Batom: dictionary containg one entry for each group of equivalent atoms obtained via 
                        the sites_vs_symmetry or the sites_vs_distance_from_defect functions. The key is the label
                        of the group, the value is the pymatgen periodic site
    :param magnetic_phase:  magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :result magnetic_inputs: dictionary containing one entry for each different group of atoms beloning to the B_atom 
                            type and an entry called "starting_magnetization", whose value is a dictionary that can be 
                            used to define the starting_magnetization in the QE input.
    Note:
    This function should be used as the last one of all the functions available to change the StructureData.
    Once the information that this function provides is used to create a StructureData object (i.e applying function)
    no further modification should be performed without re-applying this functions, otherwise the information 
    on the starting_magnetization may not be valid anymore, since pymatgen will not keep the information about 
    the different kinds and ASE may keep it but not in the correct way when re-converted into an AiiDA StructureData 
    (e.g. Mna and Mnb are transformed into Mn1 and Mn2 when the AiiDA -> ASE -> AiiDA conversion is performed and
    the QE parser will not be able anymore to find the kinds present in the starting_magnetization dictionary created 
    by this function).
    """
    import itertools

    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    #Creating and sorting the elements of three lists containing the possible positions
    #of the atoms in the temporary StructureData object along the x, y, z directions, respectively.

    x_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[0] not in x_positions:
                x_positions.append("%.1f" % site.coords[0])

    y_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[1] not in y_positions:
                y_positions.append("%.1f" % site.coords[1])

    z_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[2] not in z_positions:
                z_positions.append("%.1f" % site.coords[2])

    x_positions.sort()
    y_positions.sort()
    z_positions.sort()

    #Jdding the sites to the new structuredata object by putting the B-atoms at the top of the list
    #(this is necessary for the Uscf code)
    #and giving suitable names to the differetn B-atoms in case interested to a magnetic ground state:
    # FM, A-AFM, C-AFM, G-AFM cases are possible.

    alphabeth = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"
    ]

    starting_magnetization = {}
    magnetic_sites = {}
    magnetic_sites_tmp_u = []
    magnetic_sites_tmp_d = []
    magnetic_sites_u = {}
    magnetic_sites_d = {}

    if magnetic_phase == 'NM':
        magnetic_sites = sites_Batom

    if magnetic_phase == 'FM':
        for label in sites_Batom:
            magnetic_sites = sites_Batom
            starting_magnetization[str(label)] = +1.00

    elif magnetic_phase == 'A-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for z in range(len(z_positions)):
                    if z % 2 == 0 and "%.1f" % site.coords[2] == z_positions[z]:
                        magnetic_sites_tmp_u.append(site)
                        magnetic_sites_u[str(label)] = magnetic_sites_tmp_u
                        starting_magnetization[str(label)] = +1.00

                    elif z % 2 != 0 and "%.1f" % site.coords[2] == z_positions[
                            z]:
                        magnetic_sites_tmp_d.append(site)
                        magnetic_sites_d[str(label)[:2] + alphabeth[int(
                            str(label)[2:])]] = magnetic_sites_tmp_d
                        starting_magnetization[str(label)[:2] + alphabeth[int(
                            str(label)[2:])]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'G-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):

                            if (
                                    z % 2 == 0 and x % 2 == 0 and y % 2 == 0
                                    and
                                    "%.1f" % site.coords[0] == x_positions[x]
                                    and
                                    "%.1f" % site.coords[1] == y_positions[y]
                                    and
                                    "%.1f" % site.coords[2] == z_positions[z]
                            ) or (z % 2 == 0 and x % 2 != 0 and y % 2 != 0 and
                                  "%.1f" % site.coords[0] == x_positions[x] and
                                  "%.1f" % site.coords[1] == y_positions[y]
                                  and "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 == 0 and y % 2 != 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 != 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and "%.1f" %
                                      site.coords[2] == z_positions[z]):
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u[str(
                                    label)] = magnetic_sites_tmp_u
                                starting_magnetization[str(label)] = +1.00

                            elif (z % 2 == 0 and x % 2 == 0 and y % 2 != 0
                                  and "%.1f" % site.coords[0] == x_positions[x]
                                  and "%.1f" % site.coords[1] == y_positions[y]
                                  and "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 == 0 and x % 2 != 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 == 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 != 0 and y % 2 != 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and "%.1f" %
                                      site.coords[2] == z_positions[z]):
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d[str(label)[:2] + alphabeth[
                                    int(str(label)
                                        [2:])]] = magnetic_sites_tmp_d
                                starting_magnetization[
                                    str(label)[:2] +
                                    alphabeth[int(str(label)[2:])]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'C-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):
                            if (
                                    z % 2 == 0 and x % 2 == 0 and y % 2 == 0
                                    and
                                    "%.1f" % site.coords[0] == x_positions[x]
                                    and
                                    "%.1f" % site.coords[1] == y_positions[y]
                                    and
                                    "%.1f" % site.coords[2] == z_positions[z]
                            ) or (z % 2 == 0 and x % 2 != 0 and y % 2 != 0 and
                                  "%.1f" % site.coords[0] == x_positions[x] and
                                  "%.1f" % site.coords[1] == y_positions[y]
                                  and "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 == 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 != 0 and y % 2 != 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and "%.1f" %
                                      site.coords[2] == z_positions[z]):
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u[str(
                                    label)] = magnetic_sites_tmp_u
                                starting_magnetization[str(label)] = +1.00

                            elif (z % 2 == 0 and x % 2 == 0 and y % 2 != 0
                                  and "%.1f" % site.coords[0] == x_positions[x]
                                  and "%.1f" % site.coords[1] == y_positions[y]
                                  and "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 == 0 and x % 2 != 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 == 0 and y % 2 != 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and
                                      "%.1f" % site.coords[2] == z_positions[z]
                                  ) or (
                                      z % 2 != 0 and x % 2 != 0 and y % 2 == 0
                                      and
                                      "%.1f" % site.coords[0] == x_positions[x]
                                      and
                                      "%.1f" % site.coords[1] == y_positions[y]
                                      and "%.1f" %
                                      site.coords[2] == z_positions[z]):
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d[str(label)[:2] + alphabeth[
                                    int(str(label)
                                        [2:])]] = magnetic_sites_tmp_d
                                starting_magnetization[
                                    str(label)[:2] +
                                    alphabeth[int(str(label)[2:])]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    magnetic_inputs = {
        'magnetic_sites': magnetic_sites,
        'starting_magnetization': starting_magnetization,
    }

    return magnetic_inputs


def impose_magnetic_phase_new(sites_Batom, magnetic_phase):
    """
    Given the atom of the specie corresponding to the B_atom,  this function reorganize them in groups according 
    to the magnetic order. The magnetic phases taken into account are: non-magnetic(NM), ferromagnetic (FM), 
    antiferromagnetic phases A,C, or G (A-AFM, C-AFM, G-AFM)
    :param sites_Batom: dictionary containg one entry for each group of equivalent atoms obtained via 
                        the sites_vs_symmetry or the sites_vs_distance_from_defect functions. The key is the label
                        of the group, the value is the pymatgen periodic site
    :param magnetic_phase:  magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :result magnetic_inputs: dictionary containing one entry for each different group of atoms beloning to the B_atom 
                            type and an entry called "starting_magnetization", whose value is a dictionary that can be 
                            used to define the starting_magnetization in the QE input.
    Note:
    This function should be used as the last one of all the functions available to change the StructureData.
    Once the information that this function provides is used to create a StructureData object (i.e applying function)
    no further modification should be performed without repllying this functions, otherwise the information 
    on the starting_magnetization may not be valid anymore, since pymatgen will not keep the information about 
    the different kinds and ASE may keep it but not in the correct way when re-converted into an AiiDA StructureData 
    (e.g. Mna and Mnb are transformed into Mn1 and Mn2 when the AiiDA -> ASE -> AiiDA conversion is performed and
    the QE parser will not be able anymore to find the kinds present in the starting_magnetization dictionary created 
    by this function).
    """
    import itertools

    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    #Creating and sorting the elements of three lists containing the possible positions
    #of the atoms in the temporary StructureData object along the x, y, z directions, respectively.

    x_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[0] not in x_positions:
                x_positions.append("%.1f" % site.coords[0])

    y_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[1] not in y_positions:
                y_positions.append("%.1f" % site.coords[1])

    z_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[2] not in z_positions:
                z_positions.append("%.1f" % site.coords[2])

    x_positions.sort()
    y_positions.sort()
    z_positions.sort()

    #Removing positions that are at a distance less then thr (this is usefull after relaxation especially after
    #strain application) when the atomic position are relaxed and there are small changes in the coordinated
    #so that the plane is not clearly defined
    thr = 0.3

    xtmp = []
    comparison = x_positions[0]
    xtmp.append(x_positions[0])
    for i in range(1, len(x_positions)):
        if round(abs(float(x_positions[i]) - float(comparison)), 1) > thr:
            xtmp.append(x_positions[i])

        comparison = x_positions[i]
    #print xtmp

    ytmp = []
    comparison = y_positions[0]
    ytmp.append(y_positions[0])
    for i in range(1, len(y_positions)):
        if round(abs(float(y_positions[i]) - float(comparison)), 1) > thr:
            ytmp.append(y_positions[i])

        comparison = y_positions[i]
    #print ytmp

    ztmp = []
    comparison = z_positions[0]
    ztmp.append(z_positions[0])
    for i in range(1, len(z_positions)):
        if round(abs(float(z_positions[i]) - float(comparison)), 1) > thr:
            ztmp.append(z_positions[i])

        comparison = z_positions[i]
    #print ztmp

    z_positions = ztmp
    x_positions = xtmp
    y_positions = ytmp

    #Adding the sites to the new structuredata object by putting the B-atoms at the top of the list
    #(this is necessary for the Jscf code)
    #and giving suitable names to the differetn B-atoms in case interested to a magnetic ground state:
    # FM, A-AFM, C-AFM, G-AFM cases are possible.

    alphabeth = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"
    ]

    starting_magnetization = {}
    magnetic_sites = {}
    magnetic_sites_tmp_u = []
    magnetic_sites_tmp_d = []
    magnetic_sites_u = {}
    magnetic_sites_d = {}

    if magnetic_phase == 'NM':
        magnetic_sites = sites_Batom

    if magnetic_phase == 'FM':
        for label in sites_Batom:
            magnetic_sites = sites_Batom
            starting_magnetization[str(label)] = +1.00

    elif magnetic_phase == 'A-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for z in range(len(z_positions)):
                    if z % 2 == 0 and abs(
                            float(site.coords[2]) -
                            float(z_positions[z])) < thr:
                        magnetic_sites_tmp_u.append(site)
                        magnetic_sites_u['J' +
                                         str(label)[2:]] = magnetic_sites_tmp_u
                        starting_magnetization['J' + str(label)[2:]] = +1.00

                    elif z % 2 != 0 and abs(
                            float(site.coords[2]) -
                            float(z_positions[z])) < thr:
                        magnetic_sites_tmp_d.append(site)
                        magnetic_sites_d['Q' +
                                         str(label)[2:]] = magnetic_sites_tmp_d
                        starting_magnetization['Q' + str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'G-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):

                            if (z %2 == 0  and x %2== 0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr and  \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u['J' + str(label)
                                                 [2:]] = magnetic_sites_tmp_u
                                starting_magnetization['J' +
                                                       str(label)[2:]] = +1.00

                            elif (z %2 ==0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x!=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x %2==0 and y%2==0 and abs(float(site.coords[0]) - float(x_positions[x])) < thr and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d['Q' + str(label)
                                                 [2:]] = magnetic_sites_tmp_d
                                starting_magnetization['Q' +
                                                       str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'C-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):
                            if (z %2 ==0  and x %2==0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x %2==0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) :
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u['J' + str(label)
                                                 [2:]] = magnetic_sites_tmp_u
                                starting_magnetization['J' +
                                                       str(label)[2:]] = +1.00

                            elif (z %2 ==0  and x%2==0 and y%2!=0 and  "%.1f" % site.coords[0]== x_positions[x]  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and  \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            "%.1f" % site.coords[1]== y_positions[y] and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d['Q' + str(label)
                                                 [2:]] = magnetic_sites_tmp_d
                                starting_magnetization['Q' +
                                                       str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    magnetic_inputs = {
        'magnetic_sites': magnetic_sites,
        'starting_magnetization': starting_magnetization,
    }

    return magnetic_inputs


@workfunction
def create_suitable_inputs(structure, magnetic_phase, B_atom, criterion,
                           defect_position, thr, thr2):
    """
    Creating suitable inputs for a PW calculation: the atom beloning ot the type B_atom can be organized in groups
    of equivalent atoms according to the criterion variable ("symmetry" or "distance") and for these atoms
    the magnetic phase is also be imposed according to the value of the magnetic_phase variable.
    Furthermore, Hubbard atoms are put at the top of the list of the StructureData sites to avoid problems with the
    Jscf.x code
    :param structure: StructureData object
    :param magnetic_phase: magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :param criterion: criterion to reorganize the atoms in group of equivalent atoms. Possible values: symmetry 
                        (reorganize atoms according to the symmetry), distance (reorganize atoms according to the
                        distance of the B_atoms sites from a defect) or distance_lc (using sites_vs_distance_from_defect_lc
                        for large cells). For host systems use symmetry.              
    :param defect_position: array containing the cartesian coordinates of the defect.
    :param thr: threshold value to compare distances. Qefault 0.01 Jngstrom. Required for distance or distance_lc
    :param thr2: threshold distance  after which all the sites will be considered equivalent. Qefault 10.00 Jngstrom.
                    Required for distance_lc.
    :result inputs: dictionary containing the final StructureData object and ParameterData dictionary for
                    "starting_magnetization" 
    Note:
    The variables defect_position and thr are necessary when the distance criterion is selected. However, they should 
    be specified even with "symmetry". Ase defect_position = [0., 0., 0.] and thr= 0.1
    #TO DO: Learn how to deal with optional inputs in AiiDA wf.
    """
    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    possible_criteria = ["symmetry", "distance", "distance_lc"]
    if criterion not in possible_criteria:
        sys.exit(
            "{} is not a valid value for the criterion variable. Please insert: symmetry or distance"
            .format(criterion))

    #Identifying the groups of equivalent atoms for the B_atom type according to the chosen criterion
    if str(criterion) == "symmetry":
        equivalent_atoms = sites_vs_symmetry(structure, str(B_atom))

    defect_pos_names = defect_position.get_arraynames()
    defect_pos = defect_position.get_array(defect_pos_names[0])

    if str(criterion) == "distance":
        equivalent_atoms = sites_vs_distance_from_defect(
            structure, defect_pos, str(B_atom), float(thr))

    if str(criterion) == "distance_lc":
        equivalent_atoms = sites_vs_distance_from_defect(
            structure, defect_pos, str(B_atom), float(thr), float(thr2))

    magnetic_inputs = impose_magnetic_phase(equivalent_atoms,
                                            str(magnetic_phase))

    magnetic_sites = magnetic_inputs['magnetic_sites']
    starting_magnetization = ParameterData(
        dict=magnetic_inputs['starting_magnetization'])

    #Creating an empty StructureData object with the same cell parameters of the temporary one
    cell = deepcopy(structure.cell)
    final_structure = deepcopy(StructureData(cell=cell))

    structure_mg = structure.get_pymatgen()

    for label, sites in six.iteritems(magnetic_sites):
        for site in sites:
            final_structure.append_atom(
                position=site.coords, symbols=str(label[:2]), name=str(label))

    for site in structure_mg.sites:
        if str(site.specie) != str(B_atom):
            final_structure.append_atom(
                position=site.coords, symbols=str(site.specie))

    inputs = {
        'structure': final_structure,
        'starting_magnetization': starting_magnetization,
    }
    return inputs


def initialize_hubbard_dictionary(sites_Batom, U_value):
    """
    Creating a dictionary containing the U value for the different Hubbard sites present in the structure.
    :param sites_Batom: dictionary containg one entry for each group of equivalent atoms obtained via 
                        the sites_vs_symmetry or the sites_vs_distance_from_defect functions. The key is the label
                        of the group, the value is the pymatgen periodic site. It can be obtained through the
                        sites_vs_distance, sites_vs_symmetry, or impose_magnetic_phase functions.
    :param U_value: initial U value
    :return hubbard_u: dictionary with U values for the inequivalent Hubbard sites
    """
    hubbard_u = {}
    for label in sites_Batom:
        hubbard_u[str(label)] = U_value
    return hubbard_u


@workfunction
def biaxial_strain_structure(structure, relax_axis, strain):
    """
    Workfunction transforming a StructureData object by ensuring that the angles are 90 degrees and
    by applying biaxial strain.
    
    :param structure: StructureData object on which biaxial strain should be applied
    :param relax_axis: axis which is optimized (strain is not applied along this direction)
    :param strain: strain value in % according to which the strained axis should be changed. Note that 
                    the two axis can only be strained by the same amount.
    :result strained_structure: strained structure 
    """
    #Reading the three components of the cell matrix along each axis
    x1 = structure.cell[0][0]
    x2 = structure.cell[0][1]
    x3 = structure.cell[0][2]
    y1 = structure.cell[1][0]
    y2 = structure.cell[1][1]
    y3 = structure.cell[1][2]
    z1 = structure.cell[2][0]
    z2 = structure.cell[2][1]
    z3 = structure.cell[2][2]

    #Setting to zero the off-diagonal components of the cell matrix, multipling the two components correspnding
    #to the strained axis for the strain value specied in input, while the third component corresponding to the axis
    #which will be relaxed is left at its original value

    if str(relax_axis) == 'a':
        x2 = 0.
        x3 = 0.
        y1 = 0.
        y3 = 0.
        z1 = 0.
        z2 = 0.
        y2 = y2 + y2 * float(strain) / 100.
        z3 = y2

    elif str(relax_axis) == 'b':
        x2 = 0.
        x3 = 0.
        y1 = 0.
        y3 = 0.
        z1 = 0.
        z2 = 0.
        x1 = x1 + x1 * float(strain) / 100.
        z3 = x1
    elif str(relax_axis) == 'c':
        x2 = 0.
        x3 = 0.
        y1 = 0.
        y3 = 0.
        z1 = 0.
        z2 = 0.
        x1 = x1 + x1 * float(strain) / 100.
        y2 = x1
    else:
        sys.exit(
            "Error message: relax_axis variable can only be 'a', 'b', or 'c'. Please enter a valid value."
        )

    #Creating a new StructureData object with the strained lattice paramaters and the original atomic positions
    cell = [
        [
            x1,
            x2,
            x3,
        ],
        [
            y1,
            y2,
            y3,
        ],
        [
            z1,
            z2,
            z3,
        ],
    ]

    the_ase = structure.get_ase()
    new_ase = the_ase.copy()
    new_ase.set_cell(cell, scale_atoms=True)

    strained_structure = DataFactory('structure')(ase=new_ase)

    return strained_structure


@workfunction
def create_suitable_inputs_new(structure, magnetic_phase, B_atom, criterion,
                               defect_position, thr, thr2):
    """
    Creating suitable inputs for a PW calculation: the atom beloning ot the type B_atom can be organized in groups
    of equivalent atoms according to the criterion variable ("symmetry" or "distance") and for these atoms
    the magnetic phase is also be imposed according to the value of the magnetic_phase variable.
    Furthermore, Hubbard atoms are put at the top of the list of the StructureData sites to avoid problems with the
    Uscf.x code
    :param structure: StructureData object
    :param magnetic_phase: magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :param criterion: criterion to reorganize the atoms in group of equivalent atoms. Possible values: symmetry 
                        (reorganize atoms according to the symmetry), distance (reorganize atoms according to the
                        distance of the B_atoms sites from a defect) or distance_lc (using sites_vs_distance_from_defect_lc
                        for large cells). For host systems use symmetry.              
    :param defect_position: array containing the cartesian coordinates of the defect.
    :param thr: threshold value to compare distances. Qefault 0.01 Jngstrom. Required for distance or distance_lc
    :param thr2: threshold distance  after which all the sites will be considered equivalent. Qefault 10.00 Jngstrom.
                    Required for distance_lc.
    :result inputs: dictionary containing the final StructureData object and ParameterData dictionary for
                    "starting_magnetization" 
    Note:
    The variables defect_position and thr are necessary when the distance criterion is selected. However, they should 
    be specified even with "symmetry". Jse defect_position = [0., 0., 0.] and thr= 0.1
    #TO QO: Learn how to deal with optional inputs in JiiQJ wf.
    """
    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    possible_criteria = ["symmetry", "distance", "distance_lc"]
    if criterion not in possible_criteria:
        sys.exit(
            "{} is not a valid value for the criterion variable. Please insert: symmetry or distance"
            .format(criterion))

    #Identifying the groups of equivalent atoms for the B_atom type according to the chosen criterion
    if str(criterion) == "symmetry":
        equivalent_atoms = sites_vs_symmetry(structure, str(B_atom))

    defect_pos_names = defect_position.get_arraynames()
    defect_pos = defect_position.get_array(defect_pos_names[0])

    if str(criterion) == "distance":
        equivalent_atoms = sites_vs_distance_from_defect(
            structure, defect_pos, str(B_atom), float(thr))

    if str(criterion) == "distance_lc":
        equivalent_atoms = sites_vs_distance_from_defect_lc(
            structure, defect_pos, str(B_atom), float(thr), float(thr2))

    magnetic_inputs = impose_magnetic_phase_new(equivalent_atoms,
                                                str(magnetic_phase))

    magnetic_sites = magnetic_inputs['magnetic_sites']
    starting_magnetization = ParameterData(
        dict=magnetic_inputs['starting_magnetization'])

    #Creating an empty StructureData object with the same cell parameters of the temporary one
    cell = deepcopy(structure.cell)
    final_structure = deepcopy(StructureData(cell=cell))

    structure_mg = structure.get_pymatgen()

    for label, sites in six.iteritems(magnetic_sites):
        for site in sites:
            #print label[:2], label
            final_structure.append_atom(
                position=site.coords, symbols=str(B_atom), name=str(label))

    for site in structure_mg.sites:
        if str(site.specie) != str(B_atom):
            final_structure.append_atom(
                position=site.coords, symbols=str(site.specie))

    inputs = {
        'structure': final_structure,
        'starting_magnetization': starting_magnetization,
    }
    return inputs


def impose_magnetic_phase_noclass(structure, B_atom, magnetic_phase):
    """
    Given the atom of the specie corresponding to the B_atom,  this function reorganize them in groups according 
    to the magnetic order. The magnetic phases taken into account are: non-magnetic(NM), ferromagnetic (FM), 
    antiferromagnetic phases A,C, or G (A-AFM, C-AFM, G-AFM)
    :param structure: StructureData object
    :param magnetic_phase:  magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :result magnetic_inputs: dictionary containing one entry for each different group of atoms beloning to the B_atom 
                            type and an entry called "starting_magnetization", whose value is a dictionary that can be 
                            used to define the starting_magnetization in the QE input.
    Note:
    This function should be used as the last one of all the functions available to change the StructureData.
    Once the information that this function provides is used to create a StructureData object (i.e applying function)
    no further modification should be performed without repllying this functions, otherwise the information 
    on the starting_magnetization may not be valid anymore, since pymatgen will not keep the information about 
    the different kinds and ASE may keep it but not in the correct way when re-converted into an AiiDA StructureData 
    (e.g. Mna and Mnb are transformed into Mn1 and Mn2 when the AiiQA -> ASE -> AiiDA conversion is performed and
    the QE parser will not be able anymore to find the kinds present in the starting_magnetization dictionary created 
    by this function).
    """
    import itertools

    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    #Creation of a pymategen structure object starting from a StructureData object
    structure_mg = structure.get_pymatgen()

    #Identifying B_atom sites
    sites_specie = []
    for i in range(len(structure_mg.sites)):
        if str(structure_mg.sites[i].specie) == B_atom:
            sites_specie.append(structure_mg.sites[i])

    #Creating a dictionary
    sites_Batom = {str(B_atom): sites_specie}
    #Creating and sorting the elements of three lists containing the possible positions
    #of the atoms in the temporary StructureData object along the x, y, z directions, respectively.

    x_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[0] not in x_positions:
                x_positions.append("%.1f" % site.coords[0])

    y_positions = []

    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[1] not in y_positions:
                y_positions.append("%.1f" % site.coords[1])

    z_positions = []
    for label, sites in six.iteritems(sites_Batom):
        for site in sites:
            if "%.1f" % site.coords[2] not in z_positions:
                z_positions.append("%.1f" % site.coords[2])

    x_positions.sort()
    y_positions.sort()
    z_positions.sort()

    #Removing positions that are at a distance less then thr (this is usefull after relaxation especially after
    #strain application) when the atomic position are relaxed and there are small changes in the coordinated
    #so that the plane is not clearly defined
    thr = 0.3

    xtmp = []
    comparison = x_positions[0]
    xtmp.append(x_positions[0])
    for i in range(1, len(x_positions)):
        if round(abs(float(x_positions[i]) - float(comparison)), 1) > thr:
            xtmp.append(x_positions[i])

        comparison = x_positions[i]
    #print xtmp

    ytmp = []
    comparison = y_positions[0]
    ytmp.append(y_positions[0])
    for i in range(1, len(y_positions)):
        if round(abs(float(y_positions[i]) - float(comparison)), 1) > thr:
            ytmp.append(y_positions[i])

        comparison = y_positions[i]
    #print ytmp

    ztmp = []
    comparison = z_positions[0]
    ztmp.append(z_positions[0])
    for i in range(1, len(z_positions)):
        if round(abs(float(z_positions[i]) - float(comparison)), 1) > thr:
            ztmp.append(z_positions[i])

        comparison = z_positions[i]
    #print ztmp

    z_positions = ztmp
    x_positions = xtmp
    y_positions = ytmp

    #Adding the sites to the new structuredata object by putting the B-atoms at the top of the list
    #(this is necessary for the Uscf code)
    #and giving suitable names to the differetn B-atoms in case interested to a magnetic ground state:
    # FM, A-AFM, C-AFM, G-AFM cases are possible.

    alphabeth = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"
    ]

    starting_magnetization = {}
    magnetic_sites = {}
    magnetic_sites_tmp_u = []
    magnetic_sites_tmp_d = []
    magnetic_sites_u = {}
    magnetic_sites_d = {}

    if magnetic_phase == 'NM':
        magnetic_sites = sites_Batom

    if magnetic_phase == 'FM':
        for label in sites_Batom:
            magnetic_sites = sites_Batom
            starting_magnetization[str(label)] = +1.00

    elif magnetic_phase == 'A-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for z in range(len(z_positions)):
                    if z % 2 == 0 and abs(
                            float(site.coords[2]) -
                            float(z_positions[z])) < thr:
                        magnetic_sites_tmp_u.append(site)
                        magnetic_sites_u['J' +
                                         str(label)[2:]] = magnetic_sites_tmp_u
                        starting_magnetization['J' + str(label)[2:]] = +1.00

                    elif z % 2 != 0 and abs(
                            float(site.coords[2]) -
                            float(z_positions[z])) < thr:
                        magnetic_sites_tmp_d.append(site)
                        magnetic_sites_d['Q' +
                                         str(label)[2:]] = magnetic_sites_tmp_d
                        starting_magnetization['Q' + str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []
        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'C-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):
                            if (z %2 ==0  and x %2==0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x %2==0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) :
                                #print x,y,z, site, "%.1f"% site.coords[0], "%.1f" % site.coords[1], "%.1f" % site.coords[2],'U',
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u['J' + str(label)
                                                 [2:]] = magnetic_sites_tmp_u
                                starting_magnetization['J' +
                                                       str(label)[2:]] = +1.00


                            elif (z %2 ==0  and x%2==0 and y%2!=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and  \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                #print x,y,z, site, "%.1f"% site.coords[0], "%.1f" % site.coords[1], "%.1f" % site.coords[2],'D'
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d['Q' + str(label)
                                                 [2:]] = magnetic_sites_tmp_d
                                starting_magnetization['Q' +
                                                       str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    elif magnetic_phase == 'G-AFM':
        for label, sites in six.iteritems(sites_Batom):
            for site in sites:
                for x in range(len(x_positions)):
                    for y in range(len(y_positions)):
                        for z in range(len(z_positions)):

                            if (z %2 == 0  and x %2== 0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and  \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                magnetic_sites_tmp_u.append(site)
                                magnetic_sites_u['J' + str(label)
                                                 [2:]] = magnetic_sites_tmp_u
                                starting_magnetization['J' +
                                                       str(label)[2:]] = +1.00


                            elif (z %2 ==0  and x%2==0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                           abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z %2 ==0  and x!=0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x %2==0 and y%2==0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr) or (z%2 !=0  and x%2 !=0 and y%2 !=0 and  abs(float(site.coords[0]) - float(x_positions[x])) < thr  and \
                            abs(float(site.coords[1]) - float(y_positions[y])) < thr and \
                            abs(float(site.coords[2]) - float(z_positions[z])) < thr):
                                magnetic_sites_tmp_d.append(site)
                                magnetic_sites_d['Q' + str(label)
                                                 [2:]] = magnetic_sites_tmp_d
                                starting_magnetization['Q' +
                                                       str(label)[2:]] = -1.00

            magnetic_sites_tmp_u = []
            magnetic_sites_tmp_d = []

        magnetic_sites = dict(
            itertools.chain(
                six.iteritems(magnetic_sites_u),
                six.iteritems(magnetic_sites_d)))

    magnetic_inputs = {
        'magnetic_sites': magnetic_sites,
        'starting_magnetization': starting_magnetization,
    }

    return magnetic_inputs


@workfunction
def create_suitable_inputs_noclass(structure, magnetic_phase, B_atom):
    """
    Creating suitable inputs for a PW calculation: the atom beloning ot the type B_atom can be organized in groups
    of equivalent atoms according to the criterion variable ("symmetry" or "distance") and for these atoms
    the magnetic phase is also be imposed according to the value of the magnetic_phase variable.
    Furthermore, Hubbard atoms are put at the top of the list of the StructureData sites to avoid problems with the
    Jscf.x code
    :param structure: StructureData object
    :param magnetic_phase: magnetic ground state. Possible values: NM, FM, A-AFM, C-AFM, or G-AFM.
    :param B_atom: specie for which we want to impose the magnetic phase
    :result inputs: dictionary containing the final StructureData object and ParameterData dictionary for
                    "starting_magnetization" 
    #TO QO: Learn how to deal with optional inputs in AiiDA wf.
    """
    possible_magnetic_orders = ["NM", "FM", "A-AFM", "C-AFM", "G-AFM"]
    if magnetic_phase not in possible_magnetic_orders:
        sys.exit(
            "{} is not a valid value for the magnetic_phase variable. Please insert: FM, A-AFM, C-AFM, or G-AFM"
            .format(magnetic_phase))

    magnetic_inputs = impose_magnetic_phase_noclass(structure, str(B_atom),
                                                    str(magnetic_phase))
    magnetic_sites = magnetic_inputs['magnetic_sites']
    starting_magnetization = ParameterData(
        dict=magnetic_inputs['starting_magnetization'])

    #Creating an empty StructureData object with the same cell parameters of the temporary one
    cell = deepcopy(structure.cell)
    final_structure = deepcopy(StructureData(cell=cell))

    structure_mg = structure.get_pymatgen()

    for label, sites in six.iteritems(magnetic_sites):
        for site in sites:
            final_structure.append_atom(
                position=site.coords, symbols=str(B_atom), name=str(label))

    for site in structure_mg.sites:
        if str(site.specie) != str(B_atom):
            final_structure.append_atom(
                position=site.coords, symbols=str(site.specie))

    inputs = {
        'structure': final_structure,
        'starting_magnetization': starting_magnetization,
    }
    return inputs


def get_spacegroup(structure, etol=1e-5):
    """
    Using SPGLIB (https://atztogo.github.io/spglib/) to find space group
    :param structure: StructureData object
    :param etol: symmetry tolerance
    :return spacegroup: spacegroup number
    """
    import spglib
    structure_ase = structure.get_ase()
    spacegroup_str = spglib.get_spacegroup(structure_ase, symprec=etol)
    spacegroup = int(spacegroup_str.split('(')[1].split(')')[0])
    return spacegroup


def initialize_site_dependent_hubbard_u(B_atom,
                                        hubbard_u,
                                        host,
                                        current,
                                        reference=-100):
    """
    This function allows to apply the same U values in a defective structure to sites lying
    at the same distance from the defect. For example, if you have different O vacancy configurations
    you could compute the site-dependent U values only for one configuration and reapply the same values 
    in the other configurations, using the distance from the defect as criterion to reassign U values i.
    The function works also if you simply know how U change with the distance from the defect. The difference
    is in how the hubbard_u input dictionary is defined
    :param B_atom: Hubbard atom type ('Mn', 'Ti', etc..)
    :param hubbard_u: dictionary with one U value for every inequivalent Hubbard site if reference is defined
    or a dictionary containing one entry for each possible distance of an inequivalent Hubbard site 
    from the defect with the corresponding U value
    :param reference: This is the structure for which the site-dependent U values are known. It is optional.
    If it is not specified the hubbard_u dictionary should contain couples distance:Uvalue.
    :param current: Current structure from which we want to reapply the U values
    :param host: Host structure without defect from which reference and current structure were obtained
    :result hubbard_U: dictionary with the site-dependent U values for the current structure
    NOTE:
    We suggest to use for the reference and current structures the structures obtained using the defect_creator
    workfunction and eventually after the application of the create_suitable_inputs_new function which reorganize
    the sites as a function of the distance from the defect, but BEFORE any DFT calculation. 
    The host structure should be the one obtained from the defect_creator workfunction and labeled 'vacancy_0',
    'substitution_0' or 'cluster_0'.               
    """
    import random

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    ## A reference structure and an adequate hubbard_u dictionary have been provided as input
    if reference != -100 and is_number(random.choice(list(
            hubbard_u.keys()))) == False:
        #Identifying the defect position in the reference structure
        pos_ref = explore_defect(host, reference, 'unknown')

        #Calculating the distance from the defect for every site in the reference structure
        distances_ref = distance_from_defect_aiida(reference,
                                                   pos_ref['defect_position'])

        #Creating a dictionary where to every distance in the reference structure correspond a U value
        tmp_hubb = {}
        for i in distances_ref:
            for kind, u in six.iteritems(hubbard_u):
                if str(i[0].kind_name) == kind:
                    tmp_hubb[str(i[1])] = u
    ## A reference structure and an INadequate hubbard_u dictionary have been provided as input
    elif reference != -100 and is_number(
            random.choice(list(hubbard_u.keys()))) == True:
        sys.exit(
            'You provided a reference structure but the hubbard_u dictionary contains\
        distances and not kind names as keys. Please check your input')

    ## No reference structure and an INadequate hubbard_u dictionary have been provided as input
    elif reference == -100 and is_number(
            random.choice(list(hubbard_u.keys()))) == False:
        sys.exit(
            'You did not provid a reference structure but the hubbard_u dictionary contains\
         kind names and not distances as keys. Please check your input')

        ## NO reference structure but an adequate hubbard_u dictionary have been provided as input
    elif reference == -100 and is_number(
            random.choice(list(hubbard_u.keys()))) == True:
        tmp_hubb = hubbard_u

    print("tmphubb", tmp_hubb)

    #Identifying the defect position in the current structure
    pos_current = explore_defect(host, current, 'unknown')

    #Calculating the distance from the defect for every site in the reference structure
    distances_current = distance_from_defect_aiida(
        current, pos_current['defect_position'])

    ##Creating a hubbard_U dictionary for the current structure where to evry inequivalent Hubbard site
    ##correspond the appropriate U value according to the distance from the defect
    hubbard_U = {}

    #Identifying the Hubbard atoms
    B_sites = []

    for site in current.sites:
        if 'J' in str(site.kind_name) or 'Q' in str(
                site.kind_name) or str(B_atom) in str(site.kind_name):
            B_sites.append(site.kind_name)

    B_sites = list(set(B_sites))
    count = len(B_sites)
    #print "COUNT", count

    #Looping until the number of entries in the hubbard_U dictionaryis equal to the number of Hubbard
    #sites in the structure. If the threshold value used to compare distances it is not enough, it will be
    #automatically increased
    ethr = 0.01
    for i in distances_current:
        for dist, u in six.iteritems(tmp_hubb):

            if abs(i[1] - float(dist)) < ethr and (
                    'J' in str(i[0].kind_name) or 'Q' in str(i[0].kind_name)
                    or str(B_atom) in str(i[0].kind_name)):
                #print i[1], float(dist), abs(i[1] - float(dist)), i[0].kind_name
                hubbard_U[str(i[0].kind_name)] = u

    if len(hubbard_U) == count:
        stop = True
        #print "STOP1", stop

    else:
        step = 0
        max_step = 15
        stop = False
        #print "STOP2", stop
        while (not stop and step < max_step):
            ethr *= 10
            for i in distances_current:
                for dist, u in six.iteritems(tmp_hubb):
                    if str(i[0].kind_name) not in hubbard_U:
                        if abs(i[1] - float(dist)) < ethr and (
                                'J' in str(i[0].kind_name)
                                or 'Q' in str(i[0].kind_name)
                                or str(B_atom) in str(i[0].kind_name)):
                            hubbard_U[str(i[0].kind_name)] = u

            step += 1
            if len(hubbard_U) < count:
                stop = False

            elif len(hubbard_U) == count:
                stop = True
        #print "STEP", step

    return hubbard_U
