# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import six

from aiida import orm
from aiida.engine import WorkChain, calcfunction

#   from aiida_defects.pp.fft_tools import avg_potential_at_core


class LanyZungerAlignmentWorkchain(WorkChain):
    """
    Compute the electrostatic potential alignment via the Lany-Zunger method.
    See: S. Lany and A. Zunger, PRB 78, 235104 (2008)
    Note: Algorithm partially adapted from pylada defects (https://github.com/pylada/pylada-defects)
    """

    @classmethod
    def define(cls, spec):
        super(LanyZungerAlignmentWorkchain, cls).define(spec)
        spec.input('bulk_structure', valid_type=orm.StructureData),
        spec.input(
            'e_tol',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.2),
            help="Energy tolerance to decide which atoms to exclude to compute alignment")
        spec.input('first_potential', valid_type=orm.ArrayData)
        spec.input('second_potential', valid_type=orm.ArrayData)
        spec.input(
            'alignment_scheme',
            valid_type=orm.Str,
            default=lambda: orm.Str('lany-zunger'))
        spec.input('interpolate', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.outline(
        )
        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('alignment_required', valid_type=orm.Float, required=True)
        # Exit codes
        spec.exit_code(
            401,
            'ERROR_SUB_PROCESS_FAILED_WRONG_SHAPE',
            message=
            'the two electrostatic potentials must be the same shape, unless interpolation is allowed'
        )
        spec.exit_code(
            402,
            'ERROR_SUB_PROCESS_FAILED_INTERPOLATION',
            message='the interpolation could not be completed')
        spec.exit_code(
            403,
            'ERROR_SUB_PROCESS_FAILED_BAD_SCHEME',
            message='the alignment scheme requested is unknown')


@calcfunction
def lz_potential_alignment(bulk_structure,
                           bulk_sphere_pot,
                           bulk_symbols,
                           defect_structure,
                           defect_sphere_pot,
                           defect_symbols,
                           e_tol=0.2):
    """
    Function to compute the potential alignment correction using the average atomic electrostatic potentials
    of the bulk and defective structures. See: S. Lany and A. Zunger, PRB 78, 235104 (2008)
     S. Lany and A. Zunger, J. Appl. Phys. 100, 113725 􏰀2006􏰁.
    Note: Adapted from pylada defects (https://github.com/pylada/pylada-defects)
    Requirements: trilinear_interpolation, avg_potential_at_core. In order to use trilinear_interpolation the
    3D-FFT grid should be extracted from the FolderData node in which aiida.filplot is stored in the DB using
    the read_grid function.

    Parameters
    ----------
    bulk_structure : StructureData
        Bulk structure
    bulk_sphere_pot : Dictionary
        Sphere averaged potential corresponding to each atom in the host structure
    bulk_symbols : List
        Symbols of each atom in the host structure
    defect_sphere_pot : Dictionary
        Sphere averaged potential corresponding to each atom in the defective structure
    defect_symbols : List
        Symbols of each atom in the defect structure
    defect_structure : StructureData
        Defective structure
    defect_grid : ArrayData
        3D-FFT grid for the defect obtained from the read_grid function
    e_tol: Float
        (0.2 eV; as in S. Lany FORTRAN codes)

    Returns
    -------
    pot_align : Float
        Computed potential alignment (eV)
    """

    # Refactored logic
    # ======================================================

    bulk_structure = self.inputs.bulk_structure

    spherical_potentials = []

    # Loop over sites

    for site in bulk_structure.sites():
        spherical_pot = get_spherical_potential(site, bulk_structure)
        spherical_potentials.append(spherical_pot)

    # Original logic below:
    # ======================================================

    # Computing the average electrostatic potential per atomic site type for the host
    avg_bulk = avg_potential_at_core(bulk_sphere_pot, bulk_symbols)

    # Computing the average electrostatic potential per atomic site type for the defective structure
    avg_defect = avg_potential_at_core(defect_sphere_pot, defect_symbols)

    #Compute the difference between defect electrostatic potential and the average defect electrostatic potential
    #per atom

    diff_def = {}
    for atom, pot in six.iteritems(defect_sphere_pot):
        diff_def[atom] = float(pot) - avg_defect[atom.split('_')[0]]

    max_diff = abs(max(diff_def.values()))

    #Counting how many times a certain element appears in the list of atoms.
    #     from collections import Counter
    #     def_count = Counter(defect_symbols)
    #     host_count = Counter(bulk_symbols)

    #Identifying the list of atoms than can be used to compute the difference for which
    #diff_def is lower than max_diff or of a user energy tolerance (e_tol)
    #Substituional atoms that do not appear in the host structure are removed from the average
    acceptable_atoms = []
    for atom, value in six.iteritems(diff_def):
        if atom.split('_')[0] in bulk_symbols:
            if abs(value) < max_diff or abs(value) * 13.6058 < e_tol:
                acceptable_atoms.append(atom)

    #Avoid excluding all atoms
    while (not bool(acceptable_atoms)):
        e_tol = e_tol * 10
        print((
            "e_tol has been modified to {} in order to avoid excluding all atoms"
            .format(e_tol)))
        for atom, value in six.iteritems(diff_def):
            #if count[atom.split('_')[0]] > 1:
            if atom.split('_')[0] in bulk_symbols:
                if abs(value) < max_diff or abs(value) * 13.6058 < e_tol:
                    acceptable_atoms.append(atom)

    #Computing potential alignment avareging over all the acceptable atoms
    diff_def2 = []
    for atom, pot in six.iteritems(defect_sphere_pot):
        if atom in acceptable_atoms:
            diff_def2.append(float(pot) - avg_bulk[atom.split('_')[0]])

    pot_align = np.mean(diff_def2) * 13.6058
    return pot_align
