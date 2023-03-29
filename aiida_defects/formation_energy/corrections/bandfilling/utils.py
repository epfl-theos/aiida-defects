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
"""
Utility functions for the bandiflling workchain
"""


def find_bandgap(bandsdata, number_electrons=None, fermi_energy=None):
    """
    Tries to guess whether the bandsdata represent an insulator.
    This method is meant to be used only for electronic bands (not phonons)
    By default, it will try to use the occupations to guess the number of
    electrons and find the Fermi Energy, otherwise, it can be provided
    explicitely.
    Also, there is an implicit assumption that the kpoints grid is
    "sufficiently" dense, so that the bandsdata are not missing the
    intersection between valence and conduction band if present.
    Use this function with care!

    :param number_electrons: (optional, float) number of electrons in the unit cell
    :param fermi_energy: (optional, float) value of the fermi energy.

    :note: By default, the algorithm uses the occupations array
      to guess the number of electrons and the occupied bands. This is to be
      used with care, because the occupations could be smeared so at a
      non-zero temperature, with the unwanted effect that the conduction bands
      might be occupied in an insulator.
      Prefer to pass the number_of_electrons explicitly

    :note: Only one between number_electrons and fermi_energy can be specified at the
      same time.

    :return: (is_insulator, gap, homo, lumo), where is_insulator is a boolean, and gap, homo and lumo a
             float. The gap is None in case of a metal, zero when the homo is
             equal to the lumo (e.g. in semi-metals). For insulators and semi-metals
             returns also VBM = homo and CBM = lumo.
    Modified from the find_bandgap function in /path/aiida/orm/data/array/bands.py
    so that it returns also VBM and CBM
    """

    def nint(num):
        """
        Stable rounding function
        """
        if (num > 0):
            return int(num + .5)
        else:
            return int(num - .5)

    if fermi_energy and number_electrons:
        raise ValueError("Specify either the number of electrons or the "
                         "Fermi energy, but not both")

    try:
        stored_bands = bandsdata.get_bands()
    except KeyError:
        raise KeyError("Cannot do much of a band analysis without bands")

    if len(stored_bands.shape) == 3:
        # I write the algorithm for the generic case of having both the
        # spin up and spin down array

        # put all spins on one band per kpoint
        bands = np.concatenate([_ for _ in stored_bands], axis=1)
    else:
        bands = stored_bands

    # analysis on occupations:
    if fermi_energy is None:

        num_kpoints = len(bands)

        if number_electrons is None:
            try:
                _, stored_occupations = bandsdata.get_bands(
                    also_occupations=True)
            except KeyError:
                raise KeyError("Cannot determine metallicity if I don't have "
                               "either fermi energy, or occupations")

            # put the occupations in the same order of bands, also in case of multiple bands
            if len(stored_occupations.shape) == 3:
                # I write the algorithm for the generic case of having both the
                # spin up and spin down array

                # put all spins on one band per kpoint
                occupations = np.concatenate([_ for _ in stored_occupations],
                                             axis=1)
            else:
                occupations = stored_occupations

            # now sort the bands by energy
            # Note: I am sort of assuming that I have an electronic ground state

            # sort the bands by energy, and reorder the occupations accordingly
            # since after joining the two spins, I might have unsorted stuff
            bands, occupations = [
                np.array(y) for y in zip(*[
                    list(zip(*j)) for j in [
                        sorted(
                            zip(i[0].tolist(), i[1].tolist()),
                            key=lambda x: x[0])
                        for i in zip(bands, occupations)
                    ]
                ])
            ]
            number_electrons = int(
                round(sum([sum(i) for i in occupations]) / num_kpoints))

            homo_indexes = [
                np.where(np.array([nint(_) for _ in x]) > 0)[0][-1]
                for x in occupations
            ]
            if len(
                    set(homo_indexes)
            ) > 1:  # there must be intersections of valence and conduction bands
                return False, None, None, None
            else:
                homo = [_[0][_[1]] for _ in zip(bands, homo_indexes)]
                try:
                    lumo = [_[0][_[1] + 1] for _ in zip(bands, homo_indexes)]
                except IndexError:
                    raise ValueError(
                        "To understand if it is a metal or insulator, "
                        "need more bands than n_band=number_electrons")

        else:
            bands = np.sort(bands)
            number_electrons = int(number_electrons)

            # find the zero-temperature occupation per band (1 for spin-polarized
            # calculation, 2 otherwise)
            number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
            # gather the energies of the homo band, for every kpoint
            homo = [
                i[number_electrons / number_electrons_per_band - 1]
                for i in bands
            ]  # take the nth level
            try:
                # gather the energies of the lumo band, for every kpoint
                lumo = [
                    i[number_electrons / number_electrons_per_band]
                    for i in bands
                ]  # take the n+1th level
            except IndexError:
                raise ValueError(
                    "To understand if it is a metal or insulator, "
                    "need more bands than n_band=number_electrons")

        if number_electrons % 2 == 1 and len(stored_bands.shape) == 2:
            # if #electrons is odd and we have a non spin polarized calculation
            # it must be a metal and I don't need further checks
            return False, None, None, None

        # if the nth band crosses the (n+1)th, it is an insulator
        gap = min(lumo) - max(homo)
        if gap == 0.:
            return False, 0., max(homo), min(lumo)
        elif gap < 0.:
            return False, None, None, None
        else:
            return True, gap, max(homo), min(lumo)

    # analysis on the fermi energy
    else:
        # reorganize the bands, rather than per kpoint, per energy level

        # I need the bands sorted by energy
        bands.sort()

        levels = bands.transpose()
        max_mins = [(max(i), min(i)) for i in levels]

        if fermi_energy > bands.max():
            raise ValueError("The Fermi energy is above all band energies, "
                             "don't know what to do")
        if fermi_energy < bands.min():
            raise ValueError("The Fermi energy is below all band energies, "
                             "don't know what to do.")

        # one band is crossed by the fermi energy
        if any(i[1] < fermi_energy and fermi_energy < i[0] for i in max_mins):
            return False, None, None, None

        # case of semimetals, fermi energy at the crossing of two bands
        # this will only work if the dirac point is computed!
        elif (any(i[0] == fermi_energy for i in max_mins)
              and any(i[1] == fermi_energy for i in max_mins)):
            return False, 0., fermi_energy, fermi_energy
        # insulating case
        else:
            # take the max of the band maxima below the fermi energy
            homo = max([i[0] for i in max_mins if i[0] < fermi_energy])
            # take the min of the band minima above the fermi energy
            lumo = min([i[1] for i in max_mins if i[1] > fermi_energy])
            gap = lumo - homo
            if gap <= 0.:
                raise Exception("Something wrong has been implemented. "
                                "Revise the code!")
            return True, gap, homo, lumo


def heaviside(x):
    """
    Heaviside function
    :param x: float or int
    :return: 0 if x<0, 0.5 if x=0, and 1 if x>0
    """
    return 0.5 * np.sign(x) + 0.5


#@workfunction
def bandfilling_ms_correction(host_bandstructure, defect_bandstructure,
                              potential_alignment):
    """
    Moss Burstein or band fillling correction
    References:  Lany & Zunger PRB 72(23) 2008, Moss Proc. Phys. Soc. London Sect B 67 775 (1954),
    Burstein Phys. Rev. 93 632 (1954)
    :param host/defect_bandstructure: dictionary output result of the PwBandStructureWorkChain
    :param potential_alignment: Float obtained as result of the PotentialAlignment workchain
    :result: Float with band filling correction
    TODO: Think about the fact that host_bandstructure is a dictionary of AiiDA object and I cannot use it as a
    input for a workfunction!!!
    """
    #Extracting bandsdata and Fermi Energy for host
    host_bandsdata = host_bandstructure['band_structure']
    host_fermi_energy = host_bandstructure['scf_parameters'].dict.fermi_energy
    host_number_electrons = host_bandstructure[
        'scf_parameters'].dict.number_of_electrons

    #Finding host CBM and VBM
    bandgap = find_bandgap(host_bandsdata, host_number_electrons)
    if bandgap[0] == False and bandgap[1] == None:
        return orm.Bool(False)  #Metal systems
        #sys.exit('Are you sure to compute band filling corrections?\n
        #          Your host material is a metal.')
    else:
        E_VBM_host_align = bandgap[2] + float(potential_alignment)
        E_CBM_host_align = bandgap[3] + float(potential_alignment)

    #Extracting weights, occupations and energies from the defect band structure
    #Generalizing for both non and spin polarized calculations
    stored_occupations = defect_bandstructure['band_structure'].get_array(
        'occupations')
    stored_bands = defect_bandstructure['band_structure'].get_array('bands')
    stored_weights = defect_bandstructure['band_structure'].get_array(
        'weights')
    if len(defect_bandstructure['band_structure'].get_array(
            'occupations').shape) == 3:
        #spin polarized case: weights, band energies and occupations for both spin up and spin down
        #are concatenated in one array with the first dimension equal to the double of the number of
        #kpoints.
        occupations = np.concatenate([_ for _ in stored_occupations], axis=0)
        bands = np.concatenate([_ for _ in stored_bands], axis=0)
        weights = np.concatenate((stored_weights, stored_weights), axis=0)
        max_occupation = 1
    else:
        occupations = stored_occupations
        bands = stored_bands
        weights = stored_weights
        max_occupation = 2

    #Computing band_filling
    E_donor = 0.
    for k in range(weights.shape[0]):
        for e in range(bands.shape[1]):
            tmp = weights[k] * occupations[k][e] * (
                bands[k][e] - E_CBM_host_align) * heaviside(bands[k][e] -
                                                            E_CBM_host_align)
            E_donor += tmp

    E_acceptor = 0.
    for k in range(weights.shape[0]):
        for e in range(bands.shape[1]):
            tmp = weights[k] * (max_occupation - occupations[k][e]) * (
                E_VBM_host_align - bands[k][e]) * heaviside(E_VBM_host_align -
                                                            bands[k][e])
            E_acceptor += tmp

    return {'E_donor': E_donor, 'E_acceptor': E_acceptor}
