# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import
import sys
import argparse
import pymatgen
import numpy as np
from aiida.orm.data.upf import UpfData
from aiida.common.exceptions import NotExistent
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm import DataFactory
from aiida.orm.node import Node
from aiida.orm.code import Code
from aiida.orm import load_node

from aiida.work.run import run, submit
from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List


from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
import six
from six.moves import range
from six.moves import zip


######TODO: Apply create suitable input before submitting the PW calc


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
    import numpy
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
        bands = numpy.concatenate([_ for _ in stored_bands], axis=1)
    else:
        bands = stored_bands

    # analysis on occupations:
    if fermi_energy is None:

        num_kpoints = len(bands)

        if number_electrons is None:
            try:
                _, stored_occupations = bandsdata.get_bands(also_occupations=True)
            except KeyError:
                raise KeyError("Cannot determine metallicity if I don't have "
                               "either fermi energy, or occupations")

            # put the occupations in the same order of bands, also in case of multiple bands
            if len(stored_occupations.shape) == 3:
                # I write the algorithm for the generic case of having both the
                # spin up and spin down array

                # put all spins on one band per kpoint
                occupations = numpy.concatenate([_ for _ in stored_occupations], axis=1)
            else:
                occupations = stored_occupations

            # now sort the bands by energy
            # Note: I am sort of assuming that I have an electronic ground state

            # sort the bands by energy, and reorder the occupations accordingly
            # since after joining the two spins, I might have unsorted stuff
            bands, occupations = [numpy.array(y) for y in zip(*[list(zip(*j)) for j in
                                                                [sorted(zip(i[0].tolist(), i[1].tolist()),
                                                                        key=lambda x: x[0])
                                                                 for i in zip(bands, occupations)]])]
            number_electrons = int(round(sum([sum(i) for i in occupations]) / num_kpoints))

            homo_indexes = [numpy.where(numpy.array([nint(_) for _ in x]) > 0)[0][-1] for x in occupations]
            if len(set(homo_indexes)) > 1:  # there must be intersections of valence and conduction bands
                return False, None, None, None
            else:
                homo = [_[0][_[1]] for _ in zip(bands, homo_indexes)]
                try:
                    lumo = [_[0][_[1] + 1] for _ in zip(bands, homo_indexes)]
                except IndexError:
                    raise ValueError("To understand if it is a metal or insulator, "
                                     "need more bands than n_band=number_electrons")

        else:
            bands = numpy.sort(bands)
            number_electrons = int(number_electrons)

            # find the zero-temperature occupation per band (1 for spin-polarized
            # calculation, 2 otherwise)
            number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
            # gather the energies of the homo band, for every kpoint
            homo = [i[number_electrons / number_electrons_per_band - 1] for i in bands]  # take the nth level
            try:
                # gather the energies of the lumo band, for every kpoint
                lumo = [i[number_electrons / number_electrons_per_band] for i in bands]  # take the n+1th level
            except IndexError:
                raise ValueError("To understand if it is a metal or insulator, "
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
        elif (any(i[0] == fermi_energy for i in max_mins) and
                  any(i[1] == fermi_energy for i in max_mins)):
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
    return 0.5 * np.sign(x) +0.5



#@workfunction
def bandfilling_ms_correction(host_bandstructure, defect_bandstructure, potential_alignment):
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
    host_number_electrons = host_bandstructure['scf_parameters'].dict.number_of_electrons
    
    #Finding host CBM and VBM
    bandgap = find_bandgap(host_bandsdata, host_number_electrons)
    if bandgap[0] == False and bandgap[1] == None:
        return Bool(False) #Metal systems
        #sys.exit('Are you sure to compute band filling corrections?\n
        #          Your host material is a metal.')
    else:
        E_VBM_host_align = bandgap[2] + float(potential_alignment)
        E_CBM_host_align = bandgap[3] + float(potential_alignment)
        
    
    #Extracting weights, occupations and energies from the defect band structure
    #Generalizing for both non and spin polarized calculations
    stored_occupations = defect_bandstructure['band_structure'].get_array('occupations')
    stored_bands = defect_bandstructure['band_structure'].get_array('bands')
    stored_weights = defect_bandstructure['band_structure'].get_array('weights')
    if  len(defect_bandstructure['band_structure'].get_array('occupations').shape) == 3:
        #spin polarized case: weights, band energies and occupations for both spin up and spin down 
        #are concatenated in one array with the first dimension equal to the double of the number of 
        #kpoints.
        occupations = np.concatenate([_ for _ in stored_occupations],axis=0)
        bands = np.concatenate([_ for _ in stored_bands],axis=0)
        weights = np.concatenate((stored_weights, stored_weights),axis=0)
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
            tmp = weights[k]* occupations[k][e]*(bands[k][e] - E_CBM_host_align)*heaviside(bands[k][e] 
                                                                                           - E_CBM_host_align)
            E_donor += tmp
    
    E_acceptor = 0.
    for k in range(weights.shape[0]):
        for e in range(bands.shape[1]):
            tmp = weights[k]* (max_occupation - occupations[k][e])*(E_VBM_host_align -bands[k][e])*heaviside(
                E_VBM_host_align -bands[k][e])
            E_acceptor += tmp

    return {'E_donor' :  E_donor, 'E_acceptor' : E_acceptor}


class BandFillingCorrectionWorkChain(WorkChain):
    """
    Workchain to compute band filling corrections
    bands_NiO=run(PwBandsWorkChain,code=Code.get_from_string(codename), structure = s, pseudo_family=Str(pseudo_family),
              options=ParameterData(dict=options), settings=ParameterData(dict=settings), kpoints_mesh=kpoints,
              parameters=parameters, optimization=Bool(False), relax=relax)   
    """
    @classmethod
    def define(cls, spec):
        super(BandFillingCorrectionWorkChain, cls).define(spec)
        spec.input("code",valid_type=Code)
        spec.input("host_structure",valid_type=StructureData)
        spec.input("defect_structure",valid_type=StructureData)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input('host_parameters', valid_type=ParameterData, required=False)
        spec.input('defect_parameters', valid_type=ParameterData, required=False)
        spec.input('pseudo_family', valid_type=Str)
        spec.input('kpoints_mesh', valid_type=KpointsData, required=False)
        spec.input('kpoints_distance', valid_type=Float, default=Float(0.2))
        spec.input('potential_alignment', valid_type=Float, default=Float(0.))
        spec.input('skip_relax', valid_type=Bool, required=False, default=Bool(True))
        spec.input_group('relax')
        spec.input('host_bandstructure', valid_type=Node, required=False)
        spec.input('defect_bandstructure', valid_type=Node, required=False)
        spec.outline(
            if_(cls.should_run_host)(
            cls.run_host),
            if_(cls.should_run_defect)(
            cls.run_defect),
            cls.compute_band_filling,
            cls.retrieve_bands
            
        )
        spec.dynamic_output()
  
    def should_run_host(self):
        return not  'host_bandstructure' in self.inputs
    
    def run_host(self):
        if 'host_parameters' not in self.inputs:
            self.abort_nowait('The host bandstructure calculation was requested but the "host parameters" dictionary\
            was not provided')
        
        inputs={'code' : self.inputs.code,
                'structure' : self.inputs.host_structure,
                'options' : self.inputs.options,
                'parameters' : self.inputs.host_parameters,
                'settings' : self.inputs.settings,
                'pseudo_family' : self.inputs.pseudo_family,
            
        }
        
        if 'skip_relax' in self.inputs:
            inputs['relax'] = self.inputs.relax
        if 'kpoints_mesh' in self.inputs:
            inputs['kpoints_mesh'] = self.inputs.kpoints_mesh
            
        running = submit(PwBandsWorkChain, **inputs)
        self.report('Launching the PwBandsWorkChain for the host with PK'.format(running.pid))
        return ToContext(host_bandsworkchain=running)
       
    
    def should_run_defect(self):
        return  not  'defect_bandstructure' in self.inputs
    
    def run_defect(self):
        if 'defect_parameters' not in self.inputs:
            self.abort_nowait('The defect bandstructure calculation was requested but the "host parameters" dictionary\
            was not provided')
        
        inputs={'code' : self.inputs.code,
                'structure' : self.inputs.defect_structure,
                'options' : self.inputs.options,
                'parameters' : self.inputs.defect_parameters,
                'settings' : self.inputs.settings,
                'pseudo_family' : self.inputs.pseudo_family,
            
        }
        
        if 'skip_relax' in self.inputs:
            inputs['relax'] = self.inputs.relax
        if 'kpoints_mesh' in self.inputs:
            inputs['kpoints_mesh'] = self.inputs.kpoints_mesh
            
        running = submit(PwBandsWorkChain, **inputs)
        #print "PK", running.pk
        self.report('Launching the PwBandsWorkChain for the defect with PK'.format(running.pid))
        return ToContext(defect_bandsworkchain=running)
       
        
    def compute_band_filling(self):
        if 'host_bandstructure' in self.inputs:
            host_bandstructure = self.inputs.host_bandstructure.out
        else:
            host_bandstructure = self.ctx.host_bandsworkchain.out
        if 'defect_bandstructure' in self.inputs:
            defect_bandstructure = self.inputs.defect_bandstructure.out
        else:
            defect_bandstructure = self.ctx.defect_bandsworkchain.out
        
        
        self.ctx.band_filling = bandfilling_ms_correction(host_bandstructure,
                                                         defect_bandstructure,
                                                 float(self.inputs.potential_alignment))
        
        
    def retrieve_bands(self):
        """
        Attach the relevant output nodes from the band calculation to the workchain outputs
        for convenience
        """
        self.report('BandFillingCorrection workchain succesfully completed')
        
        self.report('The computed band filling correction is <{}> and <{}> eV for a donor and an acceptor, respectively'.format(self.ctx.band_filling['E_donor'], self.ctx.band_filling['E_acceptor']))
        
        for label, value in six.iteritems(self.ctx.band_filling):
            self.out(str(label),Float(value))
        
#         for link_label in ['primitive_structure', 'seekpath_parameters', 'scf_parameters', 'band_parameters', 'band_structure']:
#             if link_label in self.ctx.defect_bandsworkchain.out:
#                 node = self.ctx.workchain_bands.get_outputs_dict()[link_label]
#                 self.out('defect'+str(link_label), node)
#                 self.report("attaching {}<{}> as an output node with label '{}'"
#                     .format(node.__class__.__name__, node.pk, link_label))
#             if link_label in self.ctx.host_bandsworkchain.out:
#                 node = self.ctx.workchain_bands.get_outputs_dict()[link_label]
#                 self.out('host'+str(link_label), node)
#                 self.report("attaching {}<{}> as an output node with label '{}'"
#                     .format(node.__class__.__name__, node.pk, link_label))

