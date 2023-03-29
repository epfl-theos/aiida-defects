# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import
from __future__ import print_function
import sys
import pymatgen
import numpy as np
from aiida.orm.data.upf import UpfData
from aiida.common.exceptions import NotExistent
from aiida.orm.data.upf import get_pseudos_from_structure
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm import DataFactory
from aiida.orm.data.singlefile import SinglefileData

from aiida.orm.code import Code

from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida_defects.pp.pp import PpWorkChain
from aiida_quantumespresso.calculations.pw import PwCalculation

from aiida_defects.pp.fft_tools import *
import six


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
    :param bulk_structure: StructureData object fro the bulk
    :param bulk_sphere_pot: disctionary containin g the sphere averaged potential in correspondence of each atom in the host structure
    :param bulk_symbols: list of the symbol of each atom in the host structure
    :param defect_sphere_pot: disctionary containin the sphere averaged potential in correspondence of each atom in the defective structure
    :param defect_symbols: list of the symbol of each atom in the defect structure
    :param defect_structure: StructureData object fro the defect
    :param defect_grid: 3D-FFT grid for the defect obtained from the read_grif function
    :param e_tol: energy tolerance to decide which atoms to exclude to compute alignment
                (0.2 eV; as in S. Lany FORTRAN codes)
    :result pot_align: value of the potential alignment in eV
    Note: Adapted from pylada defects (https://github.com/epfl-theos/aiida-defectsada/pylada-defects)
    Requirements: trilinear_interpolation, avg_potential_at_core. In order to use trilinear_interpolation the
    3D-FFT grid should be extracted from the FolderData node in which aiida.filplot is stored in the DB using
    the read_grid function
    """
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


class PotentialAlignmentLanyZunger(WorkChain):
    """
    Computes the potential alignment for defect calculations.
    You can have three different choices:
    1) perform a pw calculation followd by a pp calculation using the PpWorkChain for either bulk
    and defective systems
    2) provide the folder of the pw calculation or the PWCalculation Node for either the bulk or the defective system
    3) for the bulk you can directly provide the fft_grid if it has been already computed
    TODO: the input variable alignment_type was created so that other alignment types different from
          Lany-Zunger could be implemented modularly using the same workchain
    """

    @classmethod
    def define(cls, spec):
        super(PotentialAlignmentLanyZunger, cls).define(spec)
        spec.input("host_structure", valid_type=StructureData)
        spec.input("defect_structure", valid_type=StructureData)
        spec.input("run_pw_host", valid_type=Bool, required=True)
        #spec.input_group('host_fftgrid', required=False)
        spec.input(
            "host_parent_folder",
            valid_type=(FolderData, RemoteData),
            required=False)
        spec.input(
            "host_parent_calculation",
            valid_type=PwCalculation,
            required=False)
        spec.input("run_pw_defect", valid_type=Bool, required=True)
        spec.input(
            "defect_parent_folder", valid_type=RemoteData, required=False)
        spec.input(
            "defect_parent_calculation",
            valid_type=PwCalculation,
            required=False)
        spec.input("code_pp", valid_type=Str)
        spec.input("code_pw", valid_type=Str, required=False)
        spec.input("pseudo_family", valid_type=Str, required=False)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData, required=False)
        spec.input(
            'parameters_pw_host', valid_type=ParameterData, required=False)
        spec.input(
            'parameters_pw_defect', valid_type=ParameterData, required=False)
        spec.input('parameters_pp', valid_type=ParameterData)
        spec.input(
            'alignment_type',
            valid_type=Str,
            required=False,
            default=Str('lany_zunger'))
        spec.outline(
            cls.validate_inputs,
            cls.run_host,
            cls.run_atomic_sphere_pot_host,
            cls.extract_values_host,
            cls.run_defect,
            cls.run_atomic_sphere_pot_defect,
            cls.extract_values_defect,
            cls.compute_alignment,
        )
        spec.dynamic_output()

    def validate_inputs(self):
        """
        To perform a PpCalculation we need to specify either the PW parent calculation or the corresponding
        remote folder for both the bulk and defective structures.
        In the case of the bulk there is also the possibility that the FFT grid has been already extracted from
        the filplot file (for workflows where more that one defect is examined it is unecessarily heavy to
        re-extract the grid and store it in the DB evry time)
        """

        if self.inputs.run_pw_host == Bool(False):
            if not ('host_parent_calculation' in self.inputs
                    or 'host_parent_folder' in self.inputs):
                self.abort_nowait(
                    'Neither the parent_calculation nor the parent_folder for the host input was defined'
                )

            if 'host_parent_calculation' in self.inputs:
                self.ctx.host_parent_folder = self.inputs.host_parent_calculation.out.remote_folder

            elif 'host_parent_folder' in self.inputs:
                self.ctx.host_parent_folder = self.inputs.host_parent_folder

            #elif 'host_fftgrid'  in self.inputs:
            #    self.ctx.host_fftgrid = self.inputs.host_fftgrid
        elif self.inputs.run_pw_host == Bool(True):
            pw_inputs = [
                'parameters_pw_host', 'kpoints', 'pseudo_family', 'code_pw'
            ]
            for i in pw_inputs:
                if not i in self.inputs:
                    self.abort_nowait(
                        'You need to provide {} for the pw calculation of the host system'
                        .format(i))

        if self.inputs.run_pw_defect == Bool(False):
            if not ('defect_parent_calculation' in self.inputs
                    or 'defect_parent_folder' in self.inputs):
                self.abort_nowait(
                    'Neither the parent_calculation nor the parent_folder input for the defect was defined'
                )

            try:
                defect_parent_folder = self.inputs.defect_parent_calculation.out.remote_folder
            except AttributeError:
                defect_parent_folder = self.inputs.defect_parent_folder

            self.ctx.defect_parent_folder = defect_parent_folder
        elif self.inputs.run_pw_defect == Bool(True):
            pw_inputs = [
                'parameters_pw_defect', 'kpoints', 'pseudo_family', 'code_pw'
            ]
            for i in pw_inputs:
                if not i in self.inputs:
                    self.abort_nowait(
                        'You need to provide {} for the pw calculation of the defective system'
                        .format(i))


#     def should_run_host(self):
#         """
#         Checking if the PpWorkChain should be run for the host system
#         """
#         return bool('host_fftgrid' not in self.inputs)

    def run_host(self):
        """
        Running the PpWorkChain to compute the electrostatic potential of the host system
        """
        #Ensure that we are computing the electrostatic potential
        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11

        inputs = {
            'structure': self.inputs.host_structure,
            'code_pp': self.inputs.code_pp,
            'options': self.inputs.options,
            'settings': self.inputs.settings,
            'parameters_pp': ParameterData(dict=param_pp),
            'pw_calc': self.inputs.run_pw_host
            #'remote_folder' : self.ctx.host_parent_folder,
        }

        if self.inputs.run_pw_host == Bool(False):
            inputs['remote_folder'] = self.ctx.host_parent_folder
        elif self.inputs.run_pw_host == Bool(True):
            inputs['code_pw'] = self.inputs.code_pw
            inputs['parameters'] = self.inputs.parameters_pw_host
            inputs['kpoints'] = self.inputs.kpoints
            inputs['pseudo_family'] = self.inputs.pseudo_family

        running = submit(PpWorkChain, **inputs)
        self.report('Launching PpWorkChain for the host. pk value {}'.format(
            running.pid))
        return ToContext(host_ppcalc=running)

    def run_atomic_sphere_pot_host(self):

        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11

        inputs = {
            'structure': self.inputs.host_structure,
            'code_pp': self.inputs.code_pp,
            'options': self.inputs.options,
            'settings': self.inputs.settings,
            #'parameters_pp' : ParameterData(dict=param_pp),
            'pw_calc': Bool(False),
            'remote_folder': self.ctx.host_ppcalc.out.remote_folder
        }

        tmp_pp = self.ctx.host_ppcalc.out.retrieved.get_file_content(
            'aiida.filplot')

        tmp_pp = tmp_pp.splitlines()[2:3][0].strip().split(' ')
        tmp_pp = [i for i in tmp_pp if i != '']
        alat = tmp_pp[1]

        alat = float(alat)

        calcs = {}
        conv_ang_to_alat = alat * 0.529177
        for i in self.inputs.host_structure.sites:

            parameters_pp = {
                'INPUTPP': {
                    'plot_num': 11
                },
                'PLOT': {
                    'iflag': 1,
                    'output_format': 0,
                    'e1(1)': 1.0,
                    'e1(2)': 0.0,
                    'e1(3)': 0.0,
                    'x0(1)': i.position[0] / conv_ang_to_alat,
                    'x0(2)': i.position[1] / conv_ang_to_alat,
                    'x0(3)': i.position[2] / conv_ang_to_alat,
                    'nx': 2
                }
            }
            inputs['parameters_pp'] = ParameterData(dict=parameters_pp)
            future = submit(PpWorkChain, **inputs)
            self.report(
                'Launching PpWorkChain for atom {} in the host system. pk value {}'
                .format(i, future.pid))
            calcs[str(i)] = Outputs(future)

        return ToContext(**calcs)

    def extract_values_host(self):
        self.ctx.host_symbols = []

        for site in self.inputs.host_structure.sites:
            for kind in self.inputs.host_structure.kinds:
                if kind.name == site.kind_name:
                    self.ctx.host_symbols.append(kind.symbol)

        self.ctx.atom_sphere_pot_host = {}

        for site in self.inputs.host_structure.sites:
            file_tmp = self.ctx[str(site)]['retrieved'].get_file_content(
                'aiida.out').splitlines()

            for kind in self.inputs.host_structure.kinds:
                if kind.name == site.kind_name:
                    symbol = kind.symbol

            for line in file_tmp:
                if 'Min, Max, imaginary charge:' in line:
                    self.ctx.atom_sphere_pot_host[str(symbol) + '_' + str(
                        site.position)] = [
                            i for i in line.strip().split(' ') if i != ''
                        ][4]

    def run_defect(self):
        """
        Running the PpWorkChain to compute the electrostatic potential of the defective system
        """
        #Ensure that we are computing the electrostatic potential
        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11

        inputs = {
            'structure': self.inputs.defect_structure,
            'code_pp': self.inputs.code_pp,
            'options': self.inputs.options,
            'settings': self.inputs.settings,
            'parameters_pp': ParameterData(dict=param_pp),
            'pw_calc': self.inputs.run_pw_defect
            #'remote_folder' : self.ctx.defect_parent_folder,
        }

        if self.inputs.run_pw_defect == Bool(False):
            inputs['remote_folder'] = self.ctx.defect_parent_folder
        elif self.inputs.run_pw_defect == Bool(True):
            inputs['code_pw'] = self.inputs.code_pw
            inputs['parameters'] = self.inputs.parameters_pw_defect
            inputs['kpoints'] = self.inputs.kpoints
            inputs['pseudo_family'] = self.inputs.pseudo_family

        running = submit(PpWorkChain, **inputs)
        self.report('Launching PpWorkChain for the defect. pk value {}'.format(
            running.pid))
        return ToContext(defect_ppcalc=running)

    def run_atomic_sphere_pot_defect(self):

        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11

        inputs = {
            'structure': self.inputs.defect_structure,
            'code_pp': self.inputs.code_pp,
            'options': self.inputs.options,
            'settings': self.inputs.settings,
            #'parameters_pp' : ParameterData(dict=param_pp),
            'pw_calc': Bool(False),
            'remote_folder': self.ctx.defect_ppcalc.out.remote_folder
        }

        tmp_pp = self.ctx.defect_ppcalc.out.retrieved.get_file_content(
            'aiida.filplot')

        tmp_pp = tmp_pp.splitlines()[2:3][0].strip().split(' ')
        tmp_pp = [i for i in tmp_pp if i != '']
        alat = tmp_pp[1]

        alat = float(alat)

        calcs = {}
        conv_ang_to_alat = alat * 0.529177
        for i in self.inputs.defect_structure.sites:

            parameters_pp = {
                'INPUTPP': {
                    'plot_num': 11
                },
                'PLOT': {
                    'iflag': 1,
                    'output_format': 0,
                    'e1(1)': 1.0,
                    'e1(2)': 0.0,
                    'e1(3)': 0.0,
                    'x0(1)': i.position[0] / conv_ang_to_alat,
                    'x0(2)': i.position[1] / conv_ang_to_alat,
                    'x0(3)': i.position[2] / conv_ang_to_alat,
                    'nx': 2
                }
            }
            inputs['parameters_pp'] = ParameterData(dict=parameters_pp)
            future = submit(PpWorkChain, **inputs)
            self.report(
                'Launching PpWorkChain for atom {} in defective system. pk value {}'
                .format(i, future.pid))
            calcs[str(i)] = Outputs(future)

        return ToContext(**calcs)

    def extract_values_defect(self):
        self.ctx.defect_symbols = []

        for site in self.inputs.defect_structure.sites:
            for kind in self.inputs.defect_structure.kinds:
                if kind.name == site.kind_name:
                    self.ctx.defect_symbols.append(kind.symbol)

        self.ctx.atom_sphere_pot_def = {}

        for site in self.inputs.defect_structure.sites:
            file_tmp = self.ctx[str(site)]['retrieved'].get_file_content(
                'aiida.out').splitlines()

            for kind in self.inputs.host_structure.kinds:
                if kind.name == site.kind_name:
                    symbol = kind.symbol

            for line in file_tmp:
                if 'Min, Max, imaginary charge:' in line:
                    self.ctx.atom_sphere_pot_def[str(symbol) + '_' + str(
                        site.position)] = [
                            i for i in line.strip().split(' ') if i != ''
                        ][4]

    def compute_alignment(self):
        """
        Computing the potential alignment
        """

        if str(self.inputs.alignment_type) == 'lany_zunger':
            pot_align = lz_potential_alignment(
                self.inputs.host_structure,
                self.ctx.atom_sphere_pot_host,
                self.ctx.host_symbols,
                self.inputs.defect_structure,
                self.ctx.atom_sphere_pot_def,
                self.ctx.defect_symbols,
                e_tol=0.2)
            self.out('pot_align', Float(pot_align))
            self.report(
                'PotentialAlignment workchain completed succesfully. The potential alignement computed with the {} scheme is {} eV'
                .format(self.inputs.alignment_type, pot_align))
