
# coding: utf-8

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

from aiida.work.run import  submit 
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_ 
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida.workflows.user.aiida_defects.pp.pp import PpWorkChain
from aiida_quantumespresso.calculations.pw import PwCalculation

from aiida.workflows.user.aiida_defects.pp.fft_tools import *

def lz_potential_alignment(bulk_structure, bulk_grid, defect_structure, defect_grid, e_tol=0.2):
    """
    Function to compute the potential alignment correction using the average atomic electrostatic potentials
    of the bulk and defective structures. See: S. Lany and A. Zunger, PRB 78, 235104 (2008)
    :param bulk_structure: StructureData object fro the bulk
    :param bulk_grid: 3D-FFT grid for the bulk obtained from the read_grif function
    :param defect_structure: StructureData object fro the defect
    :param defect_grid: 3D-FFT grid for the defect obtained from the read_grif function
    :param e_tol: energy tolerance to decide which atoms to exclude to compute alignment 
                (0.2 eV; as in S. Lany FORTRAN codes)
    :result pot_align: value of the potential alignment in eV
    Note: Adapted from pylada defects (https://github.com/pylada/pylada-defects)
    Requirements: trilinear_interpolation, avg_potential_at_core. In order to use trilinear_interpolation the 
    3D-FFT grid should be extracted from the FolderData node in which aiida.filplot is stored in the DB using 
    the read_grid function
    """
    #Extracting the potential at atomic sites from the 3D-FFT grid for the bulk
    # and computing the average per atomic site type
    bulk = trilinear_interpolation(bulk_grid, bulk_structure)
    avg_bulk = avg_potential_at_core(bulk)
    
    #Extracting the potential at atomic sites from the 3D-FFT grid for the bulk
    # and computing the average per atomic site type
    defect = trilinear_interpolation(defect_grid, defect_structure)
    avg_defect = avg_potential_at_core(defect)
    
    #Compute the difference between defect electrostatic potential and the average defect electrostatic potential
    #per atom
    
    diff_def = {}
    for atom, pot in defect['func_at_core'].iteritems():
        diff_def[atom] = pot - avg_defect[atom.split('_')[0]]
        
    max_diff = abs(max(diff_def.values()))

    #Identifying the list of atoms than can be used to compute the difference for which
    #diff_def is lower than max_diff or of a user energy tolerance (e_tol)
    acceptable_atoms = []
    for atom, value in diff_def.iteritems():
        if abs(value) < max_diff or abs(value) < e_tol:
            acceptable_atoms.append(atom)
            
    #Avoid excluding all atoms
    while (not bool(acceptable_atoms)):
        e_tol = e_tol * 10
        print("e_tol has been modified to {} in order to avoid excluding all atoms".format(e_tol))
        for atom, value in diff_def.iteritems():
            if abs(value) < max_diff or abs(value)*13.6058 < e_tol:
                acceptable_atoms.append(atom)
                
    #Computing potential alignment avareging over all the acceptable atoms            
    diff_def2 = []
    for atom, pot in defect['func_at_core'].iteritems():
        if atom in acceptable_atoms:
            diff_def2.append( pot - avg_bulk[atom.split('_')[0]])
    
    pot_align = np.mean(diff_def2)*13.6058
    return pot_align      



class PotentialAlignment(WorkChain):
    """
    Computes the potential alignment for defect calculations.
    Assumption: a PwCalculation for bulk and host has been already performed
    TODO: the input variavle alignment_type was created so that other alignment types different from 
          lany-zunger could be implemented modularly using the same workchain
    """
    @classmethod
    def define(cls, spec):
        super(PotentialAlignment, cls).define(spec)
        spec.input("host_structure",valid_type=StructureData)
        spec.input("defect_structure",valid_type=StructureData)
        spec.input_group('host_fftgrid', required=False)
        spec.input("host_parent_folder", valid_type=(FolderData,RemoteData), required = False)
        spec.input("host_parent_calculation", valid_type=PwCalculation, required = False)
        spec.input("defect_parent_folder", valid_type=RemoteData, required = False)
        spec.input("defect_parent_calculation",valid_type=PwCalculation, required = False)
        spec.input("code_pp",valid_type=Str)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input('parameters_pp', valid_type=ParameterData)
        spec.input('alignment_type', valid_type=Str, required=False, default=Str('lany_zunger'))
        spec.outline(
            cls.validate_inputs,
            if_(cls.should_run_host)(
                cls.run_host,
            ),
            cls.run_defect,
            cls.retrieve_results,
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
        if not ('host_parent_calculation' in self.inputs or 'host_parent_folder' in self.inputs or 'host_fftgrid'  in self.inputs):
            self.abort_nowait('Neither the parent_calculation nor the parent_folder nor the FFT grid for the host input was defined')
        elif not ('defect_parent_calculation' in self.inputs or 'defect_parent_folder' in self.inputs):
            self.abort_nowait('Neither the parent_calculation nor the parent_folder input for the defect was defined')
        
        if 'host_parent_calculation' in self.inputs:
            self.ctx.host_parent_folder = self.inputs.host_parent_calculation.out.remote_folder
        elif 'host_parent_folder' in self.inputs:
            self.ctx.host_parent_folder = self.inputs.host_parent_folder
        elif 'host_fftgrid'  in self.inputs:
            self.ctx.host_fftgrid = self.inputs.host_fftgrid
        
        
        try:
            defect_parent_folder = self.inputs.defect_parent_calculation.out.remote_folder
        except AttributeError:
            defect_parent_folder = self.inputs.defect_parent_folder

        self.ctx.defect_parent_folder = defect_parent_folder

    def should_run_host(self):
        """
        Checking if the PpWorkChain should be run for the host system
        """
        return bool('host_fftgrid' not in self.inputs)
            
    def run_host(self):
        """
        Running the PpWorkChain to compute the electrostatic potential of the host system
        """
        #Ensure that we are computing the electrostatic potential
        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11
        
        inputs = {'structure' : self.inputs.host_structure,
                  'code_pp' : self.inputs.code_pp,
                  'options' :self.inputs.options,
                  'settings' : self.inputs.settings,
                  'parameters_pp' : ParameterData(dict=param_pp),
                  'remote_folder' : self.ctx.host_parent_folder,
            
        }


        
        running = submit(PpWorkChain,**inputs)
        self.report('Launching PpWorkChain for the host. pk value {}'.format( running.pid))  
        return ToContext(host_ppcalc = running)
    
    def run_defect(self):
	"""
	Running the PpWorkChain to compute the electrostatic potential of the defective system
        """ 
        #Ensure that we are computing the electrostatic potential
        param_pp = self.inputs.parameters_pp.get_dict()
        param_pp['INPUTPP']['plot_num'] = 11

        inputs = {'structure' : self.inputs.defect_structure,
                  'code_pp' : self.inputs.code_pp,
                  'options' :self.inputs.options,
                  'settings' : self.inputs.settings,
                  'parameters_pp' : ParameterData(dict=param_pp),
                  'remote_folder' : self.ctx.defect_parent_folder,
            
        }
            
        running = submit(PpWorkChain,**inputs)
        self.report('Launching PpWorkChain for the defect. pk value {}'.format( running.pid))  
        return ToContext(defect_ppcalc = running)
    
    def retrieve_results(self):
        """
        Retrieving the FolderData produced by the PpCalculation of the hist and defective structures
        """
        if 'host_fftgrid' not in self.inputs:
            if any(c in self.inputs for c in ('host_parent_calculation', 'host_parent_folder')):
                self.ctx.host_fftgrid = read_grid(self.ctx.host_ppcalc.out.retrieved)
            
        
        self.ctx.defect_fftgrid = read_grid(self.ctx.defect_ppcalc.out.retrieved)

    
    def compute_alignment(self):
	"""
        Computing the potential alignment
        """
        if str(self.inputs.alignment_type) == 'lany_zunger':
            pot_align = lz_potential_alignment(self.inputs.host_structure,
                                               self.ctx.host_fftgrid,
                                               self.inputs.defect_structure,
                                               self.ctx.defect_fftgrid,
                                               e_tol=0.2)
            self.out('pot_align', Float(pot_align))
            self.report('PotentialAlignment workchain completed succesfully. The potential alignement comptuted with the {} scheme is {} eV'.format(self.inputs.alignment_type,
                                                                                      pot_align))





