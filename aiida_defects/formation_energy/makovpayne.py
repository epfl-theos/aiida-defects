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
import matplotlib.pyplot as plt
from aiida.work.run import run
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
from aiida.orm import load_node

from aiida.work.run import run, submit
from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_defects.tools.structure_manipulation import get_spacegroup


#####################################################
#                                                   #
# MAKOV-PAYNE ELECTROSTATIC CORRECTION              #
#####################################################
class MakovPayneCorrection(WorkChain):
    """
    Computes the electrostatic correction for charged defects according to the Makov-Payne
    scheme (Makov & Payne, Mod. Mat. Sci. Eng. 17 (2009) 084002):
    E_MP = [1+c_sh(1-1/epsilon_r)]*DeltaE_1 with DeltaE_1 =q**2 alpha_M/(2*epsilon_r*L)
    The correction is computed by rescaling DeltaE_1, the first order correction, computed in the case 
    of q=1 and epsilon_r=1, according to the dielectric constant of the host system (epsilon_r)
    and its geometrical properties (c_sh). Delta_1(q=1,epsilon_r=1) is obtained through the following
    procedure:
    1) we remove all the atoms from the bulk/host optimized structure
    2) we add one H atom in the middle of the cell (or in the defect position)
    3) we perform an scf calculation and DeltaE_1 corresponds to the ewald energy in the QE output

    """
    @classmethod
    def define(cls, spec):
        super(MakovPayneCorrection, cls).define(spec)
        spec.input("bulk_structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str)
        spec.input("pseudo_family",valid_type=Str)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('epsilon_r', valid_type=Float)
        spec.input('defect_charge', valid_type=Float)
        #spec.input('defect_position', valid_type=ArrayData)         
        spec.outline(
            cls.run_pw_H_structure,
            cls.retreive_ewald_energy,
            cls.compute_correction,
            )
        spec.dynamic_output()
        
    def run_pw_H_structure(self):
        """
        Running a PwBaseWorkChain for a system that has one H atom but the same cell parameter of the host
        """
        
        #Creating a structure with the same parameter of the bulk but only one H atom in the center of 
        #the cell
        import pymatgen
        lattice = pymatgen.Lattice(self.inputs.bulk_structure.cell)
        h_structure = pymatgen.Structure(lattice, ['H'], [[0.5, 0.5, 0.5]])
        
        H_structure = StructureData(pymatgen=h_structure)
            
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        kpoints = self.inputs.kpoints
        
        param = self.inputs.parameters.get_dict()
        param['CONTROL']['calculation'] ='scf'
        param['SYSTEM']['nspin'] = 2
        param['SYSTEM']['starting_magnetization'] = {'H' : 1}
        
        parameter = ParameterData(dict=param)
        
        
        inputs={'structure' : H_structure,
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }

        running = submit(PwBaseWorkChain,**inputs)
        self.report('Launching PwBaseWorkChain for the calculation of the first order Makov Payne Correction. pk value {}'.format( running.pid))  
        return ToContext(pwcalc = running)
    
    def retreive_ewald_energy(self):
	"""
        Retreiving the ewald energy for the previous PwCalculation
        """
        
        self.ctx.ewald_energy = self.ctx.pwcalc.get_outputs_dict()['output_parameters'].dict.energy_ewald

    
    def compute_correction(self):
        """
	Computing Makov Payne Correction
        """
        #Setting c_sh values
    
        c_sh_table = {'SC' : -0.369,
                    'BCC' : -0.342,
                    'FCC' : -0.343,
                    'HCP' : -0.478,
                    'OTHER' : -1./3.,
        
        }
    
        #Identifying the bulk geometry and selectring the corresponding c_sh
        #For HCP space groups see N.V. Belov, The Structure of Ionic Crystals and Metallic Phases (in Russian)
        #U.S.S.R. Ac. Sc. Press, Moscow (1947)
        space_groups = {'SC' : [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, 221, 222, 223, 224],
                        'BCC' : [199, 197, 204, 206, 211, 217, 220, 229, 230],
                        'FCC' : [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228],
                        'HCP' : [156, 164, 160, 166, 194, 187, 186],
            }

    
        space_group = get_spacegroup(self.inputs.bulk_structure)
        #print "sg", space_group

    
        for label in space_groups:
            if space_group in space_groups[label]:
                geometry = label
                break
            else:
                geometry = 'OTHER'

        c_sh = c_sh_table[geometry]

        
        #Computing Makov-Payne Correction
        epsilon_r = float(self.inputs.epsilon_r)
        charge = float(self.inputs.defect_charge)
        
        E_MP = (1. + c_sh*(1. - 1./epsilon_r)) * charge**2 * self.ctx.ewald_energy/epsilon_r

        self.out('Makov_Payne_Correction', Float(E_MP))
        self.report('Makov Payne Correction {}'.format(E_MP))
	self.report('MakovPayneCorrection workchain completed succesfully')
    

