# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
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
from aiida.orm import DataFactory
from aiida.orm.code import Code
from aiida.work.run import run, submit
from aiida.work.workchain import WorkChain, ToContext, Outputs
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_defects.tools.structure_manipulation import biaxial_strain_structure 
from aiida_defects.tools.structure_manipulation import create_suitable_inputs_noclass

strain=List()
strain._set_list([-7., -6., -5, -4, -3.,-2, -1., 0., 1., 2., 3., 4., 5., 6., 7.])

class BiaxialStrainWorkChain(WorkChain):
    """
    Workchain to apply biaxial strain to a cubic structure


    """
    @classmethod
    def define(cls, spec):
        super(BiaxialStrainWorkChain, cls).define(spec)
        spec.input("structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str)
        spec.input("pseudo_family",valid_type=Str)
        spec.input("strain",required=False,default=strain)
        spec.input("relax_axis", valid_type=Str, required=False, default=Str('c'))
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('magnetic_phase', valid_type=Str,required=False, default=Str('NM'))
        spec.input('B_atom', valid_type=Str)
        spec.input('hubbard_u', valid_type=ParameterData, required=False, default=ParameterData(dict={}))
 
        spec.outline(
            cls.initialization,
            cls.run_pw,
            cls.retrieve_results,
            )
        spec.dynamic_output()
       
    def initialization(self):  
        """
        Initializing the calculations by selecting the 
        appropiate values for the parameters necessary to the calculations
        """
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        kpoints = self.inputs.kpoints
        param = self.inputs.parameters.get_dict()
        param['CONTROL']['calculation'] = 'vc-relax'
        param['CONTROL']['tstress'] = True
        param['CONTROL']['tprnfor'] = True
        
        if self.inputs.relax_axis == 'a':
            param['CELL']['cell_dofree'] = 'epitaxial_bc'
        elif self.inputs.relax_axis == 'b':
            param['CELL']['cell_dofree'] = 'epitaxial_ac'
        elif self.inputs.relax_axis == 'c':
            param['CELL']['cell_dofree'] = 'epitaxial_ab'
        else:
            sys.exit("Error message: relax_axis variable can only be 'a', 'b', or 'c'. Please enter a valid value.")
                
        
        self.ctx.inputs={
            'code' : code_pw,
            'pseudo_family' : Str(self.inputs.pseudo_family),
            'kpoints' : kpoints,
            'parameters' : ParameterData(dict=param),
            'settings' : settings,
            'options' : options,
            'max_iterations' : Int(10),

        }
        
    
    def run_pw(self):
        """
        Creating a strained structure for each strain value and
        and running the PwBaseWorkChain to relax it.
        Calculations for each strain value are performed in parallel.
        """
        
        calcs = {}
     
        for value in self.inputs.strain:
            structure_tmp = biaxial_strain_structure(self.inputs.structure, self.inputs.relax_axis, Float(value))
            suitable_inputs=create_suitable_inputs_noclass(structure_tmp, self.inputs.magnetic_phase, self.inputs.B_atom)
            self.ctx.inputs['structure'] = suitable_inputs['structure']
            
            param = self.ctx.inputs['parameters'].get_dict()   
            magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
            if str(self.inputs.magnetic_phase)  in magnetic_phases:
                param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                param['SYSTEM']['nspin'] = 2
            U = {}
            hubbard_U= self.inputs.hubbard_u.get_dict()    
            if bool(hubbard_U) and len(hubbard_U) == 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                for site in  self.ctx.inputs['structure'].sites:
                    if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                        U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                param['SYSTEM']['hubbard_u'] = U
            elif bool(hubbard_U) and len(hubbard_U) > 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                U = hubbard_U
                param['SYSTEM']['hubbard_u'] = U
	    
            self.ctx.inputs['parameters']= ParameterData(dict=param)

            future = submit(PwBaseWorkChain,**self.ctx.inputs)
            self.report('Launching PwBaseWorkChain for the {}%  strain value. pk value {}'.format(value, future.pid))
            calcs[str(value)] = Outputs(future) 


        return ToContext(**calcs)
    
    def retrieve_results(self):
        """
        Extract the total energy and the optimized structure 
        for each strain value and attach them to the workchain outputs
        """
        energies={}
        structures={}
        
        for value in self.inputs.strain:
            tot_energy = self.ctx[str(value)]["output_parameters"].dict.energy
            self.out('energy_'+str(value),Float(tot_energy))
            opt_structure = self.ctx[str(value)]["output_structure"]
            self.out('structure_'+str(value),opt_structure)
            
        self.out('strain',self.inputs.strain)
        self.report('BiaxialStrainWorkChain succesfully completed.')
        return