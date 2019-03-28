# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import
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
from aiida.orm.code import Code
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.group import Group


from aiida.work.run import run, submit
from aiida.work.workchain import WorkChain, ToContext, if_, while_, Outputs, if_, append_

from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List

from aiida_defects.formation_energy.bandfilling import BandFillingCorrectionWorkChain
from aiida_defects.formation_energy.makovpayne import MakovPayneCorrection
from aiida_defects.formation_energy.pot_align import PotentialAlignmentLanyZunger
from aiida_defects.formation_energy.pot_align import lz_potential_alignment
from aiida_defects.tools.defects import defect_creator

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation

#DEFAULT INPUTS FOR DEFECT CREATION
#Specifing the supercell size
scale_sc_default=List()
scale_sc_default._set_list([1,1,1])
                   
#Specifying the type of defects to be created                   
vacancies_default=List()
vacancies_default._set_list([])
substitutions_default=ParameterData(dict={})

#Default values for defect formation energy corrections
corrections_default=ParameterData(dict={'makov_payne' : False,
                                        'bandfilling' : False,
                                        'potential_alignment' : False,
    
})

class DefectWorkChain(WorkChain):
    """
    Workflow to compute defect formation energy starting from a bulk structure
    """
    @classmethod
    def define(cls, spec):
        super(DefectWorkChain, cls).define(spec)
        #CODES & PSEUDOPOTENTIALS
        spec.input("code_pw",valid_type=Str)
        spec.input("code_pp",valid_type=Str)
        spec.input("pseudo_family",valid_type=Str)
        #OPTIONS & SETTINGS
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        #INPUTS FOR THE HOST CALCULATION
        spec.input("run_primitive_host",valid_type=Bool, required = False, default=Bool(True))
        spec.input("host_folder_data",valid_type=(FolderData,RemoteData),required=False)
        spec.input("type_run_primitive_host",valid_type=Str, required = False, default=Str('vc-relax'))
        spec.input("structure",valid_type=StructureData)
        spec.input('kpoints', valid_type=KpointsData, required=False)
        spec.input('kpoints_distance', valid_type=Float, default=Float(0.2))
        spec.input('kpoints_force_parity', valid_type=Bool, default=Bool(False))
        spec.input('vdw_table', valid_type=SinglefileData, required=False)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('automatic_parallelization', valid_type=ParameterData, required=False)
        spec.input('final_scf', valid_type=Bool, default=Bool(False))
        spec.input('group', valid_type=Str, required=False)
        spec.input('max_iterations', valid_type=Int, default=Int(5))
        spec.input('max_meta_convergence_iterations', valid_type=Int, default=Int(5))
        spec.input('meta_convergence', valid_type=Bool, default=Bool(True))
        spec.input('relaxation_scheme', valid_type=Str, default=Str('vc-relax'))
        spec.input('volume_convergence', valid_type=Float, default=Float(0.01))
        spec.input('clean_workdir', valid_type=Bool, default=Bool(False))
        spec.input('parameters_pp', valid_type=ParameterData)
        #INPUTS TO CREATE THE DEFECTS
        spec.input("type_run_defects",valid_type=Str, required = False, default=Str('relax'))
        spec.input('vacancies', valid_type=List, required=False,default=vacancies_default)
        spec.input('substitutions',valid_type=ParameterData,required=False,default=substitutions_default)
        spec.input('scale_sc', valid_type=List,required=False,default=scale_sc_default)
        spec.input('cluster',valid_type=Bool,required=False, default=Bool(False))
        spec.input('defect_charge',valid_type=Float,required=False,default=Float(0.00))
        spec.input('corrections',valid_type=ParameterData,default=corrections_default)
        spec.input('epsilon_r',valid_type=Float,required=False,default=Float(1.00))
        spec.input('pot_align',valid_type=Float,required=False)
        spec.input_group('bf_relax')

        spec.outline(
            if_(cls.should_run_primitive_host)(
                cls.run_primitive_host,
                cls.retrieve_host_results
            ),
             cls.create_defective_supercells,
             cls.run_defective_structures,
             cls.retrieve_defect_results,
            if_(cls.should_run_makovpayne)(
                cls.run_makovpayne
            ),

            if_(cls.should_run_bandfilling)(
                cls.run_bandfilling
            ),
             if_(cls.should_run_pot_align)(
                 cls.run_pot_align
             ),
            cls.retrieve_corrections,
        )
        spec.dynamic_output()
        
    def should_run_primitive_host(self):
        """
        Checking if the PwBaseCalculation should be run for the primitive host structure
        """
        return bool(self.inputs.run_primitive_host)
    
    def run_primitive_host(self):
        """
        Running PwBaseWorkChain for the primitive host structure
        """

        inputs={
            'code' : Code.get_from_string(str(self.inputs.code_pw)),
            'pseudo_family' : Str(self.inputs.pseudo_family),
            'parameters' : self.inputs.parameters,
            'settings' : self.inputs.settings,
            'options' : self.inputs.options,
            'structure' : self.inputs.structure

        }
        
        
        if self.inputs.type_run_primitive_host == 'scf':
            
            if not ('kpoints' in self.inputs):
                self.abort_nowait('You need to provide a kpoint inputs when type_run_primitive_host = scf' )
            
            inputs['structure'] = self.inputs.structure
            inputs['kpoints'] = self.inputs.kpoints
#             inputs={
#                 'code' : Code.get_from_string(str(self.inputs.code_pw)),
#                 'pseudo_family' : Str(self.inputs.pseudo_family),
#                 'parameters' : self.inputs.parameters,
#                 'settings' : self.inputs.settings,
#                 'options' : self.inputs.options,
#                 'structure' : self.inputs.structure,
#                 'kpoints' : self.inputs.kpoints,

#             }


            running = submit(PwBaseWorkChain,**inputs)
            self.report('Launching PwBaseWorkChain for the stoichiometric host. pk value {}'.format( running.pid))
            return ToContext(host_pwcalc = running)
            
        
        
        elif self.inputs.type_run_primitive_host == 'vc-relax' or self.inputs.type_run_primitive_host == 'relax':
            
            inputs['relaxation_scheme'] = self.inputs.type_run_primitive_host
    
            if 'kpoints' in self.inputs:
                inputs['kpoints'] = self.inputs.kpoints
            elif 'kpoints_distance' in self.inputs:
                inputs['kpoints_distance'] = self.inputs.kpoints_distance
            if 'kpoints_force_parity' in self.inputs:
                inputs['kpoints_force_parity'] = self.inputs.kpoints_force_parity
                
                
            other_inputs=['vdw_table',
                          'final_scf',
                          'group',
                          'max_iterations',
                          'max_meta__convergence_iterations',
                          'meta_convergence',
                          'volume_convergence',
                           'clean_workdir']
            
            for i in other_inputs:
                if i in self.inputs:
                    inputs[str(i)] = eval('self.inputs.'+i)

            running = submit(PwRelaxWorkChain,**inputs)
            self.report('Launching PwRelaxWorkChain for the host system. pk value {}'.format( running.pid))
            return ToContext(host_pwcalc = running)
            
    
    def retrieve_host_results(self):
        """
        Extract the total energy and the optimized structure of the host
        """        
        
        if bool(self.inputs.run_primitive_host): 
            self.ctx.total_energy_host = self.ctx.host_pwcalc.out.output_parameters.dict.energy
            self.out('energy_host',Float(self.ctx.total_energy_host))

        
        if self.inputs.type_run_primitive_host == 'vc-relax' or self.inputs.type_run_primitive_host == 'relax':
            self.ctx.structure_host = self.ctx.host_pwcalc.out.output_structure
        else:
            self.ctx.structure_host = self.inputs.structure
            
            
        self.out('structure_host',self.inputs.structure)
        

        return

    
    def create_defective_supercells(self):
        """
        Creating defective supercells
        """

        self.ctx.defective_structures = defect_creator(self.ctx.structure_host,
                                                          vacancies,
                                                          substitutions,
                                                          scale_sc,
                                                          cluster)
        
        
    def run_defective_structures(self):
        
        inputs={
            'code' : Code.get_from_string(str(self.inputs.code_pw)),
            'pseudo_family' : Str(self.inputs.pseudo_family),
            #'parameters' : self.inputs.parameters,
            'settings' : self.inputs.settings,
            'options' : self.inputs.options,
            #'structure' : self.inputs.structure

        }
        
        calcs = {}
        self.ctx.inputs_defective_structures = {}
        if self.inputs.type_run_defects == 'scf':
        
            for structure in self.ctx.defective_structures:
                if '_0' not in structure:
                    
   
                    if not ('kpoints' in self.inputs):
                        self.abort_nowait('You need to provide a kpoint inputs when type_run_defects = scf' )
            
                    param =self.inputs.parameters.get_dict()
                    param['SYSTEM']['tot_charge'] = float(self.inputs.defect_charge)
                    
                    inputs['parameters'] = ParameterData(dict=param)
                    inputs['structure'] = self.ctx.defective_structures[structure]
                    inputs['kpoints'] = self.inputs.kpoints
                    
                    future = submit(PwBaseWorkChain,**inputs)
                    self.ctx.inputs_defective_structures[structure] = inputs
                    self.report('Launching PwBaseWorkChain for the defective structure {} with charge {}. pk value {}'.format(structure,
                                                                                                                              self.inputs.defect_charge,
                                                                                                                              future.pid))
                    calcs[str(structure)] = Outputs(future)
                    
            return ToContext(**calcs)
    
        elif self.inputs.type_run_defects == 'vc-relax' or self.inputs.type_run_defects == 'relax':
            for structure in self.ctx.defective_structures:
                if '_0' not in structure:

                    inputs['relaxation_scheme'] = self.inputs.type_run_defects

                    if 'kpoints' in self.inputs:
                        inputs['kpoints'] = self.inputs.kpoints
                    elif 'kpoints_distance' in self.inputs:
                        inputs['kpoints_distance'] = self.inputs.kpoints_distance
                    if 'kpoints_force_parity' in self.inputs:
                        inputs['kpoints_force_parity'] = self.inputs.kpoints_force_parity

                    param =self.inputs.parameters.get_dict()
                    param['SYSTEM']['tot_charge'] = float(self.inputs.defect_charge)
                    inputs['parameters'] = ParameterData(dict=param)
                    inputs['structure'] = self.ctx.defective_structures[structure]


                    other_inputs=['vdw_table',
                                  'final_scf',
                                  'group',
                                  'max_iterations',
                                  'max_meta__convergence_iterations',
                                  'meta_convergence',
                                  'volume_convergence',
                                   'clean_workdir']

                    for i in other_inputs:
                        if i in self.inputs:
                            inputs[str(i)] = eval('self.inputs.'+i)


                    future = submit(PwRelaxWorkChain,**inputs)
                    self.report('Launching PwRelaxWorkChain for the defective structure {} with charge {}. pk value {}'.format(structure,
                                                                                                                          self.inputs.defect_charge,
                                                                                                                          future.pid))
                    calcs[str(structure)] = Outputs(future)

            return ToContext(**calcs)


    def retrieve_defect_results(self):
        """
        Extract the total energy and the optimized structure of the defective structures
        """  
        self.ctx.total_energy_defective_structures = {}
        for structure in self.ctx.defective_structures:
            if '_0' not in structure:
                self.ctx.total_energy_defective_structures[structure] = self.ctx[structure]["output_parameters"].dict.energy
        
        
        self.ctx.optimized_defective_structures = {}
        if self.inputs.type_run_defects == 'vc-relax' or self.inputs.type_run_defects == 'relax':
            for structure in self.ctx.defective_structures:
                if '_0' not in structure:
                    self.ctx.optimized_defective_structures[structure] = self.ctx[structure]['output_structure']
        else:
            self.ctx.optimized_defective_structures = self.ctx.defective_structures
            
        for structure in self.ctx.defective_structures:
                if '_0' not in structure:
                    self.out('structure_'+str(structure),self.ctx.optimized_defective_structures[structure])
                    self.out('energy_'+str(structure),Float(self.ctx[structure]["output_parameters"].dict.energy))


        return


    def should_run_makovpayne(self):
        """
        Checking if the Makov Payne Electrostatic correction is to be computed
        """

        return bool(self.inputs.corrections.get_dict()['makov_payne'] and self.inputs.defect_charge != 0.0)
    
    
    def run_makovpayne(self):
        """
        Computing Makov Payne Correction
    
        """
        inputs = {'bulk_structure' : self.ctx.structure_host,
                  'code_pw' : self.inputs.code_pw,
                  'settings' : self.inputs.settings,
                  'options' : self.inputs.options,
                  'pseudo_family' : Str(self.inputs.pseudo_family),
                  'kpoints' : self.ctx.host_pwcalc.inp.kpoints,
                  'parameters' : self.inputs.parameters,
                  'epsilon_r' : self.inputs.epsilon_r,
                  'defect_charge' : self.inputs.defect_charge,
                  
                  
                 }
        
        running = submit(MakovPayneCorrection,**inputs)
        self.report('Launching MakovPayneCorrection. pk value {}'.format( running.pid))
        return ToContext(makov_payne = running)
    
    def should_run_pot_align(self):
        """
        Checking if the potential alignement is to be computed.
        In case in qhich you require the bandfilling correction the workcahin will automatically
        perform the potential alignment step unless you specify the potential alignemnt 
        as input of the workchain (in the case in which it was previously computed)
        """
        if bool(self.inputs.corrections.get_dict()['potential_alignment']):
            run_pot_align = True
        elif bool(self.inputs.corrections.get_dict()['bandfilling']) and self.inputs.pot_align not in self.inputs:
            run_pot_align = True
        else:
            run_pot_align = False
        
        
        return run_pot_align
    
    def run_pot_align(self):
        """
        Computing Lany-Zunger Potential ALignment
        Add FNV method
    
        """
    
        
        inputs = {'host_structure' : self.ctx.structure_host,
                  #'host_parent_folder' : self.ctx.host_pwcalc.out.remote_folder,
                  'kpoints' : self.ctx.host_pwcalc.inp.kpoints,
                  'code_pw' : self.inputs.code_pw,
                  'code_pp' : self.inputs.code_pp,
                  'parameters_pp': self.inputs.parameters_pp,
                  'settings' : self.inputs.settings,
                  'options' : self.inputs.options,
                  'pseudo_family' : Str(self.inputs.pseudo_family),
                  'alignment_type' : Str('lany_zunger'),
                  'run_pw_host' : Bool(True),
                  'run_pw_defect' : Bool(True),
                  'parameters_pw_host' : self.inputs.parameters, 

                 }
    
        
        if  bool("run_primitive_host"):
            inputs['host_parent_folder'] = self.ctx.host_pwcalc.out.remote_folder
        else:
            inputs['host_parent_folder'] =  self.inputs.host_folder_data
        

        
        pot_align = {}
        for structure in self.ctx.optimized_defective_structures:
            
            
            if '_0' not in structure:
                inputs['defect_structure'] = self.ctx.optimized_defective_structures[structure]
                #inputs['defect_parent_folder'] = self.ctx[structure]['remote_folder'] #self.ctx[structure].inp.parameters
                inputs['parameters_pw_defect'] = self.ctx.inputs_defective_structures[structure]['parameters']
                
                future = submit(PotentialAlignmentLanyZunger,**inputs)
                self.report('Launching PotentialAlignmentLanyZunger for the defective structure {} with charge {}. pk value {}'.format(structure,self.inputs.defect_charge,future.pid))
                pot_align['pot_align_'+str(structure)] = Outputs(future)
        return ToContext(**pot_align)  
    
    
    
    
    def should_run_bandfilling(self):
        """
        Checking if the bandfilling correction is to be computed
        """

        return bool(self.inputs.corrections.get_dict()['bandfilling'])
    
    
    def run_bandfilling(self):
        """
        Computing Bandfilling Correction
    
        """
        self.ctx.potential_alignment = Float(0.00)
        
        inputs = {'host_structure' : self.ctx.structure_host,
                  'host_parameters' : self.inputs.parameters,
                  'kpoints' : self.ctx.host_pwcalc.inp.kpoints,
                  'code' : Code.get_from_string(str(self.inputs.code_pw)),
                  'settings' : self.inputs.settings,
                  'options' : self.inputs.options,
                  'pseudo_family' : Str(self.inputs.pseudo_family),
                  'skip_relax' : Bool(True),
                  'potential_alignement' : self.ctx.potential_alignment,
                  'relax' : self.inputs.bf_relax, 
                 }
        
        bandfillings = {}
        for structure in self.ctx.optimized_defective_structures:
            
            
            if '_0' not in structure:
                inputs['defect_structure'] = self.ctx.optimized_defective_structures[structure]
                inputs['defect_parameters'] = self.ctx.inputs_defective_structures[structure]['parameters'] #self.ctx[structure].inp.parameters

                future = submit(BandFillingCorrectionWorkChain,**inputs)
                self.report('Launching BandFillingCorrectionWorkChain for the defective structure {} with charge {}. pk value {}'.format(structure,self.inputs.defect_charge,future.pid))
                bandfillings['bandfilling_'+str(structure)] = Outputs(future)
        return ToContext(**bandfillings)
    
    
    def retrieve_corrections(self):

        if bool(self.inputs.corrections.get_dict()['makov_payne'] and self.inputs.defect_charge != 0.0):
            self.out('makov_payne',self.ctx.makov_payne.out.Makov_Payne_Correction)
        
        if bool(self.inputs.corrections.get_dict()['potential_alignment']): #or bool(self.inputs.corrections.get_dict()['bandfilling']):
            for structure in self.ctx.optimized_defective_structures:
                if '_0' not in structure:
                    self.out('pot_align'+str(structure),self.ctx['pot_align_'+str(structure)].out.pot_align*self.inputs.defect_charge)
        
        if bool(self.inputs.corrections.get_dict()['bandfilling']):
            for structure in self.ctx.optimized_defective_structures:
                if '_0' not in structure:
                    self.out('E_donor'+str(structure),self.ctx['bandfilling_'+str(structure)]["E_donor"])
                    self.out('E_acceptor'+str(structure),self.ctx['bandfilling_'+str(structure)]["E_acceptor"])
                
