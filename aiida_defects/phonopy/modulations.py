
# coding: utf-8

import sys
import os
import argparse
import pymatgen
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import cos, sin, radians
from aiida.work.run import run
from aiida.orm.data.upf import UpfData
from aiida.common.exceptions import NotExistent
from aiida.orm.data.upf import get_pseudos_from_structure
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
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
from aiida_defects.tools.structure_manipulation import create_suitable_inputs_noclass
from aiida_defects.phonopy.phonopy_tools import *


class PhonopyWorkChain(WorkChain):
    """
    Workchain to compute phonon band structure, find instabilities and follow structure evolution along them.
    KNOWN BUGS:
    1) the supercell to compute forces and the one for modulations should be the same
    """
    @classmethod
    def define(cls, spec):
        super(PhonopyWorkChain, cls).define(spec)
        spec.input("structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str)
        spec.input("pseudo_family",valid_type=Str)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData)#It should be the one to be used for the bulk not supercell
        spec.input('parameters', valid_type=ParameterData)
        spec.input('phonopy_input', valid_type=ParameterData)
        spec.input('magnetic_phase', valid_type=Str,required=False, default=Str('NM'))
        spec.input('B_atom', valid_type=Str)
        spec.input('hubbard_u', valid_type=ParameterData, required=False, default=ParameterData(dict={}))
        spec.input('optimization', valid_type=Str)#make possible to choose between vc-relax and relax
        spec.input('opt_modulated', valid_type=Str, required=False, default=Str('vc-relax'))
         
        spec.outline(
            cls.optimize, 
            cls.create_supercells_with_displacements,
            cls.run_pw_disp,
            cls.force_constant_calculation,
            cls.phonon_band_structure_calculation,
            cls.create_anime,
            cls.create_modulations,
            if_(cls.run_non_degenerate)(
                while_(cls.should_redo_amplitude_scan_non_degenerate)(
                    cls.amplitude_scan_nondegenerate,
                    cls.retrieve_scan_energies_nondegenerate,
                    cls.fitting_scan_energies_nondegenerate,
                ),               
            cls.optimization_nondegenerate,
            cls.retrieve_results_nondegenerate,
            ),
            if_(cls.run_degenerate)(
                while_(cls.should_redo_amplitude_scan_degenerate)(
                    cls.amplitude_scan_degenerate,
                    cls.retrieve_scan_energies_degenerate,
                    cls.fitting_scan_energies_degenerate,
                ),               
            cls.optimization_degenerate,
            cls.retrieve_results_degenerate,
            ),

            )
        spec.dynamic_output()
    
    def optimize(self):

        if 'relax' in str(self.inputs.optimization):
            code_pw = Code.get_from_string(str(self.inputs.code_pw))
            options = self.inputs.options
            settings = self.inputs.settings
            kpoints = self.inputs.kpoints
            parameter = self.inputs.parameters
        
            inputs_opt={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }

            suitable_inputs=create_suitable_inputs_noclass(self.inputs.structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom
                                                          )
            inputs_opt['structure'] = suitable_inputs['structure']
            
            param = inputs_opt['parameters'].get_dict()
            
            param['CONTROL']['calculation'] = str(self.inputs.optimization)
            #param['CONTROL']['tstress'] = True
            param['CONTROL']['tprnfor'] = True
            
            magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
            if str(self.inputs.magnetic_phase)  in magnetic_phases:
                param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                param['SYSTEM']['nspin'] = 2
                
            U = {}
            hubbard_U= self.inputs.hubbard_u.get_dict()
            if bool(hubbard_U) and len(hubbard_U) == 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                for site in  inputs_opt['structure'].sites:
                    if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                        U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                param['SYSTEM']['hubbard_u'] = U
            elif bool(hubbard_U) and len(hubbard_U) > 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                U = hubbard_U
                param['SYSTEM']['hubbard_u'] = U

            
            inputs_opt['parameters']= ParameterData(dict=param)
            

            running = submit(PwBaseWorkChain,**inputs_opt)
            self.report('Launching PwBaseWorkChain for the bulk optimization. pk value {}'.format( running.pid))

            return ToContext(optimization = running)

        
        else:
            pass
        

    def create_supercells_with_displacements(self):
        """
        Creating supercells with displacements.
        """

        if "relax" in str(self.inputs.optimization):
            
            inline_params = {"structure": self.ctx.optimization.get_outputs_dict()['output_structure'],
                            "phonopy_input": self.inputs.phonopy_input,
                             }
        else:
            suitable_inputs=create_suitable_inputs_noclass(self.inputs.structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom)
            self.ctx.structure = suitable_inputs['structure']
            
            inline_params = {"structure": self.ctx.structure,
                            "phonopy_input": self.inputs.phonopy_input,
                             } 
        self.ctx.disp_cells = create_supercells_with_displacements_inline(**inline_params)
        
    def run_pw_disp(self):
        """
        PwBaseWorkChain in the scf calculation mode for each structure with displacments
        Calculations for each displacement  are performed in parallel.
        """
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        parameter = self.inputs.parameters

            
        kpoints= self.inputs.kpoints.get_kpoints_mesh()[0]
        kpoints_x=int(kpoints[0])/int(self.inputs.phonopy_input.get_dict()['supercell'][0][0])
        kpoints_y=int(kpoints[1])/int(self.inputs.phonopy_input.get_dict()['supercell'][1][1])
        kpoints_z=int(kpoints[2])/int(self.inputs.phonopy_input.get_dict()['supercell'][2][2])

        Kpoints =  KpointsData()
        Kpoints.set_kpoints_mesh([kpoints_x, kpoints_y, kpoints_z])
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : Kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,
        }
        
        calcs = {}
        for  structure in self.ctx.disp_cells:
            suitable_inputs=create_suitable_inputs_noclass(self.ctx.disp_cells[structure],
                                                   self.inputs.magnetic_phase,
                                                   self.inputs.B_atom)
            inputs['structure'] = suitable_inputs['structure']
            
            param = inputs['parameters'].get_dict()
            param['CONTROL']['calculation'] = 'scf'
            param['CONTROL']['tprnfor'] = True
            
            magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
            if str(self.inputs.magnetic_phase)  in magnetic_phases:
                param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                param['SYSTEM']['nspin'] = 2
                
            U = {}
            hubbard_U= self.inputs.hubbard_u.get_dict()
            if bool(hubbard_U) and len(hubbard_U) == 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                for site in  inputs['structure'].sites:
                    if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                        U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                param['SYSTEM']['hubbard_u'] = U
            elif bool(hubbard_U) and len(hubbard_U) > 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                U = hubbard_U
                param['SYSTEM']['hubbard_u'] = U

            
            inputs['parameters']= ParameterData(dict=param)

            future = submit(PwBaseWorkChain,**inputs)
            self.report('Launching PwBaseWorkChain for the {} with structure with displacements. pk value {}'.format(structure, future.pid))
            calcs[structure] = Outputs(future) 

        return ToContext(**calcs)
    
    def force_constant_calculation(self):
        """
        Calculation of the force constants
        """
        if "relax" in str(self.inputs.optimization):
            self.ctx.phpy_input_structure =  self.ctx.optimization.get_outputs_dict()['output_structure']
        else:
            self.ctx.phpy_input_structure = self.ctx.structure
            

        self.ctx.inline_params = {"structure": self.ctx.phpy_input_structure,
                                  "phonopy_input": self.inputs.phonopy_input,
                             }
        for  structure in self.ctx.disp_cells:
            self.ctx.inline_params['force_'+structure.split('_')[1]] = self.ctx[structure]["output_array"]
        
        self.ctx.phonopy_data = get_force_constants_inline(**self.ctx.inline_params)
        self.ctx.inline_params['force_constants'] = self.ctx.phonopy_data['phonopy_output']
        self.out('force_constants',self.ctx.phonopy_data['phonopy_output'])
        self.report('Force constant calculation succesfull')
        
        return

    def phonon_band_structure_calculation(self):
        """
        Calculation of phonon band structure
        """
        band_structure  = phonon_band_structure(**self.ctx.inline_params)
        self.out('phonon_band_structure', band_structure)
        self.report('Phonon band structure calculation succesfull')
        
        return
    
    def create_anime(self):
        """
        Creating anime file at the q point specified in the phonopy input
        """
        
        anime = create_anime(**self.ctx.inline_params)
        self.out('anime', anime)
    
    def create_modulations(self):
        """
        Creating modulations for the imaginary phonon modes
        """
        self.ctx.modulated_structures = modulations_inspection(**self.ctx.inline_params)
        
        if 'angle_scan' in self.ctx.modulated_structures:
            self.ctx.ops = self.ctx.modulated_structures['angle_scan'].get_dict().values()
            self.out('ops_symmetries', self.ctx.modulated_structures['angle_scan'])
            del self.ctx.modulated_structures['angle_scan']
        
        if not bool(self.ctx.modulated_structures):
            self.report('No instabilities found. Workchain stopped')
            sys.exit('No instabilities found. Workchain stopped')

        elif Bool(False) in self.ctx.modulated_structures.values():
            self.report('Mode with degeneracy higher than two. Not able to treat it')
            self.ctx.modulated_structures = {key:val for key, val in 
                                             self.ctx.modulated_structures.items() if val != Bool(False)}
        
        self.ctx.non_degenerate = {}
        for modulation in self.ctx.modulated_structures:
            if 'nondeg' in modulation:
                self.ctx.non_degenerate[modulation] = self.ctx.modulated_structures[modulation]
        
        self.ctx.degenerate = {}
        for modulation in self.ctx.modulated_structures:
            if 'DEG' in modulation:
                self.ctx.degenerate[modulation] = self.ctx.modulated_structures[modulation]
                
        #Identifying the number of the non degenerate modes
        self.ctx.nondeg_modes=[]
        for key in self.ctx.non_degenerate:
            if 'nondeg' in key and  int(key.split('_')[1]) not in self.ctx.nondeg_modes:
                self.ctx.nondeg_modes.append(int(key.split('_')[1]))
        
        #Identifying the number of the  degenerate modes
        self.ctx.deg_modes = []
        #self.ctx.ops = []
        for key in self.ctx.degenerate:
            if 'DEG' in key and  key.split('_')[1] not in self.ctx.deg_modes:
                self.ctx.deg_modes.append(key.split('_')[1])
                #self.ctx.ops.append(key.split('_')[3])
        self.ctx.fitting_nondeg_ok = [False]
        self.ctx.fitting_deg_ok = [False]
        self.ctx.repeat_nondeg = {}
        self.ctx.repeat_deg = {}
        
##############################################
# NON DEGENERATE MODES ANALYSIS              #
##############################################    
    def run_non_degenerate(self):
        return bool(self.ctx.non_degenerate)
        
    def should_redo_amplitude_scan_non_degenerate(self):
        return False in self.ctx.fitting_nondeg_ok
        
    def amplitude_scan_nondegenerate(self):
        """
        Scanning the amplitude range for non degenerate immaginary phonon modes
        """

        
        
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        parameter = self.inputs.parameters

            
        kpoints= self.inputs.kpoints.get_kpoints_mesh()[0]
        kpoints_x=int(kpoints[0])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][0][0])
        kpoints_y=int(kpoints[1])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][1][1])
        kpoints_z=int(kpoints[2])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][2][2])

        Kpoints =  KpointsData()
        Kpoints.set_kpoints_mesh([kpoints_x, kpoints_y, kpoints_z])
        
        inputs={
                    'code' : code_pw,
                    'pseudo_family' : Str(self.inputs.pseudo_family),
                    'kpoints' : Kpoints,
                    'parameters' : parameter,
                    'settings' : settings,
                    'options' : options,

            }
        
        #self.ctx.opt_nondegenerate[str(mode)]
        
        calcs = {}        
        for label, structure in self.ctx.non_degenerate.iteritems():
            if not bool(self.ctx.repeat_nondeg) or self.ctx.repeat_nondeg[str(label.split('_')[1])+str(label.split('_')[3])] == False:  
                suitable_inputs=create_suitable_inputs_noclass(structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom)
                inputs['structure'] = suitable_inputs['structure']
            
                param = inputs['parameters'].get_dict()
                param['CONTROL']['calculation'] = 'scf'
                param['CONTROL']['tprnfor'] = True
            
                magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
                if str(self.inputs.magnetic_phase)  in magnetic_phases:
                    param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                    param['SYSTEM']['nspin'] = 2
                
                U = {}
                hubbard_U= self.inputs.hubbard_u.get_dict()
                if bool(hubbard_U) and len(hubbard_U) == 1:
                    param['SYSTEM']['lda_plus_u'] =True
                    param['SYSTEM']['lda_plus_u_kind'] = 0
                    for site in  inputs['structure'].sites:
                        if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                            U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                    param['SYSTEM']['hubbard_u'] = U
                elif bool(hubbard_U) and len(hubbard_U) > 1:
                    param['SYSTEM']['lda_plus_u'] =True
                    param['SYSTEM']['lda_plus_u_kind'] = 0
                    U = hubbard_U
                    param['SYSTEM']['hubbard_u'] = U

            
                inputs['parameters']= ParameterData(dict=param)

                future = submit(PwBaseWorkChain,**inputs)
                self.report('Launching PwBaseWorkChain for the {} modulation. pk value {}'.format(label, future.pid))
                calcs[label] = Outputs(future)
            

        return ToContext(**calcs)

        

        
    
    def retrieve_scan_energies_nondegenerate(self):
        """
        Extract the total energy for every modulated structure for non degenerate modes
        """
        

        #Extracting the energy for modulated structures with different aplitudes and different non degenerate modes
        self.ctx.E_nondeg={} 

        for mode in self.ctx.nondeg_modes:
            self.ctx.E_nondeg['nondeg_mode'+str(mode)] = {}
            for label, value in self.ctx.non_degenerate.iteritems():
                if "nondeg_"+str(mode) in label:
                    self.ctx.E_nondeg['nondeg_mode'+str(mode)][str(label)] = Float(self.ctx[str(label)]["output_parameters"].dict.energy)
                        #opt_structure = self.ctx[str(value)]["output_structure"]
                        #self.out('structure_'+str(value),opt_structure)
        
        #self.out('energy_nondeg',ParameterData(dict=self.ctx.E_nondeg))

    def fitting_scan_energies_nondegenerate(self):
        """
        Fitting DeltaE vs. Amplitude
        """
        
        #Emptying the list used to decide whether we need to repeat the amplitude scan
        self.ctx.fitting_nondeg_ok = []
        
        #Setting q-point 
        q = self.inputs.phonopy_input.get_dict()['modulation']['q_point']
        
        #For each non degenerate mode
        self.ctx.opt_nondegenerate = {}
        amplitude = []
        energies = []
        #if bool(self.ctx.non_degenerate):
        for mode in self.ctx.nondeg_modes:
            for label, energy in self.ctx.E_nondeg['nondeg_mode'+str(mode)].iteritems():
                amplitude.append(label.split('_')[3])
                energies.append(energy)
                
            ref_E= float(self.ctx.E_nondeg['nondeg_mode'+str(mode)]['nondeg_'+str(mode)+'_ampl_0'])
                
            #min_E = min(energies)
            energies=[float(i) for i in energies]
            DeltaE = np.array(energies)
            DeltaE = DeltaE - ref_E
            #DeltaE[:] = [E - min_E for E in energies]
            amplitude=[float(i) for i in amplitude]
            amplitudes = np.array(amplitude)
                
            #Try fitting quartic function
            name_file = str(self.inputs.structure.pk)+'_q_'+str(q)+'_mode_'+str(mode)
            popt, pcov = poly_fit(amplitudes,DeltaE, quartic_poly, name_file)
                
            r2 = r_sqrt_calc(amplitudes,DeltaE,quartic_poly,*popt)
                
            if r2 >= 0.9:
                def fit(s,*opt):
                    return quartic_poly(s,*popt)
                ampl=find_min(fit,amplitudes,DeltaE)
                ampl_min = float(ampl)

                ph_md = [[q,
                        int(mode),
                        float(ampl_min),
                        self.inputs.phonopy_input.get_dict()['modulation']['phase']]]
                
                phonon_modes = List()
                phonon_modes._set_list(ph_md)
                
                self.ctx.inline_params['phonon_modes'] = phonon_modes
                self.ctx.inline_params['name'] = Str(name_file)
                
                self.ctx.opt_nondegenerate[str(mode)] = modulations_optimal(**self.ctx.inline_params)
                
                self.ctx.fitting_nondeg_ok.append(True)
            #If quarting fitting is not ok we rescale the amplitude range, selecting in which direction
            #to rescale by checking if thecoefficient of the second order part of a quadratic fitting 
            #is
            if r2 < 0.9:
                self.ctx.non_degenerate = {}
#                 tmp = {}
#                 for key, value in self.ctx.non_degenerate.iteritems():
#                     if "nondeg_"+str(mode) not in key:
#                         tmp[key] = value
#                 self.ctx.non_degenerate = deepcopy(tmp)
                        
                del self.ctx.E_nondeg['nondeg_mode'+str(mode)]
                    
                self.ctx.fitting_nondeg_ok.append(False)
                    
                    
                name_file = str(self.inputs.structure.pk)+'_q_'+str(q)+'_mode_'+str(mode)
                popt, pcov = poly_fit(amplitudes,DeltaE, quadratic_poly, name_file)
                if popt[a]<0:
                        ampl_min = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][0]*10
                        ampl_max = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][1]*10
                        ampl_incr = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][2]*10
                        amplitudes = np.arange(ampl_min,ampl_max,ampl_incr).tolist()
                            
                        ph_md = []
                        for ampl in amplitudes:
                                
                            ph_md = [[q,
                                    int(mode),
                                    float(ampl),
                                    self.inputs.phonopy_input.get_dict()['modulation']['phase']]]
                
                            phonon_modes = List()
                            phonon_modes._set_list(ph_md)
                
                            self.ctx.inline_params['phonon_modes'] = phonon_modes
                            self.ctx.inline_params['name'] = Str(mode)
                
                            new_modulated_s = modulations_optimal(**self.ctx.inline_params)
                    
                            self.ctx.non_degenerate['nondeg_'+str(mode)+'_ampl_'+str(ampl)] = new_modulated_s[self.ctx.inline_params['name']]
                if popt[a]>0:
                        ampl_min = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][0]/10
                        ampl_max = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][1]/10
                        ampl_incr = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][2]/10
                        amplitudes = np.arange(ampl_min,ampl_max,ampl_incr).tolist()
                            
                        ph_md = []
                        for ampl in amplitudes:
                                
                            ph_md = [[q,
                                    int(mode),
                                    float(ampl),
                                    self.inputs.phonopy_input.get_dict()['modulation']['phase']]]
                
                            phonon_modes = List()
                            phonon_modes._set_list(ph_md)
                
                            self.ctx.inline_params['phonon_modes'] = phonon_modes
                            self.ctx.inline_params['name'] = Str(mode)
                
                            new_modulated_s = modulations_optimal(**self.ctx.inline_params)
                    
                            self.ctx.non_degenerate['nondeg_'+str(mode)+'_ampl_'+str(ampl)] = new_modulated_s[self.ctx.inline_params['name']]

                
            #phonon_modes._set_list([])
            amplitudes = np.array([])
            amplitude = []
            energies = []
            DeltaE = np.array([])
                
        self.ctx.repeat_nondeg = dict(zip(self.ctx.nondeg_modes,self.ctx.fitting_nondeg_ok))

    
    def optimization_nondegenerate(self):
        """
        Relaxing the modulated structure corresponding to the optimal amplitude value
        """
        #if bool(self.ctx.non_degenerate):
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        parameter = self.inputs.parameters

            
        kpoints= self.inputs.kpoints.get_kpoints_mesh()[0]
        kpoints_x=int(kpoints[0])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][0][0])
        kpoints_y=int(kpoints[1])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][1][1])
        kpoints_z=int(kpoints[2])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][2][2])

        Kpoints =  KpointsData()
        Kpoints.set_kpoints_mesh([kpoints_x, kpoints_y, kpoints_z])
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : Kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }
        
            
        self.ctx.opt_nondegenerate_flat = {}
        for mode in self.ctx.opt_nondegenerate:
            for label, structure in self.ctx.opt_nondegenerate[mode].iteritems():
                self.ctx.opt_nondegenerate_flat[label] = structure
                    
                
        calcs = {}    
        for label, structure in self.ctx.opt_nondegenerate_flat.iteritems():
            suitable_inputs=create_suitable_inputs_noclass(structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom)
            inputs['structure'] = suitable_inputs['structure']
            
            param = inputs['parameters'].get_dict()
            param['CONTROL']['calculation'] = str(self.inputs.opt_modulated)
            param['CONTROL']['tprnfor'] = True
            
            magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
            if str(self.inputs.magnetic_phase)  in magnetic_phases:
                param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                param['SYSTEM']['nspin'] = 2
                
            U = {}
            hubbard_U= self.inputs.hubbard_u.get_dict()
            if bool(hubbard_U) and len(hubbard_U) == 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                for site in  inputs['structure'].sites:
                    if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                        U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                param['SYSTEM']['hubbard_u'] = U
            elif bool(hubbard_U) and len(hubbard_U) > 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                U = hubbard_U
                param['SYSTEM']['hubbard_u'] = U

            
            inputs['parameters']= ParameterData(dict=param)

            future = submit(PwBaseWorkChain,**inputs)
            self.report('Launching PwBaseWorkChain for the modulation along {} mode. pk value {}'.format(label, future.pid))
            calcs[label] = Outputs(future) 

        return ToContext(**calcs)
        
    
    
    def retrieve_results_nondegenerate(self):
        """
        Extract the total energy for every modulated structure
        """

        #Extracting the energy for modulated structures with different apmlitudes and different non degenerate modes
        self.ctx.res_nondeg={} 
        #if bool(self.ctx.non_degenerate):
        self.out('energy_nondeg',ParameterData(dict=self.ctx.E_nondeg))
        for label in self.ctx.opt_nondegenerate_flat:
            print label, self.ctx[str(label)]
            #self.out('opt_'+str(label),self.ctx[str(label)])
#             self.ctx.res_nondeg[str(label)] = {}
#             self.ctx.res_nondeg[str(label)]['energy'] = Float(self.ctx[str(label)]["output_parameters"].dict.energy)
#             self.ctx.res_nondeg[str(label)]['structure'] = self.ctx[str(label)]["output_structure"]
#                 #self.out('modulation_',self.ctx.res_nondeg[str(label)])
#             for link_name, node in self.ctx.res_nondeg[str(label)].iteritems():
#                 self.out(link_name, node)    
    
    
##############################################
# DEGENERATE MODES ANALYSIS                  #
##############################################
    def run_degenerate(self):
        return bool(self.ctx.degenerate)
        
    def should_redo_amplitude_scan_degenerate(self):
        return False in self.ctx.fitting_deg_ok
    
    def amplitude_scan_degenerate(self):
        """
        Scanning the amplitude range for  degenerate immaginary phonon modes
        """
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        parameter = self.inputs.parameters

            
        kpoints= self.inputs.kpoints.get_kpoints_mesh()[0]
        kpoints_x=int(kpoints[0])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][0][0])
        kpoints_y=int(kpoints[1])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][1][1])
        kpoints_z=int(kpoints[2])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][2][2])

        Kpoints =  KpointsData()
        Kpoints.set_kpoints_mesh([kpoints_x, kpoints_y, kpoints_z])
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : Kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }
        calcs = {}
        for label, structure in self.ctx.degenerate.iteritems():
            if not bool(self.ctx.repeat_deg) or self.ctx.repeat_deg[str(label.split('_')[1])+str(label.split('_')[3])] == False:
                suitable_inputs=create_suitable_inputs_noclass(structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom)
                inputs['structure'] = suitable_inputs['structure']
            
                param = inputs['parameters'].get_dict()
                param['CONTROL']['calculation'] = 'scf'
                param['CONTROL']['tprnfor'] = True
            
                magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
                if str(self.inputs.magnetic_phase)  in magnetic_phases:
                    param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                    param['SYSTEM']['nspin'] = 2
                U = {}
                hubbard_U= self.inputs.hubbard_u.get_dict()
                if bool(hubbard_U) and len(hubbard_U) == 1:
                    param['SYSTEM']['lda_plus_u'] =True
                    param['SYSTEM']['lda_plus_u_kind'] = 0
                    for site in  inputs['structure'].sites:
                        if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                            U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                    param['SYSTEM']['hubbard_u'] = U
                elif bool(hubbard_U) and len(hubbard_U) > 1:
                    param['SYSTEM']['lda_plus_u'] =True
                    param['SYSTEM']['lda_plus_u_kind'] = 0
                    U = hubbard_U
                    param['SYSTEM']['hubbard_u'] = U
            
                inputs['parameters']= ParameterData(dict=param)

                future = submit(PwBaseWorkChain,**inputs)
                self.report('Launching PwBaseWorkChain for the {} modulation. pk value {}'.format(label, future.pid))
                calcs[label] = Outputs(future) 
                    
        
        return ToContext(**calcs)

        
    def retrieve_scan_energies_degenerate(self):
        """
        Extract the total energy for every modulated structure for  degenerate modes
        """
        
        
        #Extracting the energy for modulated structures with different amplitudes and different doubly degenerate modes
        self.ctx.E_deg={} 
        for mode in self.ctx.deg_modes:
            for opd in self.ctx.ops:
                self.ctx.E_deg['DEG_mode'+str(mode)+'_ang_'+str(opd)] = {}
                for label, value in self.ctx.degenerate.iteritems():
                    if "DEG_"+str(mode)+'_ang_'+str(opd) in label:
                        self.ctx.E_deg['DEG_mode'+str(mode)+'_ang_'+str(opd)][str(label)] = self.ctx[str(label)]["output_parameters"].dict.energy
        
        
        return
    
    def fitting_scan_energies_degenerate(self):
        """
        Fitting DeltaE vs. Amplitude
        """
        #Empty the list which will be used to decide wheter we should reapte the amplitude scan
        self.ctx.fitting_deg_ok = []
        
        q = self.inputs.phonopy_input.get_dict()['modulation']['q_point']
        
        #For each of the highest symmetry directions in order parameter space for each doubly degenerate mode
        self.ctx.opt_degenerate = {}
        amplitude = []
        energies = []
        #if bool(self.ctx.degenerate):
        for mode in self.ctx.deg_modes:
            for opd in self.ctx.ops:
                for label, energy in self.ctx.E_deg['DEG_mode'+str(mode)+'_ang_'+str(opd)].iteritems():
                    amplitude.append(label.split('_')[5])
                    energies.append(energy)
                    
                
                ref_E= float(self.ctx.E_deg['DEG_mode'+str(mode)+'_ang_'+str(opd)]['DEG_'+str(mode)+'_ang_'+str(opd)+'_ampl_0'])
                energies=[float(i) for i in energies]
                DeltaE = np.array(energies)
                DeltaE = DeltaE - ref_E
                #DeltaE[:] = [E - min_E for E in energies]
                amplitude=[float(i) for i in amplitude]
                amplitudes = np.array(amplitude)

                #Try fitting quartic function
                name_file = str(self.inputs.structure.pk)+'_q_'+str(q)+'_mode_'+str(mode)+'_ang_'+str(opd)
                popt, pcov = poly_fit(amplitudes,DeltaE, quartic_poly, name_file)

                r2 = r_sqrt_calc(amplitudes,DeltaE,quartic_poly,*popt)


                if r2 >= 0.9:
                    def fit(s,*opt):
                        return quartic_poly(s,*popt)
                    ampl=find_min(fit,amplitudes,DeltaE)
                    ampl_min = float(ampl)


                    ph_md = [[q,
                             int(str(mode)[1]),
                             ampl_min*cos(radians(int(opd))),
                             self.inputs.phonopy_input.get_dict()['modulation']['phase']],
                            [q,
                            int(str(mode)[4]),
                            ampl_min*sin(radians(int(opd))),
                            self.inputs.phonopy_input.get_dict()['modulation']['phase']]
                                ]

                    phonon_modes = List()
                    phonon_modes._set_list(ph_md)

                    self.ctx.inline_params['phonon_modes'] = phonon_modes
                    self.ctx.inline_params['name'] = Str(name_file)

                    self.ctx.opt_degenerate[str(mode)+'_'+str(opd)] = modulations_optimal(**self.ctx.inline_params)

                    self.ctx.fitting_deg_ok.append(True)
                #If quarting fitting is not ok we rescale the amplitude range, selecting in which direction
                #to rescale by checking if thecoefficient of the second order part of a quadratic fitting 
                #is'DEG_'+str(deg_band_index)+'_ang_'+str(opd)+'_ampl_'+str(ampl)
                if r2 < 0.9:
                    self.ctx.degenerate = {}
#                     tmp = {}
#                     for key, value in self.ctx.degenerate.iteritems():
#                         if "DEG_"+str(mode)+'_ang_'+str(opd) not in key:
#                             tmp[key] = value
#                     self.ctx.degenerate = deepcopy(tmp)

                    del self.ctx.E_deg['DEG_mode'+str(mode)+'_ang_'+str(opd)]
                    w={}

                    self.ctx.fitting_deg_ok.append(False)


                    name_file = name = str(self.inputs.structure.pk)+'_q_'+str(q)+'_mode_'+str(mode)+'_angle_'+str(opd)
                    popt, pcov = poly_fit(amplitudes,DeltaE, quadratic_poly, name_file)
                    if popt[a]<0:
                            ampl_min = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][0]*10
                            ampl_max = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][1]*10
                            ampl_incr = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][2]*10
                            amplitudes = np.arange(ampl_min,ampl_max,ampl_incr).tolist()

                            ph_md = []
                            for ampl in amplitudes:

                                ph_md = [[q,
                                    int(str(mode)[1]),
                                    ampl_min*cos(radians(int(opd))),
                                    self.inputs.phonopy_input.get_dict()['modulation']['phase']],
                                   [q,
                                   int(str(mode)[4]),
                                   ampl_min*sin(radians(int(opd))),
                                   self.inputs.phonopy_input.get_dict()['modulation']['phase']]
                                   ]

                                phonon_modes = List()
                                phonon_modes._set_list(ph_md)
                                self.ctx.inline_params['phonon_modes'] = phonon_modes
                                self.ctx.inline_params['name'] = Str(mode)

                                new_modulated_s = modulations_optimal(**self.ctx.inline_params)

                                self.ctx.degenerate['DEG_'+str(mode)+'_ang_'+str(opd)+'_ampl_'+str(ampl)] = new_modulated_s[self.ctx.inline_params['name']]
                    if popt[a]>0:
                            ampl_min = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][0]/10
                            ampl_max = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][1]/10
                            ampl_incr = self.inputs.phonopy_input.get_dict()['modulation']['amplitude'][2]/10
                            amplitudes = np.arange(ampl_min,ampl_max,ampl_incr).tolist()

                            ph_md = []
                            for ampl in amplitudes:

                                ph_md = [[q,
                                    int(str(mode)[1]),
                                    ampl_min*cos(radians(int(opd))),
                                    self.inputs.phonopy_input.get_dict()['modulation']['phase']],
                                   [q,
                                   int(str(mode)[3]),
                                   ampl_min*sin(radians(int(opd))),
                                   self.inputs.phonopy_input.get_dict()['modulation']['phase']]
                                   ]

                                phonon_modes = List()
                                phonon_modes._set_list(ph_md)

                                self.ctx.inline_params['phonon_modes'] = phonon_modes
                                self.ctx.inline_params['name'] = Str(mode)

                                new_modulated_s = modulations_optimal(**self.ctx.inline_params)
                                self.ctx.degenerate['DEG_'+str(mode)+'_ang_'+str(opd)+'_ampl_'+str(ampl)] = new_modulated_s[self.ctx.inline_params['name']]


                #phonon_modes._set_list([])
                amplitudes = np.array([])
                amplitude = []
                energies = []
                DeltaE = np.array([])
        
        self.ctx.deg_modes_ops = []
        for mode in self.ctx.deg_modes:
            for opd in self.ctx.ops:
                self.ctx.deg_modes_ops.append(str(mode)+"_"+str(opd))
        self.ctx.repeat_deg = dict(zip(self.ctx.deg_modes_ops,self.ctx.fitting_deg_ok))
                

        
    
        #return
        
    def optimization_degenerate(self):
        """
        Relaxing the modulated structures corresponding to the optimal amplitude value for each degenerate mode
        """
        #if bool(self.ctx.degenerate):
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        parameter = self.inputs.parameters

            
        kpoints= self.inputs.kpoints.get_kpoints_mesh()[0]
        kpoints_x=int(kpoints[0])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][0][0])
        kpoints_y=int(kpoints[1])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][1][1])
        kpoints_z=int(kpoints[2])/int(self.inputs.phonopy_input.get_dict()['modulation']['supercell'][2][2])
        Kpoints =  KpointsData()
        Kpoints.set_kpoints_mesh([kpoints_x, kpoints_y, kpoints_z])
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : Kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }
            
        self.ctx.opt_degenerate_flat = {}
        for mode in self.ctx.opt_degenerate:
            for label, structure in self.ctx.opt_degenerate[mode].iteritems():
                self.ctx.opt_degenerate_flat[label] = structure
        
        calcs = {}

        for label, structure in self.ctx.opt_degenerate_flat.iteritems():
            suitable_inputs=create_suitable_inputs_noclass(structure,
                                                       self.inputs.magnetic_phase,
                                                       self.inputs.B_atom)
            inputs['structure'] = suitable_inputs['structure']
            
            param = inputs['parameters'].get_dict()
            param['CONTROL']['calculation'] = str(self.inputs.opt_modulated)
            param['CONTROL']['tprnfor'] = True
            
            magnetic_phases = ["FM", "A-AFM", "C-AFM", "G-AFM"]
            if str(self.inputs.magnetic_phase)  in magnetic_phases:
                param['SYSTEM']['starting_magnetization'] = suitable_inputs['starting_magnetization'].get_dict()
                param['SYSTEM']['nspin'] = 2
                
            U = {}
            hubbard_U= self.inputs.hubbard_u.get_dict()
            if bool(hubbard_U) and len(hubbard_U) == 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                for site in  inputs['structure'].sites:
                    if site.kind_name[:2] == str(self.inputs.B_atom) or site.kind_name[:1] == 'Q' or  site.kind_name[:1] == 'J':
                        U[str(site.kind_name)] =hubbard_U[list(hubbard_U)[0]]
                param['SYSTEM']['hubbard_u'] = U
            elif bool(hubbard_U) and len(hubbard_U) > 1:
                param['SYSTEM']['lda_plus_u'] =True
                param['SYSTEM']['lda_plus_u_kind'] = 0
                U = hubbard_U
                param['SYSTEM']['hubbard_u'] = U

            
            inputs['parameters']= ParameterData(dict=param)

            future = submit(PwBaseWorkChain,**inputs)
            self.report('Launching PwBaseWorkChain for the {} modulation. pk value {}'.format(label, future.pid))
            calcs[label] = Outputs(future) 

        return ToContext(**calcs)
        
    
    def retrieve_results_degenerate(self):
        """
        Extract the total energy for every modulated structure
        """

        #Extracting the energy for modulated structures with different amplitudes and different  degenerate modes
        #TODO: I needed to change the labels in self.ctx.E_deg and self.ctx.opt_degenerate_flat because in order to
        #store these dictionaries in the AiiDA DB, their keys cannot contain the character '.'.
        #Maybe we can change that so that the two dictionaries don't have the '.' since the beginning.
        #The points come either for the angle value (maybe change self.ctx.ops) and  the q-point 
        #(maybe change q-point mesh with q-point name). For the angle I completetly removed the . and the digit
        #after it, for the q point I substitued it qith ;
        E_deg={}
        for label in self.ctx.E_deg:
            E_deg[label.split('.')[0]]= {}
            for key in self.ctx.E_deg[label]:
                E_deg[label.split('.')[0]][key.split('.')[0]+key.split('.')[1][1:]]=self.ctx.E_deg[label][key]
        
        self.out('energy_deg',ParameterData(dict=E_deg))
        self.ctx.res_deg={} 
        for label in self.ctx.opt_degenerate_flat:
            self.ctx.res_deg[label[:-2].replace('.',';')] = {}
            self.ctx.res_deg[label[:-2].replace('.',';')]['energy'] = Float(self.ctx[str(label)]["output_parameters"].dict.energy)
            self.ctx.res_deg[label[:-2].replace('.',';')]['structure'] = self.ctx[str(label)]["output_structure"]
            self.ctx.res_deg[label[:-2].replace('.',';')]['symmetry'] = Int(get_spacegroup(self.ctx.res_deg[label[:-2].replace('.',';')]['structure']))
            #self.out('modulation_'+str(label),self.ctx.res_nondeg[str(label)])
 

            for link_name, node in self.ctx.res_deg[label[:-2].replace('.',';')].iteritems():
                self.out(link_name, node) 
        
        #return










