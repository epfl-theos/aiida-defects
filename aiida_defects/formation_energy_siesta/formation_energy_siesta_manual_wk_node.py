# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.orm import Dict,Int
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida.plugins import WorkflowFactory
from aiida_siesta.workflows.base import SiestaBaseWorkChain
#from aiida.common import AttributeDict
#from aiida_siesta.calculations.tkdict import FDFDict

from aiida_defects.formation_energy_siesta.formation_energy_base import FormationEnergyWorkchainBase
from .utils import get_raw_formation_energy 
from .utils import get_corrected_formation_energy
from .utils import get_corrected_aligned_formation_energy

import numpy as np

class FormationEnergyWorkchainSIESTAManual(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA code
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTAManual, cls).define(spec)

        # DFT calculations with SIESTA are handled with different codes, so here
        # we keep track of things with two separate namespaces. An additional code, and an additional
        # namespace, is used for postprocessing
        #spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
       
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('structure'), namespace="host")
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('structure'), namespace="defect_q0")
        spec.expose_inputs(SiestaBaseWorkChain, exclude=('structure'), namespace="defect_q")

        spec.input('host_remote',
                    valid_type = orm.RemoteData, 
                    required  =False) 
        spec.input('defect_q0_remote',
                   valid_type = orm.RemoteData, 
                   required  = False) 
        spec.input('defect_q_remote',
                   valid_type = orm.RemoteData, 
                   required  = False) 

        spec.input('wk_node',
                    valid_type=orm.Int,
                    help="Restart from interrupted defect workchain",
                    required=False )
 

        spec.input('host_node', valid_type = orm.Int, required = False)
        spec.input('defect_q0_node', valid_type = orm.Int, required = False)
        spec.input('defect_q_node', valid_type = orm.Int, required = False)
        
        #===================================================================
        # The Outputs of Workflow
        #===================================================================
        spec.output('host_structure',valid_type=orm.StructureData)
        spec.output('defect_q0_structure',valid_type=orm.StructureData)
        spec.output('defect_q_structure',valid_type=orm.StructureData)
        spec.output('host_structure_wk_pk',valid_type=orm.Int)
        spec.output('defect_q0_structure_wk_pk',valid_type=orm.Int)
        spec.output('defect_q_structure_wk_pk',valid_type=orm.Int)

        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(cls.setup,
                     if_(cls.is_restart)( 
                         cls.is_relaxation_restart,
                         cls.relaxation_restart,
                         ).else_(
                                 cls.relax_structures),
                     cls.check_relaxation,
                     cls.retrieving_dft_data,
                     #cls.check_dft_potentials,
                     if_(cls.correction_required)(
                                                  if_(cls.is_gaussian_model_scheme)(
                        #cls.check_dft_potentials_gaussian_correction,
                        cls.run_gaussian_model_correction_workchain,
                        cls.check_gaussian_model_correction_workchain,
                        cls.compute_formation_energy_gaussian_model
                        )).else_(
                                 cls.compute_no_corrected_formation_energy,
                            ))
                
    
    
        spec.exit_code(200,'ERROR_MAIN_WC',
            message='The end-point relaxation SiestaBaseWorkChain failed')

    #======================================================
    # Restarting Parts
    #======================================================

    def is_restart(self):

        """
        checking if neb workchain no provided or not 
        """
        if 'wk_node' not in self.inputs :
            self.report("Starting Workchain from Scratch")
            return False
        else:
            self.report("Restarting Workchain")
            return True

    def is_relaxation_restart(self):
        """
        Checking Relaxation Restart points
        Not Sure we need it cz SiestaBaseWorkChain workchain will take care of it...
        """
        self.is_relax_host = False
        self.is_relax_defect_q0 = False
        self.is_relax_defect_q = False

        wk_node = orm.load_node(self.inputs.wk_node.value)
        self.report(f"Restarting check from node {wk_node.pk}")
        
        # For The host
        if 'host_structure' not in wk_node.outputs:
            self.report("Need to Restart Host Structure Relaxation...")
            self.is_relax_host = True
        else:
            self.report("Host Structure is Relaxed")
            self.out("host_structure_wk_pk", wk_node.outputs.host_structure_wk_pk)
            self.ctx.host_structure = wk_node.outputs.host_structure
            self.out('host_structure',self.ctx.host_structure)
            self.ctx.host_relaxation_wk = orm.load_node(wk_node.outputs.host_structure_wk_pk.value)
        
        # For the defect q0
        if 'defect_q0_structure' not in wk_node.outputs:
            self.report("Need to Restart Defected q0 Structure Relaxation...")
            self.is_relax_defect_q0 = True
        else:
            self.report("Defected q0 Structure is Relaxed")
            self.out("defect_q0_structure_wk_pk",wk_node.outputs.defect_q0_structure_wk_pk)
            self.ctx.defect_q0_structure = wk_node.outputs.defect_q0_structure
            self.out("defect_q0_structure",self.ctx.defect_q0_structure)
            self.ctx.defect_q0_relaxation_wk = orm.load_node(wk_node.outputs.defect_q0_structure_wk_pk.value)
                    
        # For the defect q
        if 'defect_q_structure' not in wk_node.outputs:
            self.report("Need to Restart Defected q Structure Relaxation...")
            self.is_relax_defect_q = True
        else:
            self.report("Defected q Structure is Relaxed")
            self.out("defect_q_structure_wk_pk",wk_node.outputs.defect_q_structure_wk_pk)
            self.ctx.defect_q_structure = wk_node.outputs.defect_q_structure
            self.out("defect_q_structure",self.ctx.defect_q_structure)
            self.ctx.defect_q_relaxation_wk = orm.load_node(wk_node.outputs.defect_q_structure_wk_pk.value)
 

    def relaxation_restart(self):
        """
        """
        calculations = {}

        # For The Host
        if self.is_relax_host:
            self.report("DEBUG: Host")
            inputs_restart =  self.exposed_inputs(SiestaBaseWorkChain, namespace='host')
            wk_node = orm.load_node(self.inputs.wk_node.value)
            restart_node_wk = orm.load_node(wk_node.outputs.host_structure_wk_pk.value)
            restart_node = orm.load_node(restart_node_wk.called[0].pk)
            self.report(f"Restarting Host strcutrure from PK {restart_node.pk}")
            restart_builder = restart_node.get_builder_restart()
            restart_builder.parent_calc_folder = restart_node.outputs.remote_folder
            inputs_restart = {'structure': restart_builder.structure,
                'parameters': restart_builder.parameters,
                'code': restart_builder.code,
                'basis': restart_builder.basis,
                'kpoints': restart_builder.kpoints,
                'pseudos':restart_builder.pseudos,
                'options': Dict(dict=self.inputs['host']['options'].get_dict()),
                'parent_calc_folder':restart_builder.parent_calc_folder,
                 }

            running = self.submit(SiestaBaseWorkChain, **inputs_restart)
            self.report(f'Restart Launched SiestaBaseWorkChain<{running.pk}> to relax the host structure.')
            calculations ['host_relaxation_wk'] = running
        
        # For The Defect q0
        if self.is_relax_defect_q0:
            self.report("DEBUG: Defect q0")
            inputs_restart =  self.exposed_inputs(SiestaBaseWorkChain, namespace='defect_q0')
            wk_node = orm.load_node(self.inputs.wk_node.value)
            restart_node_wk = orm.load_node(wk_node.outputs.defect_q0_structure_wk_pk.value)
            restart_node = orm.load_node(restart_node_wk.called[0].pk)
            self.report(f"Restarting Defect q0 strcutrure from PK {restart_node.pk}")
            restart_builder = restart_node.get_builder_restart()
            restart_builder.parent_calc_folder = restart_node.outputs.remote_folder
            inputs_restart = {'structure': restart_builder.structure,
                'parameters': restart_builder.parameters,
                'code': restart_builder.code,
                'basis': restart_builder.basis,
                'kpoints': restart_builder.kpoints,
                'pseudos':restart_builder.pseudos,
                'options': Dict(dict=self.inputs['defect_q0']['options'].get_dict()),
                'parent_calc_folder':restart_builder.parent_calc_folder,
                 }

            running = self.submit(SiestaBaseWorkChain, **inputs_restart)
            self.report(f'Restart Launched SiestaBaseWorkChain<{running.pk}> to relax the defect q0 structure.')
            calculations ['defect_q0_relaxation_wk'] = running

        # For The Defect q
        if self.is_relax_defect_q:
            self.report("DEBUG: Defect q")
            inputs_restart =  self.exposed_inputs(SiestaBaseWorkChain, namespace='defect_q')
            wk_node = orm.load_node(self.inputs.wk_node.value)
            restart_node_wk = orm.load_node(wk_node.outputs.defect_q_structure_wk_pk.value)
            restart_node = orm.load_node(restart_node_wk.called[0].pk)
            self.report(f"Restarting Defect q strcutrure from PK {restart_node.pk}")
            restart_builder = restart_node.get_builder_restart()
            restart_builder.parent_calc_folder = restart_node.outputs.remote_folder
            inputs_restart = {'structure': restart_builder.structure,
                'parameters': restart_builder.parameters,
                'code': restart_builder.code,
                'basis': restart_builder.basis,
                'kpoints': restart_builder.kpoints,
                'pseudos':restart_builder.pseudos,
                'options': Dict(dict=self.inputs['defect_q']['options'].get_dict()),
                'parent_calc_folder':restart_builder.parent_calc_folder,
                 }

            running = self.submit(SiestaBaseWorkChain, **inputs_restart)
            self.report(f'Restart Launched SiestaBaseWorkChain<{running.pk}> to relax the defect q structure.')
            calculations ['defect_q_relaxation_wk'] = running

    #=======================================================
    # Runnig DFT Calculations   
    #=======================================================
    def relax_structures(self):
        """
        """
        self.is_relax_host = True
        self.is_relax_defect_q0 = True
        self.is_relax_defect_q = True

        calculations ={}
        
        # Relax host Structure
        self.report('Preparing Host Strcuture to Relax')
        inputs = self.exposed_inputs(SiestaBaseWorkChain, namespace='host')
        inputs['structure'] = self.inputs.host_structure
        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(f'Launched SiestaBaseWorkChain<{running.pk}> to relax the host structure.')
        calculations ['host_relaxation_wk'] = running

        # Relax defect q0 Structure
        self.report('Preparing Defect q0 Structure to Relax')
        inputs = self.exposed_inputs(SiestaBaseWorkChain, namespace='defect_q0')
        inputs['structure'] = self.inputs.defect_structure
        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(f'Launched SiestaBaseWorkChain<{running.pk}> to relax the defect q0 structure.')
        calculations ['defect_q0_relaxation_wk'] = running
        
        # Relax defect q Structure
        self.report('Preparing Defect q Structure to Relax')
        inputs = self.exposed_inputs(SiestaBaseWorkChain, namespace='defect_q')
        inputs['structure'] = self.inputs.defect_structure
        inputs_parameters = Dict(dict=self.inputs.defect_q.parameters.get_dict())
        net_charge = dict={'NetCharge':self.inputs.defect_charge.value}
        inputs_parameters.update_dict(net_charge)
        inputs['parameters'] = inputs_parameters 
        
        running = self.submit(SiestaBaseWorkChain, **inputs)
        self.report(f'Launched SiestaBaseWorkChain<{running.pk}> to relax the defect q structure.')
        calculations ['defect_q_relaxation_wk'] = running

        return ToContext(**calculations)
  
    #=======================================================
    # Check DFT Calculations   
    #=======================================================
    def check_relaxation(self):
        """
        """

        if 'host_relaxation_wk' in self.ctx:
            host_wk = self.ctx.host_relaxation_wk
            host_structure_wk_pk = Int(host_wk.pk)
            host_structure_wk_pk.store()
            self.out('host_structure_wk_pk',host_structure_wk_pk)
            if not host_wk.is_finished_ok:
                self.report(f"HOST STRUCTURE WK FAILD! with PK {host_wk.pk}")
                self.out('host_structure_wk_pk',host_structure_wk_pk)
            else:
                self.ctx.host_structure = host_wk.outputs.output_structure
                self.out('host_structure',self.ctx.host_structure)
                self.report(f"Host Strcuture Relaxation WK is Okay With PK {self.ctx.host_structure.pk} ")

        if 'defect_q0_relaxation_wk' in self.ctx:
            defect_q0_wk = self.ctx.defect_q0_relaxation_wk
            defect_q0_structure_wk_pk = Int(defect_q0_wk.pk)
            defect_q0_structure_wk_pk.store()
            self.out('defect_q0_structure_wk_pk',defect_q0_structure_wk_pk)
            if not defect_q0_wk.is_finished_ok:
                self.report(f"DEFECT q0 STRUCTURE WK FAILD! with PK {defect_q0_wk.pk}")
            else:
                self.ctx.defect_q0_structure = defect_q0_wk.outputs.output_structure
                self.out('defect_q0_structure',self.ctx.defect_q0_structure)
                self.report(f"Defect q0 Structure Relaxation WK is Okay With PK {self.ctx.defect_q0_structure.pk}")

        if 'defect_q_relaxation_wk' in self.ctx:
            defect_q_wk = self.ctx.defect_q_relaxation_wk
            defect_q_structure_wk_pk = Int(defect_q_wk.pk)
            defect_q_structure_wk_pk.store()
            self.out('defect_q_structure_wk_pk',defect_q_structure_wk_pk)
            if not defect_q_wk.is_finished_ok:
                self.report(f"DEFECT q STRUCTURE WK FAILD! with PK {defect_q_wk.pk}")
            else:
                self.ctx.defect_q_structure = defect_q_wk.outputs.output_structure
                self.out('defect_q_structure',self.ctx.defect_q_structure)
                self.report(f"Defect q0 Structure Relaxation WK is Okay With PK {self.ctx.defect_q_structure.pk}")


        if not host_wk.is_finished_ok or not defect_q0_wk.is_finished_ok or not defect_q_wk.is_finished_ok:
            return self.exit_codes.ERROR_MAIN_WC

    
    #=======================================================
    # Retrieving DFT Calculations for None Correction Scheme  
    #=======================================================
    def retrieving_dft_data(self):
        """
        """
        import sisl
        from .utils import get_output_energy_manual
        from .utils import get_output_total_electrons_manual
        from .utils import get_vbm_siesta_manual_bands
        from .utils import get_fermi_siesta_from_fdf
        from .utils import get_vbm_siesta
        
        # For Host
        if self.is_relax_host:
            host_calc = self.ctx.host_relaxation_wk
        else:
            wk_node = orm.load_node(self.inputs.wk_node.value)
            host_calc = orm.load_node(wk_node.outputs.host_structure_wk_pk.value)
            self.report(f'Extracting SIESTA Run for host structure with charge 0 from node PK={host_calc.pk}')
        self.ctx.host_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['E_KS'])
        self.ctx.host_vbm = orm.Float(get_vbm_siesta(host_calc))
        self.report(f"DEBUG: VBM = {self.ctx.host_vbm.value} ")
        self.report(f"The Energy of Host is : {self.ctx.host_energy.value } eV")

        # For Defect q0
        if self.is_relax_defect_q0 :
            defect_q0_calc = self.ctx.defect_q0_relaxation_wk
        else:
            wk_node = orm.load_node(self.inputs.wk_node.value)
            defect_q0_calc = orm.load_node(wk_node.outputs.defect_q0_structure_wk_pk.value)
            self.report(f'Extracting SIESTA Run for host structure with charge 0 from node PK={defect_q0_calc.pk}')
        self.ctx.defect_q0_energy = orm.Float(defect_q0_calc.outputs.output_parameters.get_dict()['E_KS'])
        #self.ctx.defect_q0_vbm = orm.Float(get_vbm_siesta(defect_q0_calc))
        #self.report(f"DEBUG: VBM = {self.ctx.defet_q0_vbm.value} ")
        self.report(f"The Energy of Defect q0 is : {self.ctx.defect_q0_energy.value } eV")

        # For Defect q
        if self.is_relax_defect_q :
            defect_q_calc = self.ctx.defect_q_relaxation_wk
        else:
            wk_node = orm.load_node(self.inputs.wk_node.value)
            defect_q_calc = orm.load_node(wk_node.outputs.defect_q_structure_wk_pk.value)
            self.report(f'Extracting SIESTA Run for host structure with charge 0 from node PK={defect_q_calc.pk}')
        self.ctx.defect_q_energy = orm.Float(defect_q_calc.outputs.output_parameters.get_dict()['E_KS'])
        #self.ctx.defect_q_vbm = orm.Float(get_vbm_siesta(defect_q_calc))
        #self.report(f"DEBUG: VBM = {self.ctx.defet_q_vbm.value} ")
        self.report(f"The Energy of Defect q is : {self.ctx.defect_q_energy.value } eV")

        self.ctx.defect_energy = self.ctx.defect_q0_energy

    def retrieving_dft_potentials(self):
        """
        """
        import sisl
        from .utils import get_vbm_siesta_manual_bands
        from .utils import output_energy_manual
        from .utils import output_total_electrons_manual
        
        # For Host
        if is_relax_host :
            host_calc = self.ctx.host_relaxation_wk 
            self.report(f"Extracting SIESTA Potentials for host structure with charge 0 from node")
        else:
            host_calc = orm.load_node(wk_node.outputs.host_structure_wk_pk.value)
            self.report(f"Extracting SIESTA Potentials for host structure with charge 0 from node PK={wk_node.outputs.host_structure_wk_pk.value}")
        label = 'aiida'
        lVT = label+'.VT'
        lnc = label+'.nc'
        path = host_calc.outputs.remote_folder.get_remote_path()
        if (path / lVT).exists():
            VT = path / lVT
        elif (path/lnc).exists():
            VT = path / lnc
        else:
            self.exit_codes.ERROR_MAIN_WC

        data_array = sisl.get_sile(VT)
        #data_array = sisl.get_sile(host_calc.outputs.remote_folder.get_remote_path()+"/aiida.VT")
        vt = data_array.read_grid()
        vt_data = orm.ArrayData()
        vt_data_grid = orm.ArrayData()
        vt_data.set_array('data', vt.grid) # The array
        vt_data_grid.set_array('grid',np.array(vt.shape)) # The Grid Shape
        self.ctx.v_host = vt_data
        self.ctx.v_host_grid = vt_data_grid
        self.report(f"DEBUG: The host VT file Read!")
        self.report(f"DEBUG: VT Array = {self.ctx.v_host}")
        self.report(f"DEBUG: VT Array Shape = {self.ctx.v_host_grid.get_array('grid')} ")



    def  get_charge_density(self):
        """
        Obtain The Charge Density from the SIESTA calculations.
        """
        import sisl
        # Host

        if self.inputs.run_siesta_host:
            host_node = self.ctx['calc_host']
            if host_node.is_finished_ok:
                data_array = sisl.get_sile(host_node.outputs.remote_folder.get_remote_path()+"/aiida.RHO")
                rho = data_array.read_grid()
                rho_data = orm.ArrayData()
                rho_data.set_array('data', rho.grid)
                self.ctx.rho_host = rho_data
                self.report("DEBUG: The host Rho file Read!")
                self.report("DEBUG: Rho Array = "+str(self.ctx.rho_host ))
            else:
                self.report('Post processing for the host structure has failed with status {}'.format(host_node.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            HostNode = orm.load_node(self.inputs.host_node.value)
            self.report('Extracting SIESTA VT for host structure with charge {} from node PK={}'.format("0.0", self.inputs.host_node.value))
            data_array = sisl.get_sile(HostNode.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            rho=data_array.read_grid()
            rho_data = orm.ArrayData()
            rho_data.set_array('data', rho.grid)
            self.ctx.rho_host = rho_data
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.rho_host ))

         
        if self.inputs.run_siesta_defect_q0:
            defect_q0_rho = self.ctx['calc_host']
            if defect_q0_rho.is_finished_ok:
                data_array = sisl.get_sile(defect_q0_rho.outputs.remote_folder.get_remote_path()+"/aiida.RHO")
                rho = data_array.read_grid()
                rho_data = orm.ArrayData()
                rho_data.set_array('data', rho.grid)
                self.ctx.rho_host = rho_data
                self.report("DEBUG: The host Rho file Read!")
                self.report("DEBUG: Rho Array = "+str(self.ctx.rho_host ))
            else:
                self.report('Post processing for the host structure has failed with status {}'.format(defect_q0_rho.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            Defect_q0Node = orm.load_node(self.inputs.defect_q0_node.value)
            self.report('Extracting SIESTA VT for host structure with charge {} from node PK={}'.format("0.0", self.inputs.defect_q0_node.value))
            data_array = sisl.get_sile(Defect_q0Node.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            rho=data_array.read_grid()
            rho_data = orm.ArrayData()
            rho_data.set_array('data', rho.grid)
            self.ctx.rho_host = rho_data
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.rho_host ))


