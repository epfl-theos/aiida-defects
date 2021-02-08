# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida.plugins import WorkflowFactory
from aiida_siesta.workflows.base import SiestaBaseWorkChain

from aiida_defects.formation_energy_siesta.formation_energy_base import FormationEnergyWorkchainBase
#from .utils import run_siesta_calculation
from .utils import get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy

from aiida.common import AttributeDict
from aiida_siesta.calculations.tkdict import FDFDict
#from aiida_siesta.data.common import get_pseudos_from_structure
#from aiida_siesta.data.psf import PsfData
#from aiida_siesta.data.psml import PsmlData
def prepare_pseudo_inputs(structure, pseudos):
    """
    Reading Pseudos
    """
    if pseudos is None :
        raise ValueError('neither an explicit pseudos dictionary was specified')
    for kind in structure.get_kind_names():
        if kind not in pseudos:
            raise ValueError('no pseudo available for element {}'.format(kind))

    return pseudos



class FormationEnergyWorkchainSIESTA(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA code
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTA, cls).define(spec)

        # DFT and DFPT calculations with QuantumESPRESSO are handled with different codes, so here
        # we keep track of things with two separate namespaces. An additional code, and an additional
        # namespace, is used for postprocessing
        #spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
        #spec.inputs._ports['pseudos'].dynamic = True  #Temporary fix to issue #135 plumpy
        # spec.inputs._ports["pseudos.defect"].dynamic = True
        # DFT inputs for Host (SIESTA)
        
        #spec.input("use_siesta_mesh_cutoff",
        #            valid_type = orm.Bool,
        #            required = True,
        #            help = "Whether use Siesta Mesh size to Generate the Model Potential or Not ")
        spec.input_namespace("siesta.dft.supercell_host",
                             help="The siesta code to use for the calculations")
        spec.input_namespace("siesta.dft.supercell_defect_q0",
                             help="The siesta code to use for the calculations")
        spec.input_namespace("siesta.dft.supercell_defect_q",
                             help="The siesta code to use for the calculations")
        spec.input_namespace('pseudos_host', required = False, dynamic = True)
        spec.input_namespace('pseudos_defect', required = False, dynamic = True)
        
        #---------------------
        # HOST Inputs (SIESTA)
        #---------------------
        
        spec.input("siesta.dft.supercell_host.code",
                   valid_type = orm.Code,
                   help = "The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_host'
        #    valid_type = (PsfData,PsmlData),
        #    help = 'Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_host.kpoints",
                   valid_type = orm.KpointsData,
                   help = "The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_host.basis",
                   valid_type = orm.Dict,
                   help = "The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_host.parameters",
                   valid_type = orm.Dict,
                   help = "Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_host.options",
                   valid_type = orm.Dict,
                   help = "options exfor the SIESTA calcuations")
        
        #----------------------------------
        # Defect_q0 without charge (SIESTA)
        #----------------------------------
        
        spec.input("siesta.dft.supercell_defect_q0.code",
                   valid_type = orm.Code,
                   help = "The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_q0',
        #    valid_type=(PsfData,PsmlData),
        #    help='Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_defect_q0.kpoints",
                    valid_type = orm.KpointsData,
                    help = "The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_defect_q0.basis",
                   valid_type = orm.Dict,
                   help = "The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_defect_q0.parameters",
                   valid_type = orm.Dict,
                   help = "Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_defect_q0.options",
                   valid_type = orm.Dict,
                   help = "options for the SIESTA calcuations.")
        
        #------------------------------
        # Defect_q With Charge (SIESTA)
        #------------------------------

        spec.input("siesta.dft.supercell_defect_q.code",
                   valid_type = orm.Code,
                   help = "The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_q',
        #    valid_type = (PsfData,PsmlData),
        #    help = 'Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_defect_q.kpoints",
                   valid_type = orm.KpointsData,
                   help = "The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_defect_q.basis",
                   valid_type = orm.Dict,
                   help = "The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_defect_q.parameters",
                   valid_type = orm.Dict,
                   help = "Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_defect_q.options",
                   valid_type = orm.Dict,
                   help = "options for the SIESTA calcuations.")

        #===================================================================
        # What calculations to run (SIESTA) 
        #===================================================================
        spec.input('run_siesta_host', valid_type = orm.Bool, required = True)
        spec.input('run_siesta_defect_q0', valid_type = orm.Bool, required = True)
        spec.input('run_siesta_defect_q', valid_type = orm.Bool, required = True)
        spec.input('run_siesta_dfpt', valid_type = orm.Bool, required = False)

        spec.input('host_node', valid_type = orm.Int, required = False)
        spec.input('defect_q0_node', valid_type = orm.Int, required = False)
        spec.input('defect_q_node', valid_type = orm.Int, required = False)
        #spec.input("epsilon", 
        #            valid_type = orm.Float, 
        #            help = "Dielectric constant of the host", 
        #            required = False)

        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(
                cls.setup,
                cls.run_dft_calcs,
                #cls.check_dft_calcs,
                if_(cls.correction_required)(
                    if_(cls.is_gaussian_model_scheme)(
                        cls.check_dft_no_correction_scheme_calcs,
                        cls.check_dft_potentials_gaussian_correction,
                        cls.run_gaussian_model_correction_workchain,
                        cls.check_gaussian_model_correction_workchain,
                        cls.compute_formation_energy_gaussian_model
                        ),
                    if_(cls.is_gaussian_rho_scheme)(
                        cls.check_dft_potentials_gaussian_correction,
                        #cls.run_gaussian_rho_correction_workchain,
                        cls.raise_not_implemented,
                        )),
                if_(cls.is_none_scheme)(
                    cls.check_dft_no_correction_scheme_calcs,
                    cls.compute_no_corrected_formation_energy)
                    )

                        #)).else_(
                        #cls.check_dft_calcs,
                        #cls.compute_neutral_formation_energy,
                        #cls.compute_charged_formation_energy_no_corre))
                #if_(cls.is_none_scheme)(
                #    cls.check_dft_calcs,
                #    cls.compute_no_corrected_formation_energy))    
                        
                        
    
    #=========================================================================
    # This function is for running Host and defect structure neutral & charge 
    #========================================================================
    def run_dft_calcs(self):
 
        """
        Submit All DFT Calculations
        """
        self.report("Setting Up The Formation Energy Workchain ")
        #--------------
        # For the Host
        #-------------
        if self.inputs.run_siesta_host:
            siesta_inputs = self.inputs.siesta.dft.supercell_host.code.get_builder()
            #siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain)) For reusing 
            pseudos = None
            if "pseudos_host" in self.inputs:  #in case in the future Issue #142 will be solved
                if self.inputs.pseudos_host:
                    pseudos = self.inputs.pseudos_host
            #structure = None
            structure = self.inputs.host_structure
            siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
            siesta_inputs.structure = structure
            siesta_inputs.parameters = self.inputs.siesta.dft.supercell_host.parameters
            siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_host.kpoints
            siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_host.options.get_dict()
            siesta_inputs.basis = self.inputs.siesta.dft.supercell_host.basis #get_dict()
            #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
            future = self.submit(siesta_inputs)
            self.report('Launching host structure (PK={}) for SIESTA run (PK={})'.format(structure.pk, future.pk))
            #.format(self.inputs.host_structure.pk, future.pk))
            self.to_context(**{'calc_host': future})
        else:
            self.report("The host structure Data Will be Extracted from previous Runs")
        #-------------------------------------------
        # For Defect structure; neutral charge state
        #------------------------------------------
        if self.inputs.run_siesta_defect_q0:
            siesta_inputs = self.inputs.siesta.dft.supercell_defect_q0.code.get_builder()
            #siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
            pseudos = None
            if "pseudos_defect" in self.inputs:  #in case in the future Issue #142 will be solved 
                if self.inputs.pseudos_defect:
                    pseudos = self.inputs.pseudos_defect
            structure = self.inputs.defect_structure
            siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
            siesta_inputs.structure = structure
            siesta_inputs.parameters = self.inputs.siesta.dft.supercell_defect_q0.parameters
            siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_defect_q0.kpoints
            siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_defect_q0.options.get_dict()
            siesta_inputs.basis = self.inputs.siesta.dft.supercell_defect_q0.basis
            #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
            future = self.submit(siesta_inputs)
            self.report('Launching defect structure (PK={}) with charge {} for SIESTA run (PK={})'.format(structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_defect_q0': future})
        else:
            self.report("The defect structure Data with charge {} Will be Extracted from previous Runs".format("0.0"))
        #------------------------------------------
        # For Defect structure; target charge state
        #-----------------------------------------
        if self.inputs.run_siesta_defect_q:
            siesta_inputs = self.inputs.siesta.dft.supercell_defect_q.code.get_builder()
            # siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
            pseudos = None
            if "pseudos_defect" in self.inputs:  #in case in the future Issue #142 will be solved 
                if self.inputs.pseudos_defect:
                    pseudos = self.inputs.pseudos_defect
            structure = self.inputs.defect_structure 
            siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
            siesta_inputs.structure = structure
            siesta_inputs.parameters = self.inputs.siesta.dft.supercell_defect_q.parameters
            siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_defect_q.kpoints
            siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_defect_q.options.get_dict()
            siesta_inputs.basis = self.inputs.siesta.dft.supercell_defect_q.basis
            #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
            future = self.submit(siesta_inputs)
            self.report('Launching defect structure (PK={}) with charge {} for SIESTA run (PK={})'.format(structure.pk,self.inputs.defect_charge.value, future.pk))
            self.to_context(**{'calc_defect_q': future})
        else:
            self.report("The defect structure Data with charge {} Will be Extracted from previous Runs".format(self.inputs.defect_charge.value))
    
    
    #=======================================================
    # Retrieving DFT Calculations for None Correction Scheme  
    #=======================================================
    def check_dft_no_correction_scheme_calcs(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """
        import sisl
        from .utils import get_vbm_siesta        
        # We used sisl library read the Potential Grids

        self.report("Checking Up Whether DFT Caclucations are Finished ")
        # Host
        # if True Will run the Host
        if self.inputs.run_siesta_host:
            host_calc = self.ctx['calc_host']         
            if host_calc.is_finished_ok:
                self.ctx.host_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['E_KS']) # eV
                self.ctx.host_vbm = orm.Float(get_vbm_siesta(host_calc))
                self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value))
                self.report("The Energy of Host is : {} eV".format(self.ctx.host_energy.value))
            else:
                self.report('SIESTA for the host structure has failed with status {}'.format(host_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            HostNode = orm.load_node(self.inputs.host_node.value)
            self.report('Extracting SIESTA Run for host structure with charge {} from node PK={}'.format("0.0", self.inputs.host_node.value))
            self.ctx.host_energy = orm.Float(HostNode.outputs.output_parameters.get_dict()['E_KS']) # eV
            self.ctx.host_vbm = orm.Float(get_vbm_siesta(HostNode))
            self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value))
            self.report('The energy of the host is: {} eV'.format(self.ctx.host_energy.value))

            #self.ctx.host_vbm = orm.Float(HostNode.outputs.output_band.get_array('bands')[0][-1]) # eV
            #self.ctx.host_vbm = orm.Float(get_vbm(HostNode))
            #self.report('The top of valence band is: {} eV'.format(self.ctx.host_vbm.value))


        # Defect (q=0)
        # if True Will run the defect_q0
        if self.inputs.run_siesta_defect_q0:
            defect_q0_calc = self.ctx['calc_defect_q0']
            if defect_q0_calc.is_finished_ok:
                self.ctx.defect_q0_energy = orm.Float(defect_q0_calc.outputs.output_parameters.get_dict()['E_KS'])
                self.report("The Energy of Defect Structure (with charge 0)  is : {} eV".format(self.ctx.defect_q0_energy.value))
            else:
                self.report('SIESTA for the Defect Structure (with charge 0) has failed with status {}'.format(defect_q0_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            Defect_q0Node = orm.load_node(self.inputs.defect_q0_node.value)
            self.report('Extracting SIESTA Run for Defect structure with charge {} from node PK={}'.format("0.0", self.inputs.defect_q0_node.value))
            self.ctx.defect_q0_energy = orm.Float(Defect_q0Node.outputs.output_parameters.get_dict()['E_KS']) # eV
            self.report('The Energy of Defect Structure (with chatge 0) is: {} eV'.format(self.ctx.defect_q0_energy.value))




        # Defect (q=q)
        # if True Will run the defect_q0
        if self.inputs.run_siesta_defect_q:
            defect_q_calc = self.ctx['calc_defect_q']
            if defect_q_calc.is_finished_ok:
                self.ctx.defect_energy = orm.Float(defect_q_calc.outputs.output_parameters.get_dict()['E_KS']) # eV
                #self.ctx.defect_q_VT = sisl.get_sile(defect_q_calc.outputs.remote_folder.get_remote_path()+"/aiida.VT")
                self.report("The Energy of Defect Structure (with charge {})  is : {} eV".format(self.inputs.defect_charge.value ,self.ctx.defect_energy.value))
                #self.report("DEBUG: The VT file Read!")
            else:
                self.report('SIESTA for the defect structure (with charge {}) has failed with status {}'.format(self.inputs.defect_charge.value,defect_q_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            Defect_qNode = orm.load_node(self.inputs.defect_q_node.value)
            self.report('Extracting SIESTA Run for Defect structure with charge {} from node PK={}'.format(self.inputs.defect_charge.value, self.inputs.defect_q_node.value))
            self.ctx.defect_energy = orm.Float(Defect_qNode.outputs.output_parameters.get_dict()['E_KS']) # eV
            self.report('The Energy of Defect Structure with chatge {} is: {} eV'.format(self.inputs.defect_charge.value , self.ctx.defect_energy.value))




    def check_dft_potentials_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """
        import sisl
        from .utils import get_vbm_siesta
         
        # Host

        if self.inputs.run_siesta_host:
            host_vt = self.ctx['calc_host']
            if host_vt.is_finished_ok:
                data_array = sisl.get_sile(host_vt.outputs.remote_folder.get_remote_path()+"/aiida.VT")
                vt = data_array.read_grid()  
                vt_data = orm.ArrayData()
                vt_data_grid = orm.ArrayData()
                vt_data.set_array('data', vt.grid) # The array
                vt_data_grid.set_array('grid',np.array(vt.shape)) # The Grid Shape
                self.ctx.v_host = vt_data
                self.ctx.v_host_grid = vt_data_grid
                self.report("DEBUG: The host VT file Read!")
                self.report("DEBUG: VT Array = "+str(self.ctx.v_host ))
                self.report("DEBUG: VT Array Shape = "+str(self.ctx.v_host_grid.get_array('grid')))
                self.ctx.host_vbm = orm.Float(get_vbm_siesta(host_vt))
                self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value)) 
            else:
                self.report('Post processing for the host structure has failed with status {}'.format(host_vt.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            HostNode = orm.load_node(self.inputs.host_node.value)
            self.report('Extracting SIESTA VT for host structure with charge {} from node PK={}'.format("0.0", self.inputs.host_node.value))
            data_array = sisl.get_sile(HostNode.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            vt = data_array.read_grid()
            vt_data = orm.ArrayData()
            vt_data_grid = orm.ArrayData()
            vt_data.set_array('data', vt.grid) # The array
            vt_data_grid.set_array('grid',np.array(vt.shape)) # The Grid Shape
            self.ctx.v_host = vt_data
            self.ctx.v_host_grid = vt_data_grid
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.v_host ))
            self.report("DEBUG: VT Array Shape = "+str(self.ctx.v_host_grid.get_array('grid')))
            self.ctx.host_vbm = orm.Float(get_vbm_siesta(HostNode))
            self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value))



        # Defect (q=0)
        # if True Will run the defect_q0
        if self.inputs.run_siesta_defect_q0:
            defect_q0_vt = self.ctx['calc_defect_q0']
            if defect_q0_vt.is_finished_ok:
                data_array_q0 = sisl.get_sile(defect_q0_vt.outputs.remote_folder.get_remote_path()+"/aiida.VT")
                vt_q0 = data_array_q0.read_grid()        
                vt_data_q0 = orm.ArrayData()
                vt_data_q0.set_array('data', vt_q0.grid)
                self.ctx.v_defect_q0 = vt_data_q0
                self.report("DEBUG: The defect_q0 VT file Read!")
                self.report("DEBUG: VT q0 Array = "+str(self.ctx.v_defect_q0))
            else:
                self.report('Post processing for the defect structure with charge {} has failed with status {}'.format("0.0",defect_q0_vt.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            Defect_q0Node = orm.load_node(self.inputs.defect_q0_node.value)
            self.report('Extracting SIESTA VT for Defect structure with charge {} from node PK={}'.format("0.0", self.inputs.defect_q0_node.value))
            data_array = sisl.get_sile(Defect_q0Node.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            vt_q0 = data_array.read_grid()
            vt_data_q0 = orm.ArrayData()
            vt_data_q0.set_array('data', vt_q0.grid)
            self.ctx.v_defect_q0 = vt_data_q0
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT q0 Array = "+str(self.ctx.v_defect_q0))

     
            
        
        # Defect (q=q)
        if self.inputs.run_siesta_defect_q:
            defect_q_vt = self.ctx['calc_defect_q']
            if defect_q_vt.is_finished_ok:
                data_array = sisl.get_sile(defect_q_vt.outputs.remote_folder.get_remote_path()+"/aiida.VT")  
                vt_q=data_array.read_grid()
                vt_data_q = orm.ArrayData()
                vt_data_q.set_array('data', vt_q.grid)
                self.ctx.v_defect_q = vt_data_q
                self.report("DEBUG: The defect_q VT file Read!")
                self.report("DEBUG: VT Array = "+str(self.ctx.v_defect_q))
            else:
                self.report('Post processing for the defect structure (with charge {}) has failed with status {}'.format(self.inputs.defect_charge.value,defect_q_vt.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            Defect_qNode = orm.load_node(self.inputs.defect_q_node.value)
            self.report('Extracting SIESTA VT for Defect structure with charge {} from node PK={}'.format(self.inputs.defect_charge.value, self.inputs.defect_q_node.value))
            data_array = sisl.get_sile(Defect_qNode.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            vt_q=data_array.read_grid()
            vt_data_q = orm.ArrayData()
            vt_data_q.set_array('data', vt_q.grid)
            self.ctx.v_defect_q = vt_data_q
            self.report("DEBUG: The defect_q VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.v_defect_q))



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


