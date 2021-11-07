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
#from aiida_siesta.workflows.base import SiestaBaseWorkChain

from aiida_defects.formation_energy_siesta.formation_energy_base import FormationEnergyWorkchainBase
#from .utils import run_siesta_calculation
from .utils import get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy

from aiida.common import AttributeDict
from aiida_siesta.calculations.tkdict import FDFDict
#from aiida_siesta.data.common import get_pseudos_from_structure
#from aiida_siesta.data.psf import PsfData
#from aiida_siesta.data.psml import PsmlData

#def prepare_pseudo_inputs(structure, pseudos):
#    """
#    Reading Pseudos
#    """
#    if pseudos is None :
#        raise ValueError('neither an explicit pseudos dictionary was specified')
#    for kind in structure.get_kind_names():
#        if kind not in pseudos:
#            raise ValueError('no pseudo available for element {}'.format(kind))
#
#    return pseudos



class FormationEnergyWorkchainSIESTAManual(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA code
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTAManual, cls).define(spec)

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
        #spec.input_namespace("siesta.dft.supercell_host",
        #                     help="The siesta code to use for the calculations")
        #spec.input_namespace("siesta.dft.supercell_defect_q0",
        #                     help="The siesta code to use for the calculations")
        #spec.input_namespace("siesta.dft.supercell_defect_q",
        #                     help="The siesta code to use for the calculations")
        
        spec.input('host_remote',
                    valid_type = orm.RemoteData, 
                    required  =True) 
        spec.input('defect_neutral_remote',
                   valid_type = orm.RemoteData, 
                   required  = True) 
        spec.input('defect_charged_remote',
                   valid_type = orm.RemoteData, 
                   required  = True) 

        #spec.input('host_energy',
        #           valid_type = orm.Float, 
        #           required  = True) 

        #spec.input('defect_neutral_energy',
        #           valid_type = orm.Float, 
        #           required  = True) 

        #spec.input('defect_charged_energy',
        #           valid_type = orm.Float, 
        #           required  = True) 

   


        #===================================================================
        # What calculations to run (SIESTA) 
        #===================================================================
        #spec.input('run_siesta_host', valid_type = orm.Bool, required = True)
        #spec.input('run_siesta_defect_q0', valid_type = orm.Bool, required = True)
        #spec.input('run_siesta_defect_q', valid_type = orm.Bool, required = True)
        #spec.input('run_siesta_dfpt', valid_type = orm.Bool, required = False)

        spec.input('host_node', valid_type = orm.Int, required = False)
        spec.input('defect_q0_node', valid_type = orm.Int, required = False)
        spec.input('defect_q_node', valid_type = orm.Int, required = False)

        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(
                cls.setup,
                cls.check_dft_potentials_gaussian_correction,
                if_(cls.correction_required)(
                    if_(cls.is_gaussian_model_scheme)(
                        cls.check_dft_potentials_gaussian_correction,
                        cls.run_gaussian_model_correction_workchain,
                        cls.check_gaussian_model_correction_workchain,
                        cls.compute_formation_energy_gaussian_model
                        )))
                
                #cls.run_dft_calcs,
                #if_(cls.correction_required)(
                #    if_(cls.is_gaussian_model_scheme)(
                #        cls.check_dft_potentials_gaussian_correction,
                #        cls.run_gaussian_model_correction_workchain,
                #        cls.check_gaussian_model_correction_workchain,
                #        cls.compute_formation_energy_gaussian_model
                #        ),
                #    if_(cls.is_gaussian_rho_scheme)(
                #        cls.check_dft_potentials_gaussian_correction,
                #        #cls.run_gaussian_rho_correction_workchain,
                #        cls.raise_not_implemented,
                #        )),

                    #) 
                        
    
    
    #=======================================================
    # Retrieving DFT Calculations for None Correction Scheme  
    #=======================================================



    def check_dft_potentials_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """
        import sisl
        from .utils import get_vbm_siesta_manual_bands , output_energy_manual , output_total_electrons_manual
        
        self.report("The Charge is :{}".format(self.inputs.defect_charge))

         
        #----- 
        # Host
        #-----
        host = self.inputs.host_remote
        host_fdf_file = sisl.get_sile(host.get_remote_path()+"/input.fdf")
        #fdf_file_host = sisl.get_sile(host.get_remote_path()+"/input.fdf")
        host_label = host_fdf_file.get("SystemLabel")
        self.report("DEBUG: The Host System Label: {}".format(host_label) )
        
        data_array = sisl.get_sile(host.get_remote_path() +"/" + host_label + ".VT")
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
        self.number_of_electrons = output_total_electrons_manual(host)
        self.report("DEBUG: Total Number of ELectrons in Host :"+str(self.number_of_electrons))
        self.ctx.host_vbm = orm.Float(get_vbm_siesta_manual_bands(host,host_label,self.number_of_electrons))
        self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value)) 
 
        self.ctx.host_energy = orm.Float(output_energy_manual(host))
        self.report('The Energy of the Host is: {} eV'.format(self.ctx.host_energy.value))

        #---------------
        # Defect Neutral
        #---------------

        defect_q0 = self.inputs.defect_neutral_remote
        defect_q0_fdf_file = sisl.get_sile(defect_q0.get_remote_path()+"/input.fdf")
        defect_q0_label = defect_q0_fdf_file.get("SystemLabel")
        self.report("DEBUG: The Defect q0 System Label: {}".format(defect_q0_label) )

        data_array_q0 = sisl.get_sile(defect_q0.get_remote_path() +"/" + defect_q0_label + ".VT")
        vt_q0 = data_array_q0.read_grid()
        vt_q0_data = orm.ArrayData()
        vt_q0_grid_data = orm.ArrayData()
        vt_q0_data.set_array('data', vt_q0.grid)
        vt_q0_grid_data.set_array('grid', np.array(vt_q0.shape))
        self.ctx.v_defect_q0 = vt_q0_data
        self.ctx.v_defect_q0_grid = vt_q0_grid_data 
        self.report("DEBUG: The defect_q0 VT file Read!")
        self.report("DEBUG: VT q0 Array = "+str(self.ctx.v_defect_q0))
        self.report("DEBUG: VT q0 Array Shape = "+str(self.ctx.v_defect_q0_grid.get_array('grid')))
        #self.ctx.host_vbm = orm.Float(get_vbm_siesta_manual(host,host_label))
        #self.report("DEBUG: VBM = "+str(self.ctx.host_vbm.value)) 
        self.ctx.defect_q0_energy = orm.Float(output_energy_manual(defect_q0))
        self.report('The Energy of the Defect q0 is: {} eV'.format(self.ctx.defect_q0_energy.value))


        #---------------
        # Defect Charged
        #---------------

        defect_q = self.inputs.defect_charged_remote
        defect_q_fdf_file = sisl.get_sile(defect_q.get_remote_path()+"/input.fdf")
        defect_q_label = defect_q_fdf_file.get("SystemLabel")
        self.report("DEBUG: The Defect q System Label: {}".format(defect_q_label) )

        data_array_q = sisl.get_sile(defect_q.get_remote_path() +"/" + defect_q_label + ".VT")
        vt_q = data_array_q.read_grid()
        vt_q_data = orm.ArrayData()
        vt_q_grid_data = orm.ArrayData()
        vt_q_data.set_array('data', vt_q.grid)
        vt_q_grid_data.set_array('grid', np.array(vt_q.shape))
        self.ctx.v_defect_q = vt_q_data
        self.ctx.v_defect_q_grid = vt_q_grid_data 
        self.report("DEBUG: The defect_q VT file Read!")
        self.report("DEBUG: VT q Array = "+str(self.ctx.v_defect_q))
        self.report("DEBUG: VT q Array Shape = "+str(self.ctx.v_defect_q_grid.get_array('grid')))
        self.ctx.defect_energy = orm.Float(output_energy_manual(defect_q))
        self.report('The Energy of the Defect q is: {} eV'.format(self.ctx.defect_energy.value))

      


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


