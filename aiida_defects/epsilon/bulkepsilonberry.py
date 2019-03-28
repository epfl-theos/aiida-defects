# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import
from __future__ import print_function
from aiida.work.run import run, submit
from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida.orm.code import Code
from aiida.orm import load_node

from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm import DataFactory
from aiida.orm.data.singlefile import SinglefileData

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation


from aiida.work.workchain import WorkChain, ToContext, if_, while_, Outputs, if_, append_

from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List


class BulkEpsilonBerryWorkChain(WorkChain):
    """
    WorkChain to calculate the dielectric constant of a bulk structure within the Finite Electric Field approach 
    and the Berry Phase formulation. Both low and high frequency dielectric constant value can be obtained
    The method is explained in
    P. Umari and A. Pasquarello, PRL 89,157602 (2002)
    I. Souza, J.Iniguez, and D.Vanderbilt, PRL 89, 117602 (2002)
    
    TODO:
    -update once the right values will be parsed by AiiDA
    -maybe let the user specify ncycleberry?
    -check convergence wrt to kpoints
    -add low-frequency dielectric constant. YOu need to parse the ionic dielectric dipole at after the first
    scf of the relaxation step
    """
    @classmethod
    def define(cls, spec):
        super(BulkEpsilonBerryWorkChain, cls).define(spec)
        spec.input("structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str,required=False)
        spec.input("pseudo_family",valid_type=Str, required = False)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData, required = False)
        spec.input('parameters', valid_type=ParameterData, required = False)
        #spec.input('magnetic_phase', valid_type=Str,required=False, default=Str('NM'))
        #spec.input('B_atom', valid_type=Str,required=False)
        #spec.input('hubbard_u', valid_type=ParameterData, required=False, default=ParameterData(dict={}))
        spec.input('epsilon_type', valid_type=Str, required=False, default=Str('high-frequency'))
        spec.input('eamp', valid_type=Float, required=False, default=Float(0.001))
        spec.input('edir', valid_type=Int, required=False, default=Int(3))
        spec.outline(
            cls.validate_inputs,
            if_(cls.epsilon_high_frequency)(
            cls.run_pw_e0),
            cls.run_pw_e1,
            cls.retrieve_dipole,
            cls.compute_bulk_epsilon,
            )
        spec.dynamic_output()
        
    def validate_inputs(self):
        """
        Stopping if the calculation of the low-dielectric constant is requested 
        """
        
        if str(self.inputs.epsilon_type) != 'high-frequency' and str(self.inputs.epsilon_type) != 'low-frequency':
            print(str(self.inputs.epsilon_type) != 'high-frequency' and str(self.inputs.epsilon_type) != 'low-frequency')
            self.abort_nowait('Check the type of dielectric constant requested. Allowed values are: high-frequency and low-frequency')
            
        elif self.inputs.epsilon_type == Str('low-frequency'):
            self.abort_nowait('The calculation of the low-frequency dielectric constant within the Berry phase method is not yet implemented')
    
    def epsilon_high_frequency(self):
        """
        Checking with type of dielectric constant 
        the user want to compute
        """
        return (str(self.inputs.epsilon_type) == 'high-frequency')
        
    def run_pw_e0(self):
        """
        It the high frequency dielectric constant is requested, the workflow will perfom
        a first calculation with an applied external field of 0 a.u. amplitude
        """
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        kpoints = self.inputs.kpoints
        parameter = self.inputs.parameters
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }
        

        inputs['structure'] = self.inputs.structure
            
        param = inputs['parameters'].get_dict()
        
        #Making sure the calculation type is specified. If a value was not specified in the input parameters 
        #by default the workchain will perform an scf calculation resultin in the optical dielectric constant
        param['CONTROL']['calculation'] = 'scf'
        #Specifying flags for the external electric field
        param['CONTROL']['lelfield'] = True
        param['CONTROL']['nberrycyc'] = 1
        param['ELECTRONS']['startingwfc'] = 'random'
        param['ELECTRONS']['efield_cart(1)'] = 0.0
        param['ELECTRONS']['efield_cart(2)'] = 0.0
        param['ELECTRONS']['efield_cart(3)'] = 0.0

        inputs['parameters']= ParameterData(dict=param)

        running = submit(PwBaseWorkChain,**inputs)
        self.report('Launching PwBaseWorkChain for a FEF calculation with amplitude 0.0 a.u.. pk value {}'.format( running.pid))  
        return ToContext(pwcalc_e0 = running)
    
    def run_pw_e1(self):
        """
        Performing an PW calculation with an applied external field with a user defined amplitude
        """
        code_pw = Code.get_from_string(str(self.inputs.code_pw))
        options = self.inputs.options
        settings = self.inputs.settings
        kpoints = self.inputs.kpoints
        parameter = self.inputs.parameters
        
        inputs={
                'code' : code_pw,
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : kpoints,
                'parameters' : parameter,
                'settings' : settings,
                'options' : options,

            }
        

        inputs['structure'] = self.inputs.structure
            
        param = inputs['parameters'].get_dict()
        
        #Making sure the calculation type is specified. If a value was not specified in the input parameters 
        #by default the workchain will perform an scf calculation resultin in the optical dielectric constant
        if str(self.inputs.epsilon_type) == 'high-frequency':
            param['CONTROL']['calculation'] = 'scf'
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            param['CONTROL']['calculation'] = 'relax'
        #Specifying flags for the external electric field
        param['CONTROL']['lelfield'] = True
        param['CONTROL']['nberrycyc'] = 3
        param['ELECTRONS']['startingwfc'] = 'random'
        if int(self.inputs.edir) == 1:
            param['ELECTRONS']['efield_cart(1)'] = float(self.inputs.eamp)
            param['ELECTRONS']['efield_cart(2)'] = 0.0
            param['ELECTRONS']['efield_cart(3)'] = 0.0
        if int(self.inputs.edir) == 2:
            param['ELECTRONS']['efield_cart(1)'] = 0.0
            param['ELECTRONS']['efield_cart(2)'] = float(self.inputs.eamp)
            param['ELECTRONS']['efield_cart(3)'] = 0.0
        elif int(self.inputs.edir) == 3:
            param['ELECTRONS']['efield_cart(1)'] = 0.0
            param['ELECTRONS']['efield_cart(2)'] = 0.0
            param['ELECTRONS']['efield_cart(3)'] = float(self.inputs.eamp)

        inputs['parameters']= ParameterData(dict=param)

        running = submit(PwBaseWorkChain,**inputs)
        self.report('Launching PwBaseWorkChain for a FEF calculation with amplitude {} a.u.. pk value {}'.format(self.inputs.eamp,running.pid))  
        return ToContext(pwcalc_e1 = running)

    
    def retrieve_dipole(self):
        """
        Extracting information on the induced dipoles from the calculations
        """
        
        if str(self.inputs.epsilon_type) == 'high-frequency':
            self.ctx.dipole_e0 = self.ctx.pwcalc_e0.out.output_parameters.dict.electronic_dipole_cartesian_axes
            self.ctx.dipole_e1 = self.ctx.pwcalc_e1.out.output_parameters.dict.electronic_dipole_cartesian_axes
            self.ctx.volume = self.ctx.pwcalc_e0.out.output_parameters.dict.volume
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            self.ctx.dipole_e0 = self.ctx.pwcalc_e1.out.output_parameters.dict.ionic_dipole_cartesian_axes
            self.ctx.dipole_e1 = self.ctx.pwcalc_e1.out.output_parameters.dict.ionic_dipole_cartesian_axes
            self.ctx.volume = self.ctx.pwcalc_e1.out.output_parameters.dict.volume
            
        
    def compute_bulk_epsilon(self):
        """
        Calculating the dielectric constant
        """
        from math import pi
        
        volume = self.ctx.volume/(0.529177249**3)
        if  int(self.inputs.edir) == 1:
            bulk_epsilon = 4*pi*(self.ctx.dipole_e1[0]-self.ctx.dipole_e0[0])/(
            float(self.inputs.eamp) * volume) + 1
                
        elif  int(self.inputs.edir) == 2:
            bulk_epsilon = 4*pi*(self.ctx.dipole_e1[1]-self.ctx.dipole_e0[1])/(
            float(self.inputs.eamp) * volume) + 1
        elif  int(self.inputs.edir) == 3:
            bulk_epsilon = 4*pi*(self.ctx.dipole_e1[2]-self.ctx.dipole_e0[2])/(
            float(self.inputs.eamp) * volume) + 1
        self.out('epsilon', Float(bulk_epsilon))
        self.report("BulkEpsilonBerryWorkChain completed succesfully. The {} dielectric constant is {}".format(self.inputs.epsilon_type,bulk_epsilon))
