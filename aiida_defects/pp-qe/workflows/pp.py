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

from aiida.work.run import run, submit
from aiida.work.workchain import WorkChain, ToContext, if_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_defects.tools.structure_manipulation import create_suitable_inputs_noclass

#from aiida.orm import DataFactory
#from aiida.orm.data.singlefile import SinglefileData
#from aiida.work.workfunction import workfunction
#from aiida.orm import load_node
#import matplotlib.pyplot as plt
#from aiida.work.run import run
#import sys
#import argparse

class PpWorkChain(WorkChain):
    """
    WorkChain to perform a PP calculation.
    One could either decide to perfom first a PW calculation (setting pw_calc=Bool(True) followed by the PP 
    calculation, or could directly perform the PP one. In the latter case pw_calc=Bool(False) but a parent folder
    of a PwCalculation should be specified
    """
    @classmethod
    def define(cls, spec):
        super(PpWorkChain, cls).define(spec)
        spec.input("structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str,required=False)
        spec.input("code_pp",valid_type=Str)
        spec.input("pseudo_family",valid_type=Str, required = False)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData, required = False)
        spec.input('parameters', valid_type=ParameterData, required = False)
        spec.input('parameters_pp', valid_type=ParameterData)
        spec.input('pw_calc', valid_type=Bool, required=False, default=Bool(False))
        spec.input('remote_folder', valid_type=(FolderData,RemoteData), required = False)
        spec.input('parent_calculation', valid_type=PwCalculation, required=False)
        spec.outline(
            if_(cls.should_run_pw)(
                cls.run_pw,
            ),
            cls.initialize_pp,
            cls.run_pp,
            cls.retrieve_folder,
            )
        spec.dynamic_output()
    def should_run_pw(self):
        """
        Check the inputs to verify if a PW calculation is to be performed before the PP one
        """
        return bool(self.inputs.pw_calc)

    def run_pw(self):
        """
        Running the PW calculation using the PwBaseWorkChain
        """
        

        inputs={
                'code' : Code.get_from_string(str(self.inputs.code_pw)),
                'pseudo_family' : Str(self.inputs.pseudo_family),
                'kpoints' : self.inputs.kpoints,
                'parameters' : self.inputs.parameters,
                'settings' : self.inputs.settings,
                'options' : self.inputs.options,
                'structure' : self.inputs.structure

            }


        running = submit(PwBaseWorkChain,**inputs)
        self.report('Launching PwBaseWorkChain. pk value {}'.format( running.pid))
        return ToContext(pwcalc = running)

    def initialize_pp(self):
        """
        Initializing the remote folder of the PP calculation as the one of the previous PW step 
        if performed or as the one of the PwCalculation specified in the inputs or as a remote_data/folder_data
        specified in the inputs.
        """
        if bool(self.inputs.pw_calc):
            self.ctx.remote_folder = self.ctx.pwcalc.out.remote_folder
            self.out('remote_folder', self.ctx.remote_folder)
        else:
            if not ('parent_calculation' in self.inputs or 'remote_folder' in self.inputs):
                self.abort_nowait('Neither the parent_calculation nor the parent_folder input was defined')
            try:
                parent_folder = self.inputs.parent_calculation.out.remote_folder
            except AttributeError:
                parent_folder = self.inputs.remote_folder

            self.ctx.remote_folder = parent_folder

    def run_pp(self):
        """
        Running a PpCalculation
        """
        code_pp = Code.get_from_string(str(self.inputs.code_pp))

        inputs = {
        'code': code_pp,
        'parameters': self.inputs.parameters_pp,
        '_options': self.inputs.options.get_dict(), #'_options' for PwCalculation is a dictionary
        'parent_folder': self.ctx.remote_folder,
        }

        process = PpCalculation.process()
        running = submit(process,  **inputs)
        self.report('Launching a PpCalculation. pk value {}'.format(running.pid))
        return ToContext(ppcalc=running)


    def retrieve_folder(self):
        """
        Retrieving the FolderData produced by the PpCalculation.
        Since the result of the PpCalculation is not parsed, in case the calculation did not finish correctly
        the in the report of the PpWorkChain will appear a message that the filplot file contaiing the 3d-FFT
        grid was not found.
        """
        if 'aiida.filplot' not in self.ctx.ppcalc.out.retrieved.get_folder_list():
            self.report("filplot file not found. Please check your calculation")
        else:
            self.out('retrieved', self.ctx.ppcalc.out.retrieved)
            self.report("PpWorkChain  completed succesfully")
