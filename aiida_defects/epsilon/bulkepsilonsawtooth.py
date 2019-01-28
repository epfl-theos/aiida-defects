# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from aiida.work.run import run, submit
from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
    
    
from aiida_defects.pp.pp import PpWorkChain
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from aiida_defects.pp.fft_tools import planar_average, read_grid, differentiator

from aiida.work.run import run, submit
from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List

from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import  Str, Bool, Float, Int
from aiida_defects.pp.pp import PpWorkChain

from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm import DataFactory
from aiida.orm.data.singlefile import SinglefileData

    
    
from aiida_defects.pp.pp import PpWorkChain
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from aiida_defects.pp.fft_tools import planar_average, read_grid, differentiator

def linear_poly(x, m, q):
    return m *x + q

def poly_fit(xdata, ydata, func):
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt, pcov

class BulkEpsilonSawtoothWorkChain(WorkChain):
    """
    WorkChain to calculate the dielectric constant of a bulk structure within the Finite Electric Field approach 
    and a sawtooth potential. Both low and high frequency dielectric constant value can be obtained, 
    by specifying the type of calculation in the inputs of the PW calculation.
    The workchain uses a supercell 1x1x6 along the direction where the field is applied by default.
    eamp and eopreg together with the coordinate range along that direction that should be used to
    compute the dielectric constant are automatically derived. You can change this default by specifiyng
    the sc input, which is an an integer n so that a 1x1xn supercell is created,
    but please always use an even number, so that the range of the selected coordinates remains properly defined.
    WARNING: Remember to set the kpoints value for the k-grid equal to 1 for the direction along which you will apply
    the field
    TODO:
    -adapt the launching of the PpWorkChain so that eventually you can resuse a PwCalculatioon node 
    """
    @classmethod
    def define(cls, spec):
        super(BulkEpsilonSawtoothWorkChain, cls).define(spec)
        spec.input("structure",valid_type=StructureData)
        spec.input("code_pw",valid_type=Str,required=False)
        spec.input("code_pp",valid_type=Str,required=False)
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
        #spec.input('eopreg', valid_type=Float, required=False, default=Float(0.50))
        #spec.input('emaxpos', valid_type=Float, required=False, default=Float(3.5/6.0))
        spec.input('edir', valid_type=Int, required=False, default=Int(3))
        spec.input('sc', valid_type=Int, required=False,default=Int(6))
        spec.outline(
            cls.creating_input_structure,
            cls.initializing_e0,
            cls.run_ppworkchain_e0,
            cls.initializing_e1,
            cls.run_ppworkchain_e1,
            cls.initializing_e1_saw,
            cls.run_ppworkchain_e1_saw,
            cls.retrieve_potentials,
            cls.compute_bulk_epsilon,
            )
        spec.dynamic_output()

    def creating_input_structure(self):
        """
        Creating a supercell (1x1xsc) for the structure provided
        """
        structure_mg = self.inputs.structure.get_pymatgen()
        if int(self.inputs.edir) == 1:
            tructure_mg.make_supercell([int(self.inputs.sc), 1, 1])
        elif int(self.inputs.edir) == 2:
            structure_mg.make_supercell([1, int(self.inputs.sc), 1])
        elif int(self.inputs.edir) == 3:
            structure_mg.make_supercell([1, 1, int(self.inputs.sc)])

        self.ctx.structure = StructureData(pymatgen=structure_mg)

    def initializing_e0(self):
        """
        Initalizing a calculation of the electrostatic potential in absence of external field
        """

        self.ctx.parameters_pp = ParameterData(dict={'INPUTPP': {'plot_num' : 11,
                                                                }})
        parameters = self.inputs.parameters.get_dict()
        if str(self.inputs.epsilon_type) == 'high-frequency':
            parameters['CONTROL']['calculation'] = 'scf'
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            parameters['CONTROL']['calculation'] = 'relax'
        parameters['CONTROL']['tefield'] = True
        parameters['CONTROL']['dipfield'] = True
        #arameters['ELECTRON']['conv_thr'] = 1e-10
        parameters['SYSTEM']['edir'] = int(self.inputs.edir)
        parameters['SYSTEM']['emaxpos'] = 3.5/float(self.inputs.sc)
        parameters['SYSTEM']['eopreg'] = 0.5
        parameters['SYSTEM']['eamp'] = 0.0#int(self.inputs.eamp)
        self.ctx.parameters=ParameterData(dict=parameters)

        self.ctx.inputs_e0 = {'structure' : self.ctx.structure,
                      'code_pw' : self.inputs.code_pw,
                      'pseudo_family' : self.inputs.pseudo_family,
                      'kpoints' : self.inputs.kpoints,
                      'parameters' : self.ctx.parameters,
                      'parameters_pp' : self.ctx.parameters_pp,
                      'settings' : self.inputs.settings,
                      'options' : self.inputs.options,
                      'code_pp' : self.inputs.code_pp,
                      'pw_calc' : Bool(True),
                      #'B_atom' : self.inputs.B_atom,
                      #'magnetic_phase' : self.inputs.magnetic_phase,
                     }

        
    def run_ppworkchain_e0(self):
        """
        Submitting a calculation of the electrostatic potential in absence of external field
        """

        running = submit(PpWorkChain,**self.ctx.inputs_e0)
        self.report('Launching PpWorkChain for a FEF calculation with amplitude 0.0 a.u.. pk value {}'.format(
                                                                                                    running.pid))
        return ToContext(ppcalc_e0= running)

    def initializing_e1(self):
        """
        Initalizing a calculation of the electrostatic potential in presence of external field
        """

        parameters_pp = ParameterData(dict={'INPUTPP': {'plot_num' : 11,
                                                                }})
        parameters = self.inputs.parameters.get_dict()
        if str(self.inputs.epsilon_type) == 'high-frequency':
            parameters['CONTROL']['calculation'] = 'scf'
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            parameters['CONTROL']['calculation'] = 'relax'
        parameters['CONTROL']['tefield'] = True
        parameters['CONTROL']['dipfield'] = True
        #parameters['ELECTRON']['conv_thr'] = 1e-10
        parameters['SYSTEM']['edir'] = int(self.inputs.edir)
        parameters['SYSTEM']['emaxpos'] = 3.5/float(self.inputs.sc)
        parameters['SYSTEM']['eopreg'] = 0.5
        parameters['SYSTEM']['eamp'] = float(self.inputs.eamp)
        self.ctx.parameters_e1=ParameterData(dict=parameters)

        self.ctx.inputs_e1 = {'structure' : self.ctx.structure,
                      'code_pw' : self.inputs.code_pw,
                      'pseudo_family' : self.inputs.pseudo_family,
                      'kpoints' : self.inputs.kpoints,
                      'parameters' : self.ctx.parameters_e1,
                      'parameters_pp' : parameters_pp,
                      'settings' : self.inputs.settings,
                      'options' : self.inputs.options,
                      'code_pp' : self.inputs.code_pp,
                      'pw_calc' : Bool(True),
                      #'B_atom' : self.inputs.B_atom,
                      #'magnetic_phase' : self.inputs.magnetic_phase,
                     }

        
        
    def run_ppworkchain_e1(self):
        """
        Submitting a calculation of the electrostatic potential in presence of external field
        """
        running = submit(PpWorkChain,**self.ctx.inputs_e1)
        self.report('Launching PpWorkChain for a FEF calculation with amplitude {} a.u.. pk value {}'.format(
                                                                                                    self.inputs.eamp,
                                                                                                    running.pid))
        return ToContext(ppcalc_e1= running)
    
    def initializing_e1_saw(self):
        """
        Initializing sawtooth potential in presence of external field
        """


        parameters_pp = ParameterData(dict={'INPUTPP': {'plot_num' : 12,
                                                                }})
        parent_folder = self.ctx.ppcalc_e1.out.remote_folder

        self.ctx.inputs_e1_saw = {'structure' : self.ctx.structure,
                                  'parameters_pp' : parameters_pp,
                                  'settings' : self.inputs.settings,
                                  'options' : self.inputs.options,
                                  'code_pp' : self.inputs.code_pp,
                                  'pw_calc' : Bool(False),
                                  'remote_folder' : parent_folder

                     }

    def run_ppworkchain_e1_saw(self):
        """
        Computing sawtooth potential in presence of external field
        """
        running = submit(PpWorkChain,**self.ctx.inputs_e1_saw)
        self.report('Launching PpWorkChain to compute the sawtooth potential. pk value {}'.format(running.pid))
        return ToContext(ppcalc_e1_saw= running)

    def retrieve_potentials(self):
        """
        Retrieving the computed potentials
        """
        
        self.ctx.grid_e0 = read_grid(self.ctx.ppcalc_e0.out.retrieved)
        self.ctx.grid_e1 = read_grid(self.ctx.ppcalc_e1.out.retrieved)
        self.ctx.grid_e1_saw = read_grid(self.ctx.ppcalc_e1_saw.out.retrieved)
        self.out('fft_grid_V0', self.ctx.grid_e0['fft_grid'])
        self.out('fft_grid_V1', self.ctx.grid_e1['fft_grid'])
        self.out('fft_grid_V_saw', self.ctx.grid_e1_saw['fft_grid'])

        
    def compute_bulk_epsilon(self):
        """
        Computing the dielectric constant
        """
        if int(self.inputs.edir) == 1:
            axis = 'x'
        elif int(self.inputs.edir) == 2:
            axis = 'y'
        elif int(self.inputs.edir) == 3:
            axis = 'z'
   
        
        Vsaw=planar_average(self.ctx.grid_e1_saw, self.ctx.structure, axis, npt=400)
        Ve0=planar_average(self.ctx.grid_e0, self.ctx.structure, axis, npt=400)
        Ve1=planar_average(self.ctx.grid_e1, self.ctx.structure, axis, npt=400)
        Vdiff=Ve1['average']- Ve0['average']

        #print "Vsaw", Vsaw

        cell_x1 = int((1.5/float(self.inputs.sc))*Vsaw['npt'])
        cell_x2 = int((2.5/float(self.inputs.sc))*Vsaw['npt'])

        cell_x3 = int((4.5/float(self.inputs.sc))*Vsaw['npt'])
        cell_x4 = int((5.5/float(self.inputs.sc))*Vsaw['npt'])

        #print cell_x1, cell_x2, cell_x3, cell_x4
        #print type(cell_x1), type(cell_x2), type(cell_x3), type(cell_x4)

        x_pos = Ve1['average'][cell_x1:cell_x2]
        y_pos = Vdiff[cell_x1:cell_x2]
        popt, pcov = poly_fit(x_pos, y_pos, linear_poly)
        der_Vdiff = popt[0]

        y_pos = Vsaw['average'][cell_x1:cell_x2]
        popt, pcov = poly_fit(x_pos, y_pos, linear_poly)
        der_Vsaw = popt[0]

        epsilon_pos = der_Vsaw/der_Vdiff

        x_neg = Ve1['average'][cell_x3:cell_x4]
        y_neg = Vdiff[cell_x3:cell_x4]
        popt, pcov = poly_fit(x_neg, y_neg, linear_poly)
        der_Vdiff = popt[0]

        y_neg = Vsaw['average'][cell_x3:cell_x4]
        popt, pcov = poly_fit(x_neg, y_neg, linear_poly)
        der_Vsaw = popt[0]

        epsilon_neg = der_Vsaw/der_Vdiff

        epsilon = (epsilon_neg +epsilon_pos)/2
        #print epsilon_neg
        #print epsilon_pos
        #print epsilon

        self.out('epsilon', Float(epsilon))
        self.report("BulkEpsilonSawtoothWorkChain completed succesfully. The {} Epsilon value is {}".format(self.inputs.epsilon_type,
                                                                                                               epsilon))

