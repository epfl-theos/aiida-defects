# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
#from aiida_quantumespresso.workflows.pw.bands_10 import PwBandsWorkChain
from aiida_defects.formation_energy.bandfilling import BandFillingCorrectionWorkChain 
from aiida_defects.formation_energy  import bandfilling
from aiida.work.run import run
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import  Str, Bool, Float, Int


codename="pw_5.1@ubelix"  
pseudo_family='SSSP'


cell = [    [4.1699147600,    2.0849573800,    2.0849573800],
    [2.0849573800,    4.1699147600,    2.0849573800],
    [2.0849573800,    2.0849573800,    4.1699147600],

       ]
s = StructureData(cell=cell)
s.append_atom(position = [0,0,0], symbols='Ni',  name='Ni1')
s.append_atom(position = [4.169914430,         4.169914963,         4.169915010], symbols='Ni',  name='Ni2')
s.append_atom(position = [6.254871646,         6.254872444,        6.254872515], symbols='O')
s.append_atom(position = [2.084957215,         2.084957481,         2.084957505], symbols='O')


param = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            #'tstress': True,
            #'tprnfor' : True,
            #'etot_conv_thr' : 1.0e-6,
            # 'forc_conv_thr' : 1.0e-3,
        },
        'SYSTEM': {
            'ecutwfc': 50.,
            'ecutrho': 400.,
            'occupations' : 'smearing',
            'degauss' : 0.01,
            #'nspin' : 2,
            #'starting_magnetization' : {'Ni1' : 0.5, 'Ni2' :-0.5},
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
            'mixing_beta' : 0.6,
            'startingwfc' : 'atomic',
        },

    }
            
s3 = StructureData(cell=cell)
s3.append_atom(position = [0,0,0], symbols='Ni',  name='Ni1')
s3.append_atom(position = [4.169914430,         4.169914963,         4.169915010], symbols='Ni',  name='Ni2')
s3.append_atom(position = [6.254871646,         6.254872444,        6.254872515], symbols='O')
s3.append_atom(position = [2.084957215,         2.084957481,         2.084957505], symbols='O')


param = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            #'tstress': True,
            #'tprnfor' : True,
            #'etot_conv_thr' : 1.0e-6,
            # 'forc_conv_thr' : 1.0e-3,
        },
        'SYSTEM': {
            'ecutwfc': 50.,
            'ecutrho': 400.,
            'occupations' : 'smearing',
            'degauss' : 0.01,
            'nspin' : 2,
            'starting_magnetization' : {'Ni1' : 0.5, 'Ni2' :-0.5},
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
            'mixing_beta' : 0.6,
            'startingwfc' : 'atomic',
        },

    }
param2 = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            #'tstress': True,
            #'tprnfor' : True,
            #'etot_conv_thr' : 1.0e-6,
            # 'forc_conv_thr' : 1.0e-3,
        },
        'SYSTEM': {
            'ecutwfc': 50.,
            'ecutrho': 400.,
            'occupations' : 'smearing',
            'degauss' : 0.01,
            'nspin' : 2,
            'starting_magnetization' : {'Ni1' : 0.5, 'Ni2' :-0.5},
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
            'mixing_beta' : 0.6,
            'startingwfc' : 'atomic',
        },

    }

host_parameters = ParameterData(dict=param)
defect_parameters = ParameterData(dict=param2)
options={
        'resources': {
            'num_machines': 1,
            #'num_mpiprocs_per_machine': 1,
        },
         'max_wallclock_seconds' : 10800,
         'custom_scheduler_commands' : u"#SBATCH --partition=all",
         'custom_scheduler_commands' : u"#SBATCH --account=dcb",
         #'custom_scheduler_commands' : u"#SBATCH --partition=empi",
        
        }

settings={}

kpoints = KpointsData()
kpoints.set_kpoints_mesh([2,2, 2])
from copy import deepcopy
kp=KpointsData()
kp.set_kpoints_mesh([2,2, 2])
relax = {
            'kpoints': kp,
            'parameters': ParameterData(dict=param),
            'settings': ParameterData(dict=settings),
            'options': ParameterData(dict=options),
            'meta_convergence': Bool(False),
            'relaxation_scheme': Str('vc-relax'),
            'volume_convergence': Float(0.01)
        }

bf=run(BandFillingCorrection,code=Code.get_from_string(codename), 
       host_structure = s3,
       defect_structure = s,
       pseudo_family=Str(pseudo_family),
      options=ParameterData(dict=options),
       settings=ParameterData(dict=settings),
       kpoints_mesh=kpoints,
       host_parameters=host_parameters,
       defect_parameters=defect_parameters,
       skip_relax=Bool(True),
       potential_alignment=Float(0.00),
       relax=relax)
