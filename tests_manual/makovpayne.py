# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from aiida.work.run import run
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import  Str, Bool, Float, Int
from aiida_defects.formation_energy.makovpayne import MakovPayneCorrection 


structure_sd=load_node(71971)
codename="pw_6.0@ubelix"  
pseudo_family='SSSP'
#defect_position = ArrayData()
#defect_position.set_array("position", np.array([0., 0., 0.]))
defect_charge= 2.
epsilon_r =  2.

options={
        'resources': {
            'num_machines': 1,
            #'num_mpiprocs_per_machine': 1,
        },
         'max_wallclock_seconds' : 600,
         'custom_scheduler_commands' : u"#SBATCH --partition=all",
         'custom_scheduler_commands' : u"#SBATCH --account=dcb",
         #'custom_scheduler_commands' : u"#SBATCH --partition=empi",
        
        }

settings={}

kpoints = KpointsData()
kpoints.set_kpoints_mesh([8, 8, 8])


parameters = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            #'tstress': True,
            'tprnfor' : True,
            'etot_conv_thr' : 1.0e-6,
             'forc_conv_thr' : 1.0e-3,
        },
        'SYSTEM': {
            'ecutwfc': 60.,
            'ecutrho': 480.,
            'occupations' : 'smearing',
            'degauss' : 0.01,
            #'starting_magnetization' : starting_magnetization,
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
            'mixing_beta' : 0.6,
            'startingwfc' : 'atomic',
        },

    }


outputs=run(MakovPayneCorrection,
            bulk_structure=structure_sd,
            code_pw=Str(codename),
            pseudo_family=Str(pseudo_family),
            kpoints=kpoints,
            parameters=ParameterData(dict=parameters), 
            settings=ParameterData(dict=settings),
            options=ParameterData(dict=options),
            epsilon_r=Float(epsilon_r), 
            defect_charge=Float(defect_charge)
           )
    

    

