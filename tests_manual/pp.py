# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.          #
#                                                                         #
# AiiDA-Defects is hosted on GitHub at https://github.com/...             #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from aiida.orm import load_node
from aiida.work.run import run
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import  Str, Bool, Float, Int
from aiida_defects.pp.pp import PpWorkChain

from aiida.orm import load_node
inputfile='lton.cif'


from pymatgen.io import cif,aiida
from pymatgen.io.aiida import AiidaStructureAdaptor

structure_mg= cif.Structure.from_file(str(inputfile))
aiida_structure_adaptor = AiidaStructureAdaptor()
structure_sd = aiida_structure_adaptor.get_structuredata(structure_mg)
codename="pw_5.1@ubelix" 
code_pp="pp_5.1@ubelix"
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
         'max_wallclock_seconds' : 3600,
         'custom_scheduler_commands' : u"#SBATCH --partition=all",
         'custom_scheduler_commands' : u"#SBATCH --account=dcb",
         #'custom_scheduler_commands' : u"#SBATCH --partition=empi",
        
        }

settings={}

kpoints = KpointsData()
kpoints.set_kpoints_mesh([8, 8, 1])


parameters = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            #'tstress': True,
            #'tprnfor' : True,
            'nstep' : 500,
            'etot_conv_thr' : 1.0e-6,
             'forc_conv_thr' : 1.0e-3,
            'tefield'      : True,
            'dipfield'  : True
        },
        'SYSTEM': {
            'ecutwfc': 40.,
            'ecutrho': 320.,
            'occupations' : 'smearing',
            'degauss' : 0.01,
            'edir'   : 3,
            'emaxpos' : 0.75,
            'eopreg': 0.01,
            'eamp' : 0.0,

            #'starting_magnetization' : starting_magnetization,
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
            'mixing_beta' : 0.6,
            'startingwfc' : 'atomic',
        },

    }

parameters_pp = {
        'INPUTPP': {'plot_num' : 11,
        }
        }
       

outputs=run(PpWorkChain,
            structure=structure_sd,
            code_pw=Str(codename),
            pseudo_family=Str(pseudo_family),
            kpoints=kpoints,
            parameters=ParameterData(dict=parameters),
            settings=ParameterData(dict=settings),
            options=ParameterData(dict=options),
            code_pp=Str(code_pp),
            parameters_pp=ParameterData(dict=parameters_pp),
            pw_calc=Bool(True),
            B_atom=Str("Ti"),
            magnetic_phase=Str("NM"),
           )
