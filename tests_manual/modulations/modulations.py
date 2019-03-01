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
from aiida_defects.phonopy.modulations import PhonopyWorkChain 

inputfile='hktio2.cif'
codename="pw_6.0@ubelix"  
pseudo_family='SSSP'


from pymatgen.io import cif,aiida
from pymatgen.io.aiida import AiidaStructureAdaptor

structure_mg= cif.Structure.from_file(str(inputfile))
aiida_structure_adaptor = AiidaStructureAdaptor()
structure_sd = aiida_structure_adaptor.get_structuredata(structure_mg)


magnetic_phase = Str("NM")
B_atom = Str("Ti")




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

#hubbard_u = ParameterData(dict={'Mn': 3.})

phonopy_input = ParameterData(dict={'supercell': [[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 1]],
                     'distance': 0.01,
                     'mesh': [40, 40, 40],
                     'symmetry_precision': 1e-4,
                     'modulation' : {'amplitude' : [0,3,1],
                                     'phase' : 0,
                                     'E_thr' : 6,#3 to check degenerate
                                     'q_point' : [0., 0., 0.],
                                     'supercell':[[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 1]]
                                     
                                    }
                     })



outputs=run(PhonopyWorkChain,
	    structure=structure_sd,
	    code_pw=Str(codename),
            pseudo_family=Str(pseudo_family),
            kpoints=kpoints,
            parameters=ParameterData(dict=parameters), 
            settings=ParameterData(dict=settings),
            options=ParameterData(dict=options),
            magnetic_phase=Str(magnetic_phase),
            B_atom=Str(B_atom), 
            phonopy_input=phonopy_input, 
            optimization=Str('scf'))

