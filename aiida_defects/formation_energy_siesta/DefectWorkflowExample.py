#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
import aiida


# In[2]:


#from aiida import load_profile
#load_profile()

# Import commonly used functionality
import numpy as np
from aiida import orm, engine, common
from aiida.plugins import WorkflowFactory
from aiida.orm import Code


# In[3]:


#get_ipython().system('pwd')


# In[8]:


#!$JUPYTER_PATH
#!export JUPYTER_PATH="/home/aakhtar/Projects/aiida-defects-siesta/siesta-defect-formation/aiida-workflow/formation_energy/;$JUPYTER_PATH"


# In[5]:


from aiida_siesta.workflows.defect_formation.formation_energy_siesta import FormationEnergyWorkchainSIESTA
#import .formation_energy_siesta
#!pwd


# In[6]:


from aiida.orm import StructureData
#Structure Pure
cell = [[15.0, 0.0, 0.0,],
        [ 0.0,15.0, 0.0,],
        [ 0.0, 0.0,15.0,],
        ]
pure = StructureData(cell=cell)
pure.append_atom(position=( 0.000 , 0.000 , 0.000 ),symbols=['O']) #1
pure.append_atom(position=( 0.757 , 0.586 , 0.000 ),symbols=['H']) #2
pure.append_atom(position=(-0.757 , 0.586 , 0.000),symbols=['H']) #3 
pure.append_atom(position=( 0.000 , 3.500 , 0.000),symbols=['O']) #4
pure.append_atom(position=( 0.757 , 2.914 , 0.000 ),symbols=['H']) #5
pure.append_atom(position=(-0.757 , 2.914 , 0.000),symbols=['H']) #6


defect=StructureData(cell=cell)
defect.append_atom(position=( 0.000 , 0.000 , 0.000 ),symbols=['O']) #1
defect.append_atom(position=( 0.757 , 0.586 , 0.000 ),symbols=['H']) #2
defect.append_atom(position=(-0.757 , 0.586 , 0.000),symbols=['H']) #3 
defect.append_atom(position=( 0.000 , 3.500 , 0.000),symbols=['O']) #4
defect.append_atom(position=( 0.757 , 2.914 , 0.000 ),symbols=['H']) #5
defect.append_atom(position=(-0.757 , 2.914 , 0.000),symbols=['H'],name="GhostH") #6


# In[7]:


code = Code.get_from_string('siesta-psml-lua@N552VW')
charge=-2


# In[8]:


from aiida.orm import Dict
parameters_host = Dict(dict={
   "mesh-cutoff": "250 Ry",
   "dm-tolerance": "0.0001",
   "MD-TypeOfRun":   "LUA",
   "LUA-script":   "neb.lua",
   "DM-NumberPulay ":  "3",
   "DM-History-Depth":  "0",
   "SCF-Mixer-weight":  "0.02",
   "SCF-Mix":   "density",
   "SCF-Mixer-kick":  "35",
   "MD-VariableCell":  "F",
   "MD-MaxCGDispl":  "0.3 Bohr",
   "MD-MaxForceTol":  " 0.04000 eV/Ang", 
    })
parameters_defect_q0 = Dict(dict={
   "mesh-cutoff": "250 Ry",
   "dm-tolerance": "0.0001",
   "MD-TypeOfRun":   "LUA",
   "LUA-script":   "neb.lua",
   "DM-NumberPulay ":  "3",
   "DM-History-Depth":  "0",
   "SCF-Mixer-weight":  "0.02",
   "SCF-Mix":   "density",
   "SCF-Mixer-kick":  "35",
   "MD-VariableCell":  "F",
   "MD-MaxCGDispl":  "0.3 Bohr",
   "MD-MaxForceTol":  " 0.04000 eV/Ang", 
   "NetCharge": "0",
    })
parameters_defect_q = Dict(dict={
   "mesh-cutoff": "250 Ry",
   "dm-tolerance": "0.0001",
   "MD-TypeOfRun":   "LUA",
   "LUA-script":   "neb.lua",
   "DM-NumberPulay ":  "3",
   "DM-History-Depth":  "0",
   "SCF-Mixer-weight":  "0.02",
   "SCF-Mix":   "density",
   "SCF-Mixer-kick":  "35",
   "MD-VariableCell":  "F",
   "MD-MaxCGDispl":  "0.3 Bohr",
   "MD-MaxForceTol":  " 0.04000 eV/Ang",
   "NetCharge": str(charge),
    })
options_host=Dict(
    dict={
        "max_wallclock_seconds": 360,
        #'withmpi': True,
        #'account': "tcphy113c",
        #'queue_name': "DevQ",
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
    }
)
options_defect_q0=Dict(
    dict={
        "max_wallclock_seconds": 360,
        #'withmpi': True,
        #'account': "tcphy113c",
        #'queue_name': "DevQ",
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
    }
)
options_defect_q=Dict(
    dict={
        "max_wallclock_seconds": 360,
        #'withmpi': True,
        #'account': "tcphy113c",
        #'queue_name': "DevQ",
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
    }
)


# In[9]:


basis_dict_host =Dict(dict= {
'pao-basistype':'split',
'pao-splitnorm': 0.150,
'pao-energyshift': '0.020 Ry',
'%block pao-basis-sizes':
"""
GhostH DZP
H    DZP
%endblock pao-basis-sizes""",
})
basis_dict_defect_q0 = Dict(dict= {
'pao-basistype':'split',
'pao-splitnorm': 0.150,
'pao-energyshift': '0.020 Ry',
'%block pao-basis-sizes':
"""
GhostH DZP
H    DZP
%endblock pao-basis-sizes""",
})
basis_dict_defect_q =Dict(dict= {
'pao-basistype':'split',
'pao-splitnorm': 0.150,
'pao-energyshift': '0.020 Ry',
'%block pao-basis-sizes':
"""
GhostH DZP
H    DZP
%endblock pao-basis-sizes""",
})


# In[10]:


parameters_defect_q0.get_dict()


# In[11]:


kpoints_host = orm.KpointsData()
kpoints_host.set_kpoints_mesh([1,1,1]) # Definately not converged, but we want the example to run quickly
kpoints_defect_q0 = orm.KpointsData()
kpoints_defect_q0.set_kpoints_mesh([1,1,1]) # Definately not converged, but we want the example to run quickly
kpoints_defect_q = orm.KpointsData()
kpoints_defect_q.set_kpoints_mesh([1,1,1]) # Definately not converged, but we want the example to run quickly


# In[12]:


import os
from aiida_siesta.data.psf import PsfData
pseudo_file_to_species_map = [ ("H.psf", ['GhostH','H']),("O.psf", ['O'])]
pseudos_dict = {}
for fname, kinds, in pseudo_file_to_species_map:
      absname = os.path.realpath(os.path.join("./aiida_siesta/workflows/defect_formation/pseudos",fname))
      pseudo, created = PsfData.get_or_create(absname, use_first=True)
      for j in kinds:
              pseudos_dict[j]=pseudo
pseudos_dict


# In[13]:


inputs = {
    # Structures
    'host_structure': pure,
    'defect_structure': defect,
    # Defect information 
    'defect_charge' : orm.Float(-2.),  
    'defect_site' : orm.List(list=[-0.757 , 2.914 , 0.000]),    # Position of the defect in crystal coordinates
    'fermi_level' : orm.Float(0.0),               # Position of the Fermi level, with respect to the valence band maximum      
    'chemical_potential' : orm.Float(250.709), # eV, the chemical potentical of a C atom
    'gaussian_sigma':orm.Float(0.5),
    'correction_scheme' : orm.Str('gaussian'),
    "epsilon":orm.Float(1.0),
    # Computational (chosen code is QE)
    'siesta' : { 'dft': {'supercell_host':{'code': code, 'kpoints': kpoints_host, 'parameters' : parameters_host,
                                           'options':options_host,"basis": basis_dict_host},
                         'supercell_defect_q0':{'code': code, 'kpoints': kpoints_defect_q,'parameters' : parameters_defect_q0,
                                                'options':options_defect_q0,"basis": basis_dict_defect_q0},
                         'supercell_defect_q':{'code': code, 'kpoints': kpoints_defect_q,'parameters' : parameters_defect_q,
                                               'options':options_defect_q,"basis": basis_dict_defect_q}
}}}


# In[14]:


inputs


# In[15]:


workchain_future = engine.submit(FormationEnergyWorkchainSIESTA, **inputs)


# In[16]:


workchain_future


# In[17]:


#get_ipython().system('verdi process list -a')


# In[18]:


#get_ipython().system('verdi process show 799')


# In[ ]:


#get_ipython().system('verdi process report 584')


# In[ ]:


#from aiida.orm import load_node
#results=load_node('584')


# In[ ]:


#results.outputs['output_parameters'.get_dict()]


# In[ ]:


#results.outputs.output_parameters.get_dict()


# In[ ]:


#code


# In[ ]:




