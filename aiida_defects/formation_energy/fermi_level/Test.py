import json
import numpy as np
import re
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from aiida.engine import submit
from aiida.orm import StructureData
from aiida.orm import Float, Int, Str, List, Bool, Dict, ArrayData, XyData, StructureData
from aiida_defects.formation_energy.utils import get_vbm
from aiida_defects.formation_energy.fermi_level.fermi_level import FermiLevelWorkchain
from aiida_defects.formation_energy.fermi_level.utils import compute_net_charge

with open('/home/sokseiham/Documents/Defect_calculations/Li7PS6/defect_dict.json') as f:
	defect_dict = json.load(f)
	dos = np.array(defect_dict['DOS'])
	host_structure = Structure.from_dict(defect_dict['unitcell'])
	defect_data = defect_dict['defect_data']
	band_gap = defect_dict['band_gap']
	Ef = defect_dict['Ef']

# While save in json, floats in the dict key were converted into strings. We need to convert those keys back to float.
for key in defect_data:
	key_list = list(defect_data[key]['charge'].keys())
	for chg in key_list:
		defect_data[key]['charge'][float(chg)] = defect_data[key]['charge'].pop(chg)

# site = {'S_1': 1, 'S_2': 2, 'S_3': 1, 'S_oct': 1, 'S_tet': 1, 'Li_1': 2, 'Li_2': 1, 'Li_3': 2, 'Li_4': 2}
# for key in defect_data:
# 	# print(defect_data[key]['N_site'])
# 	for defect in site.keys():
# 		if defect in key:
# 			defect_data[key]['N_site'] = site[defect]
# 			break

temp = {}
aliovalent = 'Br'
for defect in defect_data:
	split = re.split('_|-', defect)
	if '-' in defect:
		# if split[0] == aliovalent:
		# 	temp[defect] = defect_data[defect]
		pass
	else:
		# if split[0] == 'V':
		temp[defect] = defect_data[defect]
defect_data = temp
#print(defect_data)

compound = 'Li7PS6'
dependent_element = 'P'
temperature = 300.0
# dopant = None

dos_node = load_node(48514)
unitcell_node = load_node(48495)
vbm = get_vbm(unitcell_node)
Dos = dos_node.outputs.output_dos
dos_x = Dos.get_x()[1] - vbm
dos_y = Dos.get_y()[1][1]
#chem_potentials = {'Li': -1.923-195.514, 'P':-191.038, 'S':-0.835-326.678}
chem_potentials = {'Li': -1.923-195.514+np.array([-1.5,0,1.5]), 'P':-191.038*np.ones(3), 'S':-0.835-326.678*np.ones(3)}
#chem_potentials = {'Li': -1.923-195.514*np.ones((3,3)), 'P':-191.038*np.ones((3,3)), 'S':-0.835-326.678*np.ones((3,3))}
input_chem_shape = np.ones_like(chem_potentials['Li'])

#f = compute_net_charge(defect_data, chem_potentials, input_chem_shape, temperature, host_structure, dos_x, dos_y, band_gap, dopant=None)
#print(f(0.2*input_chem_shape))

inputs = {
            "defect_data": Dict(dict=defect_data),
            "chem_potentials": Dict(dict=chem_potentials),
            "temperature": Float(temperature),
            "valence_band_maximum": Float(vbm),
            "number_of_electrons": Float(unitcell_node.res.number_of_electrons),
            "unitcell": StructureData(pymatgen=host_structure),
            "DOS": Dos,
            "band_gap": Float(band_gap),
            #"dopant": Dict(dict={'X_1':{'c': 1E18, 'q':-1}})
        }
#print(inputs["defect_data"].get_dict())
#defect_data = inputs["defect_data"].get_dict()
#E_Fermi = 0.0
#E_defect_formation = {}
#for defect in defect_data.keys():
#    temp = defect_data[defect]
#    Ef = {}
#    for chg in temp['charge'].keys():
#        E_formation = temp['charge'][chg]['E']-temp['E_host']+chg*(E_Fermi+temp['vbm'])+temp['charge'][chg]['E_corr']
#        for spc in temp['species'].keys():
#            E_formation -= temp['species'][spc]*chem_potentials[spc]
#        Ef[chg] = E_formation
#    E_defect_formation[defect] = Ef
#print(E_defect_formation)

#defect_data = inputs["defect_data"].get_dict()
#chem_potentials =  inputs["chem_potentials"].get_dict()
#temperature = inputs["temperature"].value
#unitcell = inputs["unitcell"].get_pymatgen_structure()
#dos_x = dos_x.get_array('data')
#dos_y = dos_y.get_array('data')
#band_gap = inputs["band_gap"].value

f = compute_net_charge(defect_data, chem_potentials, input_chem_shape, temperature, host_structure, band_gap, dos_x, dos_y, dopant=None)
print(f(0.2*input_chem_shape))

#workchain_future = submit(FermiLevelWorkchain, **inputs)
#print('Submitted workchain with PK=' + str(workchain_future.pk))
