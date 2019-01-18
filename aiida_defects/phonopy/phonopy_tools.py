# coding: utf-8


import sys
import os
import argparse
import pymatgen
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import cos, sin, radians
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm import DataFactory
from aiida.orm.data.singlefile import SinglefileData
from aiida.work.workfunction import workfunction
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida.orm import DataFactory




######################################################################
#This module contains:						     #
#1) create_supercells_with_displacements_inline			     #
#2) get_force_constants_inline					     #
#3) get_path_using_seekpath					     #
#4) phonon_band_structure  					     #
#5) create_modulations						     #
#6) modulations_inspection 					     #
#7) rescale_amplitude						     #
#8) modulations_optimal						     #
#9) quartic_poly						     #
#10) quadratic_poly						     #
#11) poly_fit							     # 
#12) find_min							     #
#13) r_sqrt_calc						     #
######################################################################

@workfunction
def create_supercells_with_displacements_inline(**kwargs):
    """
    Uses phonopy as a python module to create the StructureData objects
    corresponding to the dispaced supercells
    Adapted from wf_phonopy.py available at
    https://github.com/abelcarreras/aiida_extensions/tree/master/workflows
    Note:
    If there are site with sites a name different from the symbol, the function will 
    still be able to create the PhonopyAtoms object but the final StructureData objects
    containing the supercells will lost the information relative to the names
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    

    # Generate phonopy phonon object
    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
    positions=[site.position for site in structure.sites]
    cell=structure.cell
                
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)
    
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])

  

    phonon.generate_displacements(distance=phonopy_input['distance'])

    cells_with_disp = phonon.get_supercells_with_displacements()

    # Transform cells to StructureData and set them ready to return
    disp_cells = {}
    for i, phonopy_supercell in enumerate(cells_with_disp):
        supercell = StructureData(cell=phonopy_supercell.get_cell())
        for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                    phonopy_supercell.get_positions()):
            supercell.append_atom(position=position, symbols=symbol)
        disp_cells["structure_{}".format(i)] = supercell

    return disp_cells




@workfunction
def get_force_constants_inline(**kwargs):
    """
    Uses phonopy as a python module to compute the force_constants and force_set and stores them as
    AiiDA ArrayData objects.
    Adapted from wf_phonopy.py available at
    https://github.com/abelcarreras/aiida_extensions/tree/master/workflows
    Note:
    If there are site with sites a name different from the symbol, the function will 
    still be able to create the PhonopyAtoms object but the final StructureData objects
    containing the supercells will lost the information relative to the names
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    
    # Generate phonopy phonon object
    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
    
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])

    # Build data_sets from forces of supercells with displacments
    data_sets = phonon.get_displacement_dataset()
    
    for i, first_atoms in enumerate(data_sets['first_atoms']): 
        first_atoms['forces'] = kwargs.pop('force_{}'.format(i)).get_array('forces')[-1] 


    # Calculate and get force constants
    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()

    # force_constants = phonon.get_force_constants().tolist()
    force_constants = phonon.get_force_constants()
    

    # Set force sets and force constants array to return
    data = ArrayData()
    data.set_array('force_sets', np.array(data_sets))
    data.set_array('force_constants', force_constants)
    return {'phonopy_output': data}




def get_path_using_seekpath(structure, band_resolution=30):
    """
    Uses seekpath to create the q-point path for the phonon band structure calculation through phonopy
    """
    import seekpath

    cell = structure.cell
    positions = [site.position for site in structure.sites]
    scaled_positions = np.dot(positions, np.linalg.inv(cell))
    numbers = np.unique([site.kind_name for site in structure.sites], return_inverse=True)[1]
    structure2 = (cell, scaled_positions, numbers)
    path_data = seekpath.get_path(structure2)

    labels = path_data['point_coords']
    
    band_ranges = []
    for set in path_data['path']:
        band_ranges.append([labels[set[0]], labels[set[1]]])

    bands =[]
    for q_start, q_end in band_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return {'ranges': bands,
            'labels': path_data['path']}



@workfunction
def phonon_band_structure(**kwargs):
    """
    Return the phonon band structure. 
    :param: structure (StructureData) and force_constants (ArrayData) should be present in the phonopy_input
    dictionary we give as input
    :returns band_structure: ArrayData containing the phonon band structure.
    An image file with the phonon band structure is also saved in the folder were you will launch the script
    with the pk of the structure fro which the phono calculation is perfomed as name.
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.phonon.modulation import Modulation
    from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
    from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
    from phonopy.phonon.degeneracy import get_eigenvectors
    import collections

    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')
    
    bands = get_path_using_seekpath(structure)

    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
                
    #Generate phonopy phonon object
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])
        
    #Setting the force constants and extracting the dynamical matrix and its derivative    
    phonon.set_force_constants(force_constants)
        
    #Get band structure    
    phonon.set_band_structure(bands['ranges'])
    band_structure_phonopy = phonon.get_band_structure()
    q_points = np.array(band_structure_phonopy[0])
    q_path = np.array(band_structure_phonopy[1])
    frequencies = np.array(band_structure_phonopy[2])
    band_labels = np.array(bands['labels'])
    
    # stores band structure
    band_structure = ArrayData()
    band_structure.set_array('q_points', q_points)
    band_structure.set_array('q_path', q_path)
    band_structure.set_array('frequencies', frequencies)
    band_structure.set_array('labels', band_labels)
    
    #Setting a list of the q-points to use in order to print them as tick labels along the axis
    #of the phonon band structure plot, starting from the seekpath information
    last_label = "Nopointsyet"
    q_labels =[]
    for key in bands['labels']:
        if  last_label[0] != key[0]:
            q_labels.append(key[0])
            last_label = key
        elif last_label[0] == key[0]:
            q_labels.append(last_label[1]+"|{}".format(key[0]))
            last_label = key
    q_labels.append(key[1])
    q_labels = ["$\Gamma$" if _ == "GAMMA" else _ for _ in q_labels ]
    
    phonon.plot_band_structure(symbols=q_labels).savefig(str(structure.pk)+'.pdf',
                                                        dpi=300, transparent=True, orientation='landscape') 
                                                     
    return band_structure


@workfunction
def create_anime(**kwargs):
    """
    Creates an anime.ascii file readable with v_symm
    :parameters inline_params: dictionary containing the structure, and the phonopy_input.
                                phonopy_input should contain the q point in which we are interested and the
                                force_constants already computed
    :result anime: SinglefileData object in which in the DB is stored the anime.ascii file
                   This file can be found in the current directory with as name the PK of the structure 
                   originating it (PK.ascii)
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.phonon.modulation import Modulation
    from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
    from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
    import os
    cwd = os.getcwd()
    
    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')

    # Generate phonopy phonon object
    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
         
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])
        
    #Setting the force constants and extracting the dynamical matrix and its derivative    
    phonon.set_force_constants(force_constants)
    #dm = phonon.get_dynamical_matrix()
    #ddm = DerivativeOfDynamicalMatrix(dm)
    
    #Setting  q point
    q = phonopy_input['modulation']['q_point']
    
    phonon.write_animation(q,
                        anime_type='v_sim',
                        band_index=None,
                        amplitude=None,
                        num_div=None,
                        shift=None,
                        filename=str(structure.pk)+'.ascii')
    
    anime = SinglefileData()
    anime.set_file((cwd+'/'+str(structure.pk)+'.ascii'))

    return anime




#Generate Modulation object
def create_modulations(dm,dimension,phonon_modes):
    """
    Adjustment of the Modulation object class of phonopy
    To be used only in the workfunction modulations_inspection
    and modulations_optimal
    """
    from phonopy.phonon.modulation import Modulation
    modulation= Modulation(dm,
                            dimension,
                            phonon_modes,
                            )
    modulation.run()
    mod_and_sc=modulation.get_modulations_and_supercell()

    #Generate a modulated structure for each mode specified
    deltas =[]
    modulated_cells = {}
    for i, u in enumerate(mod_and_sc[0]):
        phonopy_supercell = modulation._get_cell_with_modulation(u)
        #supercell = StructureData(cell=phonopy_supercell.get_cell())
        #for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
        #                        phonopy_supercell.get_positions()):
        #    supercell.append_atom(position=position, symbols=symbol)
        #modulated_cells["modulated_{}".format(i)] = supercell
        deltas.append(u)

    #Generate a modulated structure by overlapping the modulation of the different modes
    sum_of_deltas = np.sum(deltas, axis=0)

    phonopy_supercell = modulation._get_cell_with_modulation(sum_of_deltas)
    supercell = StructureData(cell=phonopy_supercell.get_cell())
    for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                phonopy_supercell.get_positions()):
        supercell.append_atom(position=position, symbols=symbol)
    modulated_cells["modulated_overlap"] = supercell
    
    #Generate a structure without  modulations, but the dimension is the one that is specified
    no_modulations = np.zeros(sum_of_deltas.shape, dtype=complex)
    phonopy_supercell = modulation._get_cell_with_modulation(no_modulations)
    supercell = StructureData(cell=phonopy_supercell.get_cell())
    for symbol, position in zip(phonopy_supercell.get_chemical_symbols(),
                                phonopy_supercell.get_positions()):
        supercell.append_atom(position=position, symbols=symbol)
    modulated_cells["modulated_orig"] = supercell

    #modulation.write()
    return modulated_cells


@workfunction
def modulations_inspection(**kwargs):
    """
    Identifies the immaginary eigenvalues and create modulations to deform the structure in the direction
    indicated by the instabilities. It works for non degenerate eigenvalues and for doubly degenerate phonon modes.
    For non degenerate modes, it will create modulated structures with different amplitudes according to ampl_max, 
    ampl_min, and ampl_incr values that should be specified in the phonopy_input['modulation']['amplitude'] dictionary.
    For doubly degenerate modes the amplitude range is explored only for the highest  symmetry linear combinations
    U =ampl( u1*cos(angle)+u2*sin(angle)), that are identified by scanning the angle between 0 and 90 degrees and 
    and the amplitude in a range that goes from 0 to the a rescaled amplitude value which provides an highest
    displacement of 0.11 Angstrom and checking the symmetry of the resulting modulations. A small symmetry tolerance is used 
    (default 5e-3) and its value is updated until 3 different symmetries are finally found. 
    (For a doubly degenerate mode, there are only three possible high symmetry direction in order parameter space)
    :return modulated structure: return a dictionary of StructureData objects containing the modulated structure
    for each instable mode. If degeneracy higher than 2 is observed, a Bool(False) value is added in the dictionary
    for the degenerate mode.
    
    KNOWN BUGS:
    - at the moment the function is not able to distinguish between 4 degenerate mode (which we are not able
    to treat) and two doubly degerate modes (that we can treat)
    - the dimension of the supercell used for the band structure and for the modulation should be the same
    - change q when the ModulationWorkChain will be finished so that we can loop over all the q point with
      unstable modes
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.phonon.modulation import Modulation
    from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
    from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
    from phonopy.phonon.degeneracy import get_eigenvectors
    import collections
    from math import cos, sin, radians
    
    structureb = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')

    # Generate phonopy phonon object
    symbols=[]
    for site in structureb.sites:
        for kind in structureb.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
         
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structureb.sites],
                        cell=structureb.cell)
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])
        
    #Setting the force constants and extracting the dynamical matrix and its derivative    
    phonon.set_force_constants(force_constants)
    dm = phonon.get_dynamical_matrix()
    ddm = DerivativeOfDynamicalMatrix(dm)
    
    #Setting  q point
    q = phonopy_input['modulation']['q_point']
    
    #Calculating eigeinvalues and eigenvectors at q
    eigvals, eigvecs = get_eigenvectors(
                q,
                dm,
                ddm,
                perturbation=None,
                derivative_order=None,
                nac_q_direction=None)
    
    eignvals=eigvals.tolist()
    

    #Identifying degenerate immaginary eigenvalues
    E_thr = phonopy_input['modulation']['E_thr']
    E_thr = "%."+str(E_thr)+"f"
    degenerates={}
    dups = collections.defaultdict(list)
    
    for i, e in enumerate(eignvals):
        dups[ E_thr % e].append(i)
    
    for k, v in sorted(dups.iteritems()):
        if len(v) >= 2 and float(k)  < 0:
            degenerates[str(k)]=v
    
    degens=[]        
    for eigval, indexes in degenerates.iteritems():
        degens.append(indexes)
    deg_band_index = [item for sublist in degens for item in sublist]
    
    #Identifiyng the band indexes corresponding to non degenerate immaginary eigenvalues:
    ndeg_band_indexes=[]
    for i, value in enumerate(eignvals):
        if  float("%.5f" % value) <0 and i not in deg_band_index:
            ndeg_band_indexes.append(i)
                

    #Extracting the [axbxc] dimension list from the supercell matrix given as input to compute the dm
    dimension=[]
    dimension.append(phonopy_input['modulation']['supercell'][0][0])
    dimension.append(phonopy_input['modulation']['supercell'][1][1])
    dimension.append(phonopy_input['modulation']['supercell'][2][2])
    
    #Extracting the [axbxc] dimension list from the supercell matrix given as input to create modulated structure
    dimension_m=[]
    dimension_m.append(phonopy_input['modulation']['supercell'][0][0])
    dimension_m.append(phonopy_input['modulation']['supercell'][1][1])
    dimension_m.append(phonopy_input['modulation']['supercell'][2][2])
    
    #Creating a list with all the ampliutude values to scan
    ampl_min = phonopy_input['modulation']['amplitude'][0]
    ampl_max = phonopy_input['modulation']['amplitude'][1]
    ampl_incr = phonopy_input['modulation']['amplitude'][2]
    amplitudes = np.arange(ampl_min,ampl_max+ampl_incr,ampl_incr).tolist()
    
    #Instatiating an empty dictionary which will contain all the modulated structures
    modulated_structures = {}
    
    ################################
    # NON DEGENERATE MODES         #
    ################################
    #Setting modulation inputs for every non degenerate mode


    phonon_modes=[]        
    if len(ndeg_band_indexes) > 0:
        for index in ndeg_band_indexes:
            for ampl in amplitudes:
                phonon_modes.append([q,
                                 index,
                                 ampl,
                                 phonopy_input['modulation']['phase']])
                

    
                non_degenerate = create_modulations(dm,
                                                dimension,
                                                phonon_modes)
                modulated_structures['nondeg_'+str(index)+'_ampl_'+str(ampl)] = non_degenerate['modulated_overlap']
    
    
    ################################
    # DEGENERATE MODES             #
    ################################
    if len(deg_band_index) == 3:
        del deg_band_index[2]
    
    #Looping over the angles and the amplitude range for the linear combination of doubly degenerate modes
    #U = u1*cos(angle)+u2*sin(angle). In order to reduce the number of point of the grid for the amplitudes
    #and angle scan we will first rescale the amplitude value so that the larger displacement is 0.11 Angstrom
    #subsequently in an angle range from 0 to 90 (which is sufficient to screen all the potential energy surface
    #along this coordinate) and in an amplitude range going from 0 to the rescaled amplitude we are going to analyse
    #the symmetry of every possible linear combination to identify the highest symmetry direction in order 
    #parameter space
    
    if len(deg_band_index) == 2:
        
        #Computing rescaled amplitude
        phonon_modes_d=[]
        ampl = 1
        angle = 0
        structures =[]
        phonon_modes_d = [[q,
                            deg_band_index[0],
                            ampl*cos(radians(angle)),
                            phonopy_input['modulation']['phase']],
                            [q,
                            deg_band_index[1],
                            ampl*sin(radians(angle)),
                            phonopy_input['modulation']['phase']]
                            ]

    
        degenerate = create_modulations(dm,
                                        dimension,
                                        phonon_modes_d)
        ref = degenerate["modulated_orig"]
        s = degenerate['modulated_overlap']
        rescaled_ampl = rescale_amplitude(s, ref, ampl=1)
        
        
        
        #Creating modulations with angle values going from 0 to 90 and amplitudes  values going from 0
        #to the rescaled amplitude value
        phonon_modes_d=[]
        angles = range(0,91, 1)
        a_min = 0
        a_max = rescaled_ampl
        a_incr = a_max/(ampl_max/ampl_incr)
        ampls = np.arange(a_min,a_max+a_incr,a_incr).tolist()
        
    
        modulated_symmetry = []
        ms = []

        for angle in angles:
            for ampl in ampls:
                phonon_modes_d = [[q,
                       deg_band_index[0],
                       ampl*cos(radians(angle)),
                       phonopy_input['modulation']['phase']],
                      [q,
                       deg_band_index[1],
                       ampl*sin(radians(angle)),
                       phonopy_input['modulation']['phase']]
                     ]


                degenerate = create_modulations(dm,
                                dimension,
                                phonon_modes_d)
                structure = degenerate['modulated_overlap']
                ms.append(structure)
                
        #Checking the symmetry of the modulated structures obtained for the scan over angles and amplitudes
        #changing the symmetry tolerance (default value 5e-3) until we find three space groups 
        #different from the space group of the non modulated structure, which corresponds to the three 
        #highest symmetry directions in order parameter space
        
        etol=5e-3
        for structure in ms:
            modulated_symmetry.append(get_spacegroup(structure,etol))


        if len(list(set(modulated_symmetry))) == 4:
            order_parameter_space = True
            #Printint the scan plot
            x = np.array(ampls)
            y = np.array(angles)
            z = np.array(modulated_symmetry)
            N1=len(x)
            N2=len(y)
            Z = z.reshape(N2,N1)
            plt.imshow(Z, aspect=0.1)
            plt.ylabel(ur'Angle(degrees)', fontsize=14)#$^\circ$
            plt.xlabel(ur'Mode Amplitude (A)', fontsize=14)
            plt.title('symprec '+str(etol))
            plt.colorbar()
            plt.savefig('Deg_Modulations_Scan_'+str(structureb.pk)+'.pdf')
            plt.show()

        else:
            step = 0
            max_step = 10
            order_parameter_space = False

            while (not order_parameter_space and step < max_step):

                if len(list(set(modulated_symmetry))) < 4:
                    modulated_symmetry = []
                    etol = etol*0.1
                    for structure in ms:
                        modulated_symmetry.append(get_spacegroup(structure,etol))
                    #Printing the scan plot
                    x = np.array(ampls)
                    y = np.array(angles)
                    z = np.array(modulated_symmetry)
                    N1=len(x)
                    N2=len(y)
                    Z = z.reshape(N2,N1)
                    plt.imshow(Z, aspect=0.1)
                    plt.ylabel(ur'Angle(degrees)', fontsize=14)
                    plt.xlabel(ur'Mode Amplitude (A)', fontsize=14)
                    plt.title('symprec '+str(etol))
                    plt.colorbar()
                    plt.savefig('Deg_Modulations_Scan_'+str(structureb.pk)+'.pdf')
                    plt.show()

                    step += 1
                    order_parameter_space = False


                if len(list(set(modulated_symmetry))) > 4:
                    modulated_symmetry= []
                    etol = etol*10
                    for structure in ms:
                        modulated_symmetry.append(get_spacegroup(structure,etol))
                    #Printing the scan plot
                    x = np.array(ampls)
                    y = np.array(angles)
                    z = np.array(modulated_symmetry)
                    N1=len(x)
                    N2=len(y)
                    Z = z.reshape(N2,N1)
                    plt.imshow(Z, aspect=0.1)
                    plt.ylabel(ur'Angle(degrees)', fontsize=14)
                    plt.xlabel(ur'Mode Amplitude (A)', fontsize=14)
                    plt.title('symprec '+str(etol))
                    plt.colorbar()
                    plt.savefig('Deg_Modulations_Scan_'+str(structureb.pk)+'.pdf')
                    plt.show()

                    step += 1
                    order_parameter_space = False

                if len(list(set(modulated_symmetry))) == 4:
                    order_parameter_space = True
                    step = 11

        #Creating a dictionary with one entry for each of the three high symmetry space groups
        #each containing a list with all the angle values to which corresponds a modulation with
        #that symmetry
        sg = {}
        x = np.array(ampls)
        y = np.array(angles)
        z = np.array(modulated_symmetry)
        N1=len(x)
        N2=len(y)
        Z = z.reshape(N2,N1)
        bulk = Z[0][0]
        for angle in range(len(angles)):
            for ampl in range(len(ampls)):

                if str(Z[angle][ampl]) not in sg and Z[angle][ampl] != bulk:
                    sg[str(Z[angle][ampl])] = []
                    sg[str(Z[angle][ampl])].append(angles[angle])


                elif Z[angle][ampl] != bulk:
                    sg[str(Z[angle][ampl])].append(angles[angle])
        
        #Creating a dictionary with one entry for each high symmetry space group
        #to which correspond as value the average angle (order parameter direction)
        #at which that specific symmetry is observed. Since we scan the angle between 0 and 90 degrees
        # which is sufficient to explore all the potential energy surface, eventually applying PBC, 
        #for the space group containing the phase angle 0/90 degrees, PBC conditions are applied 
        #using negative angles in order to be able to identify the mean value for that symmetry.
        #Consider that when one space group is observed in more that one region with different angle range 
        #the region that will be selected to identify the order parametr direc tion corresponding to that
        #space group will be the one associated to the range with the lowest angle values.
        opd={}
        for label in sg:
            SG_tmp = list(set(sg[label]))
            SG_tmpr = SG_tmp[::-1]
            if SG_tmp[0] == 0:
                x0 = SG_tmp[0]
                tmp = [x0]
                tmp2 = []
                for x in range(1,len(SG_tmp)):
                    if SG_tmp[x] == x0+1:
                        tmp.append(SG_tmp[x])
                        x0 = SG_tmp[x]

                x1 = SG_tmpr[0]        
                for x in range(1,len(SG_tmpr)):
                    if SG_tmpr[x] == x1 -1:
                        tmp2.append(SG_tmpr[x])
                        x1=SG_tmpr[x]

                neg=[x-90 for x in tmp2]
                tot=tmp+neg
                opd[label]=np.mean(tot)
                SG_tmp = []
                SG_tmpr = []
                tmp = []
                tmp2 = []
            else:
                x0 = SG_tmp[0]
                tmp = [x0]
                tmp2 = []
                for x in range(1,len(SG_tmp)):
                    if SG_tmp[x] == x0+1:
                        tmp.append(SG_tmp[x])
                        x0 = SG_tmp[x]
                opd[label]=np.mean(tmp)
                SG_tmp = []
                tmp = []
        modulated_structures['angle_scan'] = ParameterData(dict=opd)
        ops = opd.values()
    #Creating modulation of the linear combination of doubly degenerate eigenvalues along
    #the highest symmetry order parameter directions
        phonon_modes_d = []
        for opd in ops:
            for ampl in amplitudes:
                phonon_modes_d = [[q,
                                   deg_band_index[0],
                                   ampl*cos(radians(opd)),
                                   phonopy_input['modulation']['phase']],
                                  [q,
                                   deg_band_index[1],
                                   ampl*sin(radians(opd)),
                                   phonopy_input['modulation']['phase']]
                                 ]
                degenerate = create_modulations(dm,
                                            dimension,
                                            phonon_modes_d)
                modulated_structures['DEG_'+str(deg_band_index)+'_ang_'+str(opd)+'_ampl_'+str(ampl)] = degenerate['modulated_overlap']
    
    elif  len(deg_band_index) > 2:
         modulated_structures['DEG_'+str(deg_band_index)]= Bool(False)
    
    
    return modulated_structures        


def rescale_amplitude(s, ref, ampl=1, ref_disp=0.11):
    """
    Rescales the amplitude of the modulation so that the higher displacement correspond to a reference value
    param s: modulated structure (StructureData)
    :param ref: non modulated structure (StructureData)
    :param ampl: amplitude used to generate the modulations
    :param ref_disp: reference value for the highest displacement
    :returns rescaled_ampl: amplitude rescaled to abtain a max displacement equal to ref_disp
    """
    from mpmath import nint

    
    #cell_x = ref.cell[0][0]
    #cell_y = ref.cell[1][1]
    #cell_z = ref.cell[2][2]
    
    cell = ref.cell
    cell_x=(sqrt(cell[0][0]**2 + cell[0][1]**2 +cell[0][2]**2))
    cell_y =(sqrt(cell[1][0]**2 + cell[1][1]**2 +cell[1][2]**2))
    cell_z=(sqrt(cell[2][0]**2 + cell[2][1]**2 +cell[2][2]**2))    

    max_disp = 0
        
    for i in range(len(ref.sites)):
        tmp = s.sites[i].position
        refe = ref.sites[i].position
        disp_x = tmp[0] -refe[0]
        disp_x = disp_x - nint(disp_x / cell_x) * cell_x

        disp_y = tmp[1] -refe[1]
        disp_y = disp_y - nint(disp_y / cell_y) * cell_y

            
        disp_z = tmp[2] -refe[2]
        disp_z = disp_z - nint(disp_z / cell_z) * cell_z


        disp_max = max(abs(disp_x), abs(disp_y), abs(disp_z))

        if disp_max > max_disp:
            max_disp = disp_max
                
        
    max_disp = float(max_disp)

    rescaled_ampl = (ref_disp*ampl)/max_disp
    return float(rescaled_ampl)


@workfunction
def modulations_optimal(**kwargs):
    """
    Workfunction to create modulated structure with the optimal amplitude selected after 
    running modulations_inspection  
    """
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    from phonopy import Phonopy
    from phonopy.phonon.modulation import Modulation
    from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
    from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
    from phonopy.phonon.degeneracy import get_eigenvectors
    import collections
    from math import cos, sin, radians
    
    structure = kwargs.pop('structure')
    phonopy_input = kwargs.pop('phonopy_input').get_dict()
    force_constants = kwargs.pop('force_constants').get_array('force_constants')
    phonon_modes = list(kwargs.pop('phonon_modes'))
    
    
    name = str(kwargs.pop('name'))

    # Generate phonopy phonon object
    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)
         
    bulk = PhonopyAtoms(symbols,
                        positions=[site.position for site in structure.sites],
                        cell=structure.cell)
    
    if 'primitive' in phonopy_input:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         primitive_matrix=phonopy_input['primitive'],
                         symprec=phonopy_input['symmetry_precision'])
    else:
        phonon = Phonopy(bulk,
                         phonopy_input['supercell'],
                         symprec=phonopy_input['symmetry_precision'])
        
    #Setting the force constants and extracting the dynamical matrix and its derivative    
    phonon.set_force_constants(force_constants)
    dm = phonon.get_dynamical_matrix()
    ddm = DerivativeOfDynamicalMatrix(dm)
    
    
    #Extracting the [axbxc] dimension list from the supercell matrix given as imput
    dimension=[]
    dimension.append(phonopy_input['modulation']['supercell'][0][0])
    dimension.append(phonopy_input['modulation']['supercell'][1][1])
    dimension.append(phonopy_input['modulation']['supercell'][2][2])
    
    modulated_structures = {}
    modulation = create_modulations(dm,
                                    dimension,
                                    phonon_modes)
    modulated_structures[str(name)] = modulation['modulated_overlap']
    
    return modulated_structures


def quartic_poly(x, a, b):
    return a *x**2 +b *x**4 

def quadratic_poly(x, a, b, c):
    return a *x**2 + b*x + c 

def poly_fit(xdata, ydata, func, name_file):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(func, xdata, ydata)
    
    max_xval = np.ceil(np.amax(xdata))
    values=np.linspace(0,max_xval,100)
    fig = plt.figure()
    plt.plot(xdata,ydata, 'o')
    plt.plot(values, func(values, *popt), 'r-', label='fit')
    
    plt.xlabel(ur'Mode Amplitude (A)', fontsize=16)#\u00c5
    plt.ylabel('Energy (eV)', fontsize=16)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.tight_layout()
    plt.show()#Remove
    fig.savefig(str(name_file)+'.pdf', dpi=300, transparent=True, orientation='landscape')
    return popt, pcov

def find_min(func, xdata, ydata):
    from scipy.optimize import fmin
    
    min_yval = np.amin(ydata)
    ylist = ydata.tolist()
    xlist = xdata.tolist()
    for i, val in enumerate(ylist):
        if val == min_yval:
            guessx = xlist[i]
            guessy = ylist[i]
            
    xmin = fmin(func,guessx)
    return float(xmin[0])

def r_sqrt_calc(xdata,ydata,func,*opt):
    """
    Computing R**2 as R**2 = 1 - RSS/TSS, where RSS=Sum((yi-func(xi))**2) and TSS = Sum((yi - average(y))**2
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    
    """
    average =np.mean(ydata)
    rss = np.sum((ydata-func(xdata,*opt))**2)
    tss = np.sum((ydata-average)**2)
    r_sqrt = 1- rss/tss
    
    
    return r_sqrt

