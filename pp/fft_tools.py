# coding: utf-8

import sys
import argparse
import pymatgen
import numpy as np
import matplotlib.pyplot as plt
from aiida.orm.data.upf import UpfData
from aiida.common.exceptions import NotExistent
from aiida.orm.data.upf import get_pseudos_from_structure
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm import DataFactory
from aiida.orm.data.singlefile import SinglefileData

from aiida.orm.code import Code
from aiida.orm import load_node

from aiida.work.workfunction import workfunction
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs, if_, append_
from aiida.orm.data.base import Float, Str, NumericType, BaseType, Int, Bool, List
from aiida_quantumespresso.calculations.pw import PwCalculation

###################################################
#This module contains:                            #
#1)read_grid(folder_data)                         #
#2)planar_average(Grid, structure, axis, npt=400) #
#3)differentiator(x,y)                            #
#4)trilinear_interpolation(Grid, structure)       #
#5)avg_potential_at_core(func)                    #
###################################################
@workfunction
def read_grid(folder_data):
    """
    Workfunction extracting the 3D-FFT grid after a PP calculation
    :param folder_data: FolderData object corresponding to the  folder produced by the PP calculation
                        containing the fileplot file
    :result grid: ArrayData object containing the 3D-FFT as an array with shape (nr1x,nr2x,nr3x)
                        
    """
    grid = folder_data.get_file_content('aiida.filplot')
    #print grid
    first_line = grid.splitlines()[1:2][0].strip().split(' ')
    first_line = [i for i in first_line if i != '']

    nr1x = int(first_line[0])
    nr2x = int(first_line[1])
    nr3x = int(first_line[2])
    nr1 = int(first_line[3])
    nr2 = int(first_line[4])
    nr3 = int(first_line[5])
    nat = int(first_line[6])
    ntyp = int(first_line[7])


    second_line = grid.splitlines()[2:3][0].strip().split(' ')
    second_line = [i for i in second_line if i != '']
    ibrav = int(second_line[0])


    if ibrav != 0:
        celldm1 = second_line[1]
        celldm2 = second_line[2]
        celldm3 = second_line[3]
        celldm4 = second_line[4]
        celldm5 = second_line[5]
        celldm6 = second_line[6]
    
        alat = celldm1
    
        lines_to_remove = 3 

    if ibrav == 0:
        alat =   second_line[1]
    
        third_line = grid.splitlines()[3:4][0].strip().split(' ')
        third_line = [i for i in third_line if i != '']
        a = [float(third_line[0]), float(third_line[1]), float(third_line[2])]
    
        fourth_line = grid.splitlines()[4:5][0].strip().split(' ')
        fourth_line = [i for i in fourth_line if i != '']
        b = [float(fourth_line[0]), float(fourth_line[1]), float(fourth_line[2])]
    
        fifth_line = grid.splitlines()[5:6][0].strip().split(' ')
        fifth_line = [i for i in fourth_line if i != '']
        c = [float(fifth_line[0]), float(fifth_line[1]), float(fifth_line[2])]
    
        lines_to_remove = 3 + nat + ntyp + 3 +1

    info = {'nr1x': nr1x,
            'nr2x': nr2x,
            'nr3x': nr3x,
            'nr1': nr1,
            'nr2': nr2,
            'nr3': nr3,
           }   
        
    #print lines_to_remove

    my_grid = []
    for line in grid.splitlines()[lines_to_remove:]:
        for element in line.strip().split(' '):
            if element != '':
                my_grid.append(float(element))
    #print my_list

    function = np.array(my_grid).reshape(nr1x,nr2x,nr3x)
    
    fft_grid = ArrayData()
    fft_grid.set_array('fft_grid_reshaped', function)
    fft_grid.set_array('fft_grid', np.array(my_grid))

    return {'fft_grid' : fft_grid, 'info' : ParameterData(dict=info)}






def planar_average(Grid, structure, axis, npt=400):
    """
    Computes the planar average in a given plane.
    See QE PP/src/average.f90
    :param Grid: node in the DB obtained applying the read_grid workfunction
    :param structure: StructureData object for which the Grid was computed
    :param axis: axis 
    :result average: planar_averaged potential in eV
    :result ax: values along the axis in Angstrom
    """
    from math import sqrt
    from scipy.interpolate import spline
    #Extracting the 3D-FFT grid and the dictionary with information on the shape of the grid
    grid = Grid['fft_grid'].get_array('fft_grid')
    info = Grid['info'].get_dict()
    
    nr1x = info['nr1x']
    nr2x = info['nr2x']
    nr3x = info['nr3x']
    nr1 = info['nr1']
    nr2 = info['nr2']
    nr3 = info['nr3']
    
    if nr1x != nr1 or nr2x != nr2 or nr3x != nr3:
        print "Thick and smooth mesh are different. Check result"
    
    #Extracting cell parameters
    cell=structure.cell
    
    #Calculating the planar average
    if axis == 'x':
        average = np.zeros(shape=(nr1))
        for i in range(nr1):
            for j in range(nr2):
                for k in range(nr3):
                    ir = i +(j-1)*nr1x +(k-1)*nr1x*nr2x
                    average[i] += grid[ir]
            average[i] = average[i] *13.6058 /(nr2*nr3)
        x =[]
        a=sqrt(cell[0][0]**2 + cell[0][1]**2 +cell[0][2]**2)
        deltax = a/nr1
        for i in range(nr1):
            x.append(i*deltax)
        ax = x
        if npt > nr1:
            axnew = np.linspace(min(ax),max(ax),npt) 
            average_smooth= spline(ax,average,axnew)
            ax = axnew
            average = average_smooth
    if axis == 'y':
        average = np.zeros(shape=(nr2))
        for j in range(nr2):
            for i in range(nr1):
                for k in range(nr3):
                    ir = i +(j-1)*nr1x +(k-1)*nr1x*nr2x
                    average[j] += grid[ir]
            average[j] = average[j] *13.6058/(nr1*nr3)
        y =[]
        b=sqrt(cell[1][0]**2 + cell[1][1]**2 +cell[1][2]**2)
        deltay = b/nr2
        for i in range(nr2):
            y.append(i*deltay)
        ax = y
        if npt > nr2:
            axnew = np.linspace(min(ax),max(ax),npt) 
            average_smooth= spline(ax,average,axnew)
            ax = axnew
            average = average_smooth
            
    if axis == 'z':
        average = np.zeros(shape=(nr3))
        for k in range(nr3):
            for j in range(nr2):
                for i in range(nr1):
                    ir = i +(j-1)*nr1x +(k-1)*nr1x*nr2x
                    average[k] += grid[ir]
            average[k] = average[k]*13.6058/(nr1*nr2)
        z =[]
        c=sqrt(cell[2][0]**2 + cell[2][1]**2 +cell[2][2]**2)
        deltaz = c/nr3
        for i in range(nr3):
            z.append(i*deltaz)
        ax = z
        if npt > nr3:
            axnew = np.linspace(min(ax),max(ax),npt) 
            average_smooth= spline(ax,average,axnew)
            ax = axnew
            average = average_smooth

    return {'average' : average,'ax': ax}

def differentiator(x,y):
    """
    First Derivative calculation
    :param x: numpy array with the independent variable
    :param y: numpa array with the depedent variable
    :return derivative: numpy array of the first derivative
    """
   
    return np.gradient(y)/np.gradient(x)


def trilinear_interpolation(Grid, structure):
    """
    Performs trilinear interpolation on the 3D-FFT grid in order to obtain the function value at a certain point
    :param Grid: node in the DB obtained applying the read_grid workfunction
    :param structure: StructureData object for which the Grid was computed
    :result func: dictionary with one entry for each atom (the label is the a string of the atom type
    +coordinates)
    """
    from scipy.interpolate import RegularGridInterpolator
    from numpy import linspace, zeros, array
    from math import sqrt
    from mpmath import nint
    
    #Extracting the 3D-FFT grid and the dictionary with information on the shape of the grid
    grid = Grid['fft_grid'].get_array('fft_grid_reshaped')
    info = Grid['info'].get_dict()
    
    nr1x = info['nr1x']
    nr2x = info['nr2x']
    nr3x = info['nr3x']
    nr1 = info['nr1']
    nr2 = info['nr2']
    nr3 = info['nr3']
   
    #Setting the points along the x, y, and z dimensions so that the unit in Angstrom
    cell = structure.cell
    x =[]
    a=sqrt(cell[0][0]**2 + cell[0][1]**2 +cell[0][2]**2)
    deltax = a/nr1
    for i in range(nr1):
        x.append(i*deltax)
    
    y =[]
    b=sqrt(cell[1][0]**2 + cell[1][1]**2 +cell[1][2]**2)
    deltay = b/nr2
    for i in range(nr2):
        y.append(i*deltay)    
    z =[]
    c=sqrt(cell[2][0]**2 + cell[2][1]**2 +cell[2][2]**2)
    deltaz = c/nr3
    for i in range(nr3):
        z.append(i*deltaz)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    #Initialization of the grid interpolation function 
    V = grid
    fn = RegularGridInterpolator((x,y,z), V)
    
    #Creating a dictionary containing
    atoms = {}
    symbols=[]
    for site in structure.sites:
        for kind in structure.kinds:
            if kind.name == site.kind_name:
                symbols.append(kind.symbol)

    
    for i, site in enumerate(structure.sites):
        x = site.position[0]
        x = x - nint(x / a) * a
        y = site.position[1]
        y = y - nint(y / b) * b
        z = site.position[2]
        z = z - nint(z / c) * c
        atoms[symbols[i]+'_'+str(site.position)] = np.array([x,y,z])
    
    func = {}
    for atom, position in atoms.iteritems():
        func[atom] = float(fn(position)[0])
    

    return {'func_at_core' : func, 'symbols' : symbols}


def avg_potential_at_core(func):
    """
    Computes the average potential per type of atom in the structure
    :param func: dictionary with potential at each core extracted from the 3D-FFT grid with trilinear_interpolation 
                and the list of symbols in the structure
    :result avg_atom_pot: average electrostatic potential for type of atom
    """
    
    potential = func['func_at_core']
    symbols = func['symbols']
    
    species = list(set(symbols))
    
    avg_pot_at_core = {}
    pot_at_core = []
    for specie in species:
        for atom, pot in potential.iteritems():
            if atom.split('_')[0] == specie:
                pot_at_core.append(pot) 
        avg_pot_at_core[str(specie)] = np.mean(pot_at_core)
        pot_at_core = []
    return avg_pot_at_core


