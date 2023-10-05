from numpy import array as nparray, zeros as npzeros, reshape
# from numpy import float as float
# from numpy import int as int
from numpy import ma
from numpy import dtype
import numpy as np
import re
import os

# from dicompylercore import dicomparser
from glob import glob
# from numericalunits import cm, mm, kg, J
# Gy = J/kg

import SimpleITK as sitk
import difflib
from typing import Optional
from collections.abc import Iterable

import pytest
# import uu
import lzma
import pickle
import pyzstd

import typer
import decimal

from tqdm import tqdm

# from rt_utils import RTStructBuilder
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
import pydicom

import json

class BrachyEgsphant:
    r"""
    Purpose:
        An object to allow for loading and manipulating the .egsphant files
        
    Attributes:
        - material_matrix:np.ndarray
        - density_matrix:np.ndarray
        - num_materials:int := the number of different material composition options a voxel has
        - material_dict:dict := a dictionary containing the name of the elements for each voxel
            and their number coding
        - num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis. 
        - vox_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters. 
        - topleft:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z] 
        - axis:np.ndarray := coorindates of grid points along z, y and x axis.  
    
    Functions:
        - load_file_to_BrachyEgsphant()
        - load_from_ctegsphant()
        - load_from_nrrd()
        - calculate_axis()
        - write_to_ctegsphant()
        - write_to_nrrd()
        - crop_by_index()
        - crop_by_body_contour()
        - assert_BrachyEgsphant_notEmpty()
        - info()
        - is_equal()
    
    Dependencies:
        numpy
        re
        os
        glob
        SimpleITK
        difflib
        typing
        collections
        pytest
        lzma
        pickle
        pyzstd
        typer
        tqdm
        DicomRTTool
        pydicom
        json
    """

    material_matrix:np.ndarray
    density_matrix:np.ndarray
    num_materials:int
    material_dict:dict
    num_voxels:np.ndarray
    vox_size:np.ndarray
    topleft:np.ndarray
    axis:np.ndarray
    _sanity_axis:np.ndarray
    
    def __init__(self, pth_egsphant_file:Optional[str]=None):
        if pth_egsphant_file is not None:
            self.load_file_to_BrachyEgsphant(pth_egsphant_file)
        else:
            self.material_matrix:np.ndarray = None
            self.density_matrix:np.ndarray = None
            self.num_materials:int = None
            self.material_dict:dict = {}
            self.num_voxels:np.ndarray = None
            self.vox_size:np.ndarray = None
            self.topleft:np.ndarray = None
            self.axis:np.ndarray = None
            self._sanity_axis:np.ndarray = None
    
    def load_file_to_BrachyEgsphant(self, pth_egsphant_file):
        pth_egsphant_file = os.path.abspath(pth_egsphant_file)
        
        assert os.path.exists(pth_egsphant_file), "The target egsphant file does not exist!"
        file_extension = os.path.splitext(pth_egsphant_file)[-1]
        
        if file_extension == ".egsphant":
            self.load_from_ctegsphant(pth_egsphant_file)
        elif file_extension == ".nrrd":
            self.load_from_nrrd(pth_egsphant_file)
        else:
            raise Exception(f"Loading from file extension {file_extension} is not supported!")

    def load_from_ctegsphant(self, pth_file:str):
        r"""
        Purpose: 
            to load a file with extension .egsphant into a BrachyEgsphant object
        Input:
            - pth_file := directory path to the .egsphant file
        """
        assert os.path.splitext(pth_file)[-1] == ".egsphant", "target file does not have .egsphant extension"
        
        with open(pth_file, "r") as egsphant:
            # first line describes how many materials are used
            self.num_materials = int(egsphant.readline().strip())
            
            # load each material line by line
            for i in range(self.num_materials):
                self.material_dict[egsphant.readline().strip()] = i
            
            egsphant.readline()
            
            # load number of voxels
            self.num_voxels = np.array([int(i) for i in egsphant.readline().strip().split()])
            
            # load the axis grid points
            self._sanity_axis = np.array([
                np.array([float(x) for x in egsphant.readline().strip().split()], dtype=np.float32),
                np.array([float(y) for y in egsphant.readline().strip().split()], dtype=np.float32),
                np.array([float(z) for z in egsphant.readline().strip().split()], dtype=np.float32)
                ], dtype=object)
            self._sanity_axis = np.flip(self._sanity_axis, axis=0)
            
            self.topleft = np.array(
                [
                    self._sanity_axis[2][0],
                    self._sanity_axis[1][0],
                    self._sanity_axis[0][0]
                ], dtype=np.float32)
            
            self.vox_size = np.array(
                [
                    self._sanity_axis[2][1] - self._sanity_axis[2][0],
                    self._sanity_axis[1][1] - self._sanity_axis[1][0],
                    self._sanity_axis[0][1] - self._sanity_axis[0][0]
                ]
            )
            
            self.axis = self.calculateAxis()
            assert np.isclose(np.concatenate(self.axis), np.concatenate(self._sanity_axis), rtol=1e-3).all(), "axis is not the same"
 
            # prepare empty matricies to hold material and density images
            self.material_matrix = npzeros((self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=int)
            self.density_matrix = npzeros((self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.float32)

            # load the material composition data in to the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.material_matrix[k][j] = list(egsphant.readline().strip())
                egsphant.readline()
                
            # load the density data into the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.density_matrix[k][j] = egsphant.readline().strip().split()
                egsphant.readline()
    
    def calculateAxis(self):
        r"""
        Purpose: will calculate the axies coordinates for a 3ddose dictionary.
        Input: 
            - dose := output of load_3ddose(). it should have the following keys and values:
                {"grid":,
                "topleft":,
                "vox_size":}
        Output: 
            - axes:numpy.array() := 
            [[z_min:vox_size:z_max],
            [y_min:vox_size:y_max],
            [x_min:vox_size:x_max]] 
        """
        # calculate the end point of axis in 3D space
        axes_end = np.array(
            self.topleft +  self.num_voxels* self.vox_size + self.vox_size # one voxel size is added because np.arange stops at an index before the end  
        )
        axes = np.empty(len(axes_end), dtype=object)
        for i in range(len(axes_end)):
            axes[i] = np.arange(self.topleft[len(axes_end)-1-i], axes_end[len(axes_end)-1-i], self.vox_size[len(axes_end)-1-i], dtype=np.float32)
        
        return axes
    
    def write_to_ctegsphant(self, fileName:str):
        r''' 
        Purpose: 
            This function will write the contents of a BrachyEgsphant onto a text file with .egsphant extension. 
        
        inputs:
            - self := a BrachyEgsphant object containing the following keys:
                num_materials:int
                material_dict:dict
                num_voxels:np.ndarray       [x, y, z]
                vox_size:np.ndarray         #Not Written
                topleft:np.ndarray          #Not Written
                axis:np.ndarray             [z, y, x] -> [x, y, z]
                material_matrix:np.ndarray  [z, y, x] -> [x, y, z]
                density_matrix:np.ndarray   [z, y, x] -> [x, y, z]
                
            - fileName := the directory path where the file will be written
        '''   
        fileName = os.path.abspath(fileName)
        num_materials = str(self.num_materials) + '\n'
        materials = "\n".join(self.material_dict.keys()) + "\n"
        spacing = "0 0 0 0 0 0 0 0 0\n"
        dimensions = ' '.join(map(str, self.num_voxels.astype(int))) + '\n'
        x_axis = ' '.join(map(str, np.round(self.axis[2], decimals=3))) + '\n'
        y_axis = ' '.join(map(str, np.round(self.axis[1], decimals=3))) + '\n'
        z_axis = ' '.join(map(str, np.round(self.axis[0], decimals=3))) + '\n'
        material_matrix = _to_single_string(self.material_matrix.astype(str))
        density_matrix = _to_single_string(self.density_matrix.astype(str), " ")
            
        with open(fileName, 'w') as file:
            lines = [num_materials, materials, spacing, dimensions, x_axis, y_axis, z_axis, material_matrix, density_matrix]
            file.writelines(lines)
    
        
    
    def is_equal(self, new_BrachyEgsphant):
        r"""
        Purpose:
            To compare if self:BrachyDose has the same attributes as an input BrachyDose
        
        Inputs:
            - new_brachyDose: another BrachyDose object whose attributes may or may not contain equal info as the attributes of self. 
        
        Outputs:
            True if attributes of new_brachyDose are the same as self
            False otherwise
        """
        assert isinstance(new_BrachyEgsphant, BrachyEgsphant), "input must be of type BrachyDose"
        assert np.array_equal(self.material_matrix, new_BrachyEgsphant.material_matrix), "material matrix is not the same" 
        assert np.array_equal(self.density_matrix, new_BrachyEgsphant.density_matrix), "density matrix is not the same" 
        assert np.isclose(np.concatenate(self.axis), np.concatenate(new_BrachyEgsphant.axis), rtol=1e-3).all(), "axis is not the same"
        assert np.array_equal(self.num_materials, new_BrachyEgsphant.num_materials), "number of materials is not the same"
        assert self.material_dict == new_BrachyEgsphant.material_dict, "the material dictionary is not the same"
        assert np.array_equal(self.num_voxels, new_BrachyEgsphant.num_voxels), "num_voxels is not the same"
        assert np.array_equal(self.vox_size, new_BrachyEgsphant.vox_size), "vox_size is not the same"
        assert np.isclose(self.topleft, new_BrachyEgsphant.topleft, rtol=1e-3).all(), "topleft is not the same"
        
        
        return np.array_equal(self.material_matrix, new_BrachyEgsphant.material_matrix) \
            and np.array_equal(self.density_matrix, new_BrachyEgsphant.density_matrix) \
            and np.isclose(np.concatenate(self.axis), np.concatenate(new_BrachyEgsphant.axis), rtol=1e-3).all() \
            and np.array_equal(self.num_materials, new_BrachyEgsphant.num_materials) \
            and self.material_dict == new_BrachyEgsphant.material_dict \
            and np.array_equal(self.num_voxels, new_BrachyEgsphant.num_voxels) \
            and np.array_equal(self.vox_size, new_BrachyEgsphant.vox_size) \
            and np.array_equal(self.topleft, new_BrachyEgsphant.topleft)
      
         
    def assert_BrachyEgsphant_notEmpty(self):
        r"""
        Purpose:
            to see which field of a brachyEgsphant object is empty
        """
        assert self.material_matrix is not None, "error: material_matrix is None"
        assert self.density_matrix is not None, "error: density_matrix is None"
        assert self.num_materials is not None, "error: num_materials is None"
        assert self.material_dict is not None, "error: material_dict is None"
        assert self.num_voxels is not None, "error: num_voxels is None"
        assert self.vox_size is not None, "error: vox_size is None"
        assert self.topleft is not None, "error: topleft is None"
        assert self.axis is not None, "error: axis is None"
        
def _to_single_string(matrix:np.ndarray, deliminator:Optional[str]=""):
    r"""
    Purpose:
        given a 3D matrix with string entries, this function concatenates all the
            entries into a single string to be written to the file. 
            "\n" is added at the end of each row and 
    
    Input:
        matrix := 3D ndarray full of string enteries
        deliminator := the string text inbetween the enteries. 
    Output:
        a single string containing all the enteries with added \n at the end of each row of 
            matrix and an addiation \n added to each slide in the input matrix
     
    """
    matrix_single_string = []
    for slide in matrix:
        slide_single_string = []
        for row in slide:
            slide_single_string.append(
                deliminator.join(row) + "\n"
                # "".join(np.pad(row, (0,1), mode='constant', constant_values=("\n")))
            )
        matrix_single_string.append(deliminator.join(slide_single_string)+"\n")
        
    return "".join(matrix_single_string)           
    
def load_egsphant(filename):
    phant = {}
    with open(filename, "r") as egsphant:
        num_media = int(egsphant.readline().strip())
        phant["media"] = []
        for i in range(num_media):
            phant["media"].append(egsphant.readline().strip())

        # dummy line
        egsphant.readline()

        phant["num_voxels"] = [int(i) for i in egsphant.readline().strip().split()]
        phant["x_voxels"] = [float(x) for x in egsphant.readline().strip().split()]
        phant["y_voxels"] = [float(y) for y in egsphant.readline().strip().split()]
        phant["z_voxels"] = [float(z) for z in egsphant.readline().strip().split()]

        phant["mat_matrix"] = npzeros((phant["num_voxels"][2], phant["num_voxels"][1], phant["num_voxels"][0]), dtype=np.int)
        phant["density_matrix"] = npzeros((phant["num_voxels"][2], phant["num_voxels"][1], phant["num_voxels"][0]), dtype=np.float32)

        for k in range(phant["num_voxels"][2]):
            for j in range(phant["num_voxels"][1]):
                phant["mat_matrix"][k][j] = list(egsphant.readline().strip())
            egsphant.readline()

        for k in range(phant["num_voxels"][2]):
            for j in range(phant["num_voxels"][1]):
                phant["density_matrix"][k][j] = egsphant.readline().strip().split()
            egsphant.readline()

    return phant

def test_write_to_egsphant():
    pth_input = "../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()
    
    egsphant_obj.write_to_ctegsphant(pth_output)
    new_egsphant_obj = BrachyEgsphant()
    new_egsphant_obj.load_from_ctegsphant(pth_output)
    
    egsphant_obj.is_equal(new_egsphant_obj)

def test_to_single_string():
    pth_input = "../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()
    
    _to_single_string(egsphant_obj.material_matrix.astype(str))     

def test_load_from_ctegsphant():
    pth_input = "../data_test/glen_prostate_p1_3mm_ct.egsphant"
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

if __name__=="__main__":
    
    # running tests top is the latest test written
    # test_to_single_string()
    test_write_to_egsphant()
    # test_load_from_ctegsphant()