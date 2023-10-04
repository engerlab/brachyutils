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

class BrachyEgsphant():
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
    
    def __init__(self, pth_egsphant_file:str):
        if pth_to_load is not None:
            self.load_file_to_BrachyEgsphant(pth_egsphant_file) 
    
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
            
            # load number of voxels
            self.num_voxels = np.array([int(i) for i in egsphant.readline().strip().split()])
            
            # load the axis grid points
            self._sanity_axis = np.array([
                [float(x) for x in egsphant.readline().strip().split()],
                [float(y) for y in egsphant.readline().strip().split()],
                [float(z) for z in egsphant.readline().strip().split()]
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
            assert np.array_equal(np.concatenate(self.axis), np.concatenate(self._sanity_axis)), "axis is not the same"
 
            # prepare empty matricies to hold material and density images
            self.material_matrix = npzeros((self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.int)
            self.density_matrix = npzeros((self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=np.int)

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
