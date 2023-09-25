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

decimal.getcontext().prec = 6

class BrachyDose:
    r"""
    Purpse: 
        This class holds information regarding a dose distribution as well as the fundamental 
    functions that are applied on the dose. All the doses are J/Gy. 
    
    Attributes:
        grid:np.ndarray := 3D numpy array holding dose at each voxel. [z, y, x]
        uncertainty:np.ndarray := 3D numpy array holding dose uncertainity at each voxel. [z, y, x] 
        num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis. 
        vox_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters. 
        topleft:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z] 
        axis:np.ndarray := coorindates of grid points along z, y and x axis.  

    Functions:
    
    Dependencies: 
    
    """
    
    grid:np.ndarray
    uncertainty:np.ndarray
    num_voxels:np.ndarray
    vox_size:np.ndarray
    topleft:np.ndarray
    axis:np.ndarray

    # def __init__(self, ):
        
    #     return None       
    
    def load_file_to_BrachyDose(self, pth_dose_file:str):
        r""" 
        Purpose: 
            given the path to a file holding dose information, it will return 
        a BrachyDose object with the populated available attributes. It will give a warning
        for the missing attributes.
        
        Inputs:
            - pth_dose_file := path directory where the file containing the dose is. The file 
                extension could be ".3ddose", ".nrrd", ".dcm", or ".minidose"
        
        Output:
        self : BrachyDose
        """
        pth_dose_file = os.path.abspath(pth_dose_file)
        
        file_extension = os.path.splitext(pth_dose_file)[-1]
        
        if file_extension == ".3ddose":
            self.load_from_3ddose(pth_dose_file)

        elif file_extension == ".nrrd":
            self.load_from_nrrd(pth_dose_file)

        elif file_extension == ".dcm":
            assert "RD" in pth_dose_file, "must be a dicom dose file starting with 'RD'"
            raise Exception("loading dose from dicom is not currently supported")
        
        elif file_extension == ".bin":
            raise Exception("loading dose from .bin file is not currently supported")
    
        return self

    def load_from_3ddose(self, filename:str):
        r""" 
        Purpose: 
            Given the path to a 3ddose file, load its content into self:BrachyDose.
        
        Input:
            - filename := path to a ".3ddose" file
        """
        assert os.path.splitext(filename)[-1] == ".3ddose", "this file should have '3ddose' extension."
        path = filename
        #print("Opening 3ddose at %s" % path)
        with open(path, "rb") as newfile:
            bench_voxels = [int(i) for i in newfile.readline().split()]
            bench_x_pos = np.round(nparray(newfile.readline().split(), dtype=np.float32), decimals=6)
            bench_y_pos = np.round(nparray(newfile.readline().split(), dtype=np.float32), decimals=6)
            bench_z_pos = np.round(nparray(newfile.readline().split(), dtype=np.float32), decimals=6)

            bench_x_spacing = (bench_x_pos[1] - bench_x_pos[0])
            bench_y_spacing = (bench_y_pos[1] - bench_y_pos[0])
            bench_slice_thick = (bench_z_pos[1] - bench_z_pos[0])

            bench_dict = {}

            huge_dose_array = nparray(newfile.readline().strip().split(), dtype=np.float32)
            bench_dose = reshape(huge_dose_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))
            try:
                huge_uncert_array = nparray(newfile.readline().strip().split(), dtype=np.float32)
                bench_uncert = reshape(huge_uncert_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))
                self.uncertainty = bench_uncert
            except:
                print("Warning: No uncertainty in the 3ddose files")

            self.grid = bench_dose
            self.num_voxels = np.array(bench_voxels, dtype=np.float32)
            self.vox_size = np.array([bench_x_spacing, bench_y_spacing, bench_slice_thick], dtype=np.float32)
            self.topleft = np.array([bench_x_pos[0], bench_y_pos[0], bench_z_pos[0]], dtype=np.float32)
            # overriding axis calculation to ignore the axis contents of 3ddose and use the function below
            self.axis = self.calculateAxis() #np.array([bench_z_pos, bench_y_pos, bench_x_pos], dtype=object)
    
    def load_from_nrrd(self, pth_nrrd:str):
        r"""
        Purpose: 
            given the path to a nrrd dose file, it will load its content into self:BrachyDose
       
        Inputs: 
            - pth_nrrd := Path to a nrrd file writtern by self.to_nrrd()
            
        Dependencies:
            - SimpleITK
            - calculateAxis()
        """
        loaded_image_nrrd = sitk.ReadImage(pth_nrrd, imageIO='NrrdImageIO')
        [dose_array, uncertainty_array] = sitk.GetArrayFromImage(loaded_image_nrrd)
        dose_array = np.swapaxes(dose_array, 0, 2)
        uncertainty_array = np.swapaxes(uncertainty_array, 0, 2)

        self.uncertainty = uncertainty_array.astype(np.float32)
        self.grid = dose_array.astype(np.float32)
        self.num_voxels = np.array(np.flip((dose_array.shape), axis=0)).astype(np.float32)
        self.vox_size = np.array(loaded_image_nrrd.GetSpacing()[1:]).astype(np.float32)
        self.topleft = np.array(loaded_image_nrrd.GetOrigin()[1:]).astype(np.float32)
        self.axis = self.calculateAxis() 
        
    def load_from_npz(self, pth_npz):
        r""" 
        Purpose: 
            Given the path to an npz file, load its content into self:BrachyDose.
        
        Input:
            - filename := path to a ".npz" file
        """
    
        assert os.path.splitext(pth_npz)[-1]==".npz", "the file extension should be npz"
        
        loaded_BrachyDose = np.load(pth_npz, allow_pickle=True)
        self.uncertainty = loaded_BrachyDose["uncertainty"]
        self.grid =  loaded_BrachyDose["grid"]
        self.num_voxels =  loaded_BrachyDose["num_voxels"]
        self.vox_size =  loaded_BrachyDose["vox_size"]
        self.topleft =  loaded_BrachyDose["topleft"]
        self.axis =  loaded_BrachyDose["axis"]
    
    # def load_from_minidose(self, pth_minidose):
        
    
    def make_profile(self, depth:float, axis:str):
        """
        Purpose: 
            Plots a profile at a given depth (z coordinate) inside a 3ddose file.
        """
        num_x, num_y, num_z = self.num_voxels
        x_size, y_size, z_size = self.vox_size
        topleft_x, topleft_y, topleft_z = self.topleft
        depth_voxel = (depth - topleft_z) / z_size
        if axis == "x":
            off_axis_values = [topleft_x + (i + 0.5) * x_size for i in range(num_x)]
            mid_y = num_y / 2
            dose_values = [self.grid[depth_voxel][mid_y][i] for i in range(num_x)]
        elif axis == "y":
            off_axis_values = [topleft_y + (i + 0.5) * y_size for i in range(num_y)]
            mid_x = num_x / 2
            dose_values = [self.grid[depth_voxel][i][mid_x] for i in range(num_y)]
        else:
            raise("Only x or y axes are recognized")

        profile_dict = {}
        # Here, x and y axis refers to the axes on a graph, not
        # the dose axes.
        profile_dict["x_axis"] = off_axis_values
        profile_dict["y_axis"] = dose_values
        return profile_dict

    def make_pdd(self):
        r"""
        Purpose:
            Documentation is missing
        """
        mid_x, mid_y, mid_z = [int(vox/2) for vox in self.num_voxels]
        x_size, y_size, z_size = self.vox_size
        z_values = [(i + 0.5) * z_size for i in range(self.num_voxels[2])]
        dose_values = [self.grid[i][mid_y][mid_x] for i in range(self.num_voxels[2])]

        pdd_dict = {}
        if self.uncertainty is not None:
            uncert_values = [self.uncert[i][mid_y][mid_x] / 2.0 for i in range(self.num_voxels[2])]
            pdd_dict["uncert"] = uncert_values

        pdd_dict["x_axis"] = z_values
        pdd_dict["y_axis"] = nparray(dose_values)
        return pdd_dict
    
    def get_average_uncert(self) -> float:
        r"""
        Purpose:
            Documentation is missing
        """
        max_dose = self.grid.max()
        dose_mask = self.grid < 0.2 * max_dose
        masked_uncert = ma.array(self.uncert, mask=dose_mask)
        masked_dose = ma.array(self.grid, mask=dose_mask)
        average_uncert = ma.average(masked_uncert / masked_dose) * 100
        return average_uncert

    def get_average_uncert_benchmark(self) -> float:
        r"""
        Purpose:
            Documentation is missing
        """
        max_dose = self.grid.max()
        dose_mask = self.grid < 0.2 * max_dose
        masked_uncert = ma.array(self.uncert, mask=dose_mask)
        average_uncert = ma.average(masked_uncert) * 100
        return average_uncert
    
    def pad_3ddose(self, new_dims:list, new_topLeft:list):
        r''' a function to padd the grid and uncertainty in BrachyDose object and bring it to the desired dimensios.
        it will update all the aspects of the dose object to match the new dimensiosn.
        The voxels must have the same size! remember, python does z, y, x. 
        inputs:
            self:BrachyDose
            
            new_dims := a 1 by 3 list containing the new x, y and z dimensions:
                [new_z_dim, new_y_dim, new_x_dim]

            new_topLeft := coordinates of the new topleft
                [x, y, z]
        '''
        assert any(new_dims > self.grid.shape), "since you are padding, the new dimensions should be larger than the input dimensions"
        
        # calculate distances between the new and old topleft voxels. 
        # if for an axis, the distance of toplefts is larger than the voxel size, use the new topleft
        # else, use the old top left
        topleft_distance = np.abs(new_topLeft - self.topleft)
        final_topleft = np.zeros(3)
        for i, distance in zip(range(3), topleft_distance):
            final_topleft[i] = new_topLeft[i] if distance > self.vox_size[i] else self.topleft[i]

        # figure out how much padding to do before and after each axis
        padding = np.zeros([3,2])
        for i in range(3):
            if final_topleft[i] == self.topleft[i]:
                # all padding goes to the end for this dose axis
                pad_before = 0
                pad_after = new_dims[2-i] - self.grid.shape[2-i]
            else:
                # all padding goes to the begining of the dose axis
                pad_before = new_dims[2-i] - self.grid.shape[2-i] 
                pad_after = 0
            padding[2-i] = [pad_before, pad_after]

        # pad the old dose grid to get the new grid!
        new_dose_grid = np.pad(self.grid, tuple(padding.astype(int)), mode='edge')
        if self.uncertainty is not None:
                new_uncert = np.pad(self.uncertainty, tuple(padding.astype(int)), mode='edge')

        # figure out the end coordinates based on the padding
        # self.vox_size is a list of x, y and z spacing, we want it to be
        # a numpy array of z, y, x spacings. 
        voxel_size = np.array(self.vox_size)[:, np.newaxis][::-1]
        end_coords_distances =  padding * np.array([[-1, 1], [-1, 1], [-1, 1]]) * voxel_size
        
        old_end_coords = np.array(
            [[self.axis[0][0],self.axis[0][-1]], 
            [self.axis[1][0],self.axis[1][-1]], 
            [self.axis[2][0],self.axis[2][-1]]])

        new_end_coords = old_end_coords + end_coords_distances

        # now padd the new axis with respect to the appropriate begin and end coordinates
        new_axis = np.array([np.zeros(new_dims[0]), np.zeros(new_dims[1]), np.zeros(new_dims[2])], dtype=object)
        
        # pad the new axis with linear ramp
        for i in range(new_axis.shape[0]):
            new_axis[i] = np.pad(self.axis[i], tuple(padding[i].astype(int)), mode='linear_ramp', end_values=new_end_coords[i])

        # fillout the new padded dose dictionary
        padded_dose = BrachyDose()
        
        padded_dose.grid = new_dose_grid 
        padded_dose.uncert = new_uncert if self.uncertainty is not None else None 
        # voxel size remains unchanged
        padded_dose.vox_size = self.vox_size 
        padded_dose.topleft = final_topleft 
        padded_dose.axis = new_axis
        
        return padded_dose
    
    def write_to_3ddose_file(self, fileName:str):
        r''' 
        Purpose: 
            This function will write the contents of a BrachyDose onto a text file with .3ddose extension. 
        
        inputs:
            - self := a BrachyDose object containing the following keys:
                grid [z, y, x]
                uncert [z, y, x] 
                vox_size [x, y, z]
                topleft [x, y, z]
                axis [z, y, x]

            - fileName := the directory path where the file will be written
        '''   
        fileName = os.path.abspath(fileName)

        dimensions = ' '.join(map(str, self.num_voxels.astype(int))) + '\n'
        x_axis = ' '.join(map(str, self.axis[2])) + '\n'
        y_axis = ' '.join(map(str, self.axis[1])) + '\n'
        z_axis = ' '.join(map(str, self.axis[0])) + '\n'
        dose_flattened = ' '.join(map(str, self.grid.flatten('C'))) + '\n'
        if self.uncertainty is not None:
            uncertainty_flattened = ' '.join(map(str, self.uncertainty.flatten('C'))) + '\n'
        else:
            uncertainty_flattened = ''
            
        with open(fileName, 'w') as file:
            lines = [dimensions, x_axis, y_axis, z_axis, dose_flattened, uncertainty_flattened]
            file.writelines(lines)
    
    def write_to_nrrd_file(self, fileName:str, metaData:Optional[dict]=None):
        r"""
            Purpose: 
                To save the contents of BrachyDose into a nrrd file. 
            inputs:
                - fileName := path where the dose nrrd file will be written to. 

                - metaData := a dictionary containing the following meta data key values (should be changed later):
                    "cancer site": 
                    "care center": 
                    "number of dwell positions": 
                    "number of segmented structures": 
                    "patient number": 
                    "Image content": "[3D dose, 3D uncertainty]"
            outputs: Void
                writes [3D dose, 3D uncertainty], voxel size, origin (topleft), and metaData to the fileName_dose.nrrd
                note that 3D dose files are written in z, y, x, but the sitk image is written in x, y, z. 
        """
        # create sitk dose image
        dose_nda = np.swapaxes(self.grid, 0, 2).astype(np.float32)
        uncertainty_nda = np.swapaxes(self.uncertainty, 0, 2).astype(np.float32)
        
        image_nrrd = sitk.JoinSeries(
            sitk.GetImageFromArray(dose_nda),
            sitk.GetImageFromArray(uncertainty_nda)
        )
        image_nrrd.SetOrigin(np.append([0],self.topleft))
        image_nrrd.SetSpacing(np.append([1],self.vox_size))
        
        # set the metadata: all sitk Images belonging to a patient will have the same meta data
        if metaData is not None:
            for key in metaData:
                image_nrrd.SetMetaData(key, metaData[key])

        # write out the files
        fileName_ospth = os.path.abspath(fileName)
        assert os.path.exists(os.path.dirname(fileName_ospth)), f"the input folder does not exist: {os.path.dirname(fileName_ospth)}"
        
        run_number = fileName_ospth.split(".")[0]

        sitk.WriteImage(image_nrrd, run_number+".nrrd", useCompression=True, compressionLevel=9)

    def write_to_npz_file(self, fileName:str):
        r"""
            Purpose: 
                To save the contents of BrachyDose into a npz file, which is numpy compressed. 
            
            inputs:
                - self := BrachyDose object
                - fileName := path where the dose npz file will be written to. 
                
            outputs: Void
                writes the contents of self:BrachyDose to the fileName. 
        """
        
        assert os.path.splitext(fileName)[-1] == ".npz"
        
        np.savez_compressed(
            file=fileName, 
            grid=self.grid,
            uncertainty=self.uncertainty,
            num_voxels=self.num_voxels, 
            vox_size=self.vox_size,
            topleft = self.topleft,
            axis=self.axis,
            )
            
    def write_to_minidose_file(self, fileName, compress_program:Optional[str]=None):
        r"""
            Purpose: 
                To save the contents of BrachyDose into a minidose file, which is just a binary file written line by line. 
            
            inputs:
                - self := BrachyDose object
                - fileName := path where the dose minidose file will be written to. 
                
            outputs: Void
                writes the contents of self:BrachyDose to the fileName. 
        """
        
        contents = bytes("", 'utf-8')
        for attribute in dir(self):
            if attribute.startswith('__') or callable(getattr(self, attribute)):
                continue
            else:
                # print(attribute)
                contents = contents + getattr(self, attribute).tobytes() + bytes('\n', 'utf-8')
                # print("breaking point was here")
                # if attribute == 'grid' or 'uncertainty':
                #     minidose_file.write(getattr(self, attribute).flatten('C').tobytes()+bytes('\n', 'utf-8'))
                #     continue
                
                # minidose_file.write(getattr(self, attribute).tobytes()+bytes('\n', 'utf-8'))
        
        if compress_program == "zstd":
            contents = pyzstd.compress(contents, 22)
        
        with open(fileName, "wb") as minidose_file:
            minidose_file.write(contents)
            
        minidose_file.close()
    
    def write_to_xz_file(self, fileName):
        assert os.path.splitext(fileName)[-1] == '.xz'
        
        with lzma.open(fileName, 'wb') as file:
            pickle.dump(self, file)
    
    def write_to_zstd_file(self, fileName):
        assert os.path.splitext(fileName)[-1] == '.zst'
        
        with pyzstd.open(fileName, "wb", level_or_option=22) as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)         
      
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
    
    def is_equal(self, new_brachyDose):
        r"""
        Purpose:
            To compare if self:BrachyDose has the same attributes as an input BrachyDose
        
        Inputs:
            - new_brachyDose: another BrachyDose object whose attributes may or may not contain equal info as the attributes of self. 
        
        Outputs:
            True if attributes of new_brachyDose are the same as self
            False otherwise
        """
        assert isinstance(new_brachyDose, BrachyDose), "input must be of type BrachyDose"
        assert np.array_equal(self.grid, new_brachyDose.grid), "dose grid is not the same" 
        assert np.array_equal(np.concatenate(self.axis), np.concatenate(new_brachyDose.axis)), "axis is not the same"
        assert np.array_equal(self.uncertainty, new_brachyDose.uncertainty), "uncertainty is not the same"
        assert np.array_equal(self.num_voxels, new_brachyDose.num_voxels), "num_voxels is not the same"
        assert np.array_equal(self.vox_size, new_brachyDose.vox_size), "vox_size is not the same"
        assert np.array_equal(self.topleft, new_brachyDose.topleft), "topleft is not the same"
        # assert np.array_equal(np.round(self.vox_size, 2), np.round(new_brachyDose.vox_size, 2)), "vox_size is not the same"
        # assert np.array_equal(np.round(self.topleft, 2), np.round(new_brachyDose.topleft), 2), "topleft is not the same"
        
        
        return np.array_equal(self.grid, new_brachyDose.grid) \
            and np.array_equal(np.concatenate(self.axis), np.concatenate(new_brachyDose.axis)) \
            and np.array_equal(self.uncertainty, new_brachyDose.uncertainty) \
            and np.array_equal(self.num_voxels, new_brachyDose.num_voxels) \
            and np.array_equal(self.vox_size, new_brachyDose.vox_size) \
            and np.array_equal(self.topleft, new_brachyDose.topleft)
            # and np.array_equal(np.round(self.vox_size, 2), np.round(new_brachyDose.vox_size, 2)) \
            # and np.array_equal(np.round(self.topleft, 2), np.round(new_brachyDose.topleft), 2)
            # np.array_equal(np.round(np.concatenate(self.axis), 2), np.concatenate(new_brachyDose.axis)) \

    def crop_by_coordinates(self, coord_range:np.array, inplace:Optional[bool]=True):
        r"""
        Purpose: 
            given a range of coordinate (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly. 
        Inputs:
            - self: BrachyDose object
            - coord_range := a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]] 
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range. 
                it will also update the num_voxels, topleft and axis. only vox_size will not change
        """
        self.assert_BrachyDose_notEmpty()
        
        # make sure the ending coordinate of the new range is larger than its origin 
        for ax in coord_range:
            assert ax[1] > ax[0], "axis ending coordinate should be larger than its begining coordinate"

        # assert coordinate range falls into the image
        axes_name = ["x axis(2)", "y axis(1)", "z axis(0)"]
        coords_flipped = np.flip(coord_range, 0) 
        for i in range(self.axis.shape[0]):
            assert coords_flipped[i][0] > self.axis[i][0], f"cropping lower limit must be larger than the lowest coordinate on {axes_name[i]}"

        # convert new coordinates to indicies for both beggingn and ending (may not be exact)
        new_origin_distance = coord_range[:, 0] - self.topleft

        new_origin_index = np.floor(new_origin_distance / self.vox_size).astype(int)

        new_ending_index = np.floor((coord_range[:, 1] - self.topleft) / self.vox_size).astype(int)

        new_index_range = np.column_stack([new_origin_index, new_ending_index])

        return self.crop_by_index(new_index_range, inplace)

    def crop_by_index(self, index_range:np.array, inplace:Optional[bool]=True):

        new_origin_index = index_range[:, 0].astype(int)
        assert np.all(new_origin_index >= 0), "new origin index cannot be negative, please report this bug"

        new_ending_index = index_range[:, 1].astype(int)
        assert np.all(new_ending_index >= 0), "new ending index cannot be negative, please report this bug"

        # update the attributes 
        if inplace:
            self.grid = self.grid[
                new_origin_index[2]:new_ending_index[2], # z
                new_origin_index[1]:new_ending_index[1], # y
                new_origin_index[0]:new_ending_index[0] # x
            ]          
            self.uncertainty = self.uncertainty[
                new_origin_index[2]:new_ending_index[2], # z
                new_origin_index[1]:new_ending_index[1], # y
                new_origin_index[0]:new_ending_index[0] # x
            ]    
            self.topleft = np.array([
                self.axis[2][new_origin_index[0]], # x
                self.axis[1][new_origin_index[1]], # y
                self.axis[0][new_origin_index[2]] # z
            ])
            self.num_voxels = np.flip(self.grid.shape, 0)    
            self.axis = self.calculateAxis()      
        else:
            new_dose_obj = BrachyDose()
            new_dose_obj.grid = self.grid[
                new_origin_index[2]:new_ending_index[2],
                new_origin_index[1]:new_ending_index[1],
                new_origin_index[0]:new_ending_index[0]
            ]          
            new_dose_obj.uncertainty = self.uncertainty[
                new_origin_index[2]:new_ending_index[2],
                new_origin_index[1]:new_ending_index[1],
                new_origin_index[0]:new_ending_index[0]
            ]    
            new_dose_obj.topleft = np.array([
                self.axis[2][new_origin_index[0]], # x
                self.axis[1][new_origin_index[1]], # y
                self.axis[0][new_origin_index[2]] # z
            ])         
            new_dose_obj.num_voxels = np.flip(self.grid.shape, 0) 
            new_dose_obj.vox_size = self.vox_size   
            new_dose_obj.axis = self.calculateAxis()
            return new_dose_obj

    def assert_BrachyDose_notEmpty(self):
        assert self.grid is not None, "error grid is None"  
        assert self.uncertainty is not None, "error uncertainty is None"
        assert self.num_voxels is not None, "error num_voxels is None"
        assert self.vox_size is not None, "error vox_size is None"
        assert self.topleft is not None, "error topleft is None"
        assert self.axis is not None, "error axis is None"
    
    def info(self):
        self.assert_BrachyDose_notEmpty()
        print(f"shape of dose grid is: {self.grid.shape}")
        print(f"shape of uncertainty matrix is: {self.uncertainty.shape}")
        print(f"num voxels attribute is: {self.num_voxels}")
        print(f"the top left (bottom left in reality) is {self.topleft}")
        print(f"the voxel size is {self.vox_size}")
        print(f"the size of the z, y and x axes are {self.axis[0].shape, self.axis[1].shape, self.axis[2].shape}")
        print(f"the range of the z axis is {self.axis[0][0], self.axis[0][-1]}")
        print(f"the range of the y axis is {self.axis[1][0], self.axis[1][-1]}")
        print(f"the range of the x axis is {self.axis[2][0], self.axis[2][-1]}")
             
app = typer.Typer()

@app.command()
def convert_many_files(input_dir: str, type_in: str, type_out: str):
    r"""
    Purpose:
        Will convert all files in the "input_dir" of type "type_in" to "type_out"
    Inputs:
        input_dir := directory where there are files to be converted 
        type_in := could be ".3ddose", ".nrrd", ".minidose", other types could be added
        type_out := could be ".3ddose", ".nrrd", ".minidose", other types could be added
    """
    input_dir = os.path.abspath(input_dir)
    assert os.path.exists(input_dir)
    file_list = glob(input_dir+"/*"+type_in)
    
    for file in tqdm(file_list):
        dose_obj = BrachyDose()
        dose_obj.load_file_to_BrachyDose(file)
        
        file_base_noExtension = os.path.splitext(file)[0]
        
        if type_out == ".3ddose":
            dose_obj.write_to_3ddose_file(file_base_noExtension+type_out)
        elif type_out == ".nrrd":
            dose_obj.write_to_nrrd_file(file_base_noExtension+type_out)
        elif type_out == ".minidose":
            dose_obj.write_to_minidose_file(file_base_noExtension+type_out)
        elif type_out == ".xz":
            dose_obj.write_to_xz_file(file_base_noExtension+type_out)
        elif type_out == ".npz":
            dose_obj.write_to_npz_file(file_base_noExtension+type_out)
        elif type_out == ".zstd":
            dose_obj.write_to_zstd_file(file_base_noExtension+type_out)

@app.command()
def padd_many_files(input_dir: str, type_in: str, dim_out:str):
    r"""
    Purpose:
        Will padd all files in the "input_dir" of type "type_in" with zeros to
            have the dimensions "dim_out"
    Inputs:
        input_dir := directory where there are files to be converted 
        type_in := could be ".3ddose", ".nrrd", ".minidose", other types could be added
        dim_out := the new dimensions in [z, y, x] format
    """
    raise Exception("This feature is not implementated yet")

def load_pmc_dose(filename):
    return load_3ddose(filename)

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

def pad_many_3ddoses(input_dir_3ddose_folder:str, output_dir_3ddose_folder:str, new_dims:list, new_topLeft:list):
    r'''Given a directory full of 3ddose maps, this function will padd them all to a user defined size. 
    inputs:
        dir_3ddose_folder := the directory of the many 3ddose files

        output_dir_3ddose_folder := the directory where each padded 3ddose file will be saved
        
        new_dims := a 1 by 3 list containing the new x, y and z dimensions:
            [new_z_dim, new_y_dim, new_x_dim]

        new_topLeft := coordinates of the new topleft
            [x, y, z]
    '''

    files = glob(input_dir_3ddose_folder+'*.3ddose')

    for file in files:
        file_name = file.split('/')[-1]
        dose_dict = load_3ddose(file)
        padded_dose_dict = pad_3ddose(dose_dict, new_dims, new_topLeft)
        write_3ddose(output_dir_3ddose_folder+file_name, padded_dose_dict)


def compare_two_3ddose_files(pth1_3ddose:str, pth2_3ddose:str):
    # old_file_dir = load_3ddose(pth1_3ddose)
    # new_file_dir = load_3ddose(pth2_3ddose)
    
    with open(pth1_3ddose, 'r') as file1, open(pth2_3ddose) as file2:
        contents1 = file1.read()
        contents2 = file2.read()

    if contents1 == contents2:
        print("write 3ddose works fine")
    else:
        print("write 3ddose does not work fine")
        print('here are the differences')
        diff_list = list(difflib.ndiff(contents1.splitlines(), contents2.splitlines()))
        print('\n'.join(diff_list))


def test_load_from_3ddose():
    # pth_3ddose =  "../data_test/run_1_old.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_from_3ddose(pth_3ddose)
    dose_obj.assert_BrachyDose_notEmpty(dose_obj)

def test_load_file_to_brachyDose():
    # pth_3ddose =  "../data_test/run_1_old.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
        
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    dose_obj.assert_BrachyDose_notEmpty(dose_obj)
# @pytest.mark.passed
def test_write_to_3ddose_file():
    # pth_3ddose =  "../data_test/run_1_old.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'_test.3ddose'
    
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)

    dose_obj.write_to_3ddose_file(pth_out)
    new_dose_obj = BrachyDose().load_file_to_BrachyDose(pth_out)
    dose_obj.is_equal(new_dose_obj)
# @pytest.mark.passed
def test_convert_to_nrrd():
    r"""
    Purpose: 
        simulatenously test write_to_nrrd() and load_from_nrrd()
    """
    # 3 mm resolution
    # pth_3ddose =  "../data_test/run_1_old.3ddose"
    # pth_nrrd = "../data_test/run_1_old.nrrd"
# 
    # 1 mm resolution
    # pth_3ddose =  "../data_test/combined.3ddose"
    # pth_nrrd = "../data_test/combined_old.nrrd"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.nrrd'
    
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
   
    dose_obj.write_to_nrrd_file(pth_out)
    
    dose_obj_From_nrrd = BrachyDose()
    dose_obj_From_nrrd.load_file_to_BrachyDose(pth_out)
    
    dose_obj.is_equal(dose_obj_From_nrrd)

def test_convert_to_npz_file():
    r"""
    Purpose: 
        simulatenously test write_to_npz_file() and load_from_npz()
    """
    # pth_3ddose =  "../data_test/combined.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.npz'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    
    dose_obj.write_to_npz_file(pth_out)
    
    new_dose_obj = BrachyDose()
    new_dose_obj.load_from_npz(pth_out)
    dose_obj.is_equal(new_dose_obj)

def test_write_to_minidose_file():
    r"""
    Purpose: 
        simulatenously test write_to_minidose_file() and load_from_minidose()
    """
    # pth_3ddose =  "../data_test/combined.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.minidose'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    
    dose_obj.write_to_minidose_file(pth_out, compress_program='zstd')
    
    new_dose_obj = BrachyDose()
    # new_dose_obj.load_from_minidose(pth_out)
    # dose_obj.is_equal(new_dose_obj)

def test_write_to_xz_file():
    
    # pth_3ddose =  "../data_test/combined.3ddose"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.xz'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    
    dose_obj.write_to_xz_file(pth_xz)

def test_write_to_zstd_file():
    
    # pth_3ddose =  "../data_test/combined.3ddose"
    # pth_zstd = "../data_test/combined.zst"

    # testing on maud's file
    pth_3ddose = "../data_test/maud.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.zstd'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    
    dose_obj.write_to_zstd_file(pth_zstd)

def test_convert_many_files():
    dir_in = "../data_test/many_files"
    type_in = ".3ddose"
    type_out = ".nrrd"
    
    convert_many_files(dir_in, type_in, type_out)
    
    dir_in = os.path.abspath(dir_in)
    nrrd_list = glob(dir_in+".nrrd")
    
    for file_nrrd in nrrd_list:
        dose_obj_nrrd = BrachyDose()
        dose_obj_nrrd.load_file_to_BrachyDose(file_nrrd)
        
        file_3ddose = os.path.splitext(file_nrrd)[0]+".3ddose"
        dose_obj_3ddose = BrachyDose()
        dose_obj_3ddose.load_file_to_BrachyDose(file_3ddose)
        
        dose_obj_3ddose.is_equal(dose_obj_nrrd)
    
def test_crop_by_coordinates():
    pth_3ddose = "../data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    dose_obj.info()

    coords=np.array([
        [-14, 8],
        [3, 15],
        [-115, -100]],dtype=np.float32)
    
    dose_obj.crop_by_coordinates(coords)
    dose_obj.info()

def test_crop_by_index():
    pth_3ddose = "../data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)
    dose_obj.info()

    index=np.array([
        [30, 90],
        [30, 90],
        [0, 94]],dtype=np.float32)
    
    dose_obj.crop_by_index(index)
    dose_obj.info()

if __name__ == "__main__":
    
    # app()

    # a Test for the following functions
    # test_load_from_3ddose()
    # test_load_file_to_brachyDose()
    # test_write_to_3ddose_file()
    # test_convert_to_nrrd()
    # test_convert_to_npz_file()
    # test_write_to_minidose_file()
    # test_write_to_xz_file()
    # test_write_to_zstd()
    # test_convert_many_files()
    # test_crop_by_coordinates()
    test_crop_by_index()
    # _test_pad_3ddose()
    # _test_write_3ddose()
    # _test_pad_many_3ddoses()
    # _test_write_nrrd()
    # _test_nrrd_to_3ddose()