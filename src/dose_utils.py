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

    def __init__(self, ):
        return None       
    
    def load_file_to_BrachyDose(self, pth_dose_file:str):
        r""" 
        Purpose: 
            given the path to a file holding dose information, it will return 
        a BrachyDose object with the populated available attributes. It will give a warning
        for the missing attributes.
        
        Inputs:
            - pth_dose_file := path directory where the file containing the dose is. The file 
                extension could be ".3ddose", ".nrrd", ".dcm", or ".bin"
        
        Output:
        self : BrachyDose
        """
        pth_dose_file = os.path.abspath(pth_dose_file)
        
        file_extension = os.path.splitext(pth_dose_file)
        
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
            bench_x_pos = nparray(newfile.readline().split(), dtype=float)
            bench_y_pos = nparray(newfile.readline().split(), dtype=float)
            bench_z_pos = nparray(newfile.readline().split(), dtype=float)

            bench_x_spacing = (bench_x_pos[1] - bench_x_pos[0])
            bench_y_spacing = (bench_y_pos[1] - bench_y_pos[0])
            bench_slice_thick = (bench_z_pos[1] - bench_z_pos[0])

            bench_dict = {}

            huge_dose_array = nparray(newfile.readline().strip().split(), dtype=float)
            bench_dose = reshape(huge_dose_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))
            try:
                huge_uncert_array = nparray(newfile.readline().strip().split(), dtype=float)
                bench_uncert = reshape(huge_uncert_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))
                self.uncertainty = bench_uncert
            except:
                print("Warning: No uncertainty in the 3ddose files")

            self.grid = bench_dose
            self.num_voxels = bench_voxels
            self.vox_size = [bench_x_spacing, bench_y_spacing, bench_slice_thick]
            self.topleft = [bench_x_pos[0], bench_y_pos[0], bench_z_pos[0]]
            self.axis = np.array([bench_z_pos, bench_y_pos, bench_x_pos], dtype=object)
    
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

        self.uncertainty = uncertainty_array
        self.grid = dose_array
        self.num_voxels = np.array(dose_array.shape)
        self.vox_size = loaded_image_nrrd.GetSpacing()[1:]
        self.topleft = loaded_image_nrrd.GetOrigin()[1:]
        self.axis = self.calculateAxis(self) 
    
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

        dimensions = ' '.join(map(str, np.array(self.grid.shape[::-1]))) + '\n'
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
    
    def write_to_nrrd_file(self, fileName:str, metaData:Optional[dict]):
        r"""
            Purpose: 
                To save the contents of BrachyDose into a nrrd file. 
            inputs:
                - fileName := path where the dose nrrd file will be written to. 
                    _dose.nrrd will be added to the basename. 
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
        for key in metaData:
            image_nrrd.SetMetaData(key, metaData[key])

        # write out the files
        fileName_ospth = os.path.abspath(fileName)
        assert os.path.exists(os.path.dirname(fileName_ospth)), f"the input folder does not exist: {os.path.dirname(fileName_ospth)}"
        
        run_number = fileName_ospth.split(".")[0]

        sitk.WriteImage(image_nrrd, run_number+"_dose.nrrd", useCompression=True, compressionLevel=9)

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
        axes_end = np.array(
            self.topleft +  np.flip(np.array(self.grid.shape), axis=0)* self.vox_size
        )
        axes = np.empty(len(axes_end), dtype=object)
        for i in range(len(axes_end)):
            # flip axes to go from x,y,z to z,y,x:
            axes[i] = np.arange(self.topleft[len(axes_end)-1-i], axes_end[len(axes_end)-1-i], self.vox_size[len(axes_end)-1-i])
        
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

        return self.grid == new_brachyDose.grid and self.axis == new_brachyDose.axis \
            and self.uncertainty == new_brachyDose.uncertainty \
            and self.num_voxels == new_brachyDose.num_voxels \
            and self.vox_size == new_brachyDose.vox_size and self.topleft == new_brachyDose.topleft

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



# def _test_nrrd_to_3ddose():
#     # 1mm 
#     # pth_nrrd = "../test_data/combined_dose.nrrd"
#     # pth_3ddose = "../test_data/combined_fromNRRD.3ddose"
#     # pth_3ddose_groundtruth = "../test_data/combined.3ddose"

#     # 3mm 
#     pth_nrrd = "../test_data/run_1_dose.nrrd"
#     pth_3ddose = "../test_data/run_1_fromNRRD.3ddose"
#     pth_3ddose_groundtruth = "../test_data/run_1_old.3ddose"
    
#     nrrd_3ddose = nrrd_to_3ddose(pth_nrrd)
#     write_3ddose(pth_3ddose, nrrd_3ddose)
    
#     compare_two_3ddose_files(pth_3ddose_groundtruth, pth_3ddose)


# def _test_pad_3ddose():
    
#     # load the 3ddose file that is to be padded
#     old_3ddose = load_3ddose('/home/majd/data/Patient_Dose_Simulations/sebastien-breast/patient_230776/run_1.3ddose')

#     # here i just give some arbitrary numbers just for developement
#     # the new dimensions must be in the z, y, x format. 
#     # voxel size and topleft must be in x, y, z forma. 
#     new_dims = nparray([78, 167, 167])

#     new_topLeft = nparray([-249., -122., 23.]) * 0.1

#     padded_dose = pad_3ddose(dose=old_3ddose, new_dims=new_dims, new_topLeft=new_topLeft)

#     print(f"size of the new grid is {padded_dose['grid'].shape}")
#     print(f"the voxel size is {padded_dose['vox_size']}")
#     print(f"the new top left is {padded_dose['topleft']} \n",
#      f"and the old top left was {old_3ddose['topleft']}")
#     print(f"the size of the new axis is {padded_dose['axis'].shape}")

#     # load the dicom files
#     # path2Dicom = "/home/majd/data/Patient_Treatment _Plans/sebastien-breast/230776_Anon/"
#     # dicom_file_path = glob(path2Dicom+'CT2.dcm')[0]
#     # loaded_dicom = dicomparser.DicomParser(dicom_file_path)
#     # print(type(rt_dose))


# def _test_write_3ddose():
#     old_file_dir = '/home/majd/data/Patient_Dose_Simulations/sebastien-breast/patient_230776/run_1.3ddose'
#     old_3ddose = load_3ddose(old_file_dir)
#     new_file_dir = './test_run_1.3ddose'

#     write_3ddose(new_file_dir, old_3ddose)

#     new_3ddose = load_3ddose(new_file_dir)

#     # print(f"the difference between original dose and written dose {new_3ddose['grid']-old_3ddose['grid']}")

#     with open(old_file_dir, 'r') as file1, open(new_file_dir) as file2:
#         contents1 = file1.read()
#         contents2 = file2.read()

#     if contents1 == contents2:
#         print("write 3ddose works fine")
#     else:
#         print("write 3ddose does not work fine")
#         print('here are the differences')
#         diff_list = list(difflib.ndiff(contents1.splitlines(), contents2.splitlines()))
#         print('\n'.join(diff_list))


#     print('okay')

# def _test_pad_many_3ddoses():
#     input_dir = '/home/majd/data/Patient_Dose_Simulations/sebastien-breast/patient_230776/'
#     output_dir = '/home/majd/data/Patient_Dose_Simulations/sebastien-breast/padded/patient_230776/'
#     new_dims = np.array([78, 167, 167])
#     new_topLeft = np.array([-249.48828125, -122.48828125, 23.5]) * 0.1

#     pad_many_3ddoses(input_dir, output_dir, new_dims, new_topLeft)

# def _test_write_nrrd():
#     # 1mm resolution
#     # pth_3ddose = "../test_data/combined.3ddose"
#     # pth_toWrite_nrrd = "../test_data/combined.nrrd"
#     # pth_toLoad_nrrd = "../test_data/combined_dose.nrrd"
#     # 3 mm resolution
#     pth_3ddose = "../test_data/run_1_old.3ddose"
#     pth_toWrite_nrrd = "../test_data/run_1.nrrd"
#     pth_toLoad_nrrd = "../test_data/run_1_dose.nrrd"
#     # creat metadata dictionary
#     meta_dict = {
#         "cancer site": "prostate",
#         "care center": "muhc glen",
#         "number of dwell positions": "100",
#         "number of segmented structures": "4",
#         "patient number": "0",
#         "Image content": "[3D dose, 3D uncertainty]"
#     }

#     # load the 3ddos file
#     dose_3ddose = load_3ddose(pth_3ddose)

#     write_nrrd(pth_toWrite_nrrd, dose_3ddose, meta_dict)

#     loaded_image_nrrd = sitk.ReadImage(pth_toLoad_nrrd, imageIO='NrrdImageIO')
#     array = sitk.GetArrayFromImage(loaded_image_nrrd)
#     print(f"dimensions of the loaded image: {loaded_image_nrrd}")
#     reader = sitk.ImageFileReader()
#     reader.SetImageIO("NrrdImageIO")

def assert_BrachyDose_notEmpty(dose_obj:BrachyDose):
    assert dose_obj.grid is not None, "error in load_from_3ddose. grid is None"  
    assert dose_obj.uncertainty is not None, "error in load_from_3ddose. uncertainty is None"
    assert dose_obj.num_voxels is not None, "error in load_from_3ddose. num_voxels is None"
    assert dose_obj.vox_size is not None, "error in load_from_3ddose. vox_size is None"
    assert dose_obj.topleft is not None, "error in load_from_3ddose. topleft is None"
    assert dose_obj.axis is not None, "error in load_from_3ddose. axis is None"


def test_load_from_3ddose():
    pth_3ddose =  "../dose_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_from_3ddose(pth_3ddose)
    assert_BrachyDose_notEmpty(dose_obj)

def test_load_file_to_brachyDose():
    pth_3ddose =  "../dose_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)

def test_write_to_3ddose_file():
    pth_3ddose =  "../dose_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_BrachyDose(pth_3ddose)

    pth_new_3ddose = "../dose_test/run_1_new.3ddose"
    dose_obj.write_to_3ddose_file(pth_new_3ddose)
    new_dose_obj = BrachyDose().load_file_to_BrachyDose(pth_new_3ddose)
    assert dose_obj.is_equal(new_dose_obj)

# if __name__ == "__main__":

    # a Test for the following functions
    # test_load_from_3ddose()
    # _test_pad_3ddose()
    # _test_write_3ddose()
    # _test_pad_many_3ddoses()
    # _test_write_nrrd()
    # _test_nrrd_to_3ddose()