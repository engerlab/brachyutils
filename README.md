# BrachyUtils

This package implements Brachytherapy dose, egsphant dicom and film dosimetry functionalities. 

## Installation

To get the package run:

`git clone https://github.com/engerlab/brachyutils.git`

If you are installing this package on a remote cluster managed by the Digital Research Alliance of Canada (Compute Canada), you need to load some required modules:

`module load StdEnv/2020`

`module load python/3.9`

`module load opencv`

Then, create a virtual envionrment and activate it by running:

`python3 -m venv ENV_brachyutils`

`source ENV_brachyutils/bin/activate`

After this process finishes, run `pip install -e .` to install the brachyutils package. 

### Optional:

`python3 -m pip install --upgrade pip`

Install SimpleITK independently by running `python3 -m pip install SimpleITK`. If you run into the error saying `skbuild` is [missing](https://bugs.python.org/issue30573), run `python3 -m pip install cmake`, then try installing SimpleITK again.


## brachyutils commands

brachyutils comes with a linux command line interface. To learn about the commands that are available run `brachyutils --help` on the command line.

At the moment, the outputs looks like the following:

```
$ brachyutils --help
Usage: brachyutils [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified
                                  shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified
                                  shell, to copy it or customize the
                                  installation.
  --help                          Show this message and exit.

Commands:
  convert-dose-many-files         Will convert all files...
  crop-dose-by-bodycontour-many-files
                                  Purpose: to crop all the...
  crop-dose-by-ratio-many-files   Purpose: Will crop all...
  crop-egsphant-by-bodycontour-many-patients
                                  Purpose: to crop the...
  get-bodycontourrange-from-dicom-many-patients
                                  Purpose: to exract body...
  get-uncertainty-one-patient     Purpose: Will calculate...
  multiply-dose-by-constant-many-files
                                  Purpose: Will scale all...
  padd-dose-many-files            Purpose: Will padd all...
```

## BrachyDose

You can import this object in your python script by running `from brachyutils import BrachyDose`. This object has the following attributes and functions:

### Purpose

This class holds information regarding a dose distribution as well as the fundamental functions that are applied on the dose. All the doses are J/Gy. 

### Attributes:
- grid:np.ndarray := 3D numpy array holding dose at each voxel. [z, y, x]
- uncertainty:np.ndarray := 3D numpy array holding dose uncertainity at each voxel. [z, y, x] 
- num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis. 
- vox_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters. 
- topleft:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z] 
- voxel_edges:np.ndarray := coorindates of voxel edges along z, y and x axis.  

### Functions:
- load_file_to_brachydose()
- load_from_3ddose()
- load_from_nrrd()
- load_from_npz()
- make_profile()
- make_pdd()
- get_average_uncert()
- get_average_uncert_benchmark()
- pad_3ddose()
- write_to_3ddose()
- write_to_nrrd()
- write_to_npz()
- write_to_minidos()
- write_to_xz()
- write_to_zstd()
- calculate_voxel_edges()
- is_equal()
- crop_by_coordinates()
- crop_by_fraction()
- crop_by_index()
- is_not_empty()
- info()
- multiply_dose_by_constant()

## BrachyEgsphant

You can import this object in your python script by running `from brachyutils import BrachyEgsphant`. This object has the following attributes and functions:

### Purpose

An object to allow for loading and manipulating the .egsphant files
        
### Attributes:
- material_matrix:np.ndarray
- density_matrix:np.ndarray
- num_materials:int := the number of different material composition options a voxel has
- material_dict:dict := a dictionary containing the name of the elements for each voxel and their number coding
- num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis. 
- vox_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters. 
- topleft:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z] 
- axis:np.ndarray := coorindates of grid points along z, y and x axis.  
    
### Functions:
- load_file_to_BrachyEgsphant()     done
- load_from_ctegsphant()            done
- load_from_nrrd()                  not implmented
- calculate_axis()                  done
- write_to_ctegsphant()             done
- write_to_nrrd()                   not implemented
- crop_by_index()                   done
- crop_by_body_contour()            
- assert_BrachyEgsphant_notEmpty()  done
- info()                            done
- is_equal()                        done

## dicom_utils 

### Purpose

dicom_utils does not implement a class, but has functions that process masks on dicom images for cropping dose or Egsphant maps or calculating DVH metrics. Currently, this module holds two functions:  

- `get_structure_index_range` gives you the range of the indicies for each structure on the DICOM images as well as the dimensions of the dicom image.

- `get_structure_mask_From_dicom` gives you the mask of a structure in the dicom image. 

For more information, please advice the documentation in the source code. 

## BrachyPlan 

You can import this object in your python script by running `from brachyutils import BrachyPlan`. This object has the following attributes and functions:

### Purpose

This class holds the information regarding the brachytherapy treatment plan as well as all the functions to support the necessary plan operations. 

### Attributes:
- num_dwells:int := the number of dwell positions in the plan
- catheter_table:list := a list of catheter dictionaries. each catheter dictionary 
contains the keys "dwells", "id", and points. the value belonging to the "dwells" key
is a list of dwell position dictionary. The dwell position dictionary contains the keys: "angle", "position", "relativePos", "rotation", "time", and "weight". for more info, look at the function BrachyPlan.load_catheterTable_json() 
- dwell_numbers:np.array := the dwell number of each dwell position in the plan
- dwell_times:np.array := the dwell time of each dwell position in the plan
- dwell_coordinates:list := a list of dictionaries. each dictionary contains the keys "position", "rotation", and "relativePos"
- organ_bounds:dict
- dose_rate_tensor:np.array := dose rate from dwell position 1 to num_dwells. 
matches the dwell_number_list. shape: (num_dwells, z, y, x)
- uncertainty_tensor:np.array := uncertainty from dwell position 1 to num_dwells. shape: (num_dwells, z, y, x)
- brachy_structure:list[BrachyStructure] := the list of patient structures in the plan

### Functions 

- load_catheterTable_json()
- extract_dwell_numbers_times_coordinates_from_catheterTable()
- load_dose_rate_tensor()
- set_dvh_metric_goals()
- create_structures()
- calculate_DVH_metrics()



We use DicomRTTools to extract info from dicom files. For more info on this package, please visit the [DicomRTTools paper and repository](https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485)

