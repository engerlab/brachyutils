# BrachyUtils

This package implements Brachytherapy dose, egsphant dicom and film dosimetry functionalities. It also interfaces with various RapidBrachy projects. If you are a developer, please take a look at the bottom of this page.

Start by clonning this repository to `YourDesiredLocation`:

```bash
git clone -b opentps https://github.com/engerlab/brachyutils.git
```

## Using Docker Image

The docker image can be downloaded from the [OneDrive Folder](https://mcgill-my.sharepoint.com/:f:/g/personal/shirin_abbasinejadenger_mcgill_ca/Elfn1nAw30xNqRhQ6xmA1cwBvxbYVmstWFjqSlJ4dptytg?e=ROqLfn).

Once the image is downloaded, you can unzip it using `zstd` and load it to docker.

```bash
# to unzip using zstd
tar -I zstd -xvf brachyutils.tar.zst
# to load the image to docker
docker load -i brachyutils.tar
```

After the image is loaded, navigate to `brachyutils/docker_src` and run `bash command_docker.sh` to start the container running the image.

To see if the container is running, try `docker ps`. A container named `BrachyUtils` should be there. You can attach to the running container using the command line (run `docker exec -it BrachyUtils bash`). BrachyUtils is installed on the root Python 3.13 environment. You can test it by running `brachyutils` on the command line or import it as a module in python.

**pro tip**
We recommend attaching using the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension on [Visual Studio Code](https://code.visualstudio.com/). Once these packages are installed and the BrachyUtils container is running:

1. open vscode, press `F1`.
2. type in `Attach to running` until it shows up in the command bar and select it.
3. You should see BrachyUtils as an option, select it.
4. A new VS code window will open, and gives you access to the files and the executables inside the container.
5. Happy coding/debugging


## Using Apptainer Image

To free the users from the hassle of installing brachutils and all its requirements, we have created an Apptainer image and a Docker image that could be downloaded from the [OneDrive Folder](https://mcgill-my.sharepoint.com/:f:/g/personal/shirin_abbasinejadenger_mcgill_ca/Elfn1nAw30xNqRhQ6xmA1cwBvxbYVmstWFjqSlJ4dptytg?e=ROqLfn).

It is recommended to use the singularity image (`brachyutils_opentps.sif`) on Compute Canada or in general on systems where `Sudo` access is **not possible** or Docker is not available. You can bind the folder where your data is located as well.

```bash
# on compute Canada only{
module load StdEnv/2023
module load apptainer
# }
apptainer run --containall --bind <YourDesiredLocation>/brachyutils:/root/brachyutils --bind <YourDataLocation>:/root/YourLocalHome brachyutils_opentps.sif
# Once apptainer is running interactively
cd /root
source .bashrc
```

The virtual enviornment called `env_brachyutils` should be activated automatically. You can make changes to the brachyutils source code by editing source files in `/root/brachyutils`. Your data can be found at `/root/YourLocalHome`.

**VS Code Support**: Using vscode, you can directly code and debug inside a docker container. Simply install the extension [Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container). While the docker container is running, open VS Code, press `F1`, type `Dev Containers: Attach to running container`. Then select the container running brachyutils. 

## Installation

### Create a Python virtual environment

We currently use python 3.13. Create a virtual envionrment and activate it:

If using [venv](https://docs.python.org/3/library/venv.html):

```bash
python3 -m venv ENV_brachyutils
source ENV_brachyutils/bin/activate
```

Else, if using [conda](https://docs.anaconda.com/miniconda/):

```bash
conda create -n ENV_brachy python=3.13
conda activate ENV_brachy
```

### Install OpenTPS from source code

The PyPi package of OpenTPS is not up to date with their Gitlab repository. Therefore, we recommend that you clone the repository and install the package.

```bash
apt install -y libbz2-dev libxrender1 python3-distutils build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev  libsqlite3-dev libgl1 libglib2.0-0
git clone https://github.com/engerlab/OpenTPS-brachyutils
cd opentps
```

Then run `pip install .` to install opentps.

### Install BrachyUtils

To get the package run:

```bash
apt install -y liblzma-dev python3-tk tk-dev
git clone https://github.com/engerlab/brachyutils.git
cd brachyutils
pip install .
```

### Optional

`python3 -m pip install --upgrade pip`

Install SimpleITK independently by running `python3 -m pip install SimpleITK`. If you run into the error saying `skbuild` is [missing](https://bugs.python.org/issue30573), run `python3 -m pip install cmake`, then try installing SimpleITK again.

## brachyutils commands

brachyutils comes with a linux command line interface. To learn about the commands that are available run `brachyutils --help` on the command line.

At the moment, the outputs looks like the following:

```bash
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

## Developer Guide

Please follow the steps when developing Brachy Utils.

1. Pull the latest code from main or the specific branch.
2. Make a new branch with an informitive name. For example, debug_dcm_dose
3. Develope as you see fit (make changes) on your new branch.
4. For every change, please commit with a descriptive message.
5. After you are done and happy with the changes, pull again from the source branch (main or others).
6. Run tests to make sure there are no bugs.
7. Push your branch to the remote repository.
8. Request to merge with the source branch

<!-- ## BrachyDose

You can import this object in your python script by running `from brachyutils import BrachyDose`. This object has the following attributes and functions:

### Purpose

This class holds information regarding a dose distribution as well as the fundamental functions that are applied on the dose. All the doses are J/Gy.

### Attributes:

- grid:np.ndarray := 3D numpy array holding dose at each voxel. [z, y, x]
- uncertainty:np.ndarray := 3D numpy array holding dose uncertainity at each voxel. [z, y, x]
- num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis.
- voxel_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters.
- origin_coordinates:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z]
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
- voxel_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters.
- origin_coordinates:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z]
- axis:np.ndarray := coorindates of grid points along z, y and x axis.

### Functions:

- load_file_to_BrachyEgsphant() done
- load_from_ctegsphant() done
- load_from_nrrd() not implmented
- calculate_axis() done
- write_to_ctegsphant() done
- write_to_nrrd() not implemented
- crop_by_index() done
- crop_by_body_contour()
- assert_BrachyEgsphant_notEmpty() done
- info() done
- is_equal() done

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

We use DicomRTTools to extract info from dicom files. For more info on this package, please visit the [DicomRTTools paper and repository](https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485) -->
