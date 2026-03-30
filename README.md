# BrachyUtils

![BrachyUtils](admin/icon-library/logos/Brachyutils-logo-2.svg)

BrachyUtils is developed to be a scripting treatment planning system for brachytherapy. The current state mostly focuses on high dose rate (HDR) brachytherapy. For full documentation, take a look at our [docs webpage](https://engerlab.github.io/brachyutils/brachyutils.html). The main modules and submodules are described below:

1. `geometry`
   1. `phantom_utils`: Handles geometry definition (Patient, segmentations or measurement setup). IO for DICOM, NRRD, and Nifti (read only). The main class here is `BrachyPhantom`.
   2. `egsphant_utils`: Handles manipulation of the data in the EGSPhant format from the EGSnrc. This class can create a `BrachyEgsphant` from `BrachyPhantom` or load it from `.egsphant` or `.seq.nrrd` files.
   3. `catheter_utils`: Implements the `CatheterTable`, which provides DICOM and JSON IO for HDR treatment plans. The `CatheterTable` numerous functionalities for adding and removing catheters and dwells and storing the dose rates and the totoal dose of the treatment plan.
   4. `registration_utils`: Provides the ability to perform image-based and contour-based registration using `reg_opentps`, `reg_plastimatch`, and `reg_simple_elastix`. All these modules makes use of the abstract class and the functionality provided in `reg_utils`.
2. `dose`
    1. `dose_utils`: Provides IO for dose and uncertainty data from DICOM, 3ddose and NRRD into `BrachyDose`. Several functinality exists for cropping and resampling dose maps exists.
    2. `dose_generation_utils`: Several classes exist here that allow for generating dose from a treatment plan. The main abstract class is `BrachyDoseGenerator`, and currently the concrete classes are: `RapidBrachyTG43`, `RapidBrachyMC`, and `BrachyUtilsTG43`.
    3. `dose_comparison_utils`: Allows for comparison between two `BrachyDose` objects based on percent error maps (according to AAPM-WGDCAB Report 372) and Gamma index analysis. The main class is `BrachyDoseComparison`
    4. `film_utils`: The class that allows for processing radiochromic film data. It has two subclasses, `CalibrationCurve` and `FilmCalibration`.
3. `planning`:
    1. `plan_utils`: This module implements `BrachyPlan` which makes use of all the previous modules to handle treatment planning operations in brachytherapy. We recommend getting started using the `load_dicom_to_plan` function.
    2. `simulation_utils`: The class to store information regarding the source in brachytherapy (IO from json and dicom) as well as simulation parameters such as the number of threads to use and the number of histories to be simulated. The two main classes here are `BrachySimulation` and `BrachySource`.
    3. `structure_utils`: This module contains `BrachyStructure`, which in additon to the structure mask, contain information regarding the associated DVH metrics for each structure and the optimization config.
    4. `optimization`: An extensive module that handles dwell time optimization using Gurobi (`optim_gurobi`), AMPL (`optim_ampl`) and ORTools (`optim_ortools`). All of these contain concrete classes that extend the abstract class `BrachyDwellTimeOptim` and use several functionalities provided in `optim_utils`. `mobo` module handles multi objective optimization of the penalty weights and the `optim_cath` module builds towards catheter placement optimization.

If you are a developer, please take a look at the bottom of this page.

## Installation

Start by clonning this repository to `YourDesiredLocation`:

```bash
git clone https://github.com/engerlab/brachyutils.git
```

### Using Docker Image

The docker image for brachyutils can be downloaded from the [OneDrive Folder](https://mcgill-my.sharepoint.com/:f:/g/personal/shirin_abbasinejadenger_mcgill_ca/IgBX59ZwMN9MTakYUOsZgNXMAb8W2FZrLVhY6kpSeHabcrY?e=XFPqqh). In addition, docker images for plastimatch and simple-elastix are provided in the same folder. If you'd like to have access to RapidBrachyMC or RapidBrachyTG43, please let us know and we will provide access to a seperate folder.

Once the image is downloaded, you can unzip it using `zstd` and load it to docker.

```bash
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
5. To debug packages using cv2 in vscode, you may need: `export QT_QPA_PLATFORM=offscreen`
6. Happy coding/debugging

### Using Apptainer Image

It is possible to create an Apptainer image (.sif file) from the docker image and run it. However, we recommend using the regular virtual enviorment or the docker images.

### Using Python virtual environment (No Container)

We currently use python 3.13. To install it, take a look at 
`./docker_src/repositories/install_python.sh`.

Once python 3.13 is installed, Create a virtual envionrment and activate it:

If using [venv](https://docs.python.org/3/library/venv.html):

```bash
python3 -m venv ENV_BU
source ENV_BU/bin/activate
```

Else, if using [conda](https://docs.anaconda.com/miniconda/):

```bash
conda create -n ENV_BU python=3.13
conda activate ENV_BU
```

#### Install BrachyUtils

To get the package run:

```bash
apt install -y build-essential zlib1g zlib1g-dev libncurses5-dev \
  libgdbm-dev pkg-config libnss3-dev libssl-dev libreadline-dev \
  libffi-dev libsqlite3-dev wget nano liblzma-dev libbz2-dev \
  libxrender1 libgl1 libglib2.0-0 tk-dev xz-utils llvm


git clone https://github.com/engerlab/brachyutils.git
cd brachyutils
# For the basic functinality
python3.13 -m pip install -e .[]
# For registration
python3.13 -m pip install -e .[reg]
# for treatment planning
python3.13 -m pip install -e .[plan]
# for the complete functionality
python3.13 -m pip install -e .[full]
```

#### Install Optimization Solvers

Solvers are currently used to run dwell time optimization. We recommend using the Gurobi solver, which requires an academic license. Another platform that one can use is AMPL, which gives you access to many solvers out there. AMPL provide a community license, but the good solvers would only be available with an academic Email.

To install Gurobi, simply run:

```bash
pip install gurobipy
```

Go to gurobi license manager and get an [Academic WSL License](https://www.gurobi.com/features/academic-wls-license/). Then download the file `gurobi.lic` and put it in your home directory (or the home directory of the docker image).

To install AMPL, run:
```bash
# Install Python API for AMPL
python -m pip install amplpy --upgrade

# Install HiGHS and Gurobi (AMPL is installed automatically with any solver)
python -m amplpy.modules install highs gurobi

# Activate your license (e.g., free https://ampl.com/ce license)
python -m amplpy.modules activate <license-uuid>

# Confirm that the license is active
python -m amplpy.modules run ampl -vvq

# Import in Python
python
>>> from amplpy import AMPL
>>> ampl = AMPL() # instantiate AMPL object

```
Click here to see a list of [AMPL solver](https://dev.ampl.com/solvers/index.html).
In my experience, Gurobi, XPRESS, and CPLEX are the only ones that work. Unfortunately, they all require and academic/industrial license.

## brachyutils commands

brachyutils comes with a linux command line interface. To learn about the commands that are available run `brachyutils --help` on the command line.

At the moment, the outputs looks like the following:

```bash
$ brachyutils --help
 Usage: brachyutils [OPTIONS] COMMAND [ARGS]...                                                                                         
                                                                                                                                        
╭─Options──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                              │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                       │
│ --help                        Show this message and exit.                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─Commands──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ convert-dose                                  Convert dose files to specified output format                                          │
│ convert-phantom                               Convert phantom (image and segmentation) files to specified output format              │
│ convert-egsphant                              Convert egsphant files to specified output format                                      │
│ crop-egsphant-by-body-contour-many-patients   Purpose: to crop the egsphant file of all patients in a directory.                     │
│ crop-dose-by-ratio-many-files                 Purpose: Will crop all files in the "input_dir" of type "type_in" and write the        │
│                                               cropped dose to file with "type_out"                                                   │
│ get-uncertainty-one-patient                   Purpose: Will calculate the uncertainty of all structures for all patients in a        │
│                                               directory                                                                              │
│ combined-dose-per-patient                     Purpose: Will combined multiple dose files for a single patient                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

For additional help with each command, run the command name with --help. For example `brachyutils convert-dose --help`

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

### Making new a Docker Image
In case you'd like to extend brachyutils with new functionality and using new libraries, you may make changes to `pyprojct.toml` requirements and update the docker image that runs brachyutils. You can make a new image following either of the process below:

### Using Dockerfile

This approach allows you to make a new image without volume mounting.

1. Push your changes to either repository, download the repo as a zip file.
2. Place the zip file inside the folder `docker_src/repositories`
3. Make sure that the name of the zipped files are written correctly inside `docker_src/Dockerfile` 
4. Inside `docker_src/docker-compose.yaml`, comment out volume mounting (lines 20-22)
5. Inside `docker_src`, run `docker-compose up --build -d BrachyUtils`

Depending on your internet speed should take about 30 minutes to 1 hour.

### Using docker commit

This approach requires you to mount volumes. If there are data in the mounted volume, it will be stored inside the docker image and cannot be deleted from the image.

1. Do Not Do It
2. If you have to, come talk to u know who