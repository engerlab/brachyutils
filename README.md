# BrachyUtils

![BrachyUtils](admin/icon-library/logos/Brachyutils-logo-2.svg)

This package implements Brachytherapy dose, egsphant dicom and film dosimetry functionalities. It also interfaces with various RapidBrachy projects. If you are a developer, please take a look at the bottom of this page.

Start by clonning this repository to `YourDesiredLocation`:

```bash
git clone https://github.com/engerlab/brachyutils.git
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
5. To debug packages using cv2 in vscode, you may need: `export QT_QPA_PLATFORM=offscreen`
6. Happy coding/debugging


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

### Install Optimization Solvers

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

### Making new a Docker Image

BrachyUtils has two main requirements, [OpenTPS-brachyutils](https://github.com/engerlab/OpenTPS-brachyutils.git), and [AI_Assisted_Brachytherapy](https://github.com/engerlab/AI_Assisted_Brachytherapy.git). You may make changes to these requirements and would like to update the docker image that runs brachyutils. In that case, you can make a new image following either of the process below:

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