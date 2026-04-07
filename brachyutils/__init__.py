"""
# BrachyUtils

![BrachyUtils](../admin/icon-library/BrachyUtils.svg)

BrachyUtils is developed to be a scripting treatment planning system for brachytherapy. The current state mostly focuses on high dose rate (HDR) brachytherapy. The main modules and submodules are described below:

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

"""

__all__ = [
    "dose",
    "geometry",
    "planning",
]
# trunk-ignore(ruff/F401)
from .dose import *
# trunk-ignore(ruff/F401)
from .geometry import *
# trunk-ignore(ruff/F401)
from .planning import *

from .planning.optimization import *  # noqa: F401, F403

from .geometry.registration_utils import *  # noqa: F401, F403

from .geometry.catheter_utils import * # noqa: F401, F403
