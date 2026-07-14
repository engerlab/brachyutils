import gc
import json
import os
import resource
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import typer
from brachyutils import BrachyDose
from brachyutils import BrachyEgsphant, _load_json
from brachyutils import BrachyPlan
from tqdm import tqdm
from typing_extensions import Annotated
from brachyutils import BrachyPhantom, BrachyDose, BrachyEgsphant
from typing import Literal, List, Dict
from enum import Enum

class DoseType(Enum):
    """Enum for dose types."""
    NRRD = ".nrrd"
    DCM = ".dcm"
    THREE_DDOSE = ".3ddose"

class EgsphantType(Enum):
    """Enum for egsphant types."""
    EGS = ".egsphant"
    NRRD = ".nrrd"

class PhantomType(Enum):
    """Enum for phantom types."""
    NRRD = ".nrrd"
    DCM = ".dcm"

def memory_limit():
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * 0.98), hard))

def get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory  # KiB

app = typer.Typer()

@app.command(name="convert-dose", help="Convert dose files to specified output format")
def convert_dose(
    pth_inputs: List[str],
    type_out: DoseType = DoseType.NRRD,
    dir_output: str = None,
    multi_proc: bool = False
):
    r"""
    ### Purpose:
    - To convert dose files to the specified output format.
    ### Inputs:
    - pth_inputs := list of paths to the input dose files. The input files can be directories or files.
    - type_out := type of the output file. Options are ".nrrd", ".dcm", ".3ddose".
    - dir_output := path to the output directory (optional)
    - multi_proc := whether to use multiprocessing (default: False)
    ### Output:
    - None: The converted dose file will be saved in the output directory.
    if the directory is not specified, it will be saved in the same directory as the input file. 
    """
    from brachyutils.dose.dose_utils import convert_dose_files
    return convert_dose_files(pth_inputs, type_out.value, dir_output, multi_proc)


@app.command(name="convert-phantom", help="Convert phantom (image and segmentation) files to specified output format")
def convert_phantom(
    pth_inputs: List[str],
    type_out: PhantomType = PhantomType.NRRD,
    dir_output: str = None,
    multi_proc: bool = False
):
    r"""
    ### Purpose:
    - To convert phantom (image and segmentation) files to the specified output format.
    ### Inputs:
    - pth_inputs := list of paths to the input phantom files. The input files can be directories or files.
    - type_out := type of the output file. Options are ".nrrd", ".dcm".
    - dir_output := path to the output directory (optional)
    - multi_proc := whether to use multiprocessing (default: False)
    ### Output:
    - None: The converted phantom file will be saved in the output directory.
    if the directory is not specified, it will be saved in the same directory as the input file. 
    """
    from brachyutils.geometry.phantom_utils import convert_phantom_files
    return convert_phantom_files(pth_inputs, type_out.value, dir_output, multi_proc)


@app.command(name="convert-egsphant", help="Convert egsphant files to specified output format")
def convert_egsphant(
    pth_inputs: List[str],
    type_out: EgsphantType = EgsphantType.EGS,
    dir_output: str = None,
    multi_proc: bool = False
):
    r"""
    ### Purpose:
    - To convert egsphant files to the specified output format.
    ### Inputs:
    - pth_inputs := list of paths to the input egsphant files. The input files can be directories or files.
    - type_out := type of the output file. Options are ".egsphant", ".nrrd".
    - dir_output := path to the output directory (optional)
    - multi_proc := whether to use multiprocessing (default: False)
    ### Output:
    - None: The converted egsphant file will be saved in the output directory.
    if the directory is not specified, it will be saved in the same directory as the input file. 
    """
    from brachyutils.geometry.egsphant_utils import convert_egsphant_files
    return convert_egsphant_files(pth_inputs, type_out.value, dir_output, multi_proc)

@app.command(
    help="""Purpose: to crop the egsphant file of all patients in a directory."""
)
def crop_egsphant_by_body_contour_many_patients(
    patient_egsphant_dir: Annotated[
        str,
        typer.Argument(
            help=""" the directory holding patient folders inside which there is .egsphant files to be cropped. \n
        Example: 
        patient_egsphant_dir/p1/ct.egsphant
        patient_egsphant_dir/p2/ct.egsphant ..."""
        ),
    ],
    patient_body_range_json: Annotated[
        str,
        typer.Argument(
            help="""a json file holding the list of the patient directory names as well as the index bounding range of the body contour and the original size of the body mask. This file can be generated by running the function dicom_utils.get_body_contour_range_from_many_patients_dicom().\n
        Example: [{"patient_number": "p1", "body_index_range": [[x_min, x_max], [y_min, y_max], [z_min,z_max]], "body_mask_shape": [512, 512, 42]}, ...] """
        ),
    ],
):
    r"""
    Purpose:
    to crop the egsphant file of all patients in a directory.
    Input:
    - patient_egsphant_dir := the directory holding patient folders inside which
    there is .egsphant files to be cropped. Example:
        patient_egsphant_dir/p1/ct.egsphant
        patient_egsphant_dir/p2/ct.egsphant
        ...
    - patient_body_range_json := a json file holding the list of the patient directory names
    as well as the index bounding range of the body contour and the original size of the body mask.
    This file can be generated by running the function dicom_utils.get_body_contour_range_from_many_patients_dicom().
    run "python dicom_utils.py get_body_contour_range_from_many_patients_dicom --help" for more details.
    Example:
    [
        {
        "patient_number": "p1",
        "body_index_range": [
            [x_min, x_max],
            [y_min, y_max],
            [z_min,z_max]
        ],
        "body_mask_shape": [512, 512, 42]
        }
    ]
    Output:
    - None: the cropped .egsphant file for each patient will be written to patient_dir/cropped_basename.egsphant
    """
    pth_egsphant_set = set(glob(patient_egsphant_dir + "/*/*.egsphant"))

    if len(pth_egsphant_set) == 0:
        raise FileNotFoundError(
            f"No .egsphant files found in the directory {patient_egsphant_dir}"
        )

    body_range_dict = _load_json(pth_json=patient_body_range_json)

    if len(body_range_dict) == 0:
        raise ValueError(
            f"No information found in .json file {patient_body_range_json}"
        )

    print(body_range_dict)
    for patient in tqdm(body_range_dict):
        print(patient["patient_number"])
        try:
            pth_egsphant = list(
                filter(lambda x: patient["patient_number"] in x, pth_egsphant_set)
            )[0]
        except IndexError as e:
            raise FileNotFoundError(
                f"No egsphant file found for patient {patient['patient_number']}"
            ) from e

        print(f"loading the patient egsphant {pth_egsphant}")
        egsphant_obj = BrachyEgsphant(pth_egsphant)
        egsphant_obj.crop_by_index(
            index_range=patient["body_index_range"],
        )
        pth_cropped_egsphant = (
            os.path.dirname(pth_egsphant) + "/cropped_" + os.path.basename(pth_egsphant)
        )
        print(f"writing the cropped egsphant to {pth_cropped_egsphant}")
        egsphant_obj.write_to_ctegsphant(pth_cropped_egsphant)

@app.command(
    help=""" Purpose: Will crop all files in the "input_dir" of type "type_in" and write the cropped dose to file with "type_out" """
)
def crop_dose_by_ratio_many_files(
    input_dir: Annotated[
        str,
        typer.Argument(
            help="""directory where there are dose files to be cropped and converted to be converted"""
        ),
    ],
    crop_ratio: Annotated[
        float,
        typer.Argument(
            help="""the fraction of the image axis that remains in the crop. for example, a crop ratio of 0.5 will keep the center of the x and y axis plus minus 0.25*dimension of the image. The x axis will not be cropped"""
        ),
    ],
    type_in: Annotated[
        str,
        typer.Argument(
            help="""could be ".3ddose", ".nrrd", ".minidos", other types could be added """
        ),
    ],
    type_out: Annotated[
        str,
        typer.Argument(
            help="""could be ".3ddose", ".nrrd", ".minidos", other types could be added """
        ),
    ],
):
    r"""
    Purpose:
        Will crop all files in the "input_dir" of type "type_in" and write the cropped dose to file with "type_out"
    Inputs:
        input_dir := directory where there are dose files to be cropped and converted to be converted
        crop_ratio := the fraction of the image axis that remains in the crop. for example, a crop ratio of 0.5 will keep
            the center of the x and y axis plus minus 0.25*dimension of the image. The x axis will not be cropped.
            +++++++++       ---------
            +++++++++       --+++++--
            +++++++++  ===> --+++++--
            +++++++++       --+++++--
            +++++++++       ---------
        type_in := could be ".3ddose", ".nrrd", ".minidos", other types could be added
        type_out := could be ".3ddose", ".nrrd", ".minidos", other types could be added
    """
    input_dir = os.path.abspath(input_dir)
    assert os.path.exists(input_dir)
    file_list = glob(input_dir + "/*" + type_in)

    for file in tqdm(file_list):
        dose_obj = BrachyDose(file)
        dose_obj.crop_by_fraction(crop_ratio)

        file_base_no_extension = os.path.splitext(file)[0]

        dose_obj.write_brachydose_to_file(file_base_no_extension + type_out)


@app.command(
    help="""Purpose: Will calculate the uncertainty of all structures for all patients in a directory"""
)
def get_uncertainty_one_patient(
    dir_doserate_maps: Annotated[
        str,
        typer.Argument(
            help="""directory containing Dose data of many patients. each folder has a subfolder for every patient. The names of the patients (subfolders) should match. """
        ),
    ],
    dir_plan: Annotated[
        str,
        typer.Argument(
            help="""directory containing Plan data of many patients. each folder has a subfolder for every patient. The names of the patients (subfolders) should match. In this folder, there should be a file named catheter_table.json that contains the catheter table."""
        ),
    ],
    dir_dicom: Annotated[
        str,
        typer.Argument(
            help="""directory containing DICOM data of many patients. each folder has a subfolder for every patient. The names of the patients (subfolders) should match."""
        ),
    ],
    pth_dvh_metric_goals_json: Annotated[
        str,
        typer.Argument(
            help="""path to the json file where the DVH metric goals are saved."""
        ),
    ],
    pth_uncertainty_json: Annotated[
        str,
        typer.Argument(
            help="""path to the json file where the uncertainty of all structures will be saved."""
        ),
    ],
    multi_proc: Annotated[
        bool,
        typer.Option(
            help="""If set to true, multiprocessing will be used to load the dose files in parallel."""
        ),
    ] = False,
):
    r"""
    Purpose:
        To loop over all patients and get the uncertainty of all structures.
    Input:
        - dir_doserate_maps := Directory containing Dose rate maps for the dwell position.
        - dir_plan := Directory containing Plan data of the patient. Inside the dir plan,
        - dir_dicom := Directory containing DICOM data of the patient.
        there should be a file named catheter_table.json that contains the catheter table.
        - pth_dvh_metric_goals_json := path to the json file where the DVH metric goals are saved.
        - pth_uncertainty_json := path to the json file where the uncertainty of all structures will be saved.
        - multi_proc := If set to true, multiprocessing will be used to load the dose files in parallel.
    Output:
        - None := Path of the json file where the uncertainty of all structures will be saved.
    Dependencies:
    """
    assert os.path.exists(dir_dicom)
    assert os.path.exists(dir_doserate_maps)

    with open(pth_dvh_metric_goals_json, "r") as dvh_target_file:
        dvh_metric_goals = json.load(dvh_target_file)

    patient = os.path.basename(os.path.normpath(dir_doserate_maps))
    pth_plan = dir_plan + "/catheter_table.json"
    assert os.path.exists(pth_plan)
    pth_dicom = dir_dicom + "/"
    assert os.path.exists(pth_dicom)
    pth_dose = dir_doserate_maps + "/"
    assert os.path.exists(pth_dose)

    structure_file = glob(pth_dicom + "/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=structure_file,
        )

    plan_obj = BrachyPlan(
        phantom=phantom_obj,
        dvh_metric_goals=dvh_metric_goals,
        catheter_table=pth_plan,
        dir_dose_rate=pth_dose,
        load_dose_or_uncertainty="uncertainty",
        multi_processing=multi_proc,
    )
    plan_obj.calculate_uncertainty_per_structure()

    patient_info = {
        "patient_id": patient,
        "pth_dicom": pth_dicom,
        "pth_dose": pth_dose,
        "pth_plan": pth_plan,
    }
    for structure in plan_obj.structure_list:
        patient_info[structure.name] = {
            "uncertainty_mean": structure.uncertainty_mean,
            "uncertainty_std": structure.uncertainty_std,
            "uncertainty_max": structure.uncertainty_max,
            "uncertainty_min": structure.uncertainty_min,
        }

    del plan_obj
    gc.collect()

    with open(pth_uncertainty_json, "w") as outfile:
        json.dump(patient_info, outfile, indent=4)


def get_dose_map(dose_file):
    r"""
    Purpose:
        Helper function for loading batch dose files for a patient that
        loads a dose file to extract the dose grid
    Input:
        - dose_file := path to the dose file
    Output:
        - dose_obj.grid := the dose grid from dose_file
    """
    try:
        # print("Loading dose file ", dose_file)
        # print("\n Start Processing", dose_file)
        dose_obj = BrachyDose(dose_file, load_uncertainty=False)
        # print("\n End Processing", dose_file)
        return dose_obj.get_dose_array()
    except (TypeError, ValueError, IndexError, IOError, StopIteration) as e:
        print("Error loading dose file ", dose_file, e)
        return None


@app.command(help="""Purpose: Will combined multiple dose files for a single patient""")
def combined_dose_per_patient(
    dir_dose_maps: Annotated[
        str, typer.Argument(help="""Directory containing dose data for a patient. """)
    ],
    type_in: Annotated[
        str,
        typer.Argument(
            help="""Extension of the files to be converted. Options are .3ddose and .nrrd. """
        ),
    ],
    type_out: Annotated[
        str,
        typer.Argument(
            help="""Extension of the output files. Options are .3ddose, .nrrd, .minidos."""
        ),
    ],
    multi_proc: Annotated[
        bool,
        typer.Option(
            help="""If set to true, multiprocessing will be used to load the dose files in parallel."""
        ),
    ] = True,
    file_name: Annotated[
        str,
        typer.Option(
            help="""Name of the output file (without the extension). If not provided, the default name will be combined"""
        ),
    ] = "combined",
    phantom_file: Annotated[
        str,
        typer.Option(
            help="""Path to the phantom file, which will be ignored in the combined if found."""
        ),
    ] = "phantom.seq.nrrd"

):
    r"""
    Purpose:
        To loop over all batches of a simulatation and create the combined 3ddose file.
    Input:
        - dir_dose_maps := Directory containing dose files.
        - type_in := Format of the dose files to be converted.
        - type_out := Format of the output file.
        - multi_proc := If set to true, multiprocessing will be used to load the dose files in parallel.
    """
    # change to absolute path since execution directory is not dir_dose_maps
    dir_dose_maps = os.path.abspath(dir_dose_maps)

    # check if the directory exists
    if not os.path.exists(dir_dose_maps):
        raise FileNotFoundError(
            f"the directory {dir_dose_maps} does not exist. Please \
        make sure that you specify the absolute path to the directory."
        )

    # make sure directory ends with a / to avoid errors
    if dir_dose_maps[-1] != "/":
        dir_dose_maps += "/"

    # prepare a list of dose files
    dose_files = [
        dir_dose_maps + file
        for file in os.listdir(dir_dose_maps)
        if file.endswith(type_in) and file != phantom_file
    ]

    n_batches = len(dose_files)

    # check if there's any dose files
    if n_batches == 0:
        raise FileNotFoundError(
            f"No {type_in} files found in the directory {dir_dose_maps}"
        )

    progress_bar_length = n_batches
    # get information about the dose grid from the first file
    dose_obj = BrachyDose(
        pth_dose_file=dose_files[0]
        )
    combined_dose_obj = BrachyDose.dose_with_empty_grid_like(dose_obj)

    combined_dose_obj.set_dose_array(dose_obj.get_dose_array())

    sum_dose = dose_obj.get_dose_array()
    uncertainty = np.zeros(dose_obj.get_dose_array().shape)

    # chunksize =
    # multiprocessing loop
    if multi_proc:
        with Pool(processes=16) as pool:
            for dose in tqdm(
                pool.imap_unordered(get_dose_map, dose_files[1:]),
                total=progress_bar_length - 1,
                desc="Extracing Dose for Mean Dose: ",
            ):
                if dose is not None:
                    sum_dose += dose
                else:
                    n_batches -= 1
            mean_dose = sum_dose / n_batches
            # print(mean_dose)
            for dose in tqdm(
                pool.imap_unordered(get_dose_map, dose_files),
                total=progress_bar_length,
                desc="Extracing Dose for Uncertainty: ",
            ):
                if dose is not None:
                    uncertainty += (dose - mean_dose) ** 2
    # if no multiprocessing, a simple loop over files
    else:
        for dose_file in tqdm(dose_files[1:]):
            dose_map = get_dose_map(dose_file)
            if dose_map is not None:
                sum_dose += dose_map
            else:
                n_batches -= 1
        mean_dose = sum_dose / n_batches
        uncertainty = np.zeros(mean_dose.shape)
        for dose_file in tqdm(dose_files):
            dose_map = get_dose_map(dose_file)
            if dose_map is not None:
                uncertainty += (dose_obj.get_dose_array() - mean_dose) ** 2

    # finish uncertainty calculation
    uncertainty = np.sqrt(uncertainty / (n_batches * (n_batches - 1)))
    uncertainty = uncertainty / (
        mean_dose + 1e-7
    )  # avoid divide by 0 with small perturbation

    # write the combined dose to file
    combined_dose_obj.set_dose_array(mean_dose)
    combined_dose_obj.set_uncertainty_array(uncertainty)

    print(
        "Combining ",
        n_batches,
        " dose files complete",
        "writing to ",
        dir_dose_maps + file_name + type_out,
    )

    if type_out == ".3ddose":
        combined_dose_obj.write_to_3ddose(dir_dose_maps + file_name + ".3ddose")
    elif type_out == ".nrrd":
        combined_dose_obj.write_to_nrrd(dir_dose_maps + file_name + ".nrrd")
    elif type_out == ".minidos":
        combined_dose_obj.write_to_minidos(dir_dose_maps + file_name + ".minidos")


def main():
    app()
    # memory_limit()
    # try:
    #     app()
    # except MemoryError:
    #     print("Memory Error. consider loading only dose or uncertainty instead of both.")
    #     sys.exit(1)
