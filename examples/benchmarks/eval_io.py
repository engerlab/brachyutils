from pathlib import Path
from time import time
from typing import Literal
import pandas as pd
from tqdm import tqdm
from brachyutils import BrachyPhantom
from brachyutils import BrachyDose
from brachyutils import BrachyEgsphant

def time_phantom_io(
        dir_out: Path,
        type_out: Literal["dicom", "nrrd", "nifti", "egs"],
        dir_dicom: Path = None,
        pth_phantom_file: Path = None,
        pth_structures_file: Path = None,
    ):
        try:
            t0_read = time()
            phantom_obj = BrachyPhantom(
                dir_dicom=dir_dicom,
                pth_phantom_file=pth_phantom_file,
                pth_structures_file=pth_structures_file
                )
            tf_read = time()
        except Exception as e:
            t0_read, tf_read = (float("nan"), float("nan"))
        try:
            if type_out == "dicom":
                t0_write = time()
                phantom_obj.export_to(dir_dicom_out=dir_out)
            elif type_out == "nrrd":
                t0_write = time()
                phantom_obj.export_to(dir_nrrd_out=dir_out)
            tf_write = time()
        except Exception as e:
            t0_write, tf_write = (float("nan"), float("nan"))
        return (tf_read - t0_read, tf_write - t0_write)

def time_dose_io(
    pth_dose_in: Path,
    pth_dose_out: Path
    ):
    try:
        t0_read = time()
        dose_obj = BrachyDose(
            pth_dose_file=pth_dose_in
        )
        tf_read = time()
    except Exception as e:
        t0_read, tf_read = (float("nan"), float("nan"))
    try:
        t0_write = time()
        dose_obj.write_brachydose_to_file(
            pth_dose_file=pth_dose_out
        )
        tf_write = time()
    except Exception as e:
        t0_write, tf_write = (float("nan"), float("nan"))
    return (tf_read - t0_read, tf_write - t0_write)

def time_egsphant_io(
    pth_egsphant_in: Path,
    pth_egsphant_out: Path
    ):
    # try:
    t0_read = time()
    egsphant_obj = BrachyEgsphant(
        pth_egsphant_file=pth_egsphant_in
    )
    tf_read = time()
    # except Exception as e:
    t0_read, tf_read = (float("nan"), float("nan"))
    try:
        t0_write = time()
        egsphant_obj.write_to_file(
            pth_output=pth_egsphant_out
        )
        tf_write = time()
    except Exception as e:
        t0_write, tf_write = (float("nan"), float("nan"))
    return (tf_read - t0_read, tf_write - t0_write)        

def eval_dicom_io(dicom_patients: Path | str, dir_out: Path | str):
    """
    Evaluate DICOM I/O performance by benchmarking read and write operations.
    This function performs timing benchmarks for various DICOM file operations including:
    - Phantom scan reading and writing (CT/MR images)
    - Phantom scan with segmentation structures reading and writing
    - Dose file reading and writing
    The function processes all DICOM directories found in the specified data path,
    measures execution times for each operation type, and generates statistical
    summaries (average and standard deviation) of the timing results.
    Directory Structure Expected:
        ~/YourLocalHome/Data/prostate-glen-2023/
        ├── patient1/
        │   ├── CT*.dcm (scan files) or MR*.dcm
        │   ├── RS*.dcm (structure set files)
        │   └── RD*.dcm (dose files)
        └── patient2/
            └── ...
    Output:
        Creates a CSV file 'timing_dicom_io.csv' in 'temp_data/dicom_io/' containing
        timing results for all operations across all processed DICOM directories.
    Raises:
        FileNotFoundError: If the specified DICOM data directory doesn't exist
        IndexError: If required DICOM files (RS*.dcm or RD*.dcm) are not found
    Note:
        Requires the following functions to be available:
        - time_phantom_io(): Times phantom reading/writing operations
        - time_dose_io(): Times dose file reading/writing operations
    """
    dicom_patients = Path(dicom_patients)
    dicom_patients = list(dicom_patients.glob("*/"))
    dir_out = Path(dir_out)
    # dir_out = Path("temp_data/dicom_io")
    timing_df = pd.DataFrame(columns=[
        "read_time_scan", "write_time_scan",
        "read_time_scan+seg", "write_time_scan+seg",
        "read_time_dose", "write_time_dose"
        ], index=[dicom.name for dicom in dicom_patients] + ["average", "std"])
    
    for dicom in tqdm(dicom_patients):
        t_read_scan, t_write_scan = time_phantom_io(
            dir_out=dir_out.joinpath(dicom.name),
            type_out="dicom",
            dir_dicom=dicom
            )
        t_read_scan_seg, t_write_scan_seg = time_phantom_io(
            dir_out=dir_out.joinpath(dicom.name),
            type_out="dicom",
            dir_dicom=dicom,
            pth_structures_file=list(dicom.glob("RS*.dcm")).pop()
        )
        pth_dose_file = list(dicom.glob("RD*.dcm")).pop()
        t_read_dose, t_write_dose = time_dose_io(
            pth_dose_in=pth_dose_file,
            pth_dose_out=dir_out.joinpath(dicom.name).joinpath(pth_dose_file.name)
        )
        timing_df.loc[dicom.name] = [
            t_read_scan, t_write_scan,
            t_read_scan_seg, t_write_scan_seg,
            t_read_dose, t_write_dose
            ]

    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_out.joinpath("timing_dicom_io.csv"))

def convert_dicom_to_nrrd(
    dicom_patients: Path | str,
    dir_out: Path | str
    ):
    """
    Convert DICOM files to NRRD format for brachytherapy data.
    This function processes DICOM directories containing brachytherapy data and converts
    them to NRRD format. It handles both phantom structures and dose data.
    The function:
    - Searches for DICOM directories in the user's home directory under 
      "YourLocalHome/Data/prostate-glen-2023"
    - Creates BrachyPhantom objects from DICOM files and structure files (RS*.dcm)
    - Exports phantom data to NRRD format in the output directory
    - Creates BrachyDose objects from dose files (RD*.dcm)
    - Writes dose data to NRRD files with .seq.nrrd extension
    Output files are saved to "temp_data/nrrd_io" directory with subdirectories
    named after the original DICOM directory names.
    Raises:
        Exception: Catches and prints any conversion errors for individual DICOM
                  directories, then continues processing remaining directories.
    Note:
        Requires BrachyPhantom and BrachyDose classes to be imported and available.
        Expects DICOM directories to contain RS*.dcm (structure) and RD*.dcm (dose) files.
    """

    dicom_patients = list(Path().home().joinpath(dicom_patients).glob("*/"))
    # first converting everything to nrrd files
    for dicom in dicom_patients:
        # Convert DICOM to NRRD
        try:
            phantom_obj = BrachyPhantom(
                dir_dicom=dicom,
                pth_structures_file=list(dicom.glob("RS*.dcm")).pop()
                )
            phantom_obj.export_to(
                dir_nrrd_out=dir_out.joinpath(dicom.name)
                )
            dose_obj = BrachyDose(
                pth_dose_file=list(dicom.glob("RD*.dcm")).pop()
            )
            dose_obj.write_brachydose_to_file(
                pth_dose_file=Path(dir_out).joinpath(dicom.name).joinpath(dicom.name+".seq.nrrd")
            )
        except Exception as e:
            print(f"Error converting {dicom.name} to NRRD: {e}")
            continue

def eval_nrrd_io(dir_nrrds: Path | str):
    """
    Evaluate NRRD file I/O performance by timing read and write operations.
    This function benchmarks the performance of reading and writing NRRD files
    for medical imaging data, including both scan data and segmentation data.
    It processes all subdirectories in the given path and measures timing for
    different I/O operations.
    Args:
        dir_nrrds (Path | str): Path to directory containing subdirectories with
            NRRD files. Each subdirectory should contain:
            - A scan file named "{subdirectory_name}.nrrd"
            - A segmentation file named "{subdirectory_name}.seg.nrrd"
    Returns:
        None: Results are saved to a CSV file named "timing_nrrd_io.csv" in the
        parent directory of the input path.
    Output CSV columns:
        - read_time_scan: Time to read scan NRRD file
        - write_time_scan: Time to write scan NRRD file
        - read_time_scan+seg: Time to read scan + segmentation NRRD files
        - write_time_scan+seg: Time to write scan + segmentation NRRD files
        - average: Mean values across all files
        - std: Standard deviation across all files
    Note:
        The function uses tqdm for progress tracking and pandas for data management.
        Each subdirectory name is used as an index in the resulting DataFrame.
    """
    
    dir_nrrds = list(Path(dir_nrrds).glob("*/"))
    timing_df = pd.DataFrame(columns=[
        "read_time_scan", "write_time_scan",
        "read_time_scan+seg", "write_time_scan+seg",
        "read_time_dose", "write_time_dose"
        ], index=[nrrd.name for nrrd in dir_nrrds] + ["average", "std"])

    for nrrd in tqdm(dir_nrrds):
        t_read_scan, t_write_scan = time_phantom_io(
            dir_out=nrrd,
            type_out="nrrd",
            pth_phantom_file=nrrd.joinpath(nrrd.name+".nrrd")
            )
        t_read_scan_seg, t_write_scan_seg = time_phantom_io(
            dir_out=nrrd,
            type_out="nrrd",
            pth_phantom_file=nrrd.joinpath(nrrd.name+".nrrd"),
            pth_structures_file=nrrd.joinpath(nrrd.name+".seg.nrrd")
        )
        try:
            pth_dose_file = list(nrrd.glob("*.seq.nrrd")).pop()
            t_read_dose, t_write_dose = time_dose_io(
                pth_dose_in=pth_dose_file,
                pth_dose_out=nrrd.joinpath(nrrd.name).joinpath(pth_dose_file.name)
            )
        except:
            timing_df.loc[nrrd.name] = [
            t_read_scan, t_write_scan,
            t_read_scan_seg, t_write_scan_seg,
            float("nan"), float("nan")
            # t_read_dose, t_write_dose
            ]
            continue
        timing_df.loc[nrrd.name] = [
            t_read_scan, t_write_scan,
            t_read_scan_seg, t_write_scan_seg,
            t_read_dose, t_write_dose
            ]
    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_nrrds[0].parent.joinpath("timing_nrrd_io.csv"))

def eval_egs_io(
    egs_patients:Path | str,
    dir_out: Path | str = None
    ):
    """
    To time the reading and writing of egsphant files.
    """
    egs_patients = Path(egs_patients)
    dir_out = Path(dir_out)
    egs_patients = list(egs_patients.glob("*/"))
    timing_df = pd.DataFrame(columns=[
        "read_time_egsphant", "write_time_egsphant",
        "read_time_egsphant_nrrd", "write_time_egsphant_nrrd"
        ], index=[egs.name for egs in egs_patients] + ["average", "std"])
    for patient in egs_patients:
        pth_ct_egsphant = list(patient.glob("ct.egsphant")).pop()
        pth_egsphant_nrrd = list(patient.glob("egsphant.seq.nrrd")).pop()
        t_read_ct_egs, t_write_ct_egs = time_egsphant_io(
            pth_egsphant_in=pth_ct_egsphant,
            pth_egsphant_out=dir_out.joinpath(f"{patient.name}/ct.egsphant")
        )
        t_read_egs_nrrd, t_write_egs_nrrd = time_egsphant_io(
            pth_egsphant_in=pth_egsphant_nrrd,
            pth_egsphant_out=dir_out.joinpath(f"{patient.name}/egsphant.seq.nrrd")
        )
        timing_df.loc[patient.name] = [
            t_read_ct_egs, t_write_ct_egs,
            t_read_egs_nrrd, t_write_egs_nrrd
        ]
        break
    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_out.joinpath("timing_egs_io.csv"))

def eval_nifti_io():
    pass

def generate_egsphants(
    nrrd_patients: Path | str,
    pth_material_dict: Path | str
    ):
    r"""
    ### Purpose:
    - To generate egsphant files from patient geometry storied in NRRD format. For each patient,
    the directory should contain:
        - A scan file named "{patient_name}.nrrd"
        - A segmentation file named "{patient_name}.seg.nrrd"
    ### Inputs:
        - nrrd_patients (Path | str): Path to directory containing subdirectories with NRRD files.
        - material_dict (dict | Path | str): A dictionary or path to a JSON file mapping structure names
          to material properties required for egsphant generation.
    ### Outputs:
        - Egsphant files saved in the same directory as the input NRRD files,
          named "egsphant.seq.nrrd" and "ct.egsphant".
    """
    nrrd_patients = Path(nrrd_patients)
    pth_material_dict = Path(pth_material_dict)

    nrrd_patients = list(Path(nrrd_patients).glob("*/"))
    for patient in tqdm(nrrd_patients):
        nrrd_files = list(patient.glob("*.nrrd"))
        for pth in nrrd_files:
            if ".seg.nrrd" in pth.name:
                    pth_seg_nrrd = pth
            elif ".seq.nrrd" in pth.name:
                pth_dose_nrrd = pth
            elif ".nrrd" in pth.name:
                pth_scan_nrrd = pth
            
        phantom_obj = BrachyPhantom(
            pth_phantom_file=pth_scan_nrrd,
            pth_structures_file=pth_seg_nrrd
            )
        phantom_obj.write_to_egsphant(
            pth_output=patient.joinpath("egsphant.seq.nrrd"),
            material_dict=pth_material_dict,
            assign_material_from_ct=False
        )
        phantom_obj.write_to_egsphant(
            pth_output=patient.joinpath("ct.egsphant"),
            material_dict=pth_material_dict,
            assign_material_from_ct=False
        )

def convert_nrrd_dose_to_dicom(
    nrrd_patients: Path | str,
    dicom_patients: Path | str
    ):
    from brachyutils import convert_dose_files
    nrrd_patients = list(Path(nrrd_patients).glob("*/"))
    dicom_patients = Path(dicom_patients)

    for nrrd in nrrd_patients:
        convert_dose_files(
            pth_inputs=[nrrd.joinpath(nrrd.name+".seq.nrrd")],
            type_out=".dcm",
            dir_output=dicom_patients.joinpath(nrrd.name)
            )

if __name__ == "__main__":
    # convert_nrrd_dose_to_dicom(
    #     "temp_data/nrrd_io",
    #     "temp_data/dicom_io"
    # )
    # eval_dicom_io(
    #     "temp_data/dicom_io",
    #     "temp_data/dicom_io"
    # )
    # convert_dicom_to_nrrd(
    #     "YourLocalHome/Data/prostate-glen-2023",
    #     "temp_data/nrrd_io"
    # )
    # eval_nrrd_io(
    #     "temp_data/nrrd_io",
    # )
    # eval_nifti_io()
    # generate_egsphants(
    #     "temp_data/nrrd_io",
    #     "admin/constants/structure_materials_prostate.json"
    # )
    eval_egs_io(
        "temp_data/nrrd_io",
        "temp_data/egs_io"
    )