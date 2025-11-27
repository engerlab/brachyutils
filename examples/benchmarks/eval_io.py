from pathlib import Path
from time import time
from typing import Literal
import pandas as pd
from tqdm import tqdm
from brachyutils import BrachyPhantom
from brachyutils import BrachyDose
from brachyutils import BrachyEgsphant

def generate_egsphants(
    nrrd_patients: Path | str,
    pth_material_dict: Path | str,
    multi_thread: bool = False,
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
    if not multi_thread:
        for patient in tqdm(nrrd_patients):
            _gen_one_egsphant(
                pth_patient=patient,
                pth_material_dict=pth_material_dict,
            )
    else:
        from multiprocessing import Pool, cpu_count
        from functools import partial
        num_cores = cpu_count() - 2
        with Pool(num_cores) as p:
            
            r = list(tqdm(p.imap(
                func=partial(
                    _gen_one_egsphant, 
                    pth_material_dict=pth_material_dict,
                ),
                iterable=nrrd_patients
                ), total=len(nrrd_patients)
            ))

def _gen_one_egsphant(
    pth_patient: Path,
    pth_material_dict: Path,
    ):
    """
    ### Purpose:
    - Generate egsphant files for a single patient from NRRD files. 
    both segmentation-based and CT-based egsphants are created.
    ### inputs:
    - pth_patient (Path): path to patient directory containing NRRD files.
    - pth_material_dict (Path): path to material dictionary file.
    ### Outputs:
    - Egsphant files saved in the patient directory,
      named "seg_egsphant.seq.nrrd" and "ct_egsphant.seq.nrrd".
    """
    nrrd_files = list(pth_patient.glob("*.nrrd"))
    for pth in nrrd_files:
        if ".seg.nrrd" in pth.name:
                pth_seg_nrrd = pth
        elif ".seq.nrrd" in pth.name:
            pth_dose_nrrd = pth
        elif ".nrrd" in pth.name and "egsphant" not in pth.name:
            pth_scan_nrrd = pth
    phantom_obj = BrachyPhantom(
        pth_phantom_file=pth_scan_nrrd,
        pth_structures_file=pth_seg_nrrd
        )
    phantom_obj.write_to_egsphant(
        pth_output=pth_patient.joinpath("seg_egsphant.seq.nrrd"),
        material_dict=pth_material_dict,
        assign_material_from_ct=False
    )
    phantom_obj.write_to_egsphant(
        pth_output=pth_patient.joinpath("ct_egsphant.seq.nrrd"),
        material_dict=pth_material_dict,
        assign_material_from_ct=True
    )


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
        file_size_mb = pth_dose_out.stat().st_size / (1024 * 1024)
    except Exception as e:
        t0_write, tf_write = (float("nan"), float("nan"))
    return (tf_read - t0_read, tf_write - t0_write, file_size_mb)

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
    #     egsphant_obj = None
    #     t0_read, tf_read = (float("nan"), float("nan"))
    # try:
        t0_write = time()
        egsphant_obj.write_to_file(
            fileName=pth_egsphant_out
        )
        tf_write = time()
        filesize_mb = pth_egsphant_out.stat().st_size / (1024 * 1024)
    # except Exception as e:
    #     t0_write, tf_write = (float("nan"), float("nan"))
        return (tf_read - t0_read, tf_write - t0_write, filesize_mb)

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

def convert_nrrd_to_dicom(
    nrrd_patients: Path | str,
    dicom_patients: Path | str
    ):
    """
    ### Purpose:
    - To convert NRRD image, segmentation and dose files to DICOM format.
    ### Inputs:
        - nrrd_patients (Path | str): Path to directory containing subdirectories with NRRD files.
        - dicom_patients (Path | str): Path to directory to save converted DICOM files.
    ### Outputs:
        - DICOM files saved in the specified output directory,
          with subdirectories named after the original NRRD directories.
    """ 
    nrrd_patients = list(Path(nrrd_patients).glob("*/"))
    for patient_nrrd in nrrd_patients:
        data_nrrd = list(patient_nrrd.glob("*.nrrd"))
        data_img = list(filter(
            lambda x: "seq" not in x.name and ".seg.nrrd" not in x.name,
            data_nrrd)).pop()
        data_seg = list(filter(lambda x: ".seg.nrrd" in x.name, data_nrrd)).pop()
        data_seq = list(filter(lambda x: ".seq.nrrd" in x.name, data_nrrd))
        data_dose = list(filter(lambda x: "egsphant" not in x.name, data_seq)).pop()
        
        # now convert phantom to dicom
        BrachyPhantom(
            pth_phantom_file=data_img,
            pth_structures_file=data_seg
        ).export_to(
            dir_dicom_out=Path(dicom_patients).joinpath(patient_nrrd.name)
        )
        # convert does to dicom
        pth_dose_dicom_out = Path(dicom_patients).joinpath(
            patient_nrrd.name).joinpath("RD.dcm")
        BrachyDose(
            pth_dose_file=data_dose
        ).write_brachydose_to_file(
            pth_dose_file=pth_dose_dicom_out
        )

def convert_nrrd_to_egs(
    nrrd_patients: Path | str,
    egs_patients: Path | str,
    ):
    """
    ### Purpose:
    - To convert egsphant and dose files from NRRD format to EGS format.
    ### Inputs:
        - nrrd_patients (Path | str): Path to directory containing subdirectories with
        ct_egsphant.seq.nrrd and seg_egsphant.seq.nrrd files.
        - egs_patients (Path | str): Path to directory to save converted EGS files.
    ### Outputs:
        - Egsphant and dose files saved in the specified output directory,
          with subdirectories named after the original NRRD directories.
    """
    nrrd_patients = list(Path(nrrd_patients).glob("*/"))
    for patient_nrrd in nrrd_patients:
        data_nrrd = list(patient_nrrd.glob("*.nrrd"))
        data_seq = list(filter(lambda x: ".seq.nrrd" in x.name, data_nrrd))
        data_egsphant = list(filter(lambda x: "egsphant" in x.name, data_seq)).pop()
        data_dose = list(filter(lambda x: "egsphant" not in x.name, data_seq)).pop()
        # now convert phantom to egsphant
        

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
        "read_time_dose", "write_time_dose", "file_size_dose"
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
        # try:
        pth_dose_file = list(nrrd.glob("*.seq.nrrd"))
        for pth in pth_dose_file:
            if "egsphant" not in pth.name:
                pth_dose_file = pth
        t_read_dose, t_write_dose, file_size_dose = time_dose_io(
            pth_dose_in=pth_dose_file,
            pth_dose_out=nrrd.joinpath(nrrd.name).joinpath(pth_dose_file.name)
        )
        # except:
        #     timing_df.loc[nrrd.name] = [
        #     t_read_scan, t_write_scan,
        #     t_read_scan_seg, t_write_scan_seg,
        #     float("nan"), float("nan")
        #     # t_read_dose, t_write_dose
        #     ]
        #     continue
        timing_df.loc[nrrd.name] = [
            t_read_scan, t_write_scan,
            t_read_scan_seg, t_write_scan_seg,
            t_read_dose, t_write_dose, file_size_dose,
            ]
    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_nrrds[0].parent.joinpath("timing_nrrd_io_dose.csv"))

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
        "read_time_egsphant", "write_time_egsphant", "file_size_egsphant",
        "read_time_egsphant_nrrd", "write_time_egsphant_nrrd", "file_size_egsphant_nrrd"
        ], index=[egs.name for egs in egs_patients] + ["average", "std"])
    for patient in tqdm(egs_patients):
        pth_ct_egsphant = list(patient.glob("ct.egsphant")).pop()
        pth_egsphant_nrrd = list(patient.glob("egsphant.seq.nrrd")).pop()
        t_read_ct_egs, t_write_ct_egs, file_size_ct = time_egsphant_io(
            pth_egsphant_in=pth_ct_egsphant,
            pth_egsphant_out=dir_out.joinpath(f"{patient.name}/ct.egsphant")
        )
        t_read_egs_nrrd, t_write_egs_nrrd, file_size_nrrd = time_egsphant_io(
            pth_egsphant_in=pth_egsphant_nrrd,
            pth_egsphant_out=dir_out.joinpath(f"{patient.name}/egsphant.seq.nrrd")
        )
        timing_df.loc[patient.name] = [
            t_read_ct_egs, t_write_ct_egs, file_size_ct,
            t_read_egs_nrrd, t_write_egs_nrrd, file_size_nrrd
        ]
        # break
    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_out.joinpath("timing_egs_io.csv"))

def eval_3ddose_io(
    patients_3ddose:Path | str,
    dir_out: Path | str = None
    ):
    """
    To time the reading and writing of 3ddose files.
    """
    patients_3ddose = Path(patients_3ddose)
    dir_out = Path(dir_out)
    patients_3ddose = list(patients_3ddose.glob("*/"))
    timing_df = pd.DataFrame(columns=[
        "read_time_3ddose", "write_time_3ddose", "file_size_3ddose"
        ], index=[pat.name for pat in patients_3ddose] + ["average", "std"])
    for patient in tqdm(patients_3ddose):
        pth_3ddose = list(patient.glob("*.3ddose")).pop()
        t_read_3ddose, t_write_3ddose = time_dose_io(
            pth_dose_in=pth_3ddose,
            pth_dose_out=dir_out.joinpath(f"{patient.name}/{pth_3ddose.name}")
        )
        file_size_3ddose = pth_3ddose.stat().st_size / (1024 * 1024)
        timing_df.loc[patient.name] = [
            t_read_3ddose, t_write_3ddose, file_size_3ddose
        ]
        # break
    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_out.joinpath("timing_3ddose_io.csv"))

if __name__ == "__main__":
    
    pth_nrrd_data = "temp_data/bench_io/nrrd_io"
    pth_dicom_data = "temp_data/bench_io/dicom_io"
    pth_egs_data = "temp_data/bench_io/egs_io"
    pth_material_dict = "admin/constants/structure_materials_prostate.json"
    
    # anonymize and convert all data to nrrd. we'll start assuming data is in nrrd.
    # generate egsphants in nrrd format
    generate_egsphants(
        nrrd_patients=pth_nrrd_data,
        pth_material_dict=pth_material_dict,
        multi_thread=False
    )

    # to benchmark dicom io, we convert all data from nrrd to dicom and egs.
    # convert_nrrd_to_dicom(
    #     pth_nrrd_data,
    #     pth_dicom_data
    # )
    # convert_nrrd_to_egs(
    #     pth_nrrd_data,
    #     pth_egs_data
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
    #     # "admin/constants/structure_materials_prostate.json",
    #     "admin/constants/CTtoDensityProstate.txt",
    #     True,
    #     True
    # )
    # eval_egs_io(
    #     "temp_data/nrrd_io",
    #     "temp_data/egs_io"
    # )
    # convert_nrrd_dose_3ddose(
    #     "temp_data/nrrd_io",
    #     "temp_data/3ddose_io"
    # )
    # eval_3ddose_io(
    #     "temp_data/3ddose_io",
    #     "temp_data/3ddose_io"
    # )