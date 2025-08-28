from pathlib import Path
from time import time
from typing import Literal
import pandas as pd
from brachyutils import BrachyPhantom
from brachyutils import BrachyDose

def time_phantom_io(
        dir_out: Path,
        type_out: Literal["dicom", "nrrd", "nifti", "egs"],
        dir_dicom: Path = None,
        pth_phantom_file: Path = None,
        pth_structures_file: Path = None,
    ):
        t0_read = time()
        phantom_obj = BrachyPhantom(
            dir_dicom=dir_dicom,
            pth_phantom_file=pth_phantom_file,
            pth_structures_file=pth_structures_file
            )
        tf_read = time()
        if type_out == "dicom":
            phantom_obj.export_to(dir_dicom_out=dir_out)
        tf_write = time()
        return (tf_read - t0_read, tf_write - tf_read)

def time_dose_io(
    pth_dose_in: Path,
    pth_dose_out: Path
    ):
    t0_read = time()
    dose_obj = BrachyDose(
        pth_dose_file=pth_dose_in
    )
    tf_read = time()
    dose_obj.write_brachydose_to_file(
        pth_dose_file=pth_dose_out
    )
    tf_write = time()
    return (tf_read - t0_read, tf_write - tf_read)

def eval_dicom_io():
    dir_dicoms = list(Path().home().joinpath("YourLocalHome/Data/prostate-glen-2023").glob("*/"))
    dir_out = Path("temp_data/dicom_io")
    timing_df = pd.DataFrame(columns=[
        "read_time_scan", "write_time_scan",
        "read_time_scan+seg", "write_time_scan+seg",
        "read_time_dose", "write_time_dose"
        ], index=[dicom.name for dicom in dir_dicoms] + ["average", "std"])
    
    for dicom in dir_dicoms:
        if "body" in dicom.name.lower():
            continue
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
        timing_df.loc[dicom.name] = [t_read_scan, t_write_scan, t_read_scan_seg, t_write_scan_seg, t_read_dose, t_write_dose]

    timing_df.loc["average"] = timing_df.mean()
    timing_df.loc["std"] = timing_df.std()
    timing_df.to_csv(dir_out.joinpath("timing_dicom_io.csv"))

def eval_nrrd_io():
    pass

def eval_nifti_io():
    pass

def eval_egs_io():
    pass


if __name__ == "__main__":
    eval_dicom_io()
    # eval_nrrd_io()
    # eval_nifti_io()
    # eval_egs_io()