from pathlib import Path
from brachyutils import BrachyDose, BrachyPlan, BrachyPhantom

if __name__ == "__main__":
    dir_test_export = Path("data_test/test_export_plan/hecktor")

    # # set the home and data directories
    dir_home = Path("/home/ubuntu/YourLocalHome")
    dir_hecktor_nifti = dir_home / "Data/HECKTOR25/test/CHUM-001"
    # # set the path to the Nifti files
    pth_ct = dir_hecktor_nifti / "CHUM-001__CT.nii.gz"
    pth_seg = dir_hecktor_nifti / "CHUM-001_seg.nii.gz"
    pth_dose = dir_hecktor_nifti / "CHUM-001__RTDOSE.nii.gz"
    
    # # load the CT and structures in the digital phantom object
    phantom_obj = BrachyPhantom(
        pth_phantom_file=pth_ct,
        pth_structures_file=pth_seg,
    )
    # # for debugging
    phantom_obj.export_to(dir_nrrd_out=dir_test_export)
    # }

    dose_obj = BrachyDose(
        pth_dose_file=pth_dose,
    )
    # # for debugging
    dose_obj.write_brachydose_to_file(
        pth_dose_file=dir_test_export/"dose.seq.nrrd")
    
    # # create a plan from phantom and dose
    plan_obj = BrachyPlan(
        phantom=phantom_obj,
        combined_dose=dose_obj,
        dvh_metric_goals=None,
        prescription_dose=None,
    )