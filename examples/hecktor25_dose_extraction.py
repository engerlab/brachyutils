from pathlib import Path
from brachyutils import BrachyDose, BrachyPlan, BrachyPhantom, Registration_OpenTPS

if __name__ == "__main__":
    dir_test_export = Path("data_test/test_export_plan/hecktor")

    # # set the home and data directories
    dir_home = Path("/home/ubuntu/YourLocalHome")
    # dir_hecktor_nifti = dir_home / "Data/HECKTOR25/test/CHUM-001"
    dir_hecktor_nifti = dir_home / "Data/HECKTOR25/CTPlanning_RTDose/MDA-402"
    # # set the path to the Nifti files
    pth_ct_static = dir_hecktor_nifti / (dir_hecktor_nifti.name + "__PlanningCT.nii.gz")
    pth_ct_moving = dir_hecktor_nifti / (dir_hecktor_nifti.name + "__CT.nii.gz")
    pth_seg_moving = dir_hecktor_nifti / (dir_hecktor_nifti.name + "_seg.nii.gz")
    pth_dose_static = dir_hecktor_nifti / (dir_hecktor_nifti.name + "__RTDOSE.nii.gz")

    # # load the CT and structures in the digital phantom object
    # phantom_obj_static = BrachyPhantom(
    #     pth_phantom_file=pth_ct_static,
    # )
    # # for debugging
    # phantom_obj_static.export_to(dir_nrrd_out=dir_test_export)
    # exported_phantom = BrachyPhantom(
    #     pth_phantom_file=dir_test_export / (dir_hecktor_nifti.name + "__PlanningCT.nrrd")
    # )
    # print("breaking point")
    # # }
    # registered the moving phantom to the static phantom
    phantom_obj_moving = BrachyPhantom(
        pth_phantom_file=pth_ct_moving,
        pth_structures_file=pth_seg_moving,
    )
    # # for debugging {
    phantom_obj_moving.export_to(
        dir_nrrd_out=dir_test_export
        )
    # }
    # exported_phantom_moving = BrachyPhantom(
    # reg_obj = Registration_OpenTPS(
    #     static_phantom=phantom_obj_static,
    #     moving_phantom=phantom_obj_moving,
    # )
    # phantom_registered = reg_obj.register()
    # # # for debugging {
    # phantom_registered.export_to(dir_nrrd_out=dir_test_export/"phantom_registered")
    # }
    
    # # load the dose file
    # dose_obj = BrachyDose(
    #     pth_dose_file=pth_dose,
    # )
    # # for debugging
    # dose_obj.write_brachydose_to_file(
    #     pth_dose_file=dir_test_export/"dose.seq.nrrd")
    
    # # create a plan from phantom and dose
    # plan_obj = BrachyPlan(
    #     phantom=phantom_obj,
    #     combined_dose=dose_obj,
    #     dvh_metric_goals=None,
    #     prescription_dose=None,
    # )