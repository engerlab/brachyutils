from pathlib import Path
from brachyutils.image_processing_utils import Registration_OpenTPS
from brachyutils.geometry_utils import BrachyPhantom

def test_register_opentps():
    # Abdominal: static = CT, moving = MR
    pth_img_static = Path("../data_test/registration/abdomin_mr_ct/tr_ct_image_0001.nii.gz")
    pth_label_static = Path("../data_test/registration/abdomin_mr_ct/tr_ct_label_0001.nii.gz")
    pth_img_moving = Path("../data_test/registration/abdomin_mr_ct/tr_mr_image_0001.nii.gz")
    pth_label_moving = Path("../data_test/registration/abdomin_mr_ct/tr_mr_label_0001.nii.gz")
    pth_output = Path("../data_test/test_export_plan/abdomin_mr_ct/registered_abdomin_ct_mr.nrrd")

    # prostate: static = US, moving = MR
    # pth_img_static = Path("../data_test/registration_prostate_mr_us/train_us_image_case000000.nii.gz")
    # pth_img_moving = Path("../data_test/registration_prostate_mr_us/train_mr_image_case000000.nii.gz")
    # pth_label_static = Path("../data_test/registration_prostate_mr_us/train_us_label_case000000.nii.gz")
    # pth_label_moving = Path("../data_test/registration_prostate_mr_us/train_mr_label_case000000.nii.gz")

    # prostate: static = MR, moving = US 
    # pth_img_static = Path("../data_test/registration_prostate_mr_us/train_mr_image_case000000.nii.gz")    
    # pth_img_moving = Path("../data_test/registration_prostate_mr_us/train_us_image_case000000.nii.gz")
    # pth_label_static = Path("../data_test/registration_prostate_mr_us/train_mr_label_case000000.nii.gz")
    # pth_label_moving = Path("../data_test/registration_prostate_mr_us/train_us_label_case000000.nii.gz")
    # pth_output = Path("../data_test/test_export_plan/prostate/registered_phantom_us_mr.nrrd")
    for pth in [pth_img_static, pth_img_moving, pth_label_static, pth_label_moving]:
        assert pth.exists(), f"File {pth} does not exist."

    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_img_static,
        pth_structures_file=pth_label_static,
        )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_img_moving,
        pth_structures_file=pth_label_moving
        )

    # mode = {"deformable": False, "algorithm": None} # this is trash
    mode = {"deformable": True, "algorithm": "quick"}
    # mode = {"deformable": True, "algorithm": "demons"}
    # mode = {"deformable": True, "algorithm": "morphons"}

    registration_obj = Registration_OpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        # register_on_contour="Segment1_Name",
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    registration_obj.register()
    registration_obj.export_to(pth_output.parent)

    static_phantom.export_to(
        dir_nrrd_out=pth_output.parent
    )
    moving_phantom.export_to(
        dir_nrrd_out=pth_output.parent
    )

def test_register_plastimatch():
    from brachyutils.image_processing_utils import Registration_Plastimatch
    # Abdominal: static = CT, moving = MR
    pth_img_static = Path("../data_test/registration/abdomin_mr_ct/tr_ct_image_0001.nii.gz")
    pth_label_static = Path("../data_test/registration/abdomin_mr_ct/tr_ct_label_0001.nii.gz")
    pth_img_moving = Path("../data_test/registration/abdomin_mr_ct/tr_mr_image_0001.nii.gz")
    pth_label_moving = Path("../data_test/registration/abdomin_mr_ct/tr_mr_label_0001.nii.gz")
    pth_output = Path("../data_test/test_export_plan/abdomin_mr_ct/registered_abdomin_ct_mr.nrrd")
    
    for pth in [pth_img_static, pth_img_moving, pth_label_static, pth_label_moving]:
        assert pth.exists(), f"File {pth} does not exist."

    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_img_static,
        pth_structures_file=pth_label_static,
        )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_img_moving,
        pth_structures_file=pth_label_moving
        )
    registration_obj = Registration_Plastimatch(
        pth_plastimatch="http://192.168.1.13:8000/plastimatch_register",
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        backend="plastimatch",
    )
    registration_obj.register()

if __name__ == "__main__":
    print("testing the registration class")
    test_register_opentps()
    # test_register_plastimatch()