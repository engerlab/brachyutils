from pathlib import Path
from brachyutils.image_processing_utils import OpenTPS
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

    # mode = {"deformable": False, "algorithm": None}
    mode = {"deformable": True, "algorithm": "quick"}
    # mode = {"deformable": True, "algorithm": "demons"}
    # mode = {"deformable": True, "algorithm": "morphons"}

    registration_obj = OpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    deformed_phantom, _ = registration_obj.register()
    deformed_phantom.write_image_to_nrrd(pth_output)
    # static_phantom.write_image_to_nrrd(Path.joinpath(pth_output.parent, pth_img_static.name.split(".")[0]+".nrrd"))
    # moving_phantom.write_image_to_nrrd(Path.joinpath(pth_output.parent, pth_img_moving.name.split(".")[0]+".nrrd"))

if __name__ == "__main__":
    print("testing the registration class")
    test_register_opentps()