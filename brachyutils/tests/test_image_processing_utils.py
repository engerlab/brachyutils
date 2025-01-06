from pathlib import Path
from brachyutils.image_processing_utils import RegistrationWithOpenTPS
from brachyutils.geometry_utils import BrachyPhantom

def test_register_opentps():
    pth_img_static = Path("../data_test/registration_prostate_mr_us/train_us_image_case000000.nii.gz")
    pth_img_moving = Path("../data_test/registration_prostate_mr_us/train_mr_image_case000000.nii.gz")
    pth_label_static = Path("../data_test/registration_prostate_mr_us/train_us_label_case000000.nii.gz")
    pth_label_moving = Path("../data_test/registration_prostate_mr_us/train_mr_label_case000000.nii.gz")
    pth_output = "../data_test/registration_prostate_mr_us/train_"

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

    mode = {"deformable": False, "algorithm": "quick"}
    mode = {"deformable": False, "algorithm": "demons"}
    mode = {"deformable": False, "algorithm": "morphons"}
    mode = {"deformable": True, "algorithm": None}

    registration_obj = RegistrationWithOpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    deformed_phantom, _ = registration_obj.register()
    deformed_phantom.write_image_to_nrrd(pth_output)

if __name__ == "__main__":
    print("testing the registration class")
    test_register_opentps()