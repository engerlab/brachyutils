from pathlib import Path
from brachyutils.geometry.registration_utils.reg_opentps import Registration_OpenTPS
from brachyutils.geometry.phantom_utils import BrachyPhantom

def test_register_opentps():
    # Abdominal: static = CT, moving = MR
    pth_img_static = Path("temp_data/registration/abdomen-mr-ct/static/AbdomenMRCT_0001.nrrd")
    # pth_img_static = Path("data_test/registration/abdomin_mr_ct/tr_ct_image_0001.nii.gz")
    pth_label_static = Path("temp_data/registration/abdomen-mr-ct/static/AbdomenMRCT_0001.seg.nrrd")
    # pth_label_static = Path("data_test/registration/abdomin_mr_ct/tr_ct_label_0001.nii.gz")
    pth_img_moving = Path("temp_data/registration/abdomen-mr-ct/moving/AbdomenMRCT_0001.nrrd")
    # pth_img_moving = Path("data_test/registration/abdomin_mr_ct/tr_mr_image_0001.nii.gz")
    pth_label_moving = Path("temp_data/registration/abdomen-mr-ct/moving/AbdomenMRCT_0001.seg.nrrd")
    # pth_label_moving = Path("data_test/registration/abdomin_mr_ct/tr_mr_label_0001.nii.gz")
    pth_output = Path("data_test/test_export_plan/abdomin_mr_ct/registered_abdomin_ct_mr.nrrd")

    # # prostate: static = US, moving = MR
    # pth_img_static = Path("data_test/registration/prostate_mr_us/train_us_image_case000000.nii.gz")
    # pth_img_moving = Path("data_test/registration/prostate_mr_us/train_mr_image_case000000.nii.gz")
    # pth_label_static = Path("data_test/registration/prostate_mr_us/train_us_label_case000000.nii.gz")
    # pth_label_moving = Path("data_test/registration/prostate_mr_us/train_mr_label_case000000.nii.gz")
    # pth_output = Path("data_test/test_export_plan/prostate/registered_phantom_us_mr.nrrd")

    # prostate: static = MR, moving = US 
    # pth_img_static = Path("data_test/registration/prostate_mr_us/train_mr_image_case000000.nii.gz")    
    # pth_img_moving = Path("data_test/registration/prostate_mr_us/train_us_image_case000000.nii.gz")
    # pth_label_static = Path("data_test/registration/prostate_mr_us/train_mr_label_case000000.nii.gz")
    # pth_label_moving = Path("data_test/registration/prostate_mr_us/train_us_label_case000000.nii.gz")
    # pth_output = Path("data_test/test_export_plan/prostate/registered_phantom_us_mr.nrrd")
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

    static_phantom.export_to(
        dir_nrrd_out=pth_output.parent
    )
    moving_phantom.export_to(
        dir_nrrd_out=pth_output.parent
    )

    # mode = {"deformable": False, "algorithm": None} # this is trash
    mode = {"deformable": True, "algorithm": "quick"}
    # mode = {"deformable": True, "algorithm": "demons"}
    # mode = {"deformable": True, "algorithm": "morphons"}

    registration_obj = Registration_OpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        register_on_contour="common",
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    registration_obj.register() 
    registration_obj.export_to(pth_output.parent)

    reg_eval = registration_obj.evaluate_on_contours()
    if reg_eval.get("Dice").get("mean") < 0.5:
        raise ValueError("Dice score is less than 0.5")

def test_register_plastimatch():
    from brachyutils.geometry.registration_utils.reg_plastimatch import Registration_Plastimatch
    # Abdominal: static = CT, moving = MR
    pth_img_static = Path("temp_data/registration/abdomen-mr-ct/static/AbdomenMRCT_0008.nrrd")
    # pth_img_static = Path("data_test/registration/abdomin_mr_ct/tr_ct_image_0008.nii.gz")
    pth_label_static = Path("temp_data/registration/abdomen-mr-ct/static/AbdomenMRCT_0008.seg.nrrd")
    # pth_label_static = Path("data_test/registration/abdomin_mr_ct/tr_ct_label_0008.nii.gz")
    pth_img_moving = Path("temp_data/registration/abdomen-mr-ct/moving/AbdomenMRCT_0008.nrrd")
    # pth_img_moving = Path("data_test/registration/abdomin_mr_ct/tr_mr_image_0008.nii.gz")
    pth_label_moving = Path("temp_data/registration/abdomen-mr-ct/moving/AbdomenMRCT_0008.seg.nrrd")
    # pth_label_moving = Path("data_test/registration/abdomin_mr_ct/tr_mr_label_0001.nii.gz")
    pth_output = Path("temp_data/registration/abdomen-mr-ct/test")

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
        pth_plastimatch="http://192.168.1.13:8000",
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        backend="plastimatch",
        # register_on_contour="common",
    )
    registration_obj.register(dir_phantom_export=pth_output)

def test_load_transformations():
    pth_transform = Path("data_test/registration/abdomin_mr_ct/plastimatch/vf.nrrd")
    pth_moving_img = Path("data_test/registration/abdomin_mr_ct/plastimatch/moving.nrrd")
    pth_output = Path("data_test/test_export_plan/abdomin_mr_ct/vf_transformed_image.nrrd")
    assert pth_transform.exists(), f"File {pth_transform} does not exist."
    from brachyutils.geometry.registration_utils.reg_plastimatch import _load_deformation_field

    transformation = _load_deformation_field(pth_transform)
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving_img,
        )
    deformed_img = transformation.deformImage(moving_phantom.image_obj)
    assert deformed_img != moving_phantom.image_obj, "Image was not deformed"
    moving_phantom.image_obj = deformed_img
    moving_phantom.export_to(pth_image_out=pth_output)


def test_register_simple_elastix():
    pth_static = Path("temp_data/registration/mr_case000000.nrrd")
    pth_moving = Path("temp_data/registration/us_case000000.nrrd")
    pth_output = Path("temp_data/registration/registered_simple_elastix.nrrd")
    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_static,
        pth_structures_file=pth_static.with_suffix(".seg.nrrd"),
        )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving,
        pth_structures_file=pth_moving.with_suffix(".seg.nrrd"),
        )
    from brachyutils.geometry.registration_utils.reg_simple_elastix import Registration_SimpleElastix

    registration_obj = Registration_SimpleElastix(
        pth_simple_elastix="http://192.168.1.14:8000",
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        backend="simple_elastix",
    )

    registration_obj.register()

if __name__ == "__main__":
    print("testing the registration class")
    # test_register_opentps()
    # test_register_plastimatch()
    # test_load_transformations()
    test_register_simple_elastix()