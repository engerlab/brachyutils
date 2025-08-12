from pathlib import Path
from brachyutils.geometry.registration_utils.reg_opentps import Registration_OpenTPS
from brachyutils.geometry.phantom_utils import BrachyPhantom

def test_register_opentps():
    pth_static = Path("temp_data/registration/us_case000000.nrrd")
    pth_moving = Path("temp_data/registration/mr_case000000.nrrd")
    dir_phant_export = Path("temp_data/registration")

    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_static,
        pth_structures_file=pth_static.with_suffix(".seg.nrrd"),
        )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving,
        pth_structures_file=pth_moving.with_suffix(".seg.nrrd")
        )

    static_phantom.export_to(
        dir_nrrd_out=dir_phant_export
    )
    moving_phantom.export_to(
        dir_nrrd_out=dir_phant_export
    )

    mode = {"deformable": False, "algorithm": None} # this is trash
    # mode = {"deformable": True, "algorithm": "quick"}
    # mode = {"deformable": True, "algorithm": "demons"}
    # mode = {"deformable": True, "algorithm": "morphons"}

    registration_obj = Registration_OpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        register_on_contour="Prostate",
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    registration_obj.register(dir_phantom_export=dir_phant_export)

    reg_eval = registration_obj.evaluate_on_contours()
    print(reg_eval)
    # if reg_eval.get("Dice").get("mean") < 0.5:
    #     raise ValueError("Dice score is less than 0.5")

def test_register_plastimatch():
    from brachyutils.geometry.registration_utils.reg_plastimatch import Registration_Plastimatch
    pth_static = Path("temp_data/registration/us_case000000.nrrd")
    pth_moving = Path("temp_data/registration/mr_case000000.nrrd")
    dir_phant_export = Path("temp_data/registration")
    
    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_static,
        pth_structures_file=pth_static.with_suffix(".seg.nrrd"),
        )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving,
        pth_structures_file=pth_moving.with_suffix(".seg.nrrd")
        )
    registration_obj = Registration_Plastimatch(
        pth_plastimatch="http://192.168.1.13:8000",
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        backend="plastimatch",
        register_on_contour="Prostate",
    )
    registration_obj.register(dir_phantom_export=dir_phant_export)

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
    pth_static = Path("temp_data/registration/us_case000000.nrrd")
    pth_moving = Path("temp_data/registration/mr_case000000.nrrd")
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
        register_on_contour="Prostate"
    )

    registration_obj.register(dir_phantom_export=pth_output.parent)
    print(registration_obj.evaluate_on_contours())

if __name__ == "__main__":
    print("testing the registration class")
    test_register_opentps()
    # test_register_plastimatch()
    # test_load_transformations()
    # test_register_simple_elastix()