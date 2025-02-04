from pathlib import Path
from glob import glob
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Dict, List, Union, Tuple
from brachyutils.registration_utils import PhantomRegistration
from brachyutils.geometry_utils import BrachyPhantom

def export_phantom_opentps_nrrd_dicom_egsphant():
    from brachyutils.geometry_utils import BrachyPhantom
    pth_img_dicom = Path("../data_test/prostate-glen-p1-dcm")
    pth_strct_dicom = glob(str(pth_img_dicom)+"/RS*.dcm")[0]
    pth_img_nrrd = Path("../data_test/test_export_plan/opentps/prostate_glen_p1.nrrd")
    pth_strct_nrrd = Path("../data_test/test_export_plan/opentps/prostate_glen_p1.seg.nrrd")
    assign_material_from_ct = True
    pth_materials = Path("../data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt")
    phantom = BrachyPhantom(
        dir_dicom=pth_img_dicom,
        pth_structures_file=pth_strct_dicom
    )
    # phantom.export_to(
    #     dir_nrrd_out=pth_img_nrrd.parent
    # )
    # phantom.export_to(
    #     dir_dicom_out=Path.joinpath(pth_img_nrrd.parent, "dicom/")
    # )
    phantom.write_to_egsphant(
        pth_output=pth_img_nrrd.parent.joinpath("egsphant.seq.nrrd"),
        material_dict=pth_materials,
        assign_material_from_ct=assign_material_from_ct
        )

def compare_dose_mc_tg43():
    from brachyutils.dose_generation_utils import DoseMonteCarlo, DoseTG43
    from brachyutils.plan_utils import BrachyPlan

def evaluate_contourBased_registration(
    dir_static: str | Path,
    dir_moving: str | Path,
    dir_registered: str | Path,
    registration_module,
    multi_thread: bool = False,
    **kwargs
):
    r"""
    Purpose:
        - register structures from MRI onto TRUS and compare it with the ground truth 
        structures on TRUS images. The registration is done based on the prostate contour
        and the transformed structures are the biopsy regions.
    
    Inputs:
        - dir_static := directory of the static images and structures. the image file and the structure
        file should have the same name. the extension of the structure file should be .seg.nrrd.
        - dir_moving := same as above, but for moving images.
        - dir_registered := the directory where the registered moving image and the structures is written to.
    """
    if not issubclass(registration_module, PhantomRegistration):
        raise ValueError("registration module should extend the abstract class PhantomRegistration")

    from brachyutils.registration_utils import Registration_OpenTPS
    dir_static = Path(dir_static)
    dir_moving = Path(dir_moving)
    dir_registered = Path(dir_registered)

    # gatheter the data in the path dict
    all_static_structs_nrrd = glob(str(dir_static.joinpath(".seg.nrrd")))
    
    # islate the segmentatoin and images for both static and moving files
    reg_data_list = List(Dict(str, str))
    for static_struct in all_static_structs_nrrd:
        common_name = Path(static_struct.stem).stem
        static_image = glob(dir_static.joinpath(f"{common_name}.nrrd"))
        moving_image = glob(dir_moving.joinpath(f"{common_name}.nrrd"))
        moving_struct = glob(dir_moving.joinpath(f"{common_name}.seg.nrrd"))

        if len(static_image) != 1 or len(moving_image) != 1 or len(moving_struct) != 1:
            warnings.warn(f"corresponding data for {static_struct} was not found")
            continue
        single_reg_data = defaultdict(Path)
        single_reg_data["static_image"] = static_image[0]
        single_reg_data["static_structure"] = static_struct
        single_reg_data["moving_image"] = moving_image[0]
        single_reg_data["moving_structure"] = moving_struct[0]
        single_reg_data["dir_registered_out"] = dir_registered
        reg_data_list.append(single_reg_data)

    print(f"number of registration instance data was {len(reg_data_list)}")

    if multi_thread:
        pass
    else:
        for single_reg_data in reg_data_list:
            eval_single_registration(
                pth_static_image = single_reg_data.get("static_image"),
                pth_static_structure = single_reg_data.get("static_structure"),
                pth_moving_image = single_reg_data.get("moving_image"),
                pth_moving_structure = single_reg_data.get("moving_structure"),
                registration_module=registration_module,
                dir_registered = single_reg_data.get("dir_registered_out")
                **kwargs
            )

def eval_single_registration(
    pth_static_image : Path,
    pth_static_structure : Path,
    pth_moving_image : Path,
    pth_moving_structure : Path,
    registration_module,
    dir_registered,
    **kwargs
):
    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_static_image,
        pth_structures_file=pth_static_structure
    )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving_image,
        pth_structures_file=pth_moving_structure
    )
    
    reg_obj = registration_module(
        static_phantom = static_phantom,
        moving_phantom = moving_phantom,
        **kwargs
    )

    reg_obj.register(
        pth_phantom_export=dir_registered
    )
    return reg_obj.evaluate_on_contours()

if __name__ == "__main__":
    export_phantom_opentps_nrrd_dicom_egsphant()