from pathlib import Path
from glob import glob
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Dict, List, Union, Tuple
import pandas as pd
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
    all_static_structs_nrrd = glob(str(dir_static.joinpath("*.seg.nrrd")))
    
    # islate the segmentatoin and images for both static and moving files
    reg_data_list = list()
    for static_struct in all_static_structs_nrrd:
        common_name = Path(Path(static_struct).stem).stem
        static_image = glob(str(dir_static.joinpath(f"{common_name}.nrrd")))
        moving_image = glob(str(dir_moving.joinpath(f"{common_name}.nrrd")))
        moving_struct = glob(str(dir_moving.joinpath(f"{common_name}.seg.nrrd")))

        if len(static_image) != 1 or len(moving_image) != 1 or len(moving_struct) != 1:
            warnings.warn(f"corresponding data for {static_struct} was not found")
            continue
        single_reg_data = defaultdict(Path)
        single_reg_data["pth_static_image"] = static_image[0]
        single_reg_data["pth_static_structure"] = static_struct
        single_reg_data["pth_moving_image"] = moving_image[0]
        single_reg_data["pth_moving_structure"] = moving_struct[0]
        single_reg_data["dir_registered"] = dir_registered
        single_reg_data["registration_module"] = registration_module
        reg_data_list.append(single_reg_data)

    print(f"number of registration instance data was {len(reg_data_list)}")

    all_dice = defaultdict()
    all_hausdorff = defaultdict()
    if multi_thread:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        async def run_in_executor(executor, single_reg_data):
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(executor, eval_single_registration, single_reg_data)
            except Exception as e:
                print(f"error in evaluating {single_reg_data.get('pth_static_image')}")
                print(e)
                return None

        async def main():
            with ThreadPoolExecutor() as executor:
                tasks = []
                for single_reg_data in reg_data_list:
                    tasks.append(run_in_executor(executor, single_reg_data))
                all_results = await asyncio.gather(*tasks)
            for eval_results in all_results:
                for key, value in eval_results.items():
                    all_dice[key] = value.get("Dice")
                    all_hausdorff[key] = value.get("Hausdorff")
            # for single_result in all_results:
                # all_dice[Path(single_result.get("static_image")).stem] = single_result.get("Dice")
                # all_hausdorff[Path(single_result.get("static_image")).stem] = single_result.get("Hausdorff")
            return all_dice, all_hausdorff

        asyncio.run(main())
    else:
        for single_reg_data in reg_data_list:
            try:
                eval_results = eval_single_registration(
                    essential_inputs = single_reg_data,
                    **kwargs
                )
                all_dice[list(eval_results.keys())[0]] = list(eval_results.values())[0].get("Dice")
                all_hausdorff[list(eval_results.keys())[0]] = list(eval_results.values())[0].get("Hausdorff")
            except Exception as e:
                print(f"error in evaluating {single_reg_data.get('pth_static_image')}")
                print(e)
                continue

    eval_df_dice = pd.DataFrame(all_dice)
    eval_df_hausdorff = pd.DataFrame(all_hausdorff)

    eval_df_dice.to_csv(dir_registered.joinpath("dice.csv"))
    eval_df_hausdorff.to_csv(dir_registered.joinpath("hausdorff.csv"))

def eval_single_registration(
    essential_inputs: Dict,
    **kwargs
):
    r"""
    Purpose:
        - evaluate the registration of the moving image and structures onto the static image.
    Inputs:
        - essential_inputs := dictionary containing the essential inputs for the registration, which are
            - pth_static_image
            - pth_static_structure
            - pth_moving_image
            - pth_moving_structure
            - registration_module
            - dir_registered
    Outputs:
        - dict containing the evaluation results
            - Dice
            - Hausdorff
    """
    static_phantom = BrachyPhantom(
        pth_phantom_file=essential_inputs.get("pth_static_image"),
        pth_structures_file=essential_inputs.get("pth_static_structure")
    )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=essential_inputs.get("pth_moving_image"),
        pth_structures_file=essential_inputs.get("pth_moving_structure")
    )
    
    reg_obj = essential_inputs.get("registration_module")(
        static_phantom = static_phantom,
        moving_phantom = moving_phantom,
        **kwargs
    )

    reg_obj.register(
        pth_phantom_export=essential_inputs.get("dir_registered"),
        **kwargs
    )
    return {Path(essential_inputs.get("pth_static_image")).stem: reg_obj.evaluate_on_contours()}

def run_registeration():
    dir_static = "../temp_data/registration/micro-reg/us-train"
    dir_moving = "../temp_data/registration/micro-reg/mr-train"
    dir_registered = "../temp_data/registration/micro-reg/reg-train"

    from brachyutils.registration_utils import Registration_OpenTPS
    
    evaluate_contourBased_registration(
        dir_static=dir_static,
        dir_moving=dir_moving,
        dir_registered=dir_registered,
        registration_module=Registration_OpenTPS,
        register_on_contour="Prostate",
        multi_thread=True
    )

if __name__ == "__main__":
    # export_phantom_opentps_nrrd_dicom_egsphant()
    run_registeration()