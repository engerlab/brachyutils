from pathlib import Path
from glob import glob
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Dict, List, Union, Tuple
import pandas as pd
from brachyutils import BrachyPhantomRegistration
from brachyutils import BrachyPhantom

def evaluate_registration(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    registration_module,
    **kwargs
):
    r"""
    ### Purpose:
        - register structures from MRI onto TRUS and compare it with the ground truth 
        structures on TRUS images. The registration is done based on the prostate contour
        and the transformed structures are the biopsy regions.    
    ### Inputs:
        - reg_data_inputs := list of dictionaries containing the paths to the static and moving images and structures
            - pth_static_image := path to the static image file
            - pth_static_structure := path to the static structure file
            - pth_moving_image := path to the moving image file
            - pth_moving_structure := path to the moving structure file
        - registration_module := the registration class that extends BrachyPhantomRegistration
        - kwargs := additional arguments for the registration module
            - dir_registered := directory where the registered moving image and structures are written to.
    ### Outputs:
        - a dictionary containing the average evaluation results, which are:
            "avg_dice", "std_dice"
            "avg_hausdorff", "std_hausdorff"
            "avg_time", "std_time"
    """
    if not issubclass(registration_module, BrachyPhantomRegistration):
        raise ValueError("registration module should extend the abstract class BrachyPhantomRegistration")

    all_dice = defaultdict()
    all_hausdorff = defaultdict()
    for single_reg_data in reg_data_inputs:
        try:
            eval_results = eval_single_registration(
                registration_module=registration_module,
                **single_reg_data,
                **kwargs
            )
            all_dice[list(eval_results.keys())[0]] = list(eval_results.values())[0].get("Dice")
            all_hausdorff[list(eval_results.keys())[0]] = list(eval_results.values())[0].get("Hausdorff")
        except Exception as e:
            print(f"error in evaluating {single_reg_data.get('pth_static_image')}")
            print(e)
            continue
        break

    eval_df_dice = pd.DataFrame(all_dice).transpose()
    eval_df_hausdorff = pd.DataFrame(all_hausdorff).transpose()
    dir_registered.mkdir(exist_ok=True, parents=True)
    eval_df_dice.to_csv(dir_registered.joinpath("dice.csv"))
    eval_df_hausdorff.to_csv(dir_registered.joinpath("hausdorff.csv"))

def eval_single_registration(
    pth_static_image: Path,
    pth_static_structure: Path,
    pth_moving_image: Path,
    pth_moving_structure: Path,
    dir_registered: Path,
    registration_module: BrachyPhantomRegistration,
    **kwargs
):
    r"""
    ### Purpose:
        - evaluate the registration of the moving image and structures onto the static image.
    ### Inputs:
        - pth_static_image := path to the static image file
        - pth_static_structure := path to the static structure file
        - pth_moving_image := path to the moving image file
        - pth_moving_structure := path to the moving structure file
        - dir_registered := directory where the registered moving image and structures are written to.
        - registration_module := the registration class that extends BrachyPhantomRegistration
        - kwargs := additional arguments for the registration module
    ### Outputs:
        - dict containing the evaluation results
            - Dice
            - Hausdorff
    """
    static_phantom = BrachyPhantom(
        pth_phantom_file=pth_static_image,
        pth_structures_file=pth_static_structure
    )
    moving_phantom = BrachyPhantom(
        pth_phantom_file=pth_moving_image,
        pth_structures_file=pth_moving_structure
    )
    reg_obj: BrachyPhantomRegistration = registration_module(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        **kwargs
    )
    reg_obj.register(
        dir_phantom_export=dir_registered,
        **kwargs
    )
    return {pth_static_image.stem: reg_obj.evaluate_on_contours()}

def eval_reg_opentps(
    reg_data_inputs: List[Dict[str, Union[str, Path]]]
):
    from brachyutils import Registration_OpenTPS
    algorithms  = ["rigid", "quick", "demons", "morphons"]
    references = ["image", "Prostate"]

    results_df = pd.DataFrame(
        columns=[
            "algorithm", "package", "reference",
            "avg_dice", "std_dice",
            "avg_hausdorff", "std_hausdorff",
            "avg_time", "std_time"
            ]
    )
    for ref in references:
        if ref == "Prostate":
            use_contour = ref
        else:
            use_contour = None

        for alg in algorithms:
            if alg ==  "rigid":
                deformable = False
            else:
                deformable = True
            reg_results = evaluate_registration(
                reg_data_inputs=reg_data_inputs,
                registration_module=Registration_OpenTPS,
                register_on_contour=use_contour,
                deformable=deformable,
                algorithm=alg
            )
        results_df.loc[len(results_df)] = {
            "algorithm": alg,
            "package": "OpenTPS",
            "reference": ref,
            "avg_dice": reg_results.get("avg_dice"),
            "std_dice": reg_results.get("std_dice"),
            "avg_hausdorff": reg_results.get("avg_hausdorff"),
            "std_hausdorff": reg_results.get("std_hausdorff"),
            "avg_time": reg_results.get("avg_time"),
            "std_time": reg_results.get("std_time")
        }

def run_registration_plastimatch():
    from brachyutils.registration_utils import Registration_Plastimatch

    # # on abdomen MR-CT
    dir_static = "temp_data/registration/abdomen-mr-ct/static"
    dir_moving = "temp_data/registration/abdomen-mr-ct/moving"
    backend = "Plastimatch"
    use_contour = ""
    dir_registered_bspline = f"temp_data/registration/abdomen-mr-ct/{backend}/{use_contour}/reg-bspline"
    pth_plastimatch = "http://192.168.1.13:8000"

    evaluate_registration(
        dir_static=dir_static,
        dir_moving=dir_moving,
        dir_registered=dir_registered_bspline,
        registration_module=Registration_Plastimatch,
        # register_on_contour=use_contour,
        pth_plastimatch=pth_plastimatch,
        # multi_thread=True,
        # deformable=True,
    )

def gen_registration_inputs_microreg(
    dir_all_data: str | Path
):
    r"""
    ### Purpose: 
        - to generate a list of registration input dictionaries from a directory containing all the data.
        we use the micro-reg prostate dataset where the static images are the US images and the moving images are the MRI images.
    ### Inputs:
        - dir_all_data := directory containing all the data
        example registration data directory structure:
        dir_all_data/
            us_case000000.nrrd
            us_case000000.seg.nrrd
            mr_case000000.nrrd
            mr_case000000.seg.nrrd
    ### Outputs:
    reg_data_inputs: List[Dict]
        - a list of registration input dictionaries
            - pth_static_image
            - pth_static_structure
            - pth_moving_image
            - pth_moving_structure
            - dir_registered
            - registration_module
    """
    dir_all_data = Path(dir_all_data)
    moving_segments = list(dir_all_data.glob("mr_*.seg.nrrd"))
    reg_data_inputs = list()
    for moving_seg in moving_segments:
        case_id = moving_seg.stem.split("mr_")[-1].split(".seg")[0]
        moving_image = dir_all_data.joinpath(f"mr_{case_id}.nrrd")
        static_image = dir_all_data.joinpath(f"us_{case_id}.nrrd")
        static_seg = dir_all_data.joinpath(f"us_{case_id}.seg.nrrd")
        if not moving_image.exists() or not static_image.exists() or not static_seg.exists():
            warnings.warn(f"corresponding data for case {case_id} was not found")
            continue
        reg_data_inputs.append({
            "pth_static_image": str(static_image),
            "pth_static_structure": str(static_seg),
            "pth_moving_image": str(moving_image),
            "pth_moving_structure": str(moving_seg),
        })
    if len(reg_data_inputs) == 0:
        raise ValueError(f"no registration data found in {dir_all_data}")
    return reg_data_inputs

if __name__ == "__main__":
    reg_data_inputs = gen_registration_inputs_microreg(
        dir_all_data="temp_data/registration/fixed-nrrd",
        )
    eval_reg_opentps(
        reg_data_inputs=reg_data_inputs
    )
    # organize_data("temp_data/registration/abdomen-mr-ct", True)
    # run_registeration_opentps()
    # run_registration_plastimatch()