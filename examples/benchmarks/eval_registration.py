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

    # # # on abdomen MR-CT
    # dir_static = "temp_data/registration/abdomen-mr-ct/static"
    # dir_moving = "temp_data/registration/abdomen-mr-ct/moving"
    # backend = "OpenTPS"
    # use_contour = "" # None
    # dir_registered_quick = f"temp_data/registration/abdomen-mr-ct/{backend}/{use_contour}/reg-quick"
    # dir_registered_demons = f"temp_data/registration/abdomen-mr-ct/{backend}/{use_contour}/reg-demons"
    # dir_registered_morphons = f"temp_data/registration/abdomen-mr-ct/{backend}/{use_contour}/reg-morphons"
    # # # on micro-reg prostate
    # # dir_static = "temp_data/registration/micro-reg/us-train"
    # # dir_moving = "temp_data/registration/micro-reg/mr-train"
    # # dir_registered = "temp_data/registration/micro-reg/reg-train"

    # # image based registration
    evaluate_registration(
        reg_data_inputs=reg_data_inputs,
        registration_module=Registration_OpenTPS,
        register_on_contour=use_contour,
        multi_thread=True,
        deformable=True,
        algorithm="quick"
    )
    # demons does not work well!
    # evaluate_registration(
    #     dir_static=dir_static,
    #     dir_moving=dir_moving,
    #     dir_registered=dir_registered_demons,
    #     registration_module=Registration_OpenTPS,
    #     # # register_on_contour="Prostate",
    #     multi_thread=False,
    #     deformable=True,
    #     algorithm="demons",
    #     tryGPU=True
    # )
    evaluate_registration(
        dir_static=dir_static,
        dir_moving=dir_moving,
        dir_registered=dir_registered_morphons,
        registration_module=Registration_OpenTPS,
        register_on_contour=use_contour,
        multi_thread=False,
        deformable=True,
        algorithm="morphons",
        tryGPU=True
    )

    # # contour based registration

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

# def organize_data(dir_out: str | Path, multi_thread: bool = False):
#     r"""
#     Purpose:
#         - to gather data from all formats and directories into one static directory,
#         one moving directory, and one registered directory. inside each directory, there
#         is one image .nrrd file and one segmentation file .seg.nrrd. per case.
    
#     Inputs:
#         - dir_out:= the path where the dir_static, dir_moving and dir_registered will be created.
    
#     Outputs:
#         - None 
#     """
#     dir_static_img = Path.home().joinpath("YourLocalHome/Data/registration/AbdomenMRCT/imagesTr")
#     dir_static_seg = Path.home().joinpath("YourLocalHome/Data/registration/AbdomenMRCT/labelsTr")
#     dir_moving_img = Path.home().joinpath("YourLocalHome/Data/registration/AbdomenMRCT/imagesTr")
#     dir_moving_seg = Path.home().joinpath("YourLocalHome/Data/registration/AbdomenMRCT/labelsTr")
    
#     all_static_img = glob(str(dir_static_img.joinpath("*_0001.nii.gz")))
#     all_moving_img = glob(str(dir_moving_img.joinpath("*_0000.nii.gz")))
#     all_static_segs = glob(str(dir_static_seg.joinpath("*_0001.nii.gz")))
#     all_moving_segs = glob(str(dir_moving_seg.joinpath("*_0000.nii.gz")))

#     all_cases = list()
#     for static_img in all_static_img:
#         static_img_name = "_".join(Path(static_img).name.split("_")[0:-1])
#         pth_static_seg = [seg for seg in all_static_segs if static_img_name in seg]
#         pth_moving_img = [img for img in all_moving_img if static_img_name in img]
#         pth_moving_seg = [seg for seg in all_moving_segs if static_img_name in seg]

#         if len(pth_static_seg) == 0 or len(pth_moving_img) == 0 or len(pth_moving_seg) == 0:
#             warnings.warn(f"no corresponding data found for {static_img_name}")
#             continue
#         all_cases.append({
#             "static_img": static_img,
#             "static_seg": pth_static_seg[0],
#             "moving_img": pth_moving_img[0],
#             "moving_seg": pth_moving_seg[0]
#         })

#     dir_out = Path(dir_out)
#     dir_out.mkdir(parents=True, exist_ok=True)
#     dir_static = dir_out.joinpath("static")
#     dir_moving = dir_out.joinpath("moving")
#     dir_registered = dir_out.joinpath("reg")   
#     if multi_thread:
#         import asyncio
#         from concurrent.futures import ThreadPoolExecutor
#         async def run_in_executor(executor, case):
#             loop = asyncio.get_event_loop()
#             try:
#                 return await loop.run_in_executor(executor, export_static_moving_phantoms, case, dir_static, dir_moving)
#             except Exception as e:
#                 print(f"error in exporting {case}")
#                 print(e)
#                 return None

#         async def main():
#             with ThreadPoolExecutor() as executor:
#                 tasks = []
#                 for case in all_cases:
#                     tasks.append(run_in_executor(executor, case))
#                 await asyncio.gather(*tasks)

#         asyncio.run(main())
#     else: 
#         for case in all_cases:
#             try:
#                 export_static_moving_phantoms(case, dir_static, dir_moving)
#                 return
#             except Exception as e:
#                 print(f"error in exporting {case}")
#                 print(e)

# def export_static_moving_phantoms(case: Dict, dir_static: Path, dir_moving: Path):
#     static_phantom = BrachyPhantom(
#         pth_phantom_file=case.get("static_img"),
#         pth_structures_file=case.get("static_seg")
#     )
#     moving_phantom = BrachyPhantom(
#         pth_phantom_file=case.get("moving_img"),
#         pth_structures_file=case.get("moving_seg")
#     )
#     static_phantom.export_to(dir_nrrd_out=dir_static)
#     moving_phantom.export_to(dir_nrrd_out=dir_moving)

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