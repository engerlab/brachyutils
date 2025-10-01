from pathlib import Path
from time import time
import warnings
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from brachyutils import BrachyPhantomRegistration
from brachyutils import BrachyPhantom

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
        register_on_contour=kwargs.get("register_on_contour", None),
        deformable=kwargs.get("deformable", False),
        algorithm=kwargs.get("algorithm", None),
        backend=kwargs.get("backend", None),
        tryGPU=kwargs.get("tryGPU", False),
        pth_executable=kwargs.get("pth_executable", None),
    )
    try:
        t0 = time()
        reg_obj.register(
            dir_phantom_export=dir_registered,
            **kwargs
        )
        t1 = time()
        measured_metrics = reg_obj.evaluate_on_contours()
    except Exception as e:
        # warnings.warn(f"registration failed for case {pth_static_image} with error {e}")
        t0 = 0
        t1 = 0
        measured_metrics = {
            "Dice": {"Prostate": np.nan, "Biopsies": np.nan},
            "Hausdorff": {"Prostate": np.nan, "Biopsies": np.nan},
        }

    return {
        "case": pth_static_image.split("/")[-1].split(".")[0],
        "time": t1 - t0,
        "dice(Prostate)": measured_metrics["Dice"]["Prostate"],
        "hausdorff(Prostate)": measured_metrics["Hausdorff"]["Prostate"],
        "dice(Biopsies)": np.mean([v for k, v in measured_metrics["Dice"].items() if k != "Prostate"]),
        "hausdorff(Biopsies)": np.mean([v for k, v in measured_metrics["Hausdorff"].items() if k != "Prostate"]),
    }

def evaluate_registration(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    registration_module,
    dir_registered: Path,
    **kwargs
):
    r"""
    ### Purpose:
        - register structures from MRI onto TRUS and compare it with the ground truth 
        structures on TRUS images. The registration is done based on the prostate contour
        and the transformed structures are the biopsy regions.    
    ### Inputs:
        - reg_data_inputs := list of dictionaries containing the paths to the static 
        and moving images and structures
            - pth_static_image := path to the static image file
            - pth_static_structure := path to the static structure file
            - pth_moving_image := path to the moving image file
            - pth_moving_structure := path to the moving structure file
        - registration_module := the registration class that extends BrachyPhantomRegistration
        - kwargs := additional arguments for the registration module
            - dir_registered := directory where the registered moving image and structures
            are written to.
    ### Outputs:
        - a dictionary containing the average evaluation results, which are:
            "avg_dice(Prostate)", "std_dice(Prostate)"
            "avg_hausdorff(Prostate)", "std_hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
            "avg_time", "std_time",
            "num_failed"
    """
    if not issubclass(registration_module, BrachyPhantomRegistration):
        raise ValueError("registration module should extend the abstract class \
 BrachyPhantomRegistration")
    results_per_case_df = pd.DataFrame(
        columns=[
            "case",
            "dice(Prostate)",
            "hausdorff(Prostate)",
            "dice(Biopsies)",
            "hausdorff(Biopsies)",
            "time"
        ]
    )
    for single_reg_data in reg_data_inputs:
        eval_results = eval_single_registration(
            registration_module=registration_module,
            dir_registered=dir_registered,
            **single_reg_data,
            **kwargs
        )
        results_per_case_df.loc[len(results_per_case_df)] = {
            "case": single_reg_data.get("case"),
            "dice(Prostate)": eval_results.get("dice(Prostate)"),
            "hausdorff(Prostate)": eval_results.get("hausdorff(Prostate)"),
            "dice(Biopsies)": eval_results.get("dice(Biopsies)"),
            "hausdorff(Biopsies)": eval_results.get("hausdorff(Biopsies)"),
            "time": eval_results.get("time")
        }
        # XXX for debugging
        # break
    return {
        "avg_dice(Prostate)": results_per_case_df["dice(Prostate)"].mean(),
        "std_dice(Prostate)": results_per_case_df["dice(Prostate)"].std(),
        "avg_hausdorff(Prostate)": results_per_case_df["hausdorff(Prostate)"].mean(),
        "std_hausdorff(Prostate)": results_per_case_df["hausdorff(Prostate)"].std(),

        "avg_dice(Biopsies)": results_per_case_df["dice(Biopsies)"].mean(),
        "std_dice(Biopsies)": results_per_case_df["dice(Biopsies)"].std(),
        "avg_hausdorff(Biopsies)": results_per_case_df["hausdorff(Biopsies)"].mean(),
        "std_hausdorff(Biopsies)": results_per_case_df["hausdorff(Biopsies)"].std(),

        "avg_time": results_per_case_df["time"].mean(),
        "std_time": results_per_case_df["time"].std(),
        "num_failed": np.sum(results_per_case_df["time"] == 0)
    }

def eval_reg_opentps(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    dir_results: Path | str
):
    from brachyutils import Registration_OpenTPS

    # for debugging
    # algorithms  = ["rigid"]
    # references = ["Prostate"]
    algorithms  = ["rigid", "quick", "demons", "morphons"]
    references = ["Image", "Prostate"]
    dir_registered = Path(dir_results)/"OpenTPS"
    results_df = pd.DataFrame(
        columns=[
            "algorithm", "package", "reference",
            "avg_dice(Prostate)", "std_dice(Prostate)",
            "avg_hausdorff(Prostate)", "std_hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
            "avg_time", "std_time",
            "num_failed"
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
#             print(f"Running OpenTPS registration with algorithm: \
# {alg}, reference: {ref}, deformable: {deformable}")
#             print(f"Results will be saved to {dir_registered/ref/alg}")
            reg_results = evaluate_registration(
                reg_data_inputs=reg_data_inputs,
                registration_module=Registration_OpenTPS,
                dir_registered=dir_registered / ref / alg,
                register_on_contour=use_contour,
                deformable=deformable,
                algorithm=alg
            )
            results_df.loc[len(results_df)] = {
                "algorithm": alg,
                "package": "OpenTPS",
                "reference": ref,

                "avg_dice(Prostate)": reg_results.get("avg_dice(Prostate)"),
                "std_dice(Prostate)": reg_results.get("std_dice(Prostate)"),
                "avg_hausdorff(Prostate)": reg_results.get("avg_hausdorff(Prostate)"),
                "std_hausdorff(Prostate)": reg_results.get("std_hausdorff(Prostate)"),

                "avg_dice(Biopsies)": reg_results.get("avg_dice(Biopsies)"),
                "std_dice(Biopsies)": reg_results.get("std_dice(Biopsies)"),
                "avg_hausdorff(Biopsies)": reg_results.get("avg_hausdorff(Biopsies)"),
                "std_hausdorff(Biopsies)": reg_results.get("std_hausdorff(Biopsies)"),

                "avg_time": reg_results.get("avg_time"),
                "std_time": reg_results.get("std_time"),
                "num_failed": reg_results.get("num_failed")
            }
            results_df.to_csv(dir_registered/"registration_results_opentps.csv", index=False)

def eval_reg_plastimatch(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    dir_results: Path | str
):
    from brachyutils import Registration_Plastimatch

    # for debugging
    # algorithms  = ["bspline"]
    # references = ["Prostate"]
    algorithms  = ["translation", "bspline"]
    references = ["Image", "Prostate"]
    dir_registered = Path(dir_results)/"Plastimatch"
    results_df = pd.DataFrame(
        columns=[
            "algorithm", "package", "reference",
            "avg_dice(Prostate)", "std_dice(Prostate)",
            "avg_hausdorff(Prostate)", "std_hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
            "avg_time", "std_time",
            "num_failed"
            ]
    )
    backend = "Plastimatch"
    pth_executable = "http://192.168.1.13:8000"
    for ref in references:
        if ref == "Prostate":
            use_contour = ref
        else:
            use_contour = None

        for alg in algorithms:
            if alg ==  "translation":
                deformable = False
                stage_params_list = [
                    {
                        "xform": "translation",
                        "impl": "plastimatch",
                        "optim": "grid_search",
                    }
                ]
            else:
                deformable = True
                stage_params_list = [
                    {
                        "xform": "bspline",
                        "impl": "plastimatch",
                        "optim": "lbfgsb",
                    }
                ]
            #print(f"Running OpenTPS registration with algorithm: \
# {alg}, reference: {ref}, deformable: {deformable}")
#             print(f"Results will be saved to {dir_registered/ref/alg}")
            reg_results = evaluate_registration(
                reg_data_inputs=reg_data_inputs,
                registration_module=Registration_Plastimatch,
                dir_registered=dir_registered / ref / alg,
                register_on_contour=use_contour,
                deformable=deformable,
                algorithm=alg,
                pth_executable=pth_executable,
                backend=backend,
                stage_params_list=stage_params_list
            )
            results_df.loc[len(results_df)] = {
                "algorithm": alg,
                "package": "OpenTPS",
                "reference": ref,

                "avg_dice(Prostate)": reg_results.get("avg_dice(Prostate)"),
                "std_dice(Prostate)": reg_results.get("std_dice(Prostate)"),
                "avg_hausdorff(Prostate)": reg_results.get("avg_hausdorff(Prostate)"),
                "std_hausdorff(Prostate)": reg_results.get("std_hausdorff(Prostate)"),

                "avg_dice(Biopsies)": reg_results.get("avg_dice(Biopsies)"),
                "std_dice(Biopsies)": reg_results.get("std_dice(Biopsies)"),
                "avg_hausdorff(Biopsies)": reg_results.get("avg_hausdorff(Biopsies)"),
                "std_hausdorff(Biopsies)": reg_results.get("std_hausdorff(Biopsies)"),

                "avg_time": reg_results.get("avg_time"),
                "std_time": reg_results.get("std_time"),
                "num_failed": reg_results.get("num_failed")
            }
            results_df.to_csv(dir_registered/"registration_results_plastimatch.csv", index=False)
            # break #XXX only do one for now

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

def get_baseline_stats_microreg(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    pth_results_csv: Path | str
):
    r"""
    ### Purpose:
        - to get the baseline stats for the moving structures without any registration.
    ### Inputs:
        - reg_data_inputs := list of dictionaries containing the paths to the static 
        and moving images and structures
            - pth_static_image := path to the static image file
            - pth_static_structure := path to the static structure file
            - pth_moving_image := path to the moving image file
            - pth_moving_structure := path to the moving structure file
    ### Outputs:
        - None:= a csv file with the following columns:
            "case",
            "volume(Prostate_US)",
            "volume(Prostate_MR)",
            "avg_Volume(Biopsies_US)", "std_Volume(Biopsies_US)",
            "avg_Volume(Biopsies_MR)", "std_Volume(Biopsies_MR)",
            "dice(Prostate)", "hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
    """
    results_df = pd.DataFrame(
        columns=[
            "case",
            "volume(Prostate_US)",
            "volume(Prostate_MR)",
            "avg_Volume(Biopsies_US)", "std_Volume(Biopsies_US)",
            "avg_Volume(Biopsies_MR)", "std_Volume(Biopsies_MR)",
            "dice(Prostate)", "hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
        ]
    )
    for data_pair in reg_data_inputs:
        static_phantom = BrachyPhantom(
            pth_phantom_file=data_pair.get("pth_static_image"),
            pth_structures_file=data_pair.get("pth_static_structure")
        )
        moving_phantom = BrachyPhantom(
            pth_phantom_file=data_pair.get("pth_moving_image"),
            pth_structures_file=data_pair.get("pth_moving_structure")
        )
        reg_obj = DummyRegistration(
            static_phantom=static_phantom,
            moving_phantom=moving_phantom
        )
        reg_obj.registered_phantom = moving_phantom.resample_to(
            origin=static_phantom.image_obj.origin,
            spacing=static_phantom.image_obj.spacing,
            gridSize=static_phantom.image_obj.gridSize,
        )
        measured_metrics = reg_obj.evaluate_on_contours()
        structure_volumes_us = reg_obj.static_phantom.get_structures_volume(
            structure_names=reg_obj.static_phantom.structure_names
            )
        structure_volumes_mr = reg_obj.moving_phantom.get_structures_volume(
            structure_names=reg_obj.moving_phantom.structure_names
            )

        results_df.loc[len(results_df)] = {
            "case": data_pair.get("pth_static_image").split("/")[-1].split(".")[0],

            "volume(Prostate_US)": structure_volumes_us.get("Prostate", np.nan),
            "avg_Volume(Biopsies_US)": np.mean([v for k, v in structure_volumes_us.items() if k != "Prostate"]),
            "std_Volume(Biopsies_US)": np.std([v for k, v in structure_volumes_us.items() if k != "Prostate"]),

            "volume(Prostate_MR)": structure_volumes_mr.get("Prostate", np.nan),
            "avg_Volume(Biopsies_MR)": np.mean([v for k, v in structure_volumes_mr.items() if k != "Prostate"]),
            "std_Volume(Biopsies_MR)": np.std([v for k, v in structure_volumes_mr.items() if k != "Prostate"]),

            "dice(Prostate)": measured_metrics["Dice"]["Prostate"],
            "hausdorff(Prostate)": measured_metrics["Hausdorff"]["Prostate"],
            "avg_dice(Biopsies)": np.mean([v for k, v in measured_metrics["Dice"].items() if k != "Prostate"]),
            "std_dice(Biopsies)": np.std([v for k, v in measured_metrics["Dice"].items() if k != "Prostate"]),
            "avg_hausdorff(Biopsies)": np.mean([v for k, v in measured_metrics["Hausdorff"].items() if k != "Prostate"]),
            "std_hausdorff(Biopsies)": np.std([v for k, v in measured_metrics["Hausdorff"].items() if k != "Prostate"]),
        }
    mean_dict = {
        "case": "mean",
        "volume(Prostate_US)": results_df["volume(Prostate_US)"].mean(),
        "volume(Prostate_MR)": results_df["volume(Prostate_MR)"].mean(),

        "avg_Volume(Biopsies_US)": results_df["avg_Volume(Biopsies_US)"].mean(),
        "std_Volume(Biopsies_US)": results_df["std_Volume(Biopsies_US)"].mean(),
        
        "avg_Volume(Biopsies_MR)": results_df["avg_Volume(Biopsies_MR)"].mean(),
        "std_Volume(Biopsies_MR)": results_df["std_Volume(Biopsies_MR)"].mean(),
        
        "dice(Prostate)": results_df["dice(Prostate)"].mean(),
        "hausdorff(Prostate)": results_df["hausdorff(Prostate)"].mean(),
        "avg_dice(Biopsies)": results_df["avg_dice(Biopsies)"].mean(),
        "std_dice(Biopsies)": results_df["std_dice(Biopsies)"].mean(),
        "avg_hausdorff(Biopsies)": results_df["avg_hausdorff(Biopsies)"].mean(),
        "std_hausdorff(Biopsies)": results_df["std_hausdorff(Biopsies)"].mean(),        
    }
    std_dict = {
        "case": "std",
        "volume(Prostate_US)": results_df["volume(Prostate_US)"].std(),
        "volume(Prostate_MR)": results_df["volume(Prostate_MR)"].std(),

        "avg_Volume(Biopsies_US)": results_df["avg_Volume(Biopsies_US)"].std(),
        "std_Volume(Biopsies_US)": results_df["std_Volume(Biopsies_US)"].std(),

        "avg_Volume(Biopsies_MR)": results_df["avg_Volume(Biopsies_MR)"].std(),
        "std_Volume(Biopsies_MR)": results_df["std_Volume(Biopsies_MR)"].std(),

        "dice(Prostate)": results_df["dice(Prostate)"].std(),
        "hausdorff(Prostate)": results_df["hausdorff(Prostate)"].std(),
        "avg_dice(Biopsies)": results_df["avg_dice(Biopsies)"].std(),
        "std_dice(Biopsies)": results_df["std_dice(Biopsies)"].std(),
        "avg_hausdorff(Biopsies)": results_df["avg_hausdorff(Biopsies)"].std(),
        "std_hausdorff(Biopsies)": results_df["std_hausdorff(Biopsies)"].std(),
    }
    # calculate the mean and std for the entire dataset
    results_df.loc[len(results_df)] = mean_dict
    results_df.loc[len(results_df)] = std_dict
    results_df.to_csv(pth_results_csv, index=False)

class DummyRegistration(BrachyPhantomRegistration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def register(self):
        pass
    def export_to(self):
        super().export_to()
    def synch_registered_phantom_with_data(self):
        super().synch_registered_phantom_with_data()
    def evaluate_on_contours(self):
        return super().evaluate_on_contours()

if __name__ == "__main__":
    dir_results = "temp_data/registration"
    reg_data_inputs = gen_registration_inputs_microreg(
        dir_all_data="temp_data/registration/fixed-nrrd",
        )
    get_baseline_stats_microreg(
        reg_data_inputs=reg_data_inputs,
        pth_results_csv=Path(dir_results)/"registration_results_baseline.csv"
    )
    # eval_reg_opentps(
    #     reg_data_inputs=reg_data_inputs,
    #     dir_results=dir_results
    # )
    # eval_reg_plastimatch(
    #     reg_data_inputs=reg_data_inputs,
    #     dir_results=dir_results
    # )
