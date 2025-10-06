from pathlib import Path
from time import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from brachyutils import BrachyPhantomRegistration
from brachyutils import BrachyPhantom
from brachyutils.geometry.phantom_utils import get_slicer_color_by_name

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
            "case": eval_results.get("case"),
            "dice(Prostate)": eval_results.get("dice(Prostate)"),
            "hausdorff(Prostate)": eval_results.get("hausdorff(Prostate)"),
            "dice(Biopsies)": eval_results.get("dice(Biopsies)"),
            "hausdorff(Biopsies)": eval_results.get("hausdorff(Biopsies)"),
            "time": eval_results.get("time")
        }
        # XXX for debugging
        # break
    out_file_name = "reg_metrics_"+ dir_registered.parent.name +"_"+ dir_registered.name+".csv"
    results_per_case_df.to_csv(dir_registered.parent.parent/out_file_name)
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
                "package": backend,
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

def eval_reg_simple_elastix(
    reg_data_inputs: List[Dict[str, Union[str, Path]]],
    dir_results: Path | str
):
    from brachyutils import Registration_SimpleElastix
    pth_executable = "http://192.168.1.14:8000"
    # for debugging
    # algorithms  = ["bspline"]
    # references = ["Prostate"]
    algorithms  = ["affine", "bspline"]
    references = ["Image", "Prostate"]
    dir_registered = Path(dir_results)/"SimpleElastix"
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
            if alg == "affine":
                deformable = False
                parameter_maps =  [
                {
                    "default_parameter_map": "translation",
                },
                {
                    "default_parameter_map": "rigid",
                    "Transform": "AffineTransform",
                }
            ]
            else:
                deformable = True
                parameter_maps =  [
                {
                    "default_parameter_map": "translation",
                },
                {
                    "default_parameter_map": "rigid",
                    "Transform": "AffineTransform",
                },
                {
                    "default_parameter_map": "bspline",
                }
            ]

            reg_results = evaluate_registration(
                reg_data_inputs=reg_data_inputs,
                registration_module=Registration_SimpleElastix,
                dir_registered=dir_registered / ref / alg,
                register_on_contour=use_contour,
                deformable=deformable,
                parameter_maps = parameter_maps,
                pth_executable=pth_executable,
                backend="SimpleElastix",
            )
            results_df.loc[len(results_df)] = {
                "algorithm": alg,
                "package": "SimpleElastix",
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
            results_df.to_csv(dir_registered/"registration_results_simple_elastix.csv", index=False)

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
            "avg_volume(Biopsies_US)", "std_volume(Biopsies_US)",
            "avg_volume(Biopsies_MR)", "std_volume(Biopsies_MR)",
            "dice(Prostate)", "hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
            "time"
    """
    results_df = pd.DataFrame(
        columns=[
            "case",
            "volume(Prostate_US)",
            "volume(Prostate_MR)",
            "avg_volume(Biopsies_US)", "std_volume(Biopsies_US)",
            "avg_volume(Biopsies_MR)", "std_volume(Biopsies_MR)",
            "dice(Prostate)", "hausdorff(Prostate)",
            "avg_dice(Biopsies)", "std_dice(Biopsies)",
            "avg_hausdorff(Biopsies)", "std_hausdorff(Biopsies)",
            "time"
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
        t0 = time()
        reg_obj.registered_phantom = moving_phantom.resample_to(
            origin=static_phantom.image_obj.origin,
            spacing=static_phantom.image_obj.spacing,
            gridSize=static_phantom.image_obj.gridSize,
        )
        tf = time()
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
            "avg_volume(Biopsies_US)": np.mean([v for k, v in structure_volumes_us.items() if k != "Prostate"]),
            "std_volume(Biopsies_US)": np.std([v for k, v in structure_volumes_us.items() if k != "Prostate"]),

            "volume(Prostate_MR)": structure_volumes_mr.get("Prostate", np.nan),
            "avg_volume(Biopsies_MR)": np.mean([v for k, v in structure_volumes_mr.items() if k != "Prostate"]),
            "std_volume(Biopsies_MR)": np.std([v for k, v in structure_volumes_mr.items() if k != "Prostate"]),

            "dice(Prostate)": measured_metrics["Dice"]["Prostate"],
            "hausdorff(Prostate)": measured_metrics["Hausdorff"]["Prostate"],
            "avg_dice(Biopsies)": np.mean([v for k, v in measured_metrics["Dice"].items() if k != "Prostate"]),
            "std_dice(Biopsies)": np.std([v for k, v in measured_metrics["Dice"].items() if k != "Prostate"]),
            "avg_hausdorff(Biopsies)": np.mean([v for k, v in measured_metrics["Hausdorff"].items() if k != "Prostate"]),
            "std_hausdorff(Biopsies)": np.std([v for k, v in measured_metrics["Hausdorff"].items() if k != "Prostate"]),
            
            "time": tf - t0
        }
    mean_dict = {
        "case": "mean",
        "volume(Prostate_US)": results_df["volume(Prostate_US)"].mean(),
        "volume(Prostate_MR)": results_df["volume(Prostate_MR)"].mean(),

        "avg_volume(Biopsies_US)": results_df["avg_volume(Biopsies_US)"].mean(),
        "std_volume(Biopsies_US)": results_df["std_volume(Biopsies_US)"].mean(),
        
        "avg_volume(Biopsies_MR)": results_df["avg_volume(Biopsies_MR)"].mean(),
        "std_volume(Biopsies_MR)": results_df["std_volume(Biopsies_MR)"].mean(),
        
        "dice(Prostate)": results_df["dice(Prostate)"].mean(),
        "hausdorff(Prostate)": results_df["hausdorff(Prostate)"].mean(),
        "avg_dice(Biopsies)": results_df["avg_dice(Biopsies)"].mean(),
        "std_dice(Biopsies)": results_df["std_dice(Biopsies)"].mean(),
        "avg_hausdorff(Biopsies)": results_df["avg_hausdorff(Biopsies)"].mean(),
        "std_hausdorff(Biopsies)": results_df["std_hausdorff(Biopsies)"].mean(),
        "time": results_df["time"].mean()
    }
    std_dict = {
        "case": "std",
        "volume(Prostate_US)": results_df["volume(Prostate_US)"].std(),
        "volume(Prostate_MR)": results_df["volume(Prostate_MR)"].std(),

        "avg_volume(Biopsies_US)": results_df["avg_volume(Biopsies_US)"].std(),
        "std_volume(Biopsies_US)": results_df["std_volume(Biopsies_US)"].std(),

        "avg_volume(Biopsies_MR)": results_df["avg_volume(Biopsies_MR)"].std(),
        "std_volume(Biopsies_MR)": results_df["std_volume(Biopsies_MR)"].std(),

        "dice(Prostate)": results_df["dice(Prostate)"].std(),
        "hausdorff(Prostate)": results_df["hausdorff(Prostate)"].std(),
        "avg_dice(Biopsies)": results_df["avg_dice(Biopsies)"].std(),
        "std_dice(Biopsies)": results_df["std_dice(Biopsies)"].std(),
        "avg_hausdorff(Biopsies)": results_df["avg_hausdorff(Biopsies)"].std(),
        "std_hausdorff(Biopsies)": results_df["std_hausdorff(Biopsies)"].std(),
        "time": results_df["time"].std()
    }
    # calculate the mean and std for the entire dataset
    mean_std_df = pd.DataFrame([mean_dict, std_dict])
    # results_df.loc[len(results_df)] = mean_dict
    # results_df.loc[len(results_df)] = std_dict
    results_df.to_csv(pth_results_csv, index=False)
    mean_std_df.to_csv(pth_results_csv.parent.joinpath("mean_std_"+pth_results_csv.name), index=False)

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

def gen_volume_plots_baseline(
    pth_baseline_results: Path | str,
):
    r"""
    ### Purpose:
        - To generate the bar plots for the volume of the prostate and biopsies in US and MR
    ### Inputs:
        - pth_baseline_results := path to the csv file containing the baseline results with the following columns:
            "case",
            "volume(Prostate_US)",
            "volume(Prostate_MR)",
            "avg_volume(Biopsies_US)",
            "avg_volume(Biopsies_MR)",
    ### Outputs:
        - boxplot_volume_prostate.svg   
        - boxplot_volume_biopsies.svg
    """
    prostate_color = get_slicer_color_by_name("prostate")
    biopsy_color = get_slicer_color_by_name("mass")

    baseline_df = pd.read_csv(pth_baseline_results)
    # box plot for volume of prostate or biopsies in US and MR
    title_prostate="Volume of Prostate in US and MR"
    xlabel="Modality"
    ylabel="Volume (cm$^3$)"
    volume_dict = baseline_df.filter(like="volume(Prostate").to_dict()
    volume_dict_prostate = {
        k.split("_")[-1].split(")")[0]: list(v.values()) 
        for k, v in volume_dict.items()
        }    
    box_plot_evals(
        title=title_prostate, xlabel=xlabel, ylabel=ylabel,
        data=volume_dict_prostate,
        pth_save=pth_baseline_results.parent/"boxplot_volume_prostate.svg",
        fig_size=(5, 5),
        font_size=14,
        use_legends=False,
        box_color=prostate_color
    )
    
    title="Volume of Biopsies in US and MR"
    xlabel="Modality"
    ylabel="Volume (cm$^3$)"
    volume_dict = baseline_df.filter(like="volume(Biopsies").to_dict()
    volume_dict = {
        k.split("_")[-1].split(")")[0]: list(v.values()) 
        for k, v in volume_dict.items()
        }    
    box_plot_evals(
        title=title, xlabel=xlabel, ylabel=ylabel,
        data=volume_dict,
        pth_save=pth_baseline_results.parent/"boxplot_volume_biopsies.svg",
        fig_size=(5, 5),
        font_size=14,
        use_legends=False,
        box_color=biopsy_color
    )

def box_plot_evals(
    title: str,
    xlabel: str,
    ylabel: str,
    data: Dict[str, List[float]],
    pth_save: Path | str = None,
    fig_size: Tuple[int, int] = (10, 6),
    font_size: int = 12,
    use_legends: bool = True,
    box_color: Tuple[float, float, float] = (0, 0, 0),
    half_tickmarks: bool = False,
):
    r"""
    ### Purpose:
        - to generate a box plot for the evaluation results
    ### Inputs:
        - title := title of the plot
        - xlabel := label for the x-axis
        - ylabel := label for the y-axis
        - data := dictionary containing the data for each method
        - pth_save := path to save the figure
        - fig_size := size of the figure
        - font_size := font size for the plot
        - use_legends := whether to use legends
        - box_color := color of the box
        - half_tickmarks := whether to use half tickmarks (groups Image and Contour together)
    The tickmarks are the keys of the dictionaries.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.dpi"] = 300

    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get method names
    methods = list(data.keys())
    
    if half_tickmarks:
        # Group Image and Contour methods together
        # Extract base method names (without -Image or -Contour)
        base_methods = []
        seen = set()
        for method in methods:
            base = method.replace("-Image", "").replace("-Contour", "")
            if base not in seen:
                base_methods.append(base)
                seen.add(base)
        
        # Separate data into Image and Contour groups
        image_data = []
        contour_data = []
        for base in base_methods:
            # Find corresponding Image and Contour methods
            img_method = None
            cont_method = None
            for method in methods:
                if base in method:
                    if "-Contour" in method:
                        cont_method = method
                    else:
                        img_method = method 
            
            # Append data (empty list if method doesn't exist)
            image_data.append(data[img_method] if img_method else [])
            contour_data.append(data[cont_method] if cont_method else [])
        
        # Create positions for grouped boxplots
        n_groups = len(base_methods)
        box_width = 0.4
        offset = 0.2

        # Positions: Image boxes at -offset, Contour boxes at +offset
        image_positions = np.array(range(n_groups)) * 2.0 - offset
        contour_positions = np.array(range(n_groups)) * 2.0 + offset

        # Create boxplots
        bp_image = ax.boxplot(image_data, positions=image_positions, 
                              widths=box_width, patch_artist=True)
        bp_contour = ax.boxplot(contour_data, positions=contour_positions, 
                                widths=box_width, patch_artist=True)

        # Customize appearance
        for patch in bp_image['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.5)
        
        for patch in bp_contour['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(1.0)
        
        # Set x-ticks at the center of each group
        ax.set_xticks(np.arange(0, n_groups * 2, 2))
        ax.set_xticklabels(base_methods, rotation=45, ha='right')
        
    else:
        # Original single-box behavior
        xtick_labels = [method.replace("-Image", "").replace("-Contour", "") 
                       for method in methods]
        alpha = [1.0 if "Contour" in method else 0.5 for method in methods]
        x_pos = range(len(methods))
        
        # Create the box plot
        bp = ax.boxplot([data[method] for method in methods], 
                        positions=x_pos, patch_artist=True)
        
        # Customize appearance
        for patch, a in zip(bp['boxes'], alpha):
            patch.set_facecolor(box_color)
            patch.set_alpha(a)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if use_legends:
        legend_elements = [
            Patch(facecolor=box_color, alpha=0.5, label='Image-based'),
            Patch(facecolor=box_color, alpha=1.0, label='Contour-based')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=font_size-2)
    
    # Tight layout to minimize margins
    plt.tight_layout()
    
    if pth_save is not None:
        plt.savefig(pth_save, dpi=300)
    else:
        plt.show()
    plt.close()

def _extract_data_to_dict(
    df: pd.DataFrame,
    metric: str,
    method_name: str,
    out_dict: Dict[str, List[float]],
):
    out_dict[method_name] = {"data": list()}
    contents = list(df.filter(like=metric).to_dict().values()).pop()
    for k, v in contents.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out_dict[method_name]["data"].append(v)
    if len(out_dict[method_name]["data"]) == 0:
        out_dict.pop(method_name)

def _reorder_keys(
    data_dict: Dict[str, Dict[str, List[float]]],
    preferred_order: List[str]
):
    ordered_dict = {k: data_dict[k] for k in preferred_order if k in data_dict}
    unordered_keys = [k for k in data_dict.keys() if k not in preferred_order]
    for k in unordered_keys:
        ordered_dict[k] = data_dict[k]
    data_dict.clear()
    data_dict.update(ordered_dict)

def get_plots_dice_hausdorff(
    pth_baseline_results: Path | str,
    list_pth_results: List[Path | str],
):
    r"""
    ### Purpose:
        - to generate the box plots for the registration evaluation results
    """
    # Load baseline results
    baseline_df = pd.read_csv(pth_baseline_results)
    # Initialize dictionaries to hold data for plotting
    dice_dict_prostate = defaultdict(dict)
    hausdorff_dict_prostate = defaultdict(dict)
    # now doing the same for the biopsies
    dice_dict_biopsies = defaultdict(dict)
    hausdorff_dict_biopsies = defaultdict(dict)
    # get that timing data too
    time_dict = defaultdict(dict)
    ordered_keys_registration_methods = [
        "No Registration \n (Resample Only)", "OpenTPS-Rigid-Image", "OpenTPS-Rigid-Contour",
        "OpenTPS-Quick-Image", "OpenTPS-Quick-Contour", "OpenTPS-Morphons-Image",
        "OpenTPS-Morphons-Contour", "Plastimatch-Translation-Image",
        "Plastimatch-Translation-Contour", "Plastimatch-Bspline-Image",
        "Plastimatch-Bspline-Contour", "SimpleElastix-Affine-Image",
        "SimpleElastix-Affine-Contour", "SimpleElastix-Bspline-Image",
        "SimpleElastix-Bspline-Contour"
    ]
    # Extract baseline data and data from each registration results file
    _extract_data_to_dict(baseline_df, "dice(Prostate)", "No Registration \n (Resample Only)", dice_dict_prostate)
    _extract_data_to_dict(baseline_df, "hausdorff(Prostate)", "No Registration \n (Resample Only)", hausdorff_dict_prostate)
    _extract_data_to_dict(baseline_df, "dice(Biopsies)", "No Registration \n (Resample Only)", dice_dict_biopsies)
    _extract_data_to_dict(baseline_df, "hausdorff(Biopsies)", "No Registration \n (Resample Only)", hausdorff_dict_biopsies)
    _extract_data_to_dict(baseline_df, "time", "No Registration \n (Resample Only)", time_dict)

    prostate_color = get_slicer_color_by_name("prostate")
    biopsy_color = get_slicer_color_by_name("mass")
    for pth_result in list_pth_results:
        method_name = pth_result.stem.split("reg_metrics_")[-1]
        package_name = pth_result.parent.name
        algorithm_name = pth_result.stem.split("_")[-1].capitalize()
        if "Demons" in algorithm_name:
            continue
        reg_based_on = pth_result.stem.split("_")[-2].capitalize()
        if reg_based_on == "Prostate":
            reg_based_on = "Contour"
        method_name = f"{package_name}-{algorithm_name}-{reg_based_on}"
        result_df = pd.read_csv(pth_result)
        _extract_data_to_dict(result_df, "dice(Prostate)", method_name, dice_dict_prostate)
        _reorder_keys(dice_dict_prostate, preferred_order=ordered_keys_registration_methods)
        _extract_data_to_dict(result_df, "hausdorff(Prostate)", method_name, hausdorff_dict_prostate)
        _reorder_keys(hausdorff_dict_prostate, preferred_order=ordered_keys_registration_methods)
        _extract_data_to_dict(result_df, "dice(Biopsies)", method_name, dice_dict_biopsies)
        _reorder_keys(dice_dict_biopsies, preferred_order=ordered_keys_registration_methods)
        _extract_data_to_dict(result_df, "hausdorff(Biopsies)", method_name, hausdorff_dict_biopsies)
        _reorder_keys(hausdorff_dict_biopsies, preferred_order=ordered_keys_registration_methods)
        _extract_data_to_dict(result_df, "time", method_name, time_dict)
        _reorder_keys(time_dict, preferred_order=ordered_keys_registration_methods)

    # # Generate box plots
    box_plot_evals(
        title="Dice Coefficient for Prostate after Registration \n(higher is better)",
        xlabel="Method",
        ylabel="Dice Coefficient",
        data={k: v["data"] for k, v in dice_dict_prostate.items()},
        pth_save=Path(pth_baseline_results).parent/"boxplot_dice_prostate.svg",
        box_color=prostate_color,
        half_tickmarks=True,
    )
    box_plot_evals(
        title="Maximum Hausdorff Distance for \n Prostate after Registration (lower is better)",
        xlabel="Method",
        ylabel="Hausdorff Distance (mm)",
        data={k: v["data"] for k, v in hausdorff_dict_prostate.items()},
        pth_save=Path(pth_baseline_results).parent/"boxplot_hausdorff_prostate.svg",
        box_color=prostate_color,
    )
    box_plot_evals(
        title="Dice Coefficient for Biopsies after Registration \n(higher is better)",
        xlabel="Method",
        ylabel="Dice Coefficient",
        data={k: v["data"] for k, v in dice_dict_biopsies.items()},
        pth_save=Path(pth_baseline_results).parent/"boxplot_dice_biopsies.svg",
        box_color=biopsy_color,
    )
    box_plot_evals(
        title="Maximum Hausdorff Distance for \n Biopsies after Registration (lower is better)",
        xlabel="Method",
        ylabel="Hausdorff Distance (mm)",
        data={k: v["data"] for k, v in hausdorff_dict_biopsies.items()},
        pth_save=Path(pth_baseline_results).parent/"boxplot_hausdorff_biopsies.svg",
        box_color=biopsy_color,
    )
    box_plot_evals(
        title="Computation Time for Registration Methods \n(lower is better)",
        xlabel="Method",
        ylabel="Time (s)",
        data={k: v["data"] for k, v in time_dict.items()},
        pth_save=Path(pth_baseline_results).parent/"boxplot_registration_time.svg",
        box_color=(0.2, 0.2, 205/255),
    )

def get_bar_plots_num_failed(
    list_pth_results: List[Path | str],
    pth_baseline_results: Path | str
):
    r"""
    ### Purpose:
        - to generate the bar plots for the number of failed registrations
    """
    # Load baseline results
    baseline_df = pd.read_csv(pth_baseline_results)
    # Initialize dictionaries to hold data for plotting
    num_failed_dict = dict()
    # Extract baseline data and data from each registration results file
    ordered_keys_registration_methods = [
        "No Registration \n (Resample Only)", "OpenTPS-Rigid-Image", "OpenTPS-Rigid-Contour",
        "OpenTPS-Quick-Image", "OpenTPS-Quick-Contour", "OpenTPS-Morphons-Image",
        "OpenTPS-Morphons-Contour", "Plastimatch-Translation-Image",
        "Plastimatch-Translation-Contour", "Plastimatch-Bspline-Image",
        "Plastimatch-Bspline-Contour", "SimpleElastix-Affine-Image",
        "SimpleElastix-Affine-Contour", "SimpleElastix-Bspline-Image",
        "SimpleElastix-Bspline-Contour"
    ]
    # _extract_data_to_dict(baseline_df, "num_failed", "No Registration \n (Resample Only)", num_failed_dict)
    for pth_result in list_pth_results:
        if "baseline" in pth_result.stem:
            continue
        result_df = pd.read_csv(pth_result)
        for row in result_df.itertuples():
            package_name = row.package
            algorithm_name = row.algorithm.capitalize()
            if "Demons" in algorithm_name:
                continue
            if row.reference == "Prostate":
                reg_based_on = "Contour"
            else:
                reg_based_on = "Image"
            method_name = f"{package_name}-{algorithm_name}-{reg_based_on}"
            num_failed_dict[method_name] = row.num_failed
    _reorder_keys(num_failed_dict, preferred_order=ordered_keys_registration_methods)

    # Generate bar plot
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(num_failed_dict.keys())
    xtick_labels = [method.replace("-Image", "").replace("-Contour", "") for method in methods]
    x_pos = range(len(methods))
    num_failed_values = [num_failed_dict[method] for method in methods]
    bars = ax.bar(x_pos, num_failed_values, color=(0.8, 0.2, 0.2))
    # add alpha to bars based on contour or image based registration
    for bar, method in zip(bars, methods):
        if "Contour" in method:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.5)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Failed Registrations")
    ax.set_title("Number of Failed Registrations by Method \n(lower is better)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    # ax.set_ylim(top=1.15*max(num_failed_values))
    ax.grid(True, axis='y', alpha=0.3)
    # Create legend elements
    legend_elements = [
        Patch(facecolor=(0.8, 0.2, 0.2), alpha=0.5, label='Image-based'),
        Patch(facecolor=(0.8, 0.2, 0.2), alpha=1.0, label='Contour-based')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    # Tight layout to minimize margins
    plt.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.dpi"] = 300
    pth_save = Path(pth_baseline_results).parent/"barplot_num_failed_registrations.svg"
    plt.savefig(pth_save, dpi=300)
    plt.close()

if __name__ == "__main__":
    dir_results = "temp_data/registration"
    # reg_data_inputs = gen_registration_inputs_microreg(
    #     dir_all_data="temp_data/registration/fixed-nrrd",
    #     )
    # get_baseline_stats_microreg(
    #     reg_data_inputs=reg_data_inputs,
    #     pth_results_csv=Path(dir_results)/"registration_results_baseline.csv"
    # )
    # eval_reg_opentps(
    #     reg_data_inputs=reg_data_inputs,
    #     dir_results=dir_results
    # )
    # eval_reg_plastimatch(
    #     reg_data_inputs=reg_data_inputs,
    #     dir_results=dir_results
    # )
    # eval_reg_simple_elastix(
    #     reg_data_inputs=reg_data_inputs,
    #     dir_results=dir_results
    # )

    # gen_volume_plots_baseline(
    #     pth_baseline_results=Path(dir_results)/"registration_results_baseline.csv",
    # )

    list_pth_results = Path(dir_results).rglob("reg_metrics_*.csv")
    get_plots_dice_hausdorff(
        list_pth_results=list_pth_results,
        pth_baseline_results=Path(dir_results)/"registration_results_baseline.csv",
    )

    # list_pth_results = list(Path(dir_results).rglob("registration_results_*.csv"))
    # get_bar_plots_num_failed(
    #     list_pth_results=list_pth_results,
    #     pth_baseline_results=Path(dir_results)/"registration_results_baseline.csv"
    # )