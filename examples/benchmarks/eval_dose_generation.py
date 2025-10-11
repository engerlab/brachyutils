from typing import Dict, Union, Literal
from pathlib import Path
from tqdm import tqdm
from brachyutils import load_dicom_to_plan
from brachyutils import DoseMonteCarlo, DoseTG43
import pandas as pd
from time import time

def run_multi_proc(function, input_list, max_workers=8):
    from multiprocessing import Pool
    
    with Pool(processes=max_workers) as pool:
        try:
            list(pool.imap_unordered(function, input_list))
        except Exception as e:
            print(f"Error in multiprocessing: {e}")

def export_single_dicom_to_plan(
    dir_dicom:Path | str,
    dir_export: Path | str,
    sim_dict: Dict[str, Union[str, int]] = None,
    content_to_export: Dict[str, bool] = None,
    ) -> Path:
    r"""
    ### Purpose:
        - Export the plans to the given directory.

    ### Inputs:
        - dir_dicom: Path | str: The path to the dicom directory for one plan. it should have images,
        and a plan file. Structure file is optional.
        - dir_export: Path | str: The directory to export the plans to. Each plan will have its own subdir.

    ### Outputs:
        - dir_export_plan: Path: The path to the exported plan.
    """
    plan_obj = load_dicom_to_plan(
        dir_dicom,
        simulation_setup=sim_dict,
        delivered_catheter_table=True
        )

    dir_export = Path(dir_export)
    dir_export.mkdir(parents=True, exist_ok=True)

    dir_export_plan = dir_export.joinpath(dir_dicom.stem)

    plan_obj.export_brachy_plan(
        dir_export=dir_export_plan,
        content_to_export=content_to_export,
    )
    return dir_export_plan

def run_export(
    dir_all_dicoms: Path | str,
    dir_export: Path | str,
    multi_proc: bool = True,
    ):
    from functools import partial
    dir_all_dicoms = Path(dir_all_dicoms)
    dir_export = Path(dir_export)
    # dir_all_dicoms = Path.home().joinpath("YourLocalHome/Data/prostate/prostate-glen-2023")
    # dir_export = Path("temp_data/tg43/prostate-glen-2023")
    # dir_export = Path("temp_data/mc/prostate-glen-2023")

    # pth_material = Path("admin/constants/CTtoDensityProstate.txt")
    # mat_from_ct = True
    pth_material = Path("admin/constants/structure_materials_prostate.json")
    mat_from_ct = False
    crop_by_contour = "body"
    sim_dict = {
        # "brachy_source": 
        # "pth_plan": "combined.plan",
        # "pth_phantom": "ct.egsphant",
        "number_histories": 1E6,
        # "total_time": 0,
        "number_of_threads": 16,
        # "PrintProgress": 10000,
        # "beam_on": 10000,
    }
    content_to_export = {
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": mat_from_ct,
        "resampled_spacing": [1., 1., 1.],
        "strict_name_match": False,
        "crop_by_contour": crop_by_contour,
        "plan": True,
        "mac": True,
        "combined_only": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }
    dir_export.mkdir(parents=True, exist_ok=True)
    all_dicoms = list(dir_all_dicoms.glob("*/"))
    partially_filled_export_func = partial(
        export_single_dicom_to_plan,
        dir_export=dir_export,
        sim_dict=sim_dict,
        content_to_export=content_to_export,
        )

    if multi_proc:
        run_multi_proc(partially_filled_export_func, all_dicoms)
    else:
        for dicom in tqdm(all_dicoms):
            partially_filled_export_func(dicom)
            # break # for debugging

def run_dose_generation(
    dir_plan_export: Path | str,
    method: Literal["tg43", "mc"] = "tg43",
):
    timing_data = pd.DataFrame(columns=[
        "plan_name", "dose_gen_method", "dose_generation_time"
        ])
    # # for TG43
    if method == "tg43":
    # dir_plan_export = Path("temp_data/tg43/prostate-glen-2023")
        dir_plans = list(dir_plan_export.glob("*/"))
        for plan in tqdm(dir_plans):
            t0 = time()
            run_single_tg43_dose_generation(plan)
            t1 = time()
            timing_data.loc[len(timing_data)] = {
                "plan_name": plan.name,
                "dose_gen_method": "tg43",
                "dose_generation_time": t1 - t0
            }
            # break # for debugging
    # # for Monte Carlo
    elif method == "mc":
        dir_plans = list(dir_plan_export.glob("*/"))
        for plan in tqdm(dir_plans):
            t0 = time()
            run_single_mc_dose_generation(plan)
            t1 = time()
            timing_data.loc[len(timing_data)] = {
                "plan_name": plan.name,
                "dose_gen_method": "mc",
                "dose_generation_time": t1 - t0
            }
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are 'tg43' and 'mc'.")

    timing_data.to_csv(dir_plan_export/f"dose_generation_timing_{method}.csv", index=False)

def run_single_tg43_dose_generation(dir_plan):
    dose_gen_obj = DoseTG43(
        dir_plan_export=dir_plan,
        pth_dose_executable="http://192.168.1.12:8000/calculate_dose_tg43"
    ).generate_dose()

def run_single_mc_dose_generation(dir_plan):
    dose_gen_obj = DoseMonteCarlo(
        dir_plan_export=dir_plan,
        pth_dose_executable="http://192.168.1.11:8000/calculate_dose_mc"
    ).generate_dose(pth_mac=dir_plan/"combined.mac")

def get_dvh_metrics_single_plan(
    dir_dicom: Path | str,
    dvh_metric_goals: Dict[str, float],
    dir_plan_export: Path | str,
    load_dose_from: Literal["dicom"] | Path | str = "dicom",
    export_combined_dose: bool = True,
) -> Dict[str, float]:
    r"""
    ### Purpose:
        - To evaluate a single plan's dose distribution based on DVH metrics.
    
    ### Inputs:
        - dir_plan_export: Path | str: The path to the exported plan.
        - dvh_metric_goals: Dict[str, float]: The DVH metrics to evaluate the plan.
        - dir_dicom: Path | str: The path to the dicom directory for one plan. it should have images,
        and structure file.

    ### Outputs:
        - dvh_metrics: Dict: The DVH metrics for the plan.
    """
    if load_dose_from == "dicom":
        plan_obj = load_dicom_to_plan(
            dir_dicom=dir_dicom,
            load_dicom_dose=True,
            dvh_metric_goals=dvh_metric_goals,
            combined_dose_only=True,
            )
    elif isinstance(load_dose_from, str) or isinstance(load_dose_from, Path):
        load_dose_from = Path(load_dose_from)
        if str(load_dose_from).endswith(".nrrd") or str(load_dose_from).endswith(".3ddose"):
            plan_obj = load_dicom_to_plan(
                dir_dicom=dir_dicom,
                load_dicom_dose=False,
                dvh_metric_goals=dvh_metric_goals,
                combined_dose = load_dose_from,
                combined_dose_only=True,
                multi_processing=True
                )
        elif load_dose_from.is_dir():
            plan_obj = load_dicom_to_plan(
            dir_dicom=dir_dicom,
            load_dicom_dose=False,
            dvh_metric_goals=dvh_metric_goals,
            dir_dose_rate=dir_plan_export,
            combined_dose_only=True,
            multi_processing=True
            )
        else:
            raise ValueError(f"Invalid load_dose_from: {load_dose_from}\
                             make it is either a path to a dose file or a directory containing\
                             a dose file per dwell position.")
    else:
        raise ValueError(f"Invalid load_dose_from: {load_dose_from}. Valid inputs\
                            are 'dicom', a path to a dose file or directory containing\
                            dose files per dwell position.")

    dvh_metrics_observed = plan_obj.get_dvh_metrics()
    if export_combined_dose:
        plan_obj.export_brachy_plan(
            dir_export=dir_plan_export,
            content_to_export={
                "dose": True,
            }
        )
    return dvh_metrics_observed

def get_dvh_metrics_all_plans(
    dir_all_dicom_folders: str | Path,
    dir_all_plan_folders: str | Path,
    dvh_metric_goals: Dict[str, float],
):
    r"""
    ### Purpose:
        - To get the dvh metrics from all the patients in the given directory.
    
    ### Inputs:
        - dir_dicom_folders: str | Path: The path to the dicom folders.
        - dir_plan_folders: str | Path: The path to the plan folders.
        - dvh_metric_goals: Dict[str, float]: The DVH metrics to evaluate the plan.
    """
    dir_all_dicom_folders = Path(dir_all_dicom_folders)
    dir_all_plan_folders = Path(dir_all_plan_folders)
    
    list_dir_plan = dir_all_plan_folders.glob("*/")
    all_dvhs = pd.DataFrame()
    for dir_plan in list_dir_plan:
        if not dir_plan.is_dir():
            continue
        dir_dicom = dir_all_dicom_folders.joinpath(dir_plan.stem)
        print(f"dvh from dicom folder: {dir_dicom}")
        try:
            dvh_metrics = get_dvh_metrics_single_plan(
                dir_dicom=dir_dicom,
                dvh_metric_goals=dvh_metric_goals,
                dir_plan_export=dir_plan,
                load_dose_from=dir_plan/"combined.seq.nrrd",
                export_combined_dose=False,
            )
            all_dvhs = pd.concat(
                    [
                    pd.DataFrame([dvh_metrics | {"name":dir_plan.name}]),
                    all_dvhs
                    ],
                ignore_index=True
                )
            
        except Exception as e:
            print(f"error in getting dvh for {dir_plan}")
            print(e)
    all_dvhs.set_index("name", inplace=True)
    all_dvhs.to_csv(dir_all_plan_folders/"dvh_metrics.csv")

def run_get_dvh_metrics_all_plans():
    # # on alien baby
    # pth_all_dicom = Path("/root/YourLocalHome/Data/prostate/prostate-glen-2023")
    # dir_all_plans = Path("/root/YourLocalHome/Data/prostate/plans-1mm/prostate-glen-2023")
    # # on photon
    pth_all_dicom = Path("/root/YourLocalHome/Data/prostate-glen-2023")
    dir_all_plans = Path("temp_data/tg43/prostate-glen-2023")
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    get_dvh_metrics_all_plans(
        dir_all_dicom_folders=pth_all_dicom,
        dir_all_plan_folders=dir_all_plans,
        dvh_metric_goals=dvh_metric_goals,
    )

def test_get_dvh_metrics_single_plan():
    pth_single_dicom = Path("/root/YourLocalHome/Data/prostate-glen-2023/p2")
    # dir_export = Path("/root/YourLocalHome/Data/prostate/plans-1mm/prostate-glen-2023/p1")
    dir_export = Path("temp_data/tg43/prostate-glen-2023/p2")
    dvh_metric_goals = {
        "D95%(ctv)": 21,
        "D1cc(rectum)": 21*0.75,
        "D0.1cc(urethra)": 21*1.25,
    }
    print(
        get_dvh_metrics_single_plan(
            dir_dicom=pth_single_dicom,
            dvh_metric_goals=dvh_metric_goals,
            dir_plan_export=dir_export)
        )

def test_export():
    pth_single_dicom = Path.home().joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p12")
    dir_export = Path("temp_data/tg43/prostate-glen-2023")
    # pth_material = Path("admin/constants/CTtoDensityProstate.txt")
    # mat_from_ct = True
    pth_material = Path("admin/constants/structure_materials_prostate.json")
    mat_from_ct = False
    crop_by_contour = "body"
    sim_dict = {
        # "source_dict": {
        #     "treatment_type": "HDR",
        #     "source_geometry": "MicroSelectronV2",
        #     "core_material": "G4_Ir",
        #     "mass_number": "192",
        #     "atomic_number": "77",
        #     "air_kerma_per_history": 1.149000e-11,
        #     "reference_air_kerma": 5e04,
        # },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 1000000,
        "total_time": 0,
        "number_of_threads": 14,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    content_to_export = {
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": mat_from_ct,
        # "resampled_spacing": [1., 1., 1.],
        "crop_by_contour": crop_by_contour,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }

    if not pth_material.exists():
        raise FileNotFoundError(f"The material file {pth_material} does not exist.")
    export_single_dicom_to_plan(
        pth_single_dicom,
        dir_export,
        content_to_export=content_to_export,
        sim_dict=sim_dict
        )

def test_dose_calc():
    # # for monte carlo
    # dir_plan_export = Path("temp_data/mc/prostate-glen-2023/p3")
    # pth_dose_executable = "http://192.168.1.11:8000/calculate_dose_mc"
    # dose_gen_obj = DoseMonteCarlo(
    #     dir_plan_export=dir_plan_export,
    #     pth_dose_executable=pth_dose_executable
    # )
    # dose_gen_obj.generate_dose()
    
    
    # # for tg43
    dir_plan_export = Path("temp_data/tg43/prostate-glen-2023/p1")
    pth_dose_executable = "http://192.168.1.12:8000/calculate_dose_tg43"
    dose_gen_obj = DoseTG43(
        dir_plan_export=dir_plan_export,
        pth_dose_executable=pth_dose_executable
    )
    dose_gen_obj.generate_dose()

def scale_by_airkerma(dir_all_plans: str | Path, dir_all_dcms: str | Path):
    r"""
        To scale the nrrd dose in a plan directory by the right air kerma.
        The wrong air keram was 5e4. each dicom plan has a different air kerma.
        the scaling factor for each dose should be plan_air_kerma/5e4.
    """
    from brachyutils import BrachySource
    from brachyutils import BrachyDose
    from functools import partial

    dir_all_plans = Path(dir_all_plans)
    dir_all_dcms = Path(dir_all_dcms)

    all_plans = list(dir_all_plans.glob("*/"))
    for plan in all_plans:
        if not plan.is_dir():
            continue
        pth_plan_dcm = list(dir_all_dcms.joinpath(plan.stem).glob("RP*.dcm"))
        if len(pth_plan_dcm) == 0:
            print(f"no plan dcm found for {plan}")
            continue
        pth_plan_dcm = pth_plan_dcm[0]
        source_obj = BrachySource(source_dict=pth_plan_dcm)
        scaling_factor = source_obj.reference_air_kerma_rate/5e4
        
        pth_dose_list = list(plan.glob("*.nrrd"))
        def scale_dose(pth_dose: Path, scaling_factor: float):
            dose_obj:BrachyDose = BrachyDose(pth_dose)
            dose_obj.set_dose_array(
                dose_obj.get_dose_array()*scaling_factor
            )
            dose_obj.write_brachydose_to_file(str(pth_dose.parent/f"scaled_{pth_dose.stem.stem}")+".seq.nrrd")
        print(f"scaling factor for {plan}: {scaling_factor}")
        run_multi_proc(
            partial(scale_dose, scaling_factor=scaling_factor),
            pth_dose_list
        )

def run_scale_by_airkerma():
    dir_all_plans = Path("temp_data/mc/prostate-glen-2023")
    dir_all_dcms = Path("/root/YourLocalHome/Data/prostate-glen-2023")
    scale_by_airkerma(dir_all_plans, dir_all_dcms)

if __name__ == "__main__":
    # test_export()
    # test_dose_calc()
    # test_get_dvh_metrics_single_plan()
    
    dir_all_dicoms = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023")
    dir_export_tg43 = Path("temp_data/tg43/prostate-glen-2023") # for tg43
    dir_export_mc = Path("temp_data/mc/prostate-glen-2023") # for Monte Carlo

    # # export all dicoms to plans
    # for dir_export in [dir_export_tg43, dir_export_mc]:
    #     run_export(
    #         dir_all_dicoms=dir_all_dicoms,
    #         dir_export=dir_export,
    #         multi_proc=False,
    #     )
        # break

    # # run dose generation for all plans
    run_dose_generation(
        dir_plan_export=dir_export_tg43,
        method="tg43"
    )
    # run_dose_generation(
    #     dir_plan_export=dir_export_mc,
    #     method="mc"
    # )
    # run_get_dvh_metrics_all_plans()
    # run_scale_by_airkerma()
