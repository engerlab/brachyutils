from typing import List, Dict, Union
from pathlib import Path
from brachyutils.plan_utils import BrachyPlan, load_dicom_to_plan
from brachyutils.dose_comparison_utils import DoseComparison
from brachyutils.dose_generation_utils import DoseGenerator, DoseMonteCarlo, DoseTG43
from pandas import DataFrame

def run_multi_proc(function, input_list):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    async def run_in_executor(executor, case):
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(executor, function, case)
        except Exception as e:
            print(f"error in exporting {case}")
            print(e)
            return None

    async def main():
        with ThreadPoolExecutor() as executor:
            tasks = []
            for case in input_list:
                tasks.append(run_in_executor(executor, case))
            await asyncio.gather(*tasks)

    asyncio.run(main())

def generate_single_dose(dir_plan_export, dose_generator_class, **kwargs):
    r"""
    Purpose:
        - To run a single dose calculation for an exported plan.
    Inputs:
        - dir_plan_export: Union[Path, str]: The path to the exported plan.
        - dose_generator_class: DoseGenerator: The dose generator class to use.
        - dir_export: Union[Path, str]: The path to export the dose calculation.
        - **kw_args: dict: Additional keyword arguments to pass to the dose generator.
    Outputs:
        - None
    """
    dose_gen_obj = dose_generator_class(
        dir_plan_export=dir_plan_export,
        pth_dose_executable=kwargs.get(
            "pth_dose_executable",
            "http://192.168.1.11:8000/calculate_dose_mc",
            ),
    )

    dose_gen_obj.generate_dose(
        pth_mac = kwargs.get("pth_mac", None),
        random_seed = kwargs.get("random_seed", 1),
    )

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
    plan_obj = load_dicom_to_plan(dir_dicom, simulation_dict=sim_dict)

    dir_export = Path(dir_export)
    dir_export.mkdir(parents=True, exist_ok=True)

    dir_export_plan = dir_export.joinpath(dir_dicom.stem)

    plan_obj.export_brachy_plan(
        dir_export=dir_export_plan,
        content_to_export=content_to_export,
    )
    return dir_export_plan

def run_export():
    from functools import partial
    pth_single_dicom = Path("/root/YourLocalHome/Data/prostate/prostate-glen-2023")
    dir_export = Path("../temp_data/mc/prostate-glen-2023")
    # pth_material = Path("../admin/constants/CTtoDensityProstate.txt")
    # mat_from_ct = True
    pth_material = Path("../admin/constants/structure_materials_prostate.json")
    mat_from_ct = False
    crop_by_contour = "body"
    sim_dict = {
        "source_dict": {
            "treatment_type": "HDR",
            "source_geometry": "MicroSelectronV2",
            "core_material": "G4_Ir",
            "mass_number": "192",
            "atomic_number": "77",
            "air_kerma_per_history": 1.149000e-11,
            "reference_air_kerma": 5e04,
        },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 1000000,
        "total_time": 0,
        "number_of_threads": 32,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    content_to_export = {
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": mat_from_ct,
        "resample_egsphant_to": [1., 1., 1.],
        "crop_by_contour": crop_by_contour,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }

    all_dicoms = list(pth_single_dicom.glob("*/"))
    all_inputs = []
    partially_filled_export_func = partial(
        export_single_dicom_to_plan,
        dir_export=dir_export,
        sim_dict=sim_dict,
        content_to_export=content_to_export,
        )

    run_multi_proc(partially_filled_export_func, all_dicoms)

def get_dvh_metrics_single_plan(
    dir_dicom: Path | str,
    dvh_metric_goals: Dict[str, float],
    dir_plan_export: Path | str,
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
    plan_obj = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_dose=False,
        dvh_metric_goals=dvh_metric_goals,
        dir_dose_rate=dir_plan_export,
        combined_dose_only=True,
        multi_processing=True
        )

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
    all_dvhs = []
    for dir_plan in list_dir_plan:
        if not dir_plan.is_dir():
            continue
        if dir_plan.stem in [
            "p1", "p2", "p3", "p4", "p5_body", "p6_body",
            ]:
            continue
        dir_dicom = dir_all_dicom_folders.joinpath(dir_plan.stem)
        print(f"dvh from dicom folder: {dir_dicom}")
        dvh_metrics = get_dvh_metrics_single_plan(
            dir_dicom=dir_dicom,
            dvh_metric_goals=dvh_metric_goals,
            dir_plan_export=dir_plan,
        )
        all_dvhs.append({dir_plan.stem: dvh_metrics})

    df_dvhs = DataFrame(all_dvhs)
    df_dvhs.to_csv(dir_all_plan_folders/"dvh_metrics.csv")

def run_get_dvh_metrics_all_plans():
    # # on alien baby
    # pth_all_dicom = Path("/root/YourLocalHome/Data/prostate/prostate-glen-2023")
    # dir_all_plans = Path("/root/YourLocalHome/Data/prostate/plans-1mm/prostate-glen-2023")
    # # on photon
    pth_all_dicom = Path("/root/YourLocalHome/Data/prostate-glen-2023")
    dir_all_plans = Path("../temp_data/mc/prostate-glen-2023")
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
    pth_single_dicom = Path("/root/YourLocalHome/Data/prostate/prostate-glen-2023/p1")
    dir_export = Path("/root/YourLocalHome/Data/prostate/plans-1mm/prostate-glen-2023/p1")
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    print(
        get_dvh_metrics_single_plan(
            dir_dicom=pth_single_dicom,
            dvh_metric_goals=dvh_metric_goals,
            dir_plan_export=dir_export)
        )

def test_export():
    pth_single_dicom = Path("/root/YourLocalHome/Data/prostate-glen-2023/p2")
    dir_export = Path("../temp_data/mc/prostate-glen-2023")
    # pth_material = Path("../admin/constants/CTtoDensityProstate.txt")
    # mat_from_ct = True
    pth_material = Path("../admin/constants/structure_materials_prostate.json")
    mat_from_ct = False
    crop_by_contour = "body"
    sim_dict = {
        "source_dict": {
            "treatment_type": "HDR",
            "source_geometry": "MicroSelectronV2",
            "core_material": "G4_Ir",
            "mass_number": "192",
            "atomic_number": "77",
            "air_kerma_per_history": 1.149000e-11,
            "reference_air_kerma": 5e04,
        },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 1000000,
        "total_time": 0,
        "number_of_threads": 32,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    content_to_export = {
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": mat_from_ct,
        "resample_egsphant_to": [1., 1., 3.],
        "crop_by_contour": crop_by_contour,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }

    if not pth_material.exists():
        raise FileNotFoundError(f"The material file {pth_material} does not exist.")
    export_single_dicom_to_plan(pth_single_dicom, dir_export)#, sim_dict, content_to_export)

def test_dose_calc():
    dir_plan_export = Path("../temp_data/mc/prostate-glen-2023/p3")
    dose_generator_class = DoseMonteCarlo
    generate_single_dose(dir_plan_export, dose_generator_class)

if __name__ == "__main__":
    test_export()
    # test_dose_calc()
    # test_get_dvh_metrics_single_plan()
    # run_export()
    # run_get_dvh_metrics_all_plans()