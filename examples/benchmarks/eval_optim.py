from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.planning.optimization.optim_utils import Optimization_Config

from brachyutils.types import BrachyPlan
from pathlib import Path
from pandas import DataFrame
from time import time
import numpy as np

def get_a_plan_to_optimize(
    pth_dicom: str | Path,
    dir_dose_rates: str | Path,
    generate_dose_rates: bool = False,
    dvh_metric_goals: dict[str, float] | None = None,
    target_dose: float = 21,
    optimization_config_list: list[Optimization_Config] | None = None,
    )->BrachyPlan:
    r"""
    ### Purpose:
    - Load a brachytherapy plan from DICOM files and prepare it for optimization.
    Optinally generate dose rate files if they do not exist.
    ### Inputs:
    - `pth_dicom`: Path to the DICOM directory containing the brachytherapy plan.
    - `dir_dose_rates`: Directory where dose rate files are stored or will be generated.
    - `generate_dose_rates`: Boolean flag to indicate whether to generate dose rate files if they do not exist.
    - `dvh_metric_goals`: Dictionary specifying dose-volume histogram (DVH) metric goals for optimization.
    - `target_dose`: Target dose for the plan.
    - `optimization_config_list`: List of `Optimization_Config` 
    objects specifying optimization parameters for different structures.
    ### Outputs:
    - `plan_obj`: A `BrachyPlan` object ready for optimization.
    """
    pth_dicom = Path(pth_dicom)
    dir_dose_rates = Path(dir_dose_rates)
    # check if the dose rate files exist
    dose_rate_files = list(dir_dose_rates.glob("*.seq.nrrd"))
    if len(dose_rate_files) < 1 and not generate_dose_rates:
        raise FileNotFoundError(f"No dose rate files found in {dir_dose_rates}. Set generate_dose_rates=True to create them.")

    if dvh_metric_goals is None:
        dvh_metric_goals = {
            "D95%(CTV)": target_dose,
            "D1cc(RECTUM)": target_dose * 0.75,
            "D0.1cc(URETHRA)": target_dose * 1.25,
            "CI(CTV)": 1.0,
            "HI(CTV)": 0.5,
        }
    if optimization_config_list is None:
        optimization_config_list=[
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=dvh_metric_goals["D95%(CTV)"],
                penalty_weight_linear=300,
                # penalty_weight_quadratic=1,
                # penalty_weight_uniformity=0,
                # penalty_weight_hotspot=1,
                # hotspot_threshold=1.5,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                # penalty_weight_quadratic=1,
                # penalty_weight_uniformity=0,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                # penalty_weight_quadratic=1,
                # penalty_weight_uniformity=0,
                mask_margin_mm=0,
                spacing_mm=3)
        ]

    plan_obj = load_dicom_to_plan(
        dir_dicom=pth_dicom,
        load_dicom_dose=False,
        strict_name_match=False,
        delivered_catheter_table=True,
        multi_processing=True,
        prescription_dose=target_dose,
        dvh_metric_goals=dvh_metric_goals,
        optimization_config_list=optimization_config_list)

    if generate_dose_rates:
        from brachyutils import DoseTG43
        pth_material = Path("admin/constants/structure_materials_prostate.json")
        mat_from_ct = False
        crop_by_contour = "body"
        content_to_export = {
            "number_histories": 1E6,
            "number_of_threads": 16,
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
        plan_obj.export_brachy_plan(
            dir_export=dir_dose_rates,
            content_to_export=content_to_export,
        )
        dose_gen_obj = DoseTG43(
            dir_plan_export=dir_dose_rates
            )
        dose_gen_obj.generate_dose(
            output_dose_per_dwell="dose_rate"
        )

    plan_obj.load_dose_rate_or_uncertainty_tensor(
        dir_dose_rate=dir_dose_rates,
        multi_processing=True,
    )

    return plan_obj

def generate_all_dose_rates(
    dir_all_dicoms: str | Path,
    dir_all_dose_rates: str | Path,
    ):
    dir_all_dicoms = Path(dir_all_dicoms)
    dir_all_dose_rates = Path(dir_all_dose_rates)

    for dicom_dir in Path(dir_all_dicoms).glob("*/"):
        get_a_plan_to_optimize(
            pth_dicom=dicom_dir,
            dir_dose_rates=dir_all_dose_rates/dicom_dir.name,
            generate_dose_rates=True,
        )
        
def eval_optim(
    dir_all_dicoms: str | Path,
    dir_all_dose_rates: str | Path,
    dvh_metric_goals: dict[str, float],
    target_dose: float = 21,
):
    r"""
    Purpose:
    - Evaluate the optimization performance of Gurobi on multiple brachytherapy plans.
    """
    dir_all_dicoms = list(Path(dir_all_dicoms).glob("*/"))
    dir_all_dose_rates = Path(dir_all_dose_rates)

    results_solver = DataFrame(columns=[
        "package", "objective_terms", "loading_time",
        "model_building_time", "solving_time", "status",
        "num_dwells"
        ]+list(dvh_metric_goals.keys()))
    
    config_variations = {
        "L": [
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=target_dose,
                penalty_weight_linear=300,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                mask_margin_mm=0,
                spacing_mm=3)
        ],
        "LQ": [
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=target_dose,
                penalty_weight_linear=300,
                penalty_weight_quadratic=1,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                mask_margin_mm=0,
                spacing_mm=3)
        ],
        "LH": [
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=target_dose,
                penalty_weight_linear=300,
                penalty_weight_hotspot=1,
                hotspot_threshold=1.5,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                mask_margin_mm=0,
                spacing_mm=3)
        ],
        "LQH": [
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=target_dose,
                penalty_weight_linear=300,
                penalty_weight_quadratic=1,
                penalty_weight_hotspot=1,
                hotspot_threshold=1.5,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                mask_margin_mm=0,
                spacing_mm=3)
        ],
        "LQU": [
            Optimization_Config(
                structure_name="CTV",
                dose_voxel_goal=target_dose,
                penalty_weight_linear=300,
                penalty_weight_quadratic=1,
                penalty_weight_uniformity=1,
                mask_margin_mm=0,
                spacing_mm=3),
            Optimization_Config(
                structure_name="URETHRA",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                penalty_weight_uniformity=1,
                mask_margin_mm=0,
                spacing_mm=1),
            Optimization_Config(
                structure_name="RECTUM",
                dose_voxel_goal=0,
                penalty_weight_linear=1,
                penalty_weight_quadratic=1,
                penalty_weight_uniformity=1,
                mask_margin_mm=0,
                spacing_mm=3)
        ]
    }

    for config_var in config_variations:
        config_list = config_variations[config_var]
        for pth_dicom in dir_all_dicoms:
            t0_loading = time()
            brachy_plan = get_a_plan_to_optimize(
                pth_dicom=pth_dicom,
                dir_dose_rates=dir_all_dose_rates/pth_dicom.name,
                generate_dose_rates=False,
                dvh_metric_goals=dvh_metric_goals,
                target_dose=target_dose,
                optimization_config_list=config_list, 
            )
            t1_loading = time()
            # XXX this part could be wrapped into a function for other optimizers
            from brachyutils import BrachyOptim_Gurobi
            try:
                t0_model_building = time()
                optim_obj = BrachyOptim_Gurobi(
                    brachy_plan=brachy_plan,
                )
                t1_model_building = time()
                t0_solving = time()
                brachy_plan = optim_obj.get_optimized_plan_from_model()
                t1_solving = time()
                result_dvh_metrics = brachy_plan.get_dvh_metrics(
                    return_percentage=True)
                status = "OPTIMIZED"
            except Exception as e:
                print(f"Optimization failed for {pth_dicom.name} with config {config_var}: {e}")
                t1_model_building = np.nan
                t1_solving = np.nan
                result_dvh_metrics = {key: np.nan for key in dvh_metric_goals.keys()}
                status = "FAILED"
            results_solver.loc[len(results_solver)] = {
                "package": "gurobi",
                "objective_terms": config_var,
                "loading_time": t1_loading - t0_loading,
                "model_building_time": t1_model_building - t0_model_building,
                "solving_time": t1_solving - t0_solving,
                "status": status,
                "num_dwells": len(optim_obj.dwell_time_variable_list),
                **result_dvh_metrics
            }
            results_solver.to_csv(
                dir_all_dose_rates/"eval_optim_gurobi_results.csv",
                index=False)
            return # for debugging only

if __name__ == "__main__":
    dir_all_dicoms = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023")
    dir_all_dose_rates = Path("temp_data/tg43/optimization") # for tg43
    target_dose = 21
    dvh_metric_goals = {
        "V100%(ctv)": 95,
        "D90%(ctv)": target_dose,
        "V150%(ctv)": 40,
        "HI(ctv)": 1,
        "CI(ctv)": 1,
        "D2cc(rectum)": 10,
        "D10%(urethra)":17,
        "D30%(urethra)": 15,
    }

    # # first, generate all the dose rate files
    # generate_all_dose_rates(
    #     dir_all_dicoms,
    #     dir_all_dose_rates,
    # )
    
    eval_optim(
        dir_all_dicoms,
        dir_all_dose_rates,
        dvh_metric_goals=dvh_metric_goals,
        target_dose=target_dose,
    )