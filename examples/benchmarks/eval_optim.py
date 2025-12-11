import pandas as pd
from copy import deepcopy
from brachyutils.planning.optimization.optim_ortools import BrachyOptim_ORTools
from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.planning.optimization.optim_utils import Optimization_Config

from brachyutils.types import BrachyPlan
from pathlib import Path
from pandas import DataFrame
from time import time
import numpy as np
from tqdm import tqdm

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

    plan_obj = load_dicom_to_plan(
        dir_dicom=pth_dicom,
        load_dicom_dose=False,
        strict_name_match=False,
        delivered_catheter_table=True,
        multi_processing=True,
        prescription_dose=target_dose,
        dvh_metric_goals=dvh_metric_goals,
        optimization_config_list=optimization_config_list
        )

    if generate_dose_rates:
        from brachyutils import DoseTG43
        pth_material = Path("admin/constants/structure_materials_prostate.json")
        mat_from_ct = False
        crop_by_contour = "body"
        timing_df = DataFrame(columns=["case_name", "time", "num_dwells"])
        # content_to_export = {
        #     "number_histories": 1E6,
        #     "number_of_threads": 16,
        # }
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
        t0_dose_gen = time()
        plan_obj.export_brachy_plan(
            dir_export=dir_dose_rates,
            content_to_export=content_to_export,
        )
        dose_gen_obj = DoseTG43(
            dir_plan_export=dir_dose_rates,
            )
        dose_gen_obj.generate_dose(
            output_dose_per_dwell="dose_rate",
            num_threads=24,
        )
        t1_dose_gen = time()
        num_dwells = plan_obj.num_dwells
        timing_df.loc[len(timing_df)] = {
            "case_name": pth_dicom.name,
            "time": t1_dose_gen - t0_dose_gen,
            "num_dwells": num_dwells,
        }
        timing_df.to_csv(
            dir_dose_rates/"dose_rate_generation_timing.csv",
            index=False,
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
    for dicom_dir in tqdm(Path(dir_all_dicoms).glob("*/")):
        # if dicom_dir.name in ["p1", "p4", "p7"]:
        #     continue
        get_a_plan_to_optimize(
            pth_dicom=dicom_dir,
            dir_dose_rates=dir_all_dose_rates/dicom_dir.name,
            generate_dose_rates=True,
        )
        # break # for debugging only

def run_optimization(
    plan,
    config_list,
    package,
    solver,
    pth_out_dose: str | Path = None,
):
    r"""
    Purpose:
    - Run optimization on a brachytherapy plan using specified package and solver.
    """
    # reset the optimization setup
    plan._reset_optimization()
    plan.optimization_config_list = config_list
    plan.setup_optimization(
        plan.optimization_config_list,
        plan.structure_list,
    )
    case_name = plan.phantom.pth_image.name
    try:
        t0_model_building = time()
        if package == "gurobi":
            from brachyutils import BrachyOptim_Gurobi
            optim_obj = BrachyOptim_Gurobi(
                plan=plan,
            )
        elif package == "ampl":
            from brachyutils import BrachyOptim_AMPL
            optim_obj = BrachyOptim_AMPL(
                plan=plan,
                solver=solver,
            )
        elif package == "ortools":
            from brachyutils import BrachyOptim_ORTools
            optim_obj = BrachyOptim_ORTools(
                plan=plan,
                solver=solver,
            )
        else:
            raise ValueError(f"Unsupported optimization package: {package}")
        t1_model_building = time()
        t0_post_proc = time()
        brachy_plan = optim_obj.get_optimized_plan_from_model()
        t1_post_proc = time()
        solve_time = optim_obj.solve_time
        result_dvh_metrics = brachy_plan.get_dvh_metrics(return_percentage=True)
        status = "OPTIMIZED"
    except Exception as e:
        print(f"Optimization failed for {case_name} with config {config_list}, package {package}, solver {solver}: {e}")
        t1_model_building = np.nan
        t1_post_proc = np.nan
        solve_time = np.nan
        result_dvh_metrics = {key: np.nan for key in plan.dvh_metric_goals.keys()}
        status = "FAILED"
    optim_trial_result = {
        "case_name": case_name,
        "package": package,
        "solver": solver,
        # "objective_terms": ",".join([cfg.structure_name for cfg in config_list]),
        "model_building_time": t1_model_building - t0_model_building,
        "solve_time": solve_time,
        "post_processing_time": t1_post_proc - t0_post_proc - solve_time,
        "status": status,
        **result_dvh_metrics
    }
    # for debugging
    # print("Dwell Times are: \n", brachy_plan.dwell_times)
    if pth_out_dose is not None and status == "OPTIMIZED":
        brachy_plan.combined_dose.write_brachydose_to_file(
            pth_out_dose
        )
    del optim_obj
    return optim_trial_result

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

    package_solver_dict = {
        "gurobi": ["gurobi"],
        "ampl": ["xpress", "cplex", "highs"],
        "ortools": ["GLOP", "PDLP","GSCIP"],
    }

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
        # "LQ": [
        #     Optimization_Config(
        #         structure_name="CTV",
        #         dose_voxel_goal=target_dose,
        #         penalty_weight_linear=300,
        #         penalty_weight_quadratic=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3),
        #     Optimization_Config(
        #         structure_name="URETHRA",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         penalty_weight_quadratic=1,
        #         mask_margin_mm=0,
        #         spacing_mm=1),
        #     Optimization_Config(
        #         structure_name="RECTUM",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         penalty_weight_quadratic=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3)
        # ],
        # "LU": [
        #     Optimization_Config(
        #         structure_name="CTV",
        #         dose_voxel_goal=target_dose,
        #         penalty_weight_linear=300,
        #         penalty_weight_uniformity=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3),
        #     Optimization_Config(
        #         structure_name="URETHRA",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=1),
        #     Optimization_Config(
        #         structure_name="RECTUM",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3)
        # ],
        # "LH": [
        #     Optimization_Config(
        #         structure_name="CTV",
        #         dose_voxel_goal=target_dose,
        #         penalty_weight_linear=300,
        #         penalty_weight_hotspot=1,
        #         hotspot_threshold=1.5,
        #         mask_margin_mm=0,
        #         spacing_mm=3),
        #     Optimization_Config(
        #         structure_name="URETHRA",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=1),
        #     Optimization_Config(
        #         structure_name="RECTUM",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3)
        # ],
        # "LT": [
        #     Optimization_Config(
        #         structure_name="CTV",
        #         dose_voxel_goal=target_dose,
        #         penalty_weight_linear=300,
        #         penalty_weight_std_time_L2=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3),
        #     Optimization_Config(
        #         structure_name="URETHRA",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=1),
        #     Optimization_Config(
        #         structure_name="RECTUM",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3)
        # ],
        # "LQUTH": [
        #     Optimization_Config(
        #         structure_name="CTV",
        #         dose_voxel_goal=target_dose,
        #         penalty_weight_linear=300,
        #         penalty_weight_quadratic=1,
        #         penalty_weight_uniformity=1,
        #         penalty_weight_hotspot=1,
        #         hotspot_threshold=1.5,
        #         penalty_weight_std_time_L2=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3),
        #     Optimization_Config(
        #         structure_name="URETHRA",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         penalty_weight_quadratic=1,
        #         mask_margin_mm=0,
        #         spacing_mm=1),
        #     Optimization_Config(
        #         structure_name="RECTUM",
        #         dose_voxel_goal=0,
        #         penalty_weight_linear=1,
        #         penalty_weight_quadratic=1,
        #         mask_margin_mm=0,
        #         spacing_mm=3)
        # ],
    }

    results_solver_dict = {}
    for package in package_solver_dict:
        results_solver_dict[package] = DataFrame(columns=[
            "case_name", "package", "solver",
            "objective_terms", "loading_time", "model_building_time",
            "solve_time", "post_processing_time", "status",
            ]+list(dvh_metric_goals.keys()))

    for pth_dicom in dir_all_dicoms:
        t0_loading = time()
        brachy_plan = get_a_plan_to_optimize(
            pth_dicom=pth_dicom,
            dir_dose_rates=dir_all_dose_rates/pth_dicom.name,
            generate_dose_rates=False,
            dvh_metric_goals=dvh_metric_goals,
            target_dose=target_dose,
            # optimization_config_list=config_list, 
            )
        t1_loading = time()
        for package in package_solver_dict:
            for solver in package_solver_dict[package]:
                for config_var in config_variations:
                    optim_trial_result = run_optimization(
                    plan=brachy_plan,
                    config_list=config_variations[config_var],
                    package=package,
                    solver=solver,
                    # pth_out_dose=dir_all_dose_rates/f"optimized_{pth_dicom.name}_{package}_{solver}_{config_var}.nrrd",
                )
                    results_solver_dict[package].loc[len(results_solver_dict[package])] = optim_trial_result | {
                        "loading_time": t1_loading - t0_loading,
                        "objective_terms": config_var
                    }
                    results_solver_dict[package].to_csv(
                        dir_all_dose_rates/f"eval_optim_results_{package}.csv",
                        index=False)
        # break # for debugging only

def gen_box_plots_solvers_results(
    results_df: DataFrame,
    filter_by: dict[str, str | float | int],
    penalty_term: str,
    dir_fig_save: str | Path,
    pth_mean_std_csv: str | Path = None,
    ):
    r"""
    ### Purpose:
    - Generate the box plots for each metric. the metrics 
    are model_building_time, solve_time, and DVH metrics.
    in each box the data from the same patients is presented.
    it is possible to filter the data, for example only the 
    columns with objective_terms == "L" will be plotted. or
    only the columns with Status == "OPTIMIZED". 
    ### Inputs:
    - `results_df`: DataFrame containing all the optimization results.
    - `filter_by`: Dictionary specifying filtering criteria for the DataFrame.
    """
    dir_fig_save = Path(dir_fig_save)
    dir_fig_save.mkdir(parents=True, exist_ok=True)
    if pth_mean_std_csv is not None:
        mean_std_df = pd.DataFrame(columns=results_df.columns)

    filtered_df = deepcopy(results_df)
    for key, val in filter_by.items():
        filtered_df = filtered_df.loc[filtered_df[key] == val]
    print("debug")
    # get unique packages and solvers
    unique_packages = filtered_df["package"].unique()
    unique_solvers = filtered_df["solver"].unique()
    
    # # filter the dataframe based on packages and solvers
    # # skip if there is no data for a package-solver combination
    data_for_box_plots = {}
    for package in unique_packages:
        for solver in unique_solvers:
            df_subset = filtered_df.loc[
                (filtered_df["package"] == package) &
                (filtered_df["solver"] == solver)
            ]
            if df_subset.empty:
                continue
            # if package == "ortools":
            #     continue
            # if solver == "gcg":
            #     continue
            # generate box plots for each metric
            metrics_to_plot = [
                "model_building_time",
            ] + [col for col in filtered_df.columns if col not in [
                "case_name", "package", "solver",
                "objective_terms", "status"
            ]]
            if pth_mean_std_csv is not None:
                mean_std_df.loc[len(mean_std_df)] = {
                    "case_name": "MEAN",
                    "package": package,
                    "solver": solver,
                    "objective_terms": df_subset["objective_terms"].iloc[0],
                    "status": df_subset["status"].iloc[0],                    
                } | df_subset[metrics_to_plot + ["loading_time"]].mean().to_dict()
                mean_std_df.loc[len(mean_std_df)] = {
                    "case_name": "STD",
                    "package": package,
                    "solver": solver,
                    "objective_terms": df_subset["objective_terms"].iloc[0],
                    "status": df_subset["status"].iloc[0],                    
                } | df_subset[metrics_to_plot + ["loading_time"]].std().to_dict()
                mean_std_df.to_csv(pth_mean_std_csv, index=False)
            # gatheter data for the box plots
            # in a dictionary.
            data_for_box_plots[f"{package}_{solver}"] = df_subset[metrics_to_plot]
    for metric in metrics_to_plot:
        package_solver_labels = []
        data_to_plot = []
        for key in data_for_box_plots:
            package_solver_labels.append(key)
            data_to_plot.append(data_for_box_plots[key][metric])
        # generate box plot
        box_plots(
            title=f"{metric} for {penalty_term} penalty",
            data=data_to_plot,
            labels=package_solver_labels,
            y_label=metric,
            x_label="Package_Solver",
            pth_save=dir_fig_save/f"boxplot_{metric}.svg"
        )

def box_plots(
    title: str,
    data: list[pd.Series],
    labels: list[str],
    y_label: str,
    x_label: str,
    pth_save: str | Path = None,
    ):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title.replace("_", " "), fontsize=20)
    plt.ylabel(y_label.replace("_", " "), fontsize=18)
    plt.xlabel(x_label.replace("_", " "), fontsize=18)
    plt.grid(True, axis="y")
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if pth_save is not None:
        plt.savefig(pth_save)
    else:
        plt.show()


def gen_box_plots_penalty_results(
    results_df: DataFrame,
    dir_fig_save: str | Path,
    pth_mean_std_csv: str | Path = None,
):
    r"""
    ### Purpose:
    - Generate box plots to compare the effect of different penalty terms on optimization results.
    ### Inputs:
    - `results_df`: DataFrame containing all the optimization results.
    - `dir_fig_save`: Directory to save the generated box plots.
    - `pth_mean_std_csv`: Path to save the mean and standard deviation of results as a CSV file.
    ### Outputs:
    - Box plots saved in the specified directory.
    """
    dir_fig_save = Path(dir_fig_save)
    dir_fig_save.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [col for col in results_df.columns if col not in [
        "case_name", "package", "solver", "objective_terms", "status"]]
    unique_penalty_terms = results_df["objective_terms"].unique()
    if pth_mean_std_csv is not None:
        mean_std_df = pd.DataFrame(columns=["case_name"] + metrics_to_plot)

    for penalty_term in unique_penalty_terms:
        df_subset = results_df.loc[
                results_df["objective_terms"] == penalty_term
            ].loc[:, metrics_to_plot]
        if df_subset.empty:
            continue
        if pth_mean_std_csv is not None:
            mean_std_df.loc[len(mean_std_df)] = {
                "case_name": f"MEAN({penalty_term})",
            } | df_subset.mean().to_dict()
            mean_std_df.loc[len(mean_std_df)] = {
                "case_name": f"STD({penalty_term})",
            } | df_subset.std().to_dict()
            mean_std_df.to_csv(pth_mean_std_csv, index=False)

    for metric in metrics_to_plot:
        penalty_term_labels = []
        data_to_plot = []
        for penalty_term in unique_penalty_terms:
            df_subset = results_df.loc[
                    results_df["objective_terms"] == penalty_term
                ]
            if df_subset.empty:
                continue
            penalty_term_labels.append(penalty_term)
            data_to_plot.append(df_subset[metric])
        # # generate box plot
        box_plots(
            title=f"{metric} for different penalty terms",
            data=data_to_plot,
            labels=penalty_term_labels,
            y_label=metric,
            x_label="Penalty Terms",
            pth_save=dir_fig_save/f"boxplot_{metric}.svg"
        )


if __name__ == "__main__":
    dir_all_dicoms = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023")
    dir_all_dose_rates = Path("temp_data/tg43/optimization") # for tg43
    target_dose = 21
    dvh_metric_goals = {
        "V100%(ctv)": 95,
        "D90%(ctv)": target_dose,
        "V150%(ctv)": 40,
        "V200%(ctv)": 10,
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
    # # evaluate the optimization performance for packages, solvers and configs
    # eval_optim(
    #     dir_all_dicoms,
    #     dir_all_dose_rates,
    #     dvh_metric_goals=dvh_metric_goals,
    #     target_dose=target_dose,
    # )

    # # load the results dataframes for all the packages
    all_results_pths = list(dir_all_dose_rates.glob("eval_optim_results_*.csv"))
    list_all_data_df = [pd.read_csv(pth) for pth in all_results_pths]
    all_data_df = pd.concat(list_all_data_df, ignore_index=True)
    # # compare package-solvers on linear only config
    gen_box_plots_solvers_results(
        all_data_df,
        filter_by={"objective_terms": "L", "status": "OPTIMIZED"},
        penalty_term="linear",
        dir_fig_save=dir_all_dose_rates/"figs_optim_results_L",
        pth_mean_std_csv=dir_all_dose_rates/"mean_std_optim_results_L.csv",
    )
    
    # # # compare the effect of different penalty terms using gurobi
    # pth_full_results_gurobi = dir_all_dose_rates/"full_eval_optim_results_gurobi.csv"
    # results_gurobi_df = pd.read_csv(pth_full_results_gurobi)
    # gen_box_plots_penalty_results(
    #     results_gurobi_df,
    #     dir_fig_save=dir_all_dose_rates/"figs_optim_results_full",
    #     pth_mean_std_csv=dir_all_dose_rates/"mean_std_optim_results_full.csv",
    # )
