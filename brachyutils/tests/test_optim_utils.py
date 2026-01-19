from pandas import DataFrame
from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.planning.optimization.optim_utils import Optimization_Config
from brachyutils.types import BrachyPlan
from pathlib import Path

def get_a_plan_to_optimize(
    pth_dicom: str | Path,
    dir_dose_rates: str | Path,
    optimization_config_list,
    generate_dose_rates: bool = False,
    )->BrachyPlan:
    pth_dicom = Path(pth_dicom)
    dir_dose_rates = Path(dir_dose_rates)
    # check if the dose rate files exist
    dose_rate_files = list(dir_dose_rates.glob("*.seq.nrrd"))
    if len(dose_rate_files) < 1 and not generate_dose_rates:
        raise FileNotFoundError(f"No dose rate files found in {dir_dose_rates}. Set generate_dose_rates=True to create them.")

    target_dose = 21
    dvh_metric_goals = {
        "D90%(CTV)": target_dose,
        "D2cc(RECTUM)": target_dose * 0.75,
        "D10%(URETHRA)": target_dose * 1.133,
        "D30%(URETHRA)": target_dose,
        "CI(CTV)": 1.0,
        "HI(CTV)": 0.5,
        "V200%(CTV)": target_dose * 0.2,
        "V150%(CTV)": target_dose * 0.4,
        "V100%(CTV)": 100.0,
    }

    plan_obj = load_dicom_to_plan(
        dir_dicom=pth_dicom,
        load_dicom_dose=False,
        strict_name_match=False,
        delivered_catheter_table=True,
        multi_processing=True,
        prescription_dose=target_dose,
        dvh_metric_goals=dvh_metric_goals,
        optimization_config_list=optimization_config_list,
        dwells_near_ptv=True,
        )

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

def test_get_a_plan_to_optimize():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dir_dose_rates = "temp_data/tg43/optim_test"
    plan_obj = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        generate_dose_rates=True,
    )
    plan_obj.get_dvh_metrics()
    print("breakpoint")

def test_DwellTime_Gurobi():
    from brachyutils.planning.optimization.optim_gurobi import DwellTime_Gurobi, Model
    model = Model("test_model")

    x = DwellTime_Gurobi(
        model=model,
        name=f"catheter_{2}_dwell_{4}",
        dwell_time=0,
        lower_bound=0,
        upper_bound=100,
        coordinates=[23, 13, 12],
        )
    model.update()
    print("dwellTimeVariable5:", x)

def test_get_optimization_roi_bounds():
    from brachyutils.planning.optimization.optim_gurobi import BrachyOptim_Gurobi

    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dicom_plan = get_a_plan_to_optimize()
    optim_obj = BrachyOptim_Gurobi(plan=dicom_plan, roi_margin_mm=[2, 2, 2])
    print(optim_obj.roi_bounds)
    print("breakpoint")

def test_run_gurobi_optim():
    from brachyutils.planning.optimization.optim_gurobi import BrachyOptim_Gurobi
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dir_dose_rates = "data_test/prostate-glen-p1-dose"
    dir_result_out = Path("data_test/test_export_plan/prostate")
    # for debugging on server
    # pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p12")
    # dir_dose_rates = Path("temp_data/tg43/optimization/p12") # for tg43
    target_dose = 21
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            penalty_weight_quadratic=1,
            penalty_weight_uniformity=1,
            # penalty_weight_hotspot=1,
            # hotspot_threshold=1.5,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="URETHRA",
            dose_voxel_goal=0,#target_dose * 1.1,
            penalty_weight_linear=1,
            penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="RECTUM",
            dose_voxel_goal=0,#target_dose * 0.75,
            penalty_weight_linear=1,
            penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3)
    ]

    solver = "gurobi"
    plan_obj = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        generate_dose_rates=False,
        optimization_config_list=optimization_config_list,
    )
    # # print out the delivered plan dose and dvh metrics
    # plan_obj.combined_dose.write_brachydose_to_file(
    #     dir_result_out.joinpath("p1_delivered.seq.nrrd")
    #     )

    # delivered_dvh_metrics = plan_obj.get_dvh_metrics(return_percentage=True)
    # DataFrame([
    #     delivered_dvh_metrics | {
    #         "mean_dwell_times": plan_obj.dwell_times.mean(),
    #         "std_dwell_times": plan_obj.dwell_times.std(),
    #         }]).to_csv(
    #         dir_result_out.joinpath("p1_delivered_dvh_metrics.csv")
    #         )

    # # Optimize the plan and get the data
    results = DataFrame(
        columns=[
            "solver", "status",
            "mean(dwell_times)", "std(dwell_times)",
            "solve_time"] + list(plan_obj.dvh_metric_goals.keys())
        )

    optim_obj = BrachyOptim_Gurobi(plan=plan_obj, multi_processing=True)
    optimized_plan = optim_obj.get_optimized_plan_from_model()
    dvh_metrics = optimized_plan.get_dvh_metrics(return_percentage=True)
    results.loc[len(results)] = {
        "solver": solver,
        "status": "Solved" if optim_obj.solution_found else "Failed",
        "mean(dwell_times)": optimized_plan.dwell_times.mean(),
        "std(dwell_times)": optimized_plan.dwell_times.std(),
        "solve_time": optim_obj.solve_time} | dvh_metrics
    results.to_csv(dir_result_out.joinpath("gurobi_noHS.csv"))
    print(optimized_plan.dwell_times)
    # export phantom
    # plan_obj.phantom.export_to(
    #     dir_nrrd_out=dir_result_out,
        # dir_dicom_out=dir_result_out.joinpath("dcm")
        # )
    # export optimized dose
    # plan_obj.combined_dose.write_brachydose_to_file(
    #     dir_result_out.joinpath("p1_HSgurobi.seq.nrrd")
    #     )

    # optimized_plan.export_brachy_plan(
    #     dir_export="data_test/test_export_plan/prostate",
    #     content_to_export={
    #         "dose": True,
    #     }
    #     )

    # # test setting a bound on a specific dwell time variable
    # optim_obj.bound_dwell_time(
    #     name="catheter_0_dwell_0",
    #     lower_bound=1,
    #     upper_bound=1
    # )
    # optimized_plan = optim_obj.get_optimized_plan_from_model()
    # print(optimized_plan.get_dvh_metrics())
    # print(optimized_plan.dwell_times)

def test_dwellTime_AMPL():
    from brachyutils.planning.optimization.optim_ampl import DwellTime_AMPL
    from amplpy import AMPL
    
    model = AMPL()
    x = DwellTime_AMPL(
        model=model,
        name=f"catheter_{2}_dwell_{4}",
        dwell_time=0,
        lower_bound=0,
        upper_bound=100,
        coordinates=[23, 13, 12],
        )
    print(x._model_variable)

    x.set_bounds(model=model, lower_bound=32, upper_bound=45)
    print(x._model_variable.lb(), x._model_variable.ub())

def test_run_ampl_optim():
    from brachyutils.planning.optimization.optim_ampl import BrachyOptim_AMPL
    from pandas import DataFrame
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dir_dose_rates = "data_test/prostate-glen-p1-dose"
    dir_result_out = Path("data_test/test_export_plan/prostate")
    # for debugging on server
    # pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p12")
    # dir_dose_rates = Path("temp_data/tg43/optimization/p12") # for tg43
    target_dose = 21
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            # penalty_weight_quadratic=1,
            # penalty_weight_uniformity=1,
            # penalty_weight_hotspot=100,
            # hotspot_threshold=1.5,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="URETHRA",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            # penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=1),
        Optimization_Config(
            structure_name="RECTUM",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            # penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3)
    ]
    plan_obj = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        generate_dose_rates=False,
        optimization_config_list=optimization_config_list,
    )
    
    results = DataFrame(
        columns=[
            "solver", "status",
            "mean(dwell_times)", "std(dwell_times)",
            "solve_time"] + list(plan_obj.dvh_metric_goals.keys())
        )
    for solver in ["gurobi"]: # [
        # "couenne", "bonmin", "copt", 
        # "mosek" "ipopt", "xpress",
        # "cplex", "highs", "scip",
        # "gurobi", "gcg"]:
        # try:
        optim_obj = BrachyOptim_AMPL(plan=plan_obj, solver=solver, verbose=True)
        optimized_plan = optim_obj.get_optimized_plan_from_model(inplace=False)
        dvh_metrics = optimized_plan.get_dvh_metrics(return_percentage=True)
        results.loc[len(results)] = {
            "solver": solver,
            "status": "Solved" if optim_obj.solution_found else "Failed",
            "mean(dwell_times)": optimized_plan.dwell_times.mean(),
            "std(dwell_times)": optimized_plan.dwell_times.std(),
            "solve_time": optim_obj.solve_time} | dvh_metrics
        del optimized_plan
        del optim_obj
        del plan_obj
        # except Exception as e:
        #     # raise e
        #     continue
    results.to_csv("data_test/test_export_plan/prostate/ampl_lin.csv")
    # results.to_csv("data_test/test_export_plan/prostate/solvers_quadObj.csv")

def test_dwelltime_orTools():
    from brachyutils.planning.optimization.optim_ortools import DwellTime_ORTools
    from ortools.math_opt.python import mathopt

    solver = mathopt.Model(name="test_model")
    x = DwellTime_ORTools(
        model=solver,
        name=f"catheter_{2}_dwell_{4}",
        dwell_time=0,
        lower_bound=0,
        upper_bound=100,
        coordinates=[23, 13, 12],
        )
    print(x._model_variable)
    x.set_bounds(lower_bound=32, upper_bound=45)
    print(x._model_variable.lower_bound, x._model_variable.upper_bound)

def test_run_ortool_optim():
    from brachyutils.planning.optimization.optim_ortools import BrachyOptim_ORTools
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dir_dose_rates = "data_test/prostate-glen-p1-dose"
    dir_result_out = Path("data_test/test_export_plan/prostate")
    # for debugging on server
    # pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p12")
    # dir_dose_rates = Path("temp_data/tg43/optimization/p12") # for tg43
    target_dose = 21
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            # penalty_weight_quadratic=1,
            # penalty_weight_uniformity=1,
            # penalty_weight_hotspot=100,
            # hotspot_threshold=1.5,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="URETHRA",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            # penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=1),
        Optimization_Config(
            structure_name="RECTUM",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            # penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3)
    ]
    plan_obj = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        generate_dose_rates=False,
        optimization_config_list=optimization_config_list,
    )

    results = DataFrame(
        columns=[
            "solver", "status",
            "mean(dwell_times)", "std(dwell_times)",
            "solve_time"] + list(plan_obj.dvh_metric_goals.keys())
        )
    for solver in ["GLOP"]: #["GLOP", "PDLP","GSCIP", "GLPK"]:
        # try:
        optim_obj = BrachyOptim_ORTools(plan=plan_obj, solver=solver)
        optimized_plan = optim_obj.get_optimized_plan_from_model(solver=solver, inplace=False)
        dvh_metrics = optimized_plan.get_dvh_metrics(return_percentage=True)        
        results.loc[len(results)] = {
            "solver": solver,
            "status": "Solved" if optim_obj.solution_found else "Failed",
            "mean(dwell_times)": optimized_plan.dwell_times.mean(),
            "std(dwell_times)": optimized_plan.dwell_times.std(),
            "solve_time": optim_obj.solve_time} | dvh_metrics
        # except Exception as e:
        #     # raise e
        #     continue
        results.to_csv(dir_result_out.joinpath("ortools_lin.csv"))
        # except:
        #     print(f"Solver {solver} failed.")
        #     results.loc[len(results)] = {
        #         "solver": solver,
        #         "status": "Failed",
        #         "dvh_metrics": "N/A",
        #         "mean(dwell_times)": "N/A",
        #         "std(dwell_times)": "N/A",
        #         "solve_time": 0
        #         }
        #     continue
    # results.to_csv("data_test/test_export_plan/prostate/ortools_solvers.csv")

def test_hotspot_estimators():
    from brachyutils.planning.optimization.optim_gurobi import BrachyOptim_Gurobi
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dir_dose_rates = "data_test/prostate-glen-p1-dose"
    # for debugging on server
    # pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p1")
    # dir_dose_rates = Path("temp_data/tg43/optimization/p1") # for tg43
    target_dose = 21
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            penalty_weight_quadratic=1,
            # penalty_weight_uniformity=0,
            penalty_weight_hotspot=100,
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
    ]

    solver = "gurobi"
    plan_obj = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        generate_dose_rates=False,
        optimization_config_list=optimization_config_list,
    )
    print(plan_obj.dwell_times)
    # export the phantom structures to make sure the hotspot estimators are working correctly
    # will write them to both nrrd and dicom format.
    # plan_obj.phantom.export_to(
    #     dir_nrrd_out="data_test/test_export_plan/prostate",
    #     dir_dicom_out="data_test/test_export_plan/prostate/dcm"
    #     )
    # plan_obj.combined_dose.write_brachydose_to_file("data_test/test_export_plan/prostate/p1_dose.seq.nrrd")
    # results = DataFrame(
    #     columns=[
    #         "solver", "status",
    #         "mean(dwell_times)", "std(dwell_times)",
    #         "solve_time"] + list(plan_obj.dvh_metric_goals.keys())
    #     )

    # optim_obj = BrachyOptim_Gurobi(plan=plan_obj)
    # optimized_plan = optim_obj.get_optimized_plan_from_model()
    # dvh_metrics = optimized_plan.get_dvh_metrics(return_percentage=True)
    # results.loc[len(results)] = {
    # "solver": solver,
    # "status": "Solved" if optim_obj.solution_found else "Failed",
    # "mean(dwell_times)": optimized_plan.dwell_times.mean(),
    # "std(dwell_times)": optimized_plan.dwell_times.std(),
    # "solve_time": optim_obj.solve_time} | dvh_metrics
    # results.to_csv("data_test/test_export_plan/prostate/solvers_linObj_gurobi.csv")
    # print(optimized_plan.dwell_times)

def test_scaling_plan_to_objective():
    from brachyutils.planning.optimization.optim_utils import scale_to_objective
    from brachyutils.geometry.phantom_utils import BrachyPhantom
    from brachyutils.planning.plan_utils import BrachyPlan

    from ai_assisted_brachy.utils.utils import sitk_bb_to_opentps_range
    
    import json
    import os 
    import glob

    ai_assisted_brachy_pipeline_out_folder = "/home/sebq/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_validation_results_oar_lw_up_to_500_hotspotupto200_contraints_by_voxel_hotspothr2_more_runs_focusoptim50cc_scalingplans_mostcriteria/cat_Dataset010_oar_Dataset030/val_benchmark_fold_0/592479/"
    exp_folder = os.path.join(ai_assisted_brachy_pipeline_out_folder, "manual_clinical_structures__clinical_dwellpos__auto_optimization")
    phantom = BrachyPhantom(
        pth_phantom_file=os.path.join(ai_assisted_brachy_pipeline_out_folder, "ct_isotropic_1mm.nrrd"),
        pth_structures_file=os.path.join(ai_assisted_brachy_pipeline_out_folder, "manual_clinical_structures_isotropic_1mm.seg.nrrd"),
    )
    # Cropping the phantom to dose shape 
    cropped_bounds_potential_path = glob.glob(os.path.join(exp_folder, "Doses", "*_cropped_bounds.json"))
    assert len(cropped_bounds_potential_path) > 0, (
        f"TG43 and MC DL doses already exist but could not find the cropped bounds file in {os.path.join(exp_folder, 'Doses')}."
    )
    with open(cropped_bounds_potential_path[0], "r") as file:
        bounds_used = json.load(file)
    bounds_index_range = sitk_bb_to_opentps_range(bounds_used)
    phantom.crop_by_index(index_range=bounds_index_range, inplace=True, no_margin=True)
    
    dvh_goal_metrics_path = os.path.join(
        ai_assisted_brachy_pipeline_out_folder, 
        "manual_clinical_structures__clinical_dwellpos__auto_optimization",
        "dvh_metric_goals.json")
    with open(dvh_goal_metrics_path, 'r') as f:
        dvh_metric_goals = json.load(f)

    plan = BrachyPlan(
        phantom=phantom,
        dvh_metric_goals=dvh_metric_goals,
        prescription_dose=dvh_metric_goals["target_dose"],
        catheter_table=os.path.join(exp_folder, "optimized_catheter_table_30_randominit_20_moboiter_opt_on_tg43.json"),
        # dir_dose_rate=os.path.join(ai_assisted_brachy_pipeline_out_folder, "test_numdosepoints/cat_Dataset010_oar_Dataset030/val_benchmark_fold_0/592479/AI_structures__AI_catheter_contour_created_dwellpos__auto_optimization/Doses/TG43_doses"),
        combined_dose=os.path.join(exp_folder, "Doses/TG43_doses/combined_30_randominit_20_moboiter_opt_on_tg43.seq.nrrd"),
    )

    print("Before scaling:", plan.get_dvh_metrics(return_percentage=False, bin_size=0.0001))
    plan, scale_factor = scale_to_objective(
        plan=plan,
        objective_to_scale_to={"V100%(PTV)": 81}, # 87.1169}, # {"D90%(CTV)": 6.}, #
    )
    ## If you loaded dose rates
    # plan.update_plan_from_catheter_table()

    ## If you loaded combined dose directly
    plan.combined_dose.dose_image.imageArray = plan.combined_dose.dose_image.imageArray * scale_factor
    print("After scaling:", plan.get_dvh_metrics(return_percentage=False, bin_size=0.0001))

    plan, scale_factor = scale_to_objective(
        plan=plan,
        objective_to_scale_to={"V100%(PTV)": 81}, # 87.1169}, # {"D90%(CTV)": 6.}, #
    )
    plan.combined_dose.dose_image.imageArray = plan.combined_dose.dose_image.imageArray * scale_factor
    print("After second scaling:", plan.get_dvh_metrics(return_percentage=False, bin_size=0.0001))


if __name__ == "__main__":
    # test_get_a_plan_to_optimize()
    # test_DwellTime_Gurobi()
    # test_get_optimization_roi_bounds()
    # test_run_gurobi_optim()
    # test_dwellTime_AMPL()
    # test_run_ampl_optim()
    # test_dwelltime_orTools()
    # test_run_ortool_optim()
    # test_hotspot_estimators()
    test_scaling_plan_to_objective()