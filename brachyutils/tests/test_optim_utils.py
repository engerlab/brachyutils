from pandas import DataFrame
from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.planning.optimization.optim_utils import Optimization_Config
from brachyutils.types import BrachyPlan
from pathlib import Path
from time import time

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
        # add_hotspots_to_phantom=True,
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
    # pth_dicom = "data_test/prostate-glen-p1-dcm"
    # dir_dose_rates = "data_test/prostate-glen-p1-dose"
    dir_result_out = Path("data_test/test_export_plan/prostate")
    # for debugging on server
    pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p1")
    dir_dose_rates = Path("temp_data/tg43/optimization/p1") # for tg43
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
            # penalty_weight_std_time_L2=1,
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
            + ["model_build_time"]
        )
    t0=time()
    optim_obj = BrachyOptim_Gurobi(plan=plan_obj, multi_processing=True)
    t1 = time()
    optimized_plan = optim_obj.get_optimized_plan_from_model()
    dvh_metrics = optimized_plan.get_dvh_metrics(return_percentage=True)
    results.loc[len(results)] = {
        "solver": solver,
        "status": "Solved" if optim_obj.solution_found else "Failed",
        "mean(dwell_times)": optimized_plan.dwell_times.mean(),
        "std(dwell_times)": optimized_plan.dwell_times.std(),
        "solve_time": optim_obj.solve_time,
        "model_build_time": t1-t0,
        } | dvh_metrics
    results.to_csv(dir_result_out.joinpath("gurobi_lin_HS.csv"))
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
    # pth_dicom = "data_test/prostate-glen-p1-dcm"
    # dir_dose_rates = "data_test/prostate-glen-p1-dose"
    dir_result_out = Path("data_test/test_export_plan/prostate")
    # for debugging on server
    pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p1")
    dir_dose_rates = Path("temp_data/tg43/optimization/p1") # for tg43
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
        print(optimized_plan.dwell_times)
        # except Exception as e:
        #     # raise e
        #     continue
    results.to_csv(dir_result_out/"ampl_lin.csv")
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
    # pth_dicom = Path("/home/ubuntu").joinpath("YourLocalHome/Data/prostate/prostate-glen-2023/p1")
    # dir_dose_rates = Path("temp_data/tg43/optimization/p1") # for tg43
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
            spacing_mm=3),
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


if __name__ == "__main__":
    # test_get_a_plan_to_optimize()
    # test_DwellTime_Gurobi()
    # test_get_optimization_roi_bounds()
    test_run_gurobi_optim()
    # test_dwellTime_AMPL()
    # test_run_ampl_optim()
    # test_dwelltime_orTools()
    # test_run_ortool_optim()
    # test_hotspot_estimators()