from brachyutils.planning.optim_utils import DwellTimeVariable, Gurobi_Optimization, Model
from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.planning.optim_utils import Optimization_Config
from brachyutils.types import BrachyPlan
def get_a_plan_to_optimize()->BrachyPlan:
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_dir_dose_rate = "data_test/prostate-glen-p1-dose"
    target_dose = 21
    dvh_metric_goals = {
        "target_dose": target_dose,
        "D95%(ctv)": target_dose,
        "D1cc(rectum)": target_dose * 0.75,
        "D0.1cc(urethra)": target_dose * 1.25,
        "CI(ctv)": 1.0,
        "HI(ctv)": 0.5,
    }
    optimization_config_list=[
        Optimization_Config(
            structure_name="ctv",
            dose_voxel_goal=dvh_metric_goals["D95%(ctv)"],
            penalty_weight_linear=300,
            penalty_weight_hotspot=1,
            hotspot_threshold=1.5,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="urethra",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            mask_margin_mm=0,
            spacing_mm=1),
        Optimization_Config(
            structure_name="rectum",
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            mask_margin_mm=0,
            spacing_mm=3
        )
    ]
    plan_obj = load_dicom_to_plan(
        dir_dicom=pth_dicom,
        load_dicom_dose=False,
        delivered_catheter_table=True,
        dir_dose_rate=pth_dir_dose_rate,
        multi_processing=True,
        prescription_dose=target_dose,
        dvh_metric_goals=dvh_metric_goals,
        optimization_config_list=optimization_config_list)
    return plan_obj

def test_DwellTimeVariable():
    model = Model("test_model")

    x = DwellTimeVariable(
        model=model,
        name=f"catheter_{2}_dwell_{4}",
        dwell_time=0,
        lower_bound=0,
        upper_bound=100,
        coordinates=[23, 13, 12],
        )
    print("dwellTimeVariable5:", x)

def test_get_optimization_roi_bounds():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dicom_plan = load_dicom_to_plan(pth_dicom)
    optim_obj = Gurobi_Optimization(plan=dicom_plan, roi_margin_mm=[2, 2, 2])
    print(optim_obj.roi_bounds)
    print("breakpoint")

def test_run_gurobi_optim():
    plan_obj = get_a_plan_to_optimize()
    optim_obj = Gurobi_Optimization(plan=plan_obj)
    optimized_plan = optim_obj.get_optimized_plan_from_model()
    # print(optimized_plan.get_dvh_metrics())
    # print(optimized_plan.dwell_times)

    # test setting a bound on a specific dwell time variable
    optim_obj.bound_dwell_time(
        name="catheter_0_dwell_0",
        lower_bound=1,
        upper_bound=1
    )
    optimized_plan = optim_obj.get_optimized_plan_from_model()
    print(optimized_plan.dwell_times)

def test_run_ampl_optim():
    from brachyutils.planning.optim_utils import AMPL_Optimization
    plan_obj = get_a_plan_to_optimize()
    optim_obj = AMPL_Optimization(plan=plan_obj)
    # optimized_plan = optim_obj.get_optimized_plan_from_model()
    # print(optimized_plan.get_dvh_metrics())
    # print(optimized_plan.dwell_times)

if __name__ == "__main__":
    # test_DwellTimeVariable()
    # test_get_optimization_roi_bounds()
    # test_run_gurobi_optim()
    test_run_ampl_optim()