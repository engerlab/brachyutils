from pathlib import Path
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterVar_Gurobi, CatheterTableOptim_Gurobi)
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from gurobipy import Model
from brachyutils.tests.test_optim_utils import get_a_plan_to_optimize
from brachyutils.tests.test_plan_utils import get_a_plan
from brachyutils.planning.optimization.optim_utils import Optimization_Config
from pandas import DataFrame
def test_catheter_gurobi_initialization():
    # we need a catheter table first!
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    cat_table = CatheterTable(catheter_list=list(pth_dicom.glob("RP*.dcm"))[0])

    catheter_vars = []
    model = Model("test_model")
    for catheter in cat_table:
        catheter_vars.append(
            CatheterVar_Gurobi(
            catheter=catheter,
            model=model,
            )
        )
    model.update()
    model.printStats()

def test_catheter_table_optim():
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("data_test/test_export_plan/prostate").resolve()
    target_dose = 21

    # # for loading the delivered dose rates. 
    dir_dose_rates = Path("data_test/prostate-glen-p1-dose").resolve()
    gen_dose_rates = False
    from_delivered_dwellpositions=True

    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            is_target=True,
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            penalty_weight_quadratic=1,
            penalty_weight_uniformity=1,
            penalty_weight_hotspot=1,
            hotspot_threshold=1.5,
            penalty_weight_variance_time=1,
            mask_margin_mm=0,
            spacing_mm=3,
            catheter_recommendaion=True),
        Optimization_Config(
            structure_name="URETHRA",
            is_target=False,
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3),
        Optimization_Config(
            structure_name="RECTUM",
            is_target=False,
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=3,
            )
    ]
    plan = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        from_delivered_dwellpositions=from_delivered_dwellpositions,
        optimization_config_list=optimization_config_list,
        generate_dose_rates=gen_dose_rates,
        )
    catheter_optim_obj = CatheterTableOptim_Gurobi(
        plan=plan,
        multi_processing=True,
        )
    optimized_plan = catheter_optim_obj.get_optimized_plan_from_model()
    optimized_plan.export_brachy_plan(
        dir_export=dir_export,
        content_to_export={
            "dose": True
        }
    )
    dvh_metrics_dict = optimized_plan.get_dvh_metrics(return_percentage=True)
    cat_table = optimized_plan.catheter_table.write_to_slicer_markup(
        pth_json = dir_export / "catheter_table.mrk.json")
    DataFrame([dvh_metrics_dict]).to_csv(
        dir_export / "dvh_metrics.csv", index=False)

def test_dynamic_plan_generation():
    r"""
    ### Purpose:
    """
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("data_test/test_export_plan/prostate").resolve()
    target_dose = 21
    
    # # for generating the dose rates on the fly
    dir_dose_rates=dir_export/"dose_rates"
    gen_dose_rates = True
    from_delivered_dwellpositions=False

    # # get the full catheter table.
    full_cat_table = CatheterTable(catheter_list=list(pth_dicom.glob("RP*.dcm"))[0])
    # # now split this catheter table into two.
    cat_table_p1 = full_cat_table[:len(full_cat_table)//2]
    cat_table_p2 = full_cat_table[len(full_cat_table)//2:]
    cat_table_p2.reset_index()
    cat_table_p1.info()
    cat_table_p2.info()

    # plan = get_a_plan(
    #     pth_dicom=pth_dicom,
    #     dir_dose_rates=dir_dose_rates,
    #     from_delivered_dwellpositions=from_delivered_dwellpositions,
    #     generate_dose_rates=gen_dose_rates,
    #     )

if __name__ == "__main__":
    # test_catheter_gurobi_initialization()
    # test_catheter_table_optim()
    test_dynamic_plan_generation()