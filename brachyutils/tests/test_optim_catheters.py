from pathlib import Path
from time import time
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterVar_Gurobi, CatheterTableOptim_Gurobi)
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from gurobipy import Model
from brachyutils.tests.test_optim_utils import get_a_plan_to_optimize
from brachyutils.tests.test_plan_utils import get_a_plan
from brachyutils.planning.optimization.optim_configs import Optimization_Config
from pandas import DataFrame
from brachyutils.dose.dose_generation_utils import RapidBrachyTG43
def test_catheter_gurobi_initialization():
    # we need a catheter table first!
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    cat_table = CatheterTable(catheters_dict=list(pth_dicom.glob("RP*.dcm"))[0])

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

def test_catheter_table_optim(
    retrun_optim_obj:bool = False
    ):
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    # dir_export = Path("data_test/test_export_plan/prostate").resolve()
    target_dose = 21
    from_delivered_dwellpositions=False
    multi_processing = False
    gen_dose_rates = False
    catheter_recommendaion=False

    # # For generating the dose rates
    dir_dose_rate = Path("temp_data/tg43/cat-optim/test").resolve()
    dir_export = dir_dose_rate

    dvh_metric_names = [
        "D90%(CTV)", "D2cc(RECTUM)", "D10%(URETHRA)",
        "D30%(URETHRA)", "CI(CTV)", "HI(CTV)",
        "V200%(CTV)", "V150%(CTV)", "V100%(CTV)"
    ]

    plan = get_a_plan(
        dir_dicom=pth_dicom,
        from_delivered_dwellpositions=from_delivered_dwellpositions,
        dvh_metric_goals=dvh_metric_names,
        )

    if gen_dose_rates:
        from brachyutils.dose.dose_generation_utils import RapidBrachyTG43
        init_export_config = {
            "dir_export": dir_dose_rate,
            "export_config_egsphant": {
                "strict_name_match": False,
                "crop_by_contour": ["ctv", "urethra", "rectum"]}
            }
        plan.export_brachy_plan(
            content_to_export=init_export_config
        )
        dose_generator = RapidBrachyTG43(
            dir_plan_export=dir_dose_rate
        )
        plan = dose_generator.run_dose_generation(
            plan=plan,
            generate_dose_rate_maps=True,
            export_config_brachyplan={
                "dir_export": dir_dose_rate,
                "export_config_plan_and_mac": True,
            }
        )
        # plan.export_brachy_plan(
        # content_to_export={
        #     "dir_export": dir_export,
        #     "export_config_dose": True,
        #     "export_config_cathetertable": {
        #             "file_extension": ".json",
        #         },
        #     }
        # )
    else:
        plan.catheter_table.load_dose_rates(
            dir_dose_rate=dir_dose_rate,
            multi_processing=multi_processing
        )
        
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
            catheter_recommendaion=catheter_recommendaion),
        Optimization_Config(
            structure_name="URETHRA",
            is_target=False,
            dose_voxel_goal=0,
            penalty_weight_linear=1,
            penalty_weight_quadratic=1,
            mask_margin_mm=0,
            spacing_mm=1),
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
    plan.setup_optimization(
        optimization_config_list=optimization_config_list,
        structure_list=plan.structure_list,
        strict_name_match=False,
    )

    catheter_optim_obj = CatheterTableOptim_Gurobi(
        plan=plan,
        multi_processing=multi_processing,
        )

    if retrun_optim_obj:
        return catheter_optim_obj

    optimized_plan = catheter_optim_obj.get_optimized_plan_from_model()
    optimized_plan.export_brachy_plan(
        content_to_export={
            "dir_export": dir_export,
            "export_config_dose": True,
            "export_config_cathetertable": {
                    "file_extension": ".json",
                },
        }
    )
    dvh_metrics_dict = optimized_plan.get_dvh_metrics(return_percentage=True)
    cat_table = optimized_plan.catheter_table.write_to_slicer_markup(
        pth_mrk_json = dir_export / "catheter_table.mrk.json")
    DataFrame([dvh_metrics_dict]).to_csv(
        dir_export / "dvh_metrics.csv", index=False)

def test_dynamic_plan_generation():
    r"""
    ### Purpose:
    """
    pth_dicom = Path("data_test/prostate-glen-p1-dcm")
    dir_export = Path("temp_data/tg43/optimization")
    target_dose = 21    
    # # for generating the dose rates on the fly
    dir_dose_rate=dir_export/"test"

    # # get a plan without catheter table.
    plan = get_a_plan(
        dir_dicom=pth_dicom,
        load_dicom_catheter_table=False,
        )
    # ensure that the plan does not have a catheter table.
    assert plan.catheter_table is None, "The plan should not have a catheter table."
    # export the egsphant with cropping for dose generation
    init_export_config = {
        "dir_export": dir_dose_rate,
        "export_config_egsphant": {
            "strict_name_match": False,
            "crop_by_contour": ["ctv", "urethra", "rectum"]}
        }
    plan.export_brachy_plan(
        content_to_export=init_export_config
    )
    # # get the full catheter table.
    full_cat_table = CatheterTable(catheters_dict=list(pth_dicom.glob("RP*.dcm"))[0])
    # # now split this catheter table into two.
    cat_table_p1 = full_cat_table[:len(full_cat_table)//2]
    cat_table_p2 = full_cat_table[len(full_cat_table)//2:]
    # cat_table_p2.reset_index()
    # cat_table_p1.info()
    # cat_table_p2.info()

    # # let's load catheter table, and calculate dose rates for the first half of the catheters.
    plan.set_catheter_table(
        catheter_table=cat_table_p1,
    )
    # plan.catheter_table.info()
    # # now generate dose rates for the first half of the catheters.
    # # initialize the dose generator object
    dose_generator = RapidBrachyTG43(
        dir_plan_export=dir_dose_rate
    )
    plan = dose_generator.run_dose_generation(
        plan=plan,
        generate_dose_rate_maps=True,
        export_config_brachyplan={
            "dir_export": dir_dose_rate,
            "export_config_plan_and_mac": {
                "name_combined": "cat_p1"
            },
        }
    )

    plan.set_catheter_table(
        catheter_table=cat_table_p1+cat_table_p2,        
    )
    plan = dose_generator.run_dose_generation(
        plan=plan,
        generate_dose_rate_maps=True,
        export_config_brachyplan={
            "dir_export": dir_dose_rate,
            "export_config_plan_and_mac": {
                "name_combined": "cat_p2"
            },
        }
    )
    plan.catheter_table.write_to_slicer_markup(dir_dose_rate/"catheter_table.mrk.json")
    plan.combined_dose.write_to_nrrd(dir_dose_rate/"combined_dose.seq.nrrd")
    print("dynamic plan generation test completed successfully.")


def test_set_constraints():
    dir_dose_rate = Path("temp_data/tg43/cat-optim/test").resolve()
    dir_export = dir_dose_rate

    from brachyutils.planning.optimization.optim_configs import Constraint_Config
    constraint_obj = Constraint_Config(
            constraint_type="bound",
            variable_type="dwell",
            variable_name_ids="1_1_0",
            equal=100,
        )
    constraint_dict = {
        constraint_obj.name: constraint_obj 
    }
    ti_build = time()
    catheter_optim_obj = test_catheter_table_optim(retrun_optim_obj=True)
    catheter_optim_obj.set_constraints(
        constraint_config_dict=constraint_dict
    )
    tf_build = time()
    optimized_plan = catheter_optim_obj.get_optimized_plan_from_model()
    t_solve = time()
    optimized_plan.export_brachy_plan(
        content_to_export={
            "dir_export": dir_export,
            "export_config_dose": True,
            "export_config_cathetertable": {
                    "file_extension": ".json",
                },
        }
    )
    dvh_metrics_dict = optimized_plan.get_dvh_metrics(return_percentage=True)
    DataFrame([dvh_metrics_dict]).to_csv(
        dir_export / "dvh_metrics.csv", index=False)
    print("time to build model with bound variables: ", tf_build - ti_build)
    print("time to solve model with bound variables: ", t_solve - tf_build)

if __name__ == "__main__":
    from viztracer import VizTracer
    tracer = VizTracer()
    tracer.start()
    # test_catheter_gurobi_initialization()
    test_catheter_table_optim()
    # test_dynamic_plan_generation()
    # test_set_constraints()
