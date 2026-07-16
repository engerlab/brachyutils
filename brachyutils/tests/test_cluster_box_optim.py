from time import time
from brachyutils.planning.optimization.optim_cath.cluster_box_optim import get_geometric_constraints
from brachyutils.planning.optimization.optim_configs import Optimization_Config
from brachyutils.tests.test_cluster_box import test_cluster_box
from brachyutils.planning.plan_utils import load_dicom_to_plan
from pathlib import Path
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Catheter_Rotation, Config_ClusterBox
from brachyutils.planning.optimization.optim_cath.cluster_box_optim import ClusterBoxOptim, get_voxel_dose_rate_goal
    
def test_get_geometric_constraints():
    cbox = test_cluster_box(return_box=True)
    constraint_dict = get_geometric_constraints(cluster_box=cbox)
    print("debug here")

def test_cluster_box_optim(
    num_decision_planes = None,
    config_angle = None,
    return_output:bool = False,
    export_cluster_box:bool = False,
    run_optimization:bool = False,
    solve_strategy="simultaneous",
    dwell_reach_mm=None,):
    
    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    outdir = Path("data_test/test_export_plan/prostate/clusterbox_optim")
    target_dose = 15
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            is_target=True,
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            penalty_weight_quadratic=1,
            penalty_weight_uniformity=1,
            # penalty_weight_hotspot=1,
            # hotspot_threshold=1.5,
            # penalty_weight_variance_time=1,
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

    if config_angle is None:
        config_angle = Config_Catheter_Rotation(
            x_angle_max=0,
            x_angle_step=0,
            y_angle_max=0,
            y_angle_step=0,
        )
    if num_decision_planes is None:
        num_decision_planes = 2
    config_cluster_box = Config_ClusterBox(
        num_physical_catheters=15,
        insertion_point_spacing_mm=5,
        num_decision_planes=num_decision_planes,
        config_angle=config_angle,
        box_margin_mm=5,
    )

    # # build a plan without catheter table but have optimizatio constraints.
    plan = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_catheter_table=False,
        load_dicom_prescription_dose=False,
        optimization_config_list=optimization_config_list,
        strict_name_match=False,
        prescription_dose=target_dose,
        )

    cbox_optim = ClusterBoxOptim(
        plan=plan,
        config_cluster_box=config_cluster_box,
        solve_strategy = solve_strategy,
        dwell_reach_mm = dwell_reach_mm,
    )
    # # export the cluster
    if export_cluster_box:
        cbox_optim.export_to(
            out_dir=outdir,
            dose_normalization_constant=target_dose
        )
    optimized_plan = None
    if run_optimization:
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
        optimized_plan = cbox_optim.get_optimized_plan_from_model()
        optimized_plan.set_dvh_metric_goals(
            dvh_metric_goals=dvh_metric_goals,
            strict_name_match=False
        )
        observed_dvh_metrics = optimized_plan.get_dvh_metrics()
        print("DVH Metrics are:")
        print(observed_dvh_metrics)
        optimized_plan.catheter_table.combined_dose.write_brachydose_to_file(
            outdir/"combined_dose.seq.nrrd"
        )

    if return_output:
        return cbox_optim, optimized_plan
    
def test_constraint_catheter_number():
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=2,
        return_output=True, run_optimization=True)
    num_non_zero_catheters = 0
    for catheter in optimized_plan.catheter_table:
        if catheter.channel_total_time == 0:
            continue
        num_non_zero_catheters += 1
    if num_non_zero_catheters != cbox_optim.cluster_box.num_physical_catheters:
        raise AssertionError("The constraint on the number of catheters was not respected!")
    print("constarining the number of catheters works!")

def test_constraint_uniqueness():
    cbox_optim, optimized_plan = test_cluster_box_optim(
        return_output=True,
        export_cluster_box=True,
        run_optimization=True,)
    for uniquness in cbox_optim.geometric_constraint_dict["uniqueness"].values():
        num_non_zero_segments = 0
        catheters = optimized_plan.catheter_table.get_catheters_by_ids(
            uniquness.variable_name_ids)

        for cath in catheters:
            channel_time = cath.channel_total_time
            if channel_time == 0:
                continue
            num_non_zero_segments += 1
        if num_non_zero_segments > 1:
            raise AssertionError(f"The uniqueness constraint is not respected for {uniquness.name_id}")
    print("uniqueness constraints passed")

def test_constraint_collision():
    cbox_optim, optimized_plan = test_cluster_box_optim(
        return_output=True,
        export_cluster_box=True,
        run_optimization=True,)
    for collision in cbox_optim.geometric_constraint_dict["collision"].values():
        num_non_zero_segments = 0
        catheters = optimized_plan.catheter_table.get_catheters_by_ids(
            collision.variable_name_ids)

        for cath in catheters:
            channel_time = cath.channel_total_time
            if channel_time == 0:
                continue
            num_non_zero_segments += 1
        if num_non_zero_segments > 1:
            raise AssertionError(f"The collision constraint is not respected for {collision.name_id}")
    print("collision constraints passed")

def test_constraint_continuity():
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=4,
        return_output=True,
        export_cluster_box=True,
        run_optimization=True,)
    for continuity in cbox_optim.geometric_constraint_dict["continuity"].values():
        num_non_zero_segments = 0
        candidate_catheters = optimized_plan.catheter_table.get_catheters_by_ids(
            continuity.variable_name_ids
        )
        for cath in candidate_catheters:
            channel_time = cath.channel_total_time
            if channel_time == 0:
                continue
            num_non_zero_segments +=1

        if num_non_zero_segments == 0:
            continue
        elif num_non_zero_segments > 1:
            raise AssertionError(f"The uniqueness constraint has failed for cluster {continuity.name_id}")
        elif num_non_zero_segments == 1:
            num_non_zero_parents = 0            
            parent_catheters = optimized_plan.catheter_table.get_catheters_by_ids(
                continuity.parent_catheter_name_ids
            )
            for cath in parent_catheters:
                channel_time = cath.channel_total_time
                if channel_time == 0:
                    continue
                num_non_zero_parents += 1
            if num_non_zero_parents != len(parent_catheters):
                raise AssertionError(f"The continuity constraint is not respected for {continuity.name_id}")
        else:
            raise ValueError("SOMETHING IS VERY WRONG")
    print("continuity constraints passed")

def test_get_voxel_dose_rate_goal():
    cbox_optim, _ = test_cluster_box_optim(
        num_decision_planes=2,
        return_output=True,
        export_cluster_box=False,
        run_optimization=False,)
    plan = cbox_optim.plan
    dwell_reaches = range(5, 30, 5)
    dwell_reaches = [1]+list(dwell_reaches)
    for reach in dwell_reaches:
        voxel_dose_rate_goal = get_voxel_dose_rate_goal(
            dwell_reach_mm=reach,
            dose_voxel_goal=1,
            dwell_position=plan.catheter_table.all_dwells[0]
        )
        print(voxel_dose_rate_goal)

def test_get_optimized_plan_from_model():
    cbox_optim, _ = test_cluster_box_optim(
        num_decision_planes=2,
        return_output=True,
        export_cluster_box=True,
        run_optimization=False,
        solve_strategy="cascaded",
        dwell_reach_mm=5)
    t0_sim = time()
    # optimized_plan = cbox_optim.get_optimized_plan_from_model(
    #     solve_strategy='simultaneous')
    t1_sim = time()
    
    t0_cas = time()
    optimized_plan = cbox_optim.get_optimized_plan_from_model()
    t1_cas = time()
    print("---------------------------")    
    print("time for simultaenous was:")
    print(t1_sim - t0_sim)
    print("time for cascaded was:")
    print(t1_cas - t0_cas)
    
if __name__ == "__main__":
    print("Testing cluster box optimization")
    # test_get_geometric_constraints()
    # test_cluster_box_optim()
    # test_constraint_catheter_number()
    # test_constraint_uniqueness()
    # test_constraint_collision()
    # test_constraint_continuity()
    # test_get_voxel_dose_rate_goal()
    test_get_optimized_plan_from_model()