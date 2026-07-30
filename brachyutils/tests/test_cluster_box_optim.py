from brachyutils.planning.optimization.optim_cath.cluster_box_optim import get_geometric_constraints
from brachyutils.planning.optimization.optim_configs import Optimization_Config
from brachyutils.tests.test_cluster_box import test_cluster_box
from brachyutils.planning.plan_utils import load_dicom_to_plan
from pathlib import Path
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Catheter_Rotation, Config_ClusterBox
from brachyutils.planning.optimization.optim_cath.cluster_box_optim import (
    ClusterBoxOptim, run_experiment_sequential, disturbe_catheter_table,
    _activate_this_cluster, _count_physical_catheters_used)
from time import time
import random

def test_get_geometric_constraints():
    cbox = test_cluster_box(return_box=True)
    constraint_dict = get_geometric_constraints(cluster_box=cbox)
    print("debug here")

def test_cluster_box_optim(
    num_decision_planes = None,
    num_physical_catheters = None,
    insertion_point_spacing_mm = None,
    config_catheter_rotation = None,
    return_output:bool = False,
    export_cluster_box:bool = False,
    run_optimization:bool = False,):
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

    if config_catheter_rotation is None:
        config_catheter_rotation = Config_Catheter_Rotation(
            x_angle_max=0,
            x_angle_step=0,
            y_angle_max=10,
            y_angle_step=10,
        )
    if num_decision_planes is None:
        num_decision_planes = 2
    if num_physical_catheters is None:
        num_physical_catheters = 5
    if insertion_point_spacing_mm is None:
        insertion_point_spacing_mm = 15
    config_cluster_box = Config_ClusterBox(
        num_physical_catheters=num_physical_catheters,
        insertion_point_spacing_mm=insertion_point_spacing_mm,
        num_decision_planes=num_decision_planes,
        config_catheter_rotation=config_catheter_rotation,
        box_margin_mm=5,
    )
    dvh_metric_goals = {
        "D90%(CTV)": 100.0,
        "D2cc(RECTUM)": 100.0 * 0.75,
        "D10%(URETHRA)": 100.0 * 1.133,
        "D30%(URETHRA)": 100.0,
        "CI(CTV)": 1.0,
        "HI(CTV)": 0.5,
        "V200%(CTV)": 100.0 * 0.2,
        "V150%(CTV)": 100.0 * 0.4,
        "V100%(CTV)": 100.0,
    }

    # # build a plan without catheter table but have optimizatio constraints.
    plan = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_catheter_table=False,
        load_dicom_prescription_dose=False,
        optimization_config_list=optimization_config_list,
        strict_name_match=False,
        prescription_dose=target_dose,
        dvh_metric_goals = dvh_metric_goals,
        )

    cbox_optim = ClusterBoxOptim(
        plan=plan,
        config_cluster_box=config_cluster_box
    )
    optimized_plan = None
    if run_optimization:
        optimized_plan = cbox_optim.get_optimized_plan_from_model()
        optimized_plan.set_dvh_metric_goals(
            dvh_metric_goals=dvh_metric_goals,
            strict_name_match=False
        )
        observed_dvh_metrics = optimized_plan.get_dvh_metrics()
        print("DVH Metrics are:")
        print(observed_dvh_metrics)
        # optimized_plan.catheter_table.combined_dose.write_brachydose_to_file(
        #     outdir/"combined_dose.seq.nrrd"
        # )
    # # export the cluster
    if export_cluster_box:
        cbox_optim.export_to(
            out_dir=outdir,
            dose_normalization_constant=target_dose
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
    config_catheter_rotation = Config_Catheter_Rotation(
        x_angle_max=0,
        x_angle_step=0,
        y_angle_max=0,
        y_angle_step=0,
    )
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=2,
        config_catheter_rotation=config_catheter_rotation,
        export_cluster_box=True,
        run_optimization=False,
        insertion_point_spacing_mm=5,
        return_output=True,
        num_physical_catheters=[10, 14],
    )
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
        export_cluster_box=False,
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

def test_modify_constraint():
    config_catheter_rotation = Config_Catheter_Rotation(
        x_angle_max=0,
        x_angle_step=0,
        y_angle_max=0,
        y_angle_step=0,
    )
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=2,
        num_physical_catheters = 4,
        insertion_point_spacing_mm = 15,
        return_output=True,
        export_cluster_box=False,
        run_optimization=False,
        config_catheter_rotation=config_catheter_rotation)
    
    cbox_optim.cluster_box.num_physical_catheters = cbox_optim.cluster_box.num_physical_catheters + 3
    cbox_optim.geometric_constraint_dict.get(
        "num_catheters").get("num_catheters").maximum = cbox_optim.cluster_box.num_physical_catheters
    cbox_optim.set_geometric_constraints(
        cluster_box=cbox_optim.cluster_box,
        optim_obj=cbox_optim.optimization_object,
        constraint_dict=cbox_optim.geometric_constraint_dict["num_catheters"]
        )

def test_run_experiment_sequential():
    outdir=Path("data_test/test_export_plan/prostate/clusterbox_optim")
    max_num_physical_catheters = 21
    step_num_physical_catheters = 3
    initial_num_physical_catheters = 18
    prob_catheter_deviation = 0
    prepandicular_catheters = True
    config_catheter_rotation = Config_Catheter_Rotation(
        x_angle_max=0,
        x_angle_step=0,
        y_angle_max=4,
        y_angle_step=4,
    )
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=3,
        num_physical_catheters = initial_num_physical_catheters,
        insertion_point_spacing_mm = 10,        
        return_output=True,
        export_cluster_box=False,
        run_optimization=False,
        config_catheter_rotation=config_catheter_rotation,)

    run_experiment_sequential(
        cbox_optim=cbox_optim,
        max_num_physical_catheters = max_num_physical_catheters,
        step_num_physical_catheters = step_num_physical_catheters,
        initial_num_physical_catheters = initial_num_physical_catheters,
        prob_catheter_deviation = prob_catheter_deviation,
        prepandicular_catheters=prepandicular_catheters,
        dir_output=outdir)

def test_disturbe_catheter_table():
    outdir = Path("data_test/test_export_plan/prostate/disturb_catheters")
    initial_num_physical_catheters = 2
    prob_catheter_deviation = 0.6
    config_catheter_rotation = Config_Catheter_Rotation(
        x_angle_max=4,
        x_angle_step=8,
        y_angle_max=4,
        y_angle_step=8,
    )
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=2,
        num_physical_catheters = initial_num_physical_catheters,
        insertion_point_spacing_mm = 10,        
        return_output=True,
        export_cluster_box=True,
        run_optimization=False,
        config_catheter_rotation=config_catheter_rotation,)
    cluster_box = cbox_optim.cluster_box
    catheter_table = cbox_optim.plan.catheter_table

    # randomly activate some of the clusters in this box
    clusters_depth_zero = []
    for cluster in cluster_box.cluster_dict.values():
        if cluster.depth == 0:
            clusters_depth_zero.append(cluster)
    selected_clusters = random.sample(clusters_depth_zero, 5)
    for cluster in selected_clusters:
        _activate_this_cluster(
            cluster.insert_position,
            catheter_table,
            cluster_box)
    assert _count_physical_catheters_used(catheter_table, cluster_box) == 5
    catheter_table.combined_dose.write_brachydose_to_file(
        pth_dose_file=outdir/"before.seq.nrrd"
    )
    catheter_table.write_to_json(
        pth_json=outdir/"before.json"
    )
    diturbed_table = disturbe_catheter_table(
        catheter_table=catheter_table,
        prob_catheter_deviation=prob_catheter_deviation,
        cluster_box=cluster_box
    )
    catheter_table.combined_dose.write_brachydose_to_file(
        pth_dose_file=outdir/"after.seq.nrrd"
    )
    diturbed_table.write_to_json(
        pth_json=outdir/"after.json"
    )

def test_get_physical_catheter_tabel():
    outdir = Path("data_test/test_export_plan/prostate/clusterbox_optim")
    initial_num_physical_catheters = 2
    config_catheter_rotation = Config_Catheter_Rotation(
        x_angle_max=0,
        x_angle_step=0,
        y_angle_max=4,
        y_angle_step=8,
    )
    cbox_optim, optimized_plan = test_cluster_box_optim(
        num_decision_planes=4,
        num_physical_catheters = initial_num_physical_catheters,
        insertion_point_spacing_mm = 10,        
        return_output=True,
        export_cluster_box=True,
        run_optimization=False,
        config_catheter_rotation=config_catheter_rotation,)
    cluster_box = cbox_optim.cluster_box
    catheter_table = cbox_optim.plan.catheter_table

    # randomly activate some of the clusters in this box
    clusters_depth_zero = []
    for cluster in cluster_box.cluster_dict.values():
        if cluster.depth == 0:
            clusters_depth_zero.append(cluster)
    selected_clusters = random.sample(clusters_depth_zero, 5)
    for cluster in selected_clusters:
        _activate_this_cluster(
            cluster.insert_position,
            catheter_table,
            cluster_box)
    # assert _count_physical_catheters_used(catheter_table, cluster_box) == 5
    physical_catheter_table = cbox_optim.get_physical_catheter_table(
        catheter_table=catheter_table)
    physical_catheter_table.write_to_slicer_markup(
        pth_mrk_json=outdir/"physical_catheter_table.mrk.json")

if __name__ == "__main__":
    print("Testing cluster box optimization")
    # test_get_geometric_constraints()
    # config_catheter_rotation = Config_Catheter_Rotation(
    #     x_angle_max=0,
    #     x_angle_step=0,
    #     y_angle_max=0,
    #     y_angle_step=0,
    # )
    # t0 = time()
    # test_cluster_box_optim(
    #     num_decision_planes=2,
    #     config_catheter_rotation=config_catheter_rotation,
    #     export_cluster_box=True,
    #     run_optimization=True,
    #     insertion_point_spacing_mm=5,
    #     return_output=False,
    #     num_physical_catheters=[10, 14],
    # )
    # t1 = time()
    # print("--------")
    # print("time for the entire pipeline")
    # print(t1-t0)
    # test_constraint_catheter_number()
    # test_constraint_uniqueness()
    # test_constraint_collision()
    # test_constraint_continuity()
    # test_modify_constraint()
    test_run_experiment_sequential()
    # test_disturbe_catheter_table()
    # test_get_physical_catheter_tabel()
