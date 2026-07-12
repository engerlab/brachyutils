from brachyutils.planning.optimization.optim_cath.cluster_box_optim import get_geometric_constraints
from brachyutils.planning.optimization.optim_configs import Optimization_Config
from brachyutils.tests.test_cluster_box import test_cluster_box
from brachyutils.planning.plan_utils import load_dicom_to_plan
from pathlib import Path
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen, Config_ClusterBox
from brachyutils.planning.optimization.optim_cath.cluster_box_optim import ClusterBoxOptim
    
def test_get_geometric_constraints():
    cbox = test_cluster_box(return_box=True)
    constraint_dict = get_geometric_constraints(cluster_box=cbox)
    print("debug here")

def test_cluster_box_optim():
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

    config_angle = Config_Angled_CathGen(
        x_angle_max=4,
        x_angle_step=4,
        y_angle_max=4,
        y_angle_step=4
    )

    config_cluster_box = Config_ClusterBox(
        num_physical_catheters=12,
        insertion_point_spacing_mm=10,
        num_decision_planes=2,
        config_angle=config_angle,
        box_margin_mm=10,
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

    # TODO priority 1: complete the tester here.
    cbox_optim = ClusterBoxOptim(
        plan=plan,
        config_cluster_box=config_cluster_box
    )
    # # export the cluster 
    cbox_optim.cluster_box.to_ply(
        out_ply_dir=outdir)
    # # export the normalized combined dose rate
    cbox_optim.plan.catheter_table.reset_dwelltimes_to(reset_value=1)
    combined_dose_rate = cbox_optim.plan.combined_dose
    combined_dose_rate.dose_image.imageArray = combined_dose_rate.dose_image.imageArray / target_dose
    combined_dose_rate.write_brachydose_to_file(
        pth_dose_file=outdir/"combined_dose.seq.nrrd"
    )
    # cbox_optim.solve()
    
    print("debug here")

if __name__ == "__main__":
    print("Testing cluster box optimization")
    # test_get_geometric_constraints()
    test_cluster_box_optim()