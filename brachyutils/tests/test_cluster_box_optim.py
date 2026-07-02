from brachyutils.planning.optimization.optim_cath.cluster_box_optim import get_geometric_constraints
from brachyutils.planning.optimization.optim_configs import Optimization_Config
from brachyutils.tests.test_cluster_box import test_cluster_box
from brachyutils.planning.plan_utils import load_dicom_to_plan
from pathlib import Path
    
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
            spacing_mm=2),
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

    # # build a plan without catheter table but have optimizatio constraints.
    plan = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_catheter_table=False,
        load_dicom_prescription_dose=False,
        
        )

if __name__ == "__main__":
    print("Testing cluster box optimization")
    # test_get_geometric_constraints()
    test_cluster_box_optim()