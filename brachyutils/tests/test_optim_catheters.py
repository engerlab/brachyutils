from pathlib import Path
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterVar_Gurobi, CatheterTableOptim_Gurobi)
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from gurobipy import Model
from brachyutils.tests.test_optim_utils import get_a_plan_to_optimize
from brachyutils.planning.optimization.optim_utils import Optimization_Config
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
    cat_table = CatheterTable(catheter_list=list(pth_dicom.glob("RP*.dcm"))[0])
    dir_dose_rates = Path("data_test/prostate-glen-p1-dose").resolve()
    target_dose = 21
    gen_dose_rates = False
    # XXX to try later: Generate dose rates from cropped egsphant, consider all dwell positions
    optimization_config_list=[
        Optimization_Config(
            structure_name="CTV",
            dose_voxel_goal=target_dose,
            penalty_weight_linear=300,
            # penalty_weight_quadratic=1,
            # penalty_weight_uniformity=1,
            penalty_weight_hotspot=1,
            hotspot_threshold=1.5,
            # penalty_weight_variance_time=1,
            mask_margin_mm=0,
            spacing_mm=3,
            catheter_recommendaion=True),
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
    plan = get_a_plan_to_optimize(
        pth_dicom=pth_dicom,
        dir_dose_rates=dir_dose_rates,
        optimization_config_list=optimization_config_list,
        generate_dose_rates=gen_dose_rates,)
    catheter_optim_obj = CatheterTableOptim_Gurobi(
        plan=plan,
    )

    
if __name__ == "__main__":
    test_catheter_gurobi_initialization()