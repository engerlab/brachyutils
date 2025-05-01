from brachyutils.planning.optim_utils import DwellTimeVariable, DwellTimeOptimizer, Model
from brachyutils.planning.plan_utils import load_dicom_to_plan


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
    optim_obj = DwellTimeOptimizer(plan=dicom_plan, roi_margin_mm=[2, 2, 2])
    print(optim_obj.roi_bounds)
    print("breakpoint")

def test_set_penalty_function():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_dir_dose_rate = ?
    plan_obj = load_dicom_to_plan(
        dir_dicom=pth_dicom,
        load_dicom_dose=False,
        dir_dose_rate=pth_dir_dose_rate,
        multi_processing=True
        )
    
    optim_obj = DwellTimeOptimizer(plan=plan_obj)

if __name__ == "__main__":
    # test_DwellTimeVariable()
    test_get_optimization_roi_bounds()