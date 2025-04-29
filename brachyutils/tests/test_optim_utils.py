from brachyutils.planning.optim_utils import DwellTimeOptimizer
from brachyutils.planning.plan_utils import load_dicom_to_plan


def test_get_variables_from_plan():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dicom_plan = load_dicom_to_plan(pth_dicom)
    optim_obj = DwellTimeOptimizer(plan=dicom_plan)
    print("breakpoint")

def test_get_optimization_roi_bounds():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    dicom_plan = load_dicom_to_plan(pth_dicom)
    optim_obj = DwellTimeOptimizer(plan=dicom_plan, roi_margin_mm=[2, 2, 2])
    
    print("breakpoint")
if __name__ == "__main__":
    # test_get_variables_from_plan()
    test_get_optimization_roi_bounds()