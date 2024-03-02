import os

import numpy as np
from egsphant_utils import BrachyEgsphant, _to_single_string


def test_crop_by_body_contour():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    print("pth_dicomRS: ".format(pth_dicomRS))

    pth_input = (
        "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/ct.egsphant"
    )
    # pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()

    egsphant_obj.crop_by_body_contour(pth_dir_dicom=pth_dicomRS)
    egsphant_obj.info()


def test_crop_by_index():
    pth_input = (
        "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/ct.egsphant"
    )
    pth_output = os.path.dirname(pth_input) + "/test_" + os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()

    index = np.array([[30, 90], [30, 90], [0, 94]], dtype=np.float32)

    egsphant_obj.crop_by_index(index)
    egsphant_obj.info()
    egsphant_obj.write_to_ctegsphant(pth_output)


def test_write_to_egsphant():
    pth_input = (
        "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/ct.egsphant"
    )
    pth_output = os.path.dirname(pth_input) + "/test_" + os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

    egsphant_obj.write_to_ctegsphant(pth_output)
    new_egsphant_obj = BrachyEgsphant()
    new_egsphant_obj.load_from_ctegsphant(pth_output)

    egsphant_obj.is_equal(new_egsphant_obj)


def test_to_single_string():
    pth_input = (
        "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/ct.egsphant"
    )
    # pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

    _to_single_string(egsphant_obj.material_matrix.astype(str))


def test_load_from_ctegsphant():
    pth_input = (
        "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/ct.egsphant"
    )

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()
