import os

import numpy as np

from brachyutils.dicom_utils import BrachyDicom
from brachyutils.egsphant_utils import BrachyEgsphant, _to_single_string


def test_crop_by_body_contour():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    # print("pth_dicomRS: ".format(pth_dicomRS))

    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()

    egsphant_obj.crop_by_body_contour(pth_dir_dicom=pth_dicomRS)
    egsphant_obj.info()


def test_crop_by_index():
    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_" + os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()

    index = np.array([[30, 90], [30, 90], [0, 94]], dtype=np.float32)

    egsphant_obj.crop_by_index(index)
    egsphant_obj.info()
    egsphant_obj.write_to_ctegsphant(pth_output)


def test_write_to_egsphant():
    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_" + os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

    egsphant_obj.write_to_ctegsphant(pth_output)
    new_egsphant_obj = BrachyEgsphant()
    new_egsphant_obj.load_from_ctegsphant(pth_output)

    egsphant_obj.is_equal(new_egsphant_obj)


def test_to_single_string():
    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

    _to_single_string(egsphant_obj.material_matrix.astype(str))


def test_load_from_ctegsphant():
    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()


def test_create_egsphant_from_images():
    dir_images = "../../data_test/rectal-jgh-dcm"
    pth_output = "../../data_test/test_export_plan/rectal_from_images_ct.egsphant"
    materials = {
        "Blair": {"density": 0.001225, "HU_limit": -1000.0},
        "Air": {"density": 0.001225, "HU_limit": 1000.0},
        "Adipose": {"density": 0.95, "HU_limit": 1100.0},
        "Water": {"density": 1.0, "HU_limit": 1200.0},
        "SoftTissue": {"density": 1.02, "HU_limit": 1300.0},
        "SoftTissue_Male": {"density": 1.03, "HU_limit": 1400.0},
        "Rectum": {"density": 1.03, "HU_limit": 1500.0},
        "Bladder_Filled": {"density": 1.03, "HU_limit": 1600.0},
        "SiliconRubber": {"density": 1.14, "HU_limit": 1700.0},
        "Bone_Cortical": {"density": 1.92, "HU_limit": 1800.0},
        "Bone": {"density": 3.0, "HU_limit": 1900.0},
        "Metal": {"density": 19.0, "HU_limit": 2000.0},
    }

    dicom_obj = BrachyDicom(
        pth_dir_dicom=dir_images,
        load_structure=True,
    )
    egsphant_obj = BrachyEgsphant(
        image=dicom_obj,
        material_dict=materials,
        assign_material_from_ct=True,
    )
    egsphant_obj.write_to_ctegsphant(pth_output)


if __name__ == "__main__":
    test_create_egsphant_from_images()
