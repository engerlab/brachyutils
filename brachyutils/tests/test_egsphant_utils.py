import os
from time import time

import numpy as np
from checkpointing import checkpoint

from brachyutils.dicom_utils import BrachyDicom
from brachyutils.egsphant_utils import (
    BrachyEgsphant,
    _load_material_dict,
    _to_single_string,
)


def test_crop_by_body_contour():
    pth_dicomRS = "../data_test/prostate-glen-p1-dcm/"
    # print("pth_dicomRS: ".format(pth_dicomRS))

    pth_input = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"
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
    pth_input = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_output = (
        "../data_test/test_export_plan" + "/test_" + os.path.basename(pth_input)
    )

    egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_input)
    # egsphant_obj.load_from_ctegsphant(pth_input)
    # egsphant_obj.assert_BrachyEgsphant_notEmpty()

    egsphant_obj.write_to_ctegsphant(pth_output)
    new_egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_output)
    # new_egsphant_obj.load_from_ctegsphant(pth_output)

    egsphant_obj.is_equal(new_egsphant_obj)


def test_to_single_string():
    pth_input = "../../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

    _to_single_string(egsphant_obj.material_matrix.astype(str))


def test_load_from_ctegsphant():
    pth_input = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"

    egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_input)
    egsphant_obj.info()


def test_create_egsphant_from_images():
    # dir_images = "../data_test/rectal-jgh-dcm"
    dir_images = "../data_test/prostate-glen-p1-dcm"
    # materials from CT
    pth_materials = "../data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt"
    pth_output = "../data_test/test_export_plan/prostate_from_images_ct.egsphant"
    load_structure = False
    assign_material_from_ct = True
    # # materials from contours
    # pth_materials = "../data_test/prostate_material_dict.json"
    # pth_output = "../data_test/test_export_plan/prostate_from_contours.egsphant"
    # load_structure = True
    # assign_material_from_ct = False
    # dicom_obj = BrachyDicom(
    # pth_dir_dicom=dir_images,
    # load_structure=True,
    # )

    start_time = time()
    egsphant_obj = test_egsphant_constructor(
        dir_images,
        pth_materials,
        load_structure,
        assign_material_from_ct,
    )
    # breakpoint()
    egsphant_obj.write_to_ctegsphant(pth_output)
    end_time = time()
    print("Time elapsed: ", end_time - start_time)
    # egsphant_obj.export_material_dict(
    # os.path.join(
    # os.path.dirname(pth_output),
    # "test_materials.json")
    # )


def text_load_material_dict():
    pth_input = "../../data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt"
    materials_dict = _load_material_dict(pth_input)
    print(materials_dict)

def test_crop_by_coordinates():
    pth_input = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_output = (
        "../data_test/test_export_plan" + "/cropped" + os.path.basename(pth_input)
    )
    egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_input)
    egsphant_obj.info()
    crop_coordinates = np.array([[-1248.5, -1227.5], [0, 200], [-100, 100]], dtype=np.float32)
    egsphant_obj.crop_by_coordinates(coordinate_range=crop_coordinates)
    egsphant_obj.info()
    egsphant_obj.write_to_ctegsphant(pth_output)

@checkpoint()
def test_egsphant_constructor(
    dir_images,
    pth_materials,
    load_structure=True,
    assign_material_from_ct=False,
):
    dicom_obj = BrachyDicom(
        pth_dir_dicom=dir_images,
        load_structure=load_structure,
    )
    return BrachyEgsphant(
        dicom_image=dicom_obj,
        material_dict=pth_materials,
        assign_material_from_ct=assign_material_from_ct,
    )


if __name__ == "__main__":
    # test_load_from_ctegsphant()
    # test_write_to_egsphant()
    # test_create_egsphant_from_images()
    # text_load_material_dict()
    test_crop_by_coordinates()
