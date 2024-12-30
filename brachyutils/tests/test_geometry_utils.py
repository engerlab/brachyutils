from glob import glob
from pathlib import Path

import numpy as np

from brachyutils import BrachyPhantom
from brachyutils.geometry_utils import BrachyApplicator

def test_brachy_phantom():
    # pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_nrrd = "../data_test/prostate_glen_p1_ct.nrrd"
    # pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        # dir_dicom=pth_dicom,
        pth_phantom_file=pth_nrrd,
        # pth_structures_file=pth_structure
    )
    phantom_obj.info()


def test_get_structure_mask():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    print(phantom_obj.get_structure_mask(["ctv"], mask_type=np.ndarray))


def test_write_image_to_dicom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_nrrd = "../data_test/prostate_glen_p1_ct.nrrd"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan/test_p1_ct"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        # pth_phantom_file=pth_nrrd,
        pth_structures_file=pth_structure
    )
    phantom_obj.write_image_to_dicom(pth_out)

    new_phantom = BrachyPhantom(
        dir_dicom=pth_out,
    )
    new_phantom.is_equal(phantom_obj)


def test_write_image_to_nrrd():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_out = "../data_test/prostate_glen_p1_ct.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom)
    phantom_obj.write_image_to_nrrd(pth_out)


def test_write_structures_to_nrrd():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "../data_test/prostate_glen_p1_structs.seg.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    phantom_obj.write_structures_to_nrrd(pth_out, True)


def test_write_structures_to_dicom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan/test_p1_dcm"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    phantom_obj.write_image_to_dicom(pth_out)
    phantom_obj.write_structures_to_dicom(pth_out)


def test_read_structures_from_nrrd():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structures = "../data_test/prostate_glen_p1_structs.seg.nrrd"
    pth_out = "../data_test/test_export_plan/test_p1_dcm/rs.seg.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structures)
    # phantom_obj.write_image_to_nrrd(pth_out)
    phantom_obj.write_structures_to_nrrd(pth_out)


def test_write_to_egsphant():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan/test_ct.egsphant"
    # pth_materials = "../data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt"
    # assign_material_from_ct = True
    pth_materials = "../data_test/prostate_material_dict.json"
    assign_material_from_ct = False

    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure,
    )
    phantom_obj.write_to_egsphant(
        pth_output=pth_out,
        material_dict=pth_materials,
        assign_material_from_ct=assign_material_from_ct,
    )


def test_load_egsphant():
    pth_egsphant = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_out = "../data_test/test_export_plan/test_ct.egsphant"

    phantom_obj = BrachyPhantom(pth_egsphant_file=pth_egsphant)
    phantom_obj.write_to_egsphant(
        pth_output=pth_out,
    )


def test_crop_phantom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    crop_coordinates = np.array([[-100, 100], [-150, 200], [-1280, -1140]])
    phantom_obj.crop_by_coordinates(crop_coordinates)
    phantom_obj.write_image_to_nrrd(pth_out + "/test_ct.nrrd")
    phantom_obj.write_structures_to_nrrd(pth_out + "/test_rs.seg.nrrd")


def test_catheter_table():
    from brachyutils.geometry_utils import CatheterTable

    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]

    catheter_table = CatheterTable(pth_catheter_table=pth_plan)
    catheter_table.info()


def test_BrachyApplicator():
    pth_applicator_stl = "../data_test/rectal-jgh-planFiles/applicator_0.stl"
    applicator_obj = BrachyApplicator(pth_applicator_stl)
    applicator_obj.info()


def test_BrachyApplicator_to_mac():
    pth_applicator_stl = "../data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    rotation = np.array([0, 0, 0])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "../data_test/test_export_plan/applicator_0.mac"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        rotation=rotation,
    )
    applicator_obj.to_mac(pth_outfile)


def test_BrachyApplicator_to_stl():
    pth_applicator_stl = "../data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    rotation = np.array([90, 1, 0, 0])
    coordinates = np.array([0, 0, 0])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "../data_test/test_export_plan/applicator_0_tilted.stl"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        rotation=rotation,
        coordinates=coordinates,
    )
    applicator_obj.to_stl(pth_outfile)


def test_BrachyApplicator_set_rotation():
    pth_applicator_stl = "../data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    coordinates = np.array([50, 50, 50])
    rotation = np.array([90, 0, 1, 0])
    rotation_origin = np.array([50, 50, 50])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "../data_test/test_export_plan/applicator_0_tilted.stl"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        coordinates=coordinates,
    )
    applicator_obj.set_rotation(rotation, rotation_origin)
    applicator_obj.to_stl(pth_outfile)

def test_load_nifti_image_file():
    # mri images
    # pth_img_nifti = Path("../data_test/registration_prostate_mr_us/train_mr_image_case000000.nii.gz")
    # pth_img_out = Path("../data_test/test_export_plan/test_mr_image_case000000.nrrd")
    # pth_label_nifti = Path("../data_test/registration_prostate_mr_us/train_mr_label_case000000.nii.gz")
    # pth_label_out = Path("../data_test/test_export_plan/test_mr_label_case000000.nrrd")
    
    # ultrasound images
    pth_img_nifti = Path("../data_test/registration_prostate_mr_us/train_us_image_case000000.nii.gz")
    pth_img_out = Path("../data_test/test_export_plan/test_us_image_case000000.nrrd")
    pth_label_nifti = Path("../data_test/registration_prostate_mr_us/train_us_label_case000000.nii.gz")
    pth_label_out = Path("../data_test/test_export_plan/test_us_label_case000000.nrrd")
    
    phantom_obj = BrachyPhantom(
        pth_phantom_file=pth_img_nifti,
        pth_structures_file=pth_label_nifti
        )
    phantom_obj.info()
    phantom_obj.write_image_to_nrrd(pth_img_out)
    phantom_obj.write_structures_to_nrrd(pth_label_out)

if __name__ == "__main__":
    # print("testing BrachyPhantom")
    # test_brachy_phantom()
    # test_get_structure_mask()
    # test_write_image_to_dicom()
    # test_write_image_to_nrrd()
    # test_write_structures_to_nrrd()
    # test_write_structures_to_dicom()
    # test_read_structures_from_nrrd()
    # test_write_to_egsphant()
    # test_load_egsphant()
    # test_crop_phantom()
    # print("testing CatheterTable")
    # test_catheter_table()
    # print("testing BrachyApplicator")
    # test_BrachyApplicator()
    # test_BrachyApplicator_to_mac()
    # test_BrachyApplicator_to_stl()
    # test_BrachyApplicator_set_rotation()
    test_load_nifti_image_file()