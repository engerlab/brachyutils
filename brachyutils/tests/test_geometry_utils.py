from brachyutils import BrachyPhantom
from glob import glob
from pathlib import Path
import numpy as np

def test_brachy_phantom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm" 
    pth_nrrd = "../data_test/prostate_glen_p1_ct.nrrd"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        # dir_dicom=pth_dicom,
        pth_phantom_file=pth_nrrd,
        # pth_structures_file=pth_structure
        )
    phantom_obj.info()

def test_get_structure_mask():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    print(phantom_obj.get_structure_mask(['ctv'], mask_type=np.ndarray))

def test_write_image_to_dicom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_nrrd = "../data_test/prostate_glen_p1_ct.nrrd"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan/test_p1_ct"
    phantom_obj = BrachyPhantom(
        # dir_dicom=pth_dicom,
        pth_phantom_file=pth_nrrd,
        # pth_structures_file=pth_structure
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
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    pth_out = "../data_test/prostate_glen_p1_structs.seg.nrrd"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    phantom_obj.write_structures_to_nrrd(pth_out, True)

def test_write_structures_to_dicom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan/test_p1_dcm"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    phantom_obj.write_image_to_dicom(pth_out)
    phantom_obj.write_structures_to_dicom(pth_out)

def test_read_structures_from_nrrd():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structures = "../data_test/prostate_glen_p1_structs.seg.nrrd"
    pth_out = "../data_test/test_export_plan/test_p1_dcm/rs.seg.nrrd"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structures
        )
    # phantom_obj.write_image_to_nrrd(pth_out)
    phantom_obj.write_structures_to_nrrd(pth_out)
    
def test_write_to_egsphant():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
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
        assign_material_from_ct=assign_material_from_ct
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
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    pth_out = "../data_test/test_export_plan"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    crop_coordinates = np.array([[-100, 100], [-150, 200], [-1280, -1140]])
    phantom_obj.crop_by_coordinates(crop_coordinates)
    phantom_obj.write_image_to_nrrd(pth_out+"/test_ct.nrrd")
    phantom_obj.write_structures_to_nrrd(pth_out+"/test_rs.seg.nrrd")

if __name__ == "__main__":
    print("testing BrachyPhantom")
    # test_brachy_phantom()
    # test_get_structure_mask()
    # test_write_image_to_dicom()
    # test_write_image_to_nrrd()
    # test_write_structures_to_nrrd()
    # test_write_structures_to_dicom()
    # test_read_structures_from_nrrd()
    # test_write_to_egsphant()
    # test_load_egsphant()
    test_crop_phantom()
