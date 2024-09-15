from brachyutils import BrachyPhantom
from glob import glob
from pathlib import Path

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
    print(phantom_obj.get_structure_mask(['ctv']))

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
    pth_out = "../data_test/prostate_glen_p1_structs.nrrd"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    phantom_obj.write_structures_to_nrrd(pth_out)

if __name__ == "__main__":
    print("testing BrachyPhantom")
    # test_brachy_phantom()
    # test_get_structure_mask()
    # test_write_image_to_dicom()
    # test_write_image_to_nrrd()
    test_write_structures_to_nrrd()