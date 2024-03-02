from dicom_utils import get_structure_index_range
from dicom_utils import get_strcuture_mask_from_dicom

def test_get_structure_index_range():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    print(get_structure_index_range(pth_dicomRS, ['body', 'urethra', 'rectum', 'ctv']))

def test_get_strcuture_mask_from_dicom():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    get_strcuture_mask_from_dicom(pth_dicomRS, ['urethra', 'rectum', 'ctv'])
