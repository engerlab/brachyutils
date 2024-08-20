# import numpy as np
from brachyutils.dicom_utils import BrachyDicom


def test_load_dicom():
    # CT Prostate image:
    pth_dicom = "../../data_test/prostate-glen-p1-dcm/"
    # MR Prostate image:
    # pth_dicom = "../../data_test/prostate-glen-p5-dcm/"
    # CT rectal image:
    # pth_dicom = "../../data_test/rectal-jgh-dcm"
    # especial patient case
    dicom_obj = BrachyDicom(
        pth_dicom,
        load_image=True,
        load_structure=True,
        load_dose=True,
        load_plan=False,
    )
    dicom_obj.info()


def test_get_strcuture_mask_from_dicom():
    # ct images
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    dicom_obj = BrachyDicom(pth_dicomRS, load_dose=True)
    strcuture_masks = dicom_obj.get_strcuture_mask_from_dicom(
        ["urethra", "rectum", "ctv"]
    )
    # assert np.sum(strcuture_masks) != 0, "structure masks are empty"
    dicom_obj.info()
    print(strcuture_masks)
    print(strcuture_masks.keys())


def test_get_structure_index_range():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    dicom_obj = BrachyDicom(pth_dicomRS)
    structure_index_range = dicom_obj.get_structure_index_range(
        ["urethra", "rectum", "ctv"]
    )
    print(structure_index_range)
    # assert structure_index_range is not None, "structure index range is empty"


if __name__ == "__main__":
    print("running tests")
    test_load_dicom()
    # test_get_strcuture_mask_from_dicom()
    # test_get_structure_index_range()
    # test_get_dvh_metrics_from_dicom_dose()
