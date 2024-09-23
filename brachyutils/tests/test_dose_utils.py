import logging
import os
import sys

import numpy as np

from brachyutils.dose_utils import BrachyDose, DoseComparison


def test_load_from_3ddose():
    pth_file = "../data_test/rectal-jgh-planFiles/combined.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_from_3ddose(pth_file)
    dose_obj.is_not_empty()
    dose_obj.info()


def test_load_file_to_brachydose():
    pth_3ddose = "../data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.is_not_empty()


def test_load_from_dicom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm/RD1.3.6.1.4.1.2452.6.350102904.1117384417.1751574951.1257637737.dcm"
    dose_obj = BrachyDose(pth_dicom)
    dose_obj.info()
    dose_obj.is_not_empty()


def test_write_to_3ddose():
    # pth_3ddose =  "../../data_test/run_1_old.3ddose"

    # testing on maude's file
    pth_file = "../data_test/rectal-jgh-planFiles/combined.3ddose"
    dir_out = "../data_test/test_export_plan"

    dose_obj = BrachyDose(pth_file)

    dose_obj.write_to_3ddose(os.path.join(dir_out, "test" + os.path.basename(pth_file)))
    new_dose_obj = BrachyDose(
        os.path.join(dir_out, "test" + os.path.basename(pth_file))
    )
    print(dose_obj.is_equal(new_dose_obj))


def test_load_from_nrrd():
    pth_input = "../data_test/new_nrrd/PreOptimization/run_1_1_0.nrrd"
    # pth_input = "../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"

    dose_obj = BrachyDose(pth_input)
    dose_obj.info()


def test_write_to_nrrd():
    r"""
    Purpose:
        simulatenously test write_to_nrrd() and load_from_nrrd()
    """
    pth_out = "../data_test/test_export_plan"
    pth_input = "../data_test/rectal-jgh-planFiles/combined.3ddose"
    pth_out = os.path.join(pth_out, os.path.splitext(os.path.basename(pth_input))[0] + ".nrrd")
    dose_obj = BrachyDose(pth_input)
    dose_obj.write_to_nrrd(pth_out)
    dose_obj_from_nrrd = BrachyDose(pth_out)
    print(dose_obj.is_equal(dose_obj_from_nrrd))


def test_convert_to_npz_file():
    r"""
    Purpose:
        simulatenously test write_to_npz() and load_from_npz()
    """
    # pth_3ddose =  "../../data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".npz"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_npz(pth_out)

    new_dose_obj = BrachyDose()
    new_dose_obj.load_from_npz(pth_out)
    dose_obj.is_equal(new_dose_obj)


def test_write_to_xz():

    # pth_3ddose =  "../../data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".xz"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_xz(pth_out)


def test_write_to_zstd():

    # pth_3ddose =  "../../data_test/combined.3ddose"
    # pth_zstd = "../../data_test/combined.zst"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".zst"
    print(pth_out)
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_zstd(pth_out)


def test_crop_by_coordinates():
    pth_input = "../data_test/batch_uncertainty/combined.3ddose"
    # pth_input = "../../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    coords = np.array([[2, 7], [2, 7], [2, 7]], dtype=np.float32)

    dose_obj.crop_by_coordinates(coords)
    dose_obj.info()


def test_crop_by_index():
    pth_input = "../data_test/batch_uncertainty/combined.3ddose"
    # pth_input = "../../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    index = np.array([[0, 9], [3, 9], [3, 9]], dtype=np.float32)

    dose_obj.crop_by_index(index)

    dose_obj.info()


def test_crop_by_fraction():
    pth_input = "../data_test/batch_uncertainty/combined.3ddose"
    # pth_input = "../../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    fraction = [0.3, 0.5, 1]

    dose_obj.crop_by_fraction(fraction)
    dose_obj.info()


def test_convert_to_minidos():
    pth_input = "../../data_test/dwell1_1mm.nrrd"
    pth_minidos = os.path.splitext(pth_input)[0] + ".minidos"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_input)
    dose_obj.write_to_minidos(pth_minidos)


def test_dose_comparison():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    pth_3ddose = "../../data_test/run_1_old.3ddose"
    pth_3ddose2 = "../../data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj2 = BrachyDose()
    dose_obj2.load_file_to_brachydose(pth_3ddose2)
    dose_comparison = DoseComparison(dose_obj, dose_obj2, 1, 1)
    # evaluate that the grid contains only 0
    assert not np.any(dose_comparison.percent_difference.grid)
    # dose_comparison.compare_dose_distributions_2D(
    #    dose_obj.voxel_edges[2], dose_obj.voxel_edges[1], dose_obj.voxel_edges[0][0], 'z')


def test_crop_by_dicom_structure():
    pth_dicomRS = "../data_test/rectal-jgh-dcm/"
    pth_input = "../data_test/rectal-jgh-planFiles/combined.3ddose"
    # dir_out = "../data_test/test_export_plan"

    dose_obj = BrachyDose(pth_input)
    dose_obj.info()
    cropped_dose = dose_obj.crop_by_dicom_structure(
        pth_dir_dicom=pth_dicomRS, structure_name="body", inplace=False
    )
    cropped_dose.info()


if __name__ == "__main__":
    # test_load_from_3ddose()
    # test_load_from_dicom()
    # test_load_from_nrrd()
    # test_write_to_3ddose()
    test_write_to_nrrd()
    # test_crop_by_coordinates()
    # test_crop_by_fraction()
    # test_crop_by_index()
    # test_crop_by_dicom_structure()
