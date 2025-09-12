import logging
import os
import sys

import numpy as np

from brachyutils.dose.dose_utils import BrachyDose
from brachyutils.dose.dose_comparison_utils import BrachyDoseComparison
from pathlib import Path
from brachyutils.geometry.phantom_utils import BrachyPhantom

def make_dose_from_image():
    pth_dicom = Path("data_test/prostate-glen-p1-dcm")
    pth_dose_nrrd = Path("data_test/test_export_plan/dose_image.seq.nrrd")

    image = BrachyPhantom(
        dir_dicom=pth_dicom,
    )
    image.info()
    dose = BrachyDose()

    dose.dose_image = image.image_obj
    dose.uncertainty_image = image.image_obj
    dose.get_voxel_edges()
    dose.info()
    dose.write_to_nrrd(pth_dose_nrrd)

def test_load_from_3ddose():
    pth_file = "data_test/rectal-jgh-planFiles/combined.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_from_3ddose(pth_file)
    dose_obj.is_not_empty()
    dose_obj.info()


def test_load_file_to_brachydose():
    pth_3ddose = "data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.is_not_empty()


def test_load_from_dicom():
    pth_dicom = "data_test/prostate-glen-p1-dcm/RD1.3.6.1.4.1.2452.6.350102904.1117384417.1751574951.1257637737.dcm"
    dose_obj = BrachyDose(pth_dicom)
    dose_obj.info()
    dose_obj.is_not_empty()


def test_write_to_3ddose():
    # pth_file = "data_test/prostate-glen-p1-dose/combined.seq.nrrd"
    pth_file ="data_test/test_export_plan/prostate/combined.3ddose" 
    dir_out = "data_test/test_export_plan/prostate"
    
    pth_file = Path(pth_file)
    dir_out = Path(dir_out)
    dose_obj = BrachyDose(pth_file)

    # dose_obj.write_to_3ddose(dir_out / (pth_file.name.replace(".seq.nrrd",".3ddose")))
    dose_obj.write_to_3ddose(dir_out / ("test_"+pth_file.name))
    new_dose_obj = BrachyDose(
        # dir_out / (pth_file.name.replace(".seq.nrrd",".3ddose"))
        dir_out / ("test_"+pth_file.name)
    )
    print(dose_obj.is_equal(new_dose_obj))


def test_load_from_nrrd():
    pth_input = "data_test/prostate-glen-p1-dose/scaled_run_1.seq.nrrd"
    # pth_input = "data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    from time import time
    start_time = time()
    dose_obj = BrachyDose(pth_input)
    end_time = time()
    print(f"Loading took {end_time - start_time} seconds")
    
    dose_obj.info()


def test_write_to_nrrd():
    r"""
    Purpose:
        simulatenously test write_to_nrrd() and load_from_nrrd()
    """
    pth_out = Path("data_test/test_export_plan/prostate/test_combined.seq.nrrd")
    pth_input = Path("data_test/test_export_plan/prostate/combined.seq.nrrd")

    dose_obj = BrachyDose(pth_input)
    dose_obj.write_to_nrrd(pth_out)
    new_dose_obj = BrachyDose(pth_out)
    dose_obj.is_equal(new_dose_obj)


def test_convert_to_npz_file():
    r"""
    Purpose:
        simulatenously test write_to_npz() and load_from_npz()
    """
    # pth_3ddose =  "data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".npz"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_npz(pth_out)

    new_dose_obj = BrachyDose()
    new_dose_obj.load_from_npz(pth_out)
    dose_obj.is_equal(new_dose_obj)


def test_write_to_xz():

    # pth_3ddose =  "data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".xz"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_xz(pth_out)


def test_write_to_zstd():

    # pth_3ddose =  "data_test/combined.3ddose"
    # pth_zstd = "data_test/combined.zst"

    # testing on maude's file
    pth_3ddose = "data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0] + ".zst"
    print(pth_out)
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_zstd(pth_out)


def test_crop_by_coordinates():
    pth_input = "data_test/rectal-jgh-planFiles/combined.3ddose"
    # pth_input = "data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    coords = np.array([[-150, 150], [-300, -100], [50, 200]], dtype=np.float32)

    dose_obj.crop_by_coordinates(coords)
    dose_obj.info()


def test_crop_by_index():
    pth_input = "data_test/rectal-jgh-planFiles/combined.3ddose"
    # pth_input = "data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    index = np.array([[0, 9], [3, 9], [3, 9]], dtype=np.float32)

    dose_obj.crop_by_index(index)

    dose_obj.info()


def test_crop_by_fraction():
    pth_input = "data_test/rectal-jgh-planFiles/combined.3ddose"
    # pth_input = "data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_obj = BrachyDose(pth_input)
    dose_obj.info()

    fraction = [0.3, 0.5, 1]

    dose_obj.crop_by_fraction(fraction)
    dose_obj.info()


def test_convert_to_minidos():
    pth_input = "data_test/dwell1_1mm.nrrd"
    pth_minidos = os.path.splitext(pth_input)[0] + ".minidos"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_input)
    dose_obj.write_to_minidos(pth_minidos)


def test_dose_comparison():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    pth_3ddose = "data_test/run_1_old.3ddose"
    pth_3ddose2 = "data_test/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj2 = BrachyDose()
    dose_obj2.load_file_to_brachydose(pth_3ddose2)
    dose_comparison = BrachyDoseComparison(dose_obj, dose_obj2, 1, 1)
    # evaluate that the grid contains only 0
    assert not np.any(dose_comparison.percent_difference.grid)
    # dose_comparison.compare_dose_distributions_2D(
    #    dose_obj.voxel_edges[2], dose_obj.voxel_edges[1], dose_obj.voxel_edges[0][0], 'z')


def test_crop_by_dicom_structure():
    pth_dicomRS = "data_test/rectal-jgh-dcm/"
    pth_input = "data_test/rectal-jgh-planFiles/combined.3ddose"
    # dir_out = "data_test/test_export_plan"

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
    test_write_to_3ddose()
    # test_write_to_nrrd()
    # test_crop_by_coordinates()
    # test_crop_by_fraction()
    # test_crop_by_index()
    # test_crop_by_dicom_structure()
