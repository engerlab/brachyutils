from brachyutils.geometry_utils import BrachyPhantom
from brachyutils.dose_utils import BrachyDose

from pathlib import Path

pth_dicom = "../data_test/prostate-glen-p1-dcm"
pth_dose_nrrd = "../data_test/test_export_plan/dose_image.seq.nrrd"

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
