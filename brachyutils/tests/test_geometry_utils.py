from brachyutils import BrachyPhantom
from glob import glob
from pathlib import Path

def test_BrachyPhantom():
    pth_dicom = "../data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom+"/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure
        )
    phantom_obj.info()

if __name__ == "__main__":
    print("testing BrachyPhantom")
    test_BrachyPhantom()