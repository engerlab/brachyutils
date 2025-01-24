from brachyutils.geometry_utils import BrachyPhantom
from pathlib import Path
from glob import glob
def export_phantom_opentps_nrrd_dicom_egsphant():
    pth_img_dicom = Path("../data_test/prostate-glen-p1-dcm")
    pth_strct_dicom = glob(str(pth_img_dicom)+"/RS*.dcm")[0]
    pth_img_nrrd = Path("../data_test/test_export_plan/opentps/prostate_glen_p1.nrrd")
    pth_strct_nrrd = Path("../data_test/test_export_plan/opentps/prostate_glen_p1.seg.nrrd")
    assign_material_from_ct = True
    pth_materials = Path("../data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt")
    phantom = BrachyPhantom(
        dir_dicom=pth_img_dicom,
        pth_structures_file=pth_strct_dicom
    )
    # phantom.export_to(
    #     dir_nrrd_out=pth_img_nrrd.parent
    # )
    # phantom.export_to(
    #     dir_dicom_out=Path.joinpath(pth_img_nrrd.parent, "dicom/")
    # )
    phantom.write_to_egsphant(
        pth_output=pth_img_nrrd.parent.joinpath("egsphant.seq.nrrd"),
        material_dict=pth_materials,
        assign_material_from_ct=assign_material_from_ct
        )

def register_opentps_on_image_and_contour():
    

if __name__ == "__main__":
    export_phantom_opentps_nrrd_dicom_egsphant()