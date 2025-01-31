from pathlib import Path
import numpy as np
from brachyutils.geometry_utils import BrachyPhantom

def fix_one_image_structure(
    pth_image: Path | str,
    pth_structure: Path | str,
    pth_output: Path | str
    ):
    r"""
    Purpose:
        - load the phantom image, swap the first and last axis and take a look
    """
    pth_image = Path(pth_image)
    if pth_image.exists() is False:
        raise FileNotFoundError(f"File {pth_image} does not exist.")

    phantom_obj = BrachyPhantom(
        pth_phantom_file=pth_image,
        pth_structures_file=pth_structure
        )

    # swap the first and last axis for images
    img_array = phantom_obj.get_image_array()
    img_array = img_array.swapaxes(0, 2)
    img_array = img_array.swapaxes(1, 2)
    phantom_obj.set_image_array(img_array)

    # swap the first and last axis for structures
    structure_set = phantom_obj.get_structure_mask(
        phantom_obj.structure_names,
        np.ndarray
        )
    for struc in structure_set:
        struc_array = structure_set[struc]
        struc_array = struc_array.swapaxes(0, 2)
        struc_array = struc_array.swapaxes(1, 2)
        structure_set[struc] = struc_array
    phantom_obj.set_structure_set(structure_set)

    phantom_obj.write_image_to_nrrd(pth_output)
    phantom_obj.write_structures_to_nrrd(pth_output.parent.joinpath("corrected_mr_label.nrrd"))

if __name__ == "__main__":
    pth_sample_image = Path("../data_test/registration/prostate_mr_us/train_mr_image_case000000.nii.gz")
    pth_out = Path("../data_test/test_export_plan/prostate/corrected_mr_image.nrrd")
    pth_sample_structure = Path("../data_test/registration/prostate_mr_us/train_mr_label_case000000.nii.gz")

    fix_one_image_structure(pth_sample_image, pth_sample_structure, pth_out)
