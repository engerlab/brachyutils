from pathlib import Path

from brachyutils.geometry_utils import BrachyPhantom

def fix_one_image(
    pth_image: Path | str,
    pth_output: Path | str
    ):
    r"""
    Purpose:
        - load the phantom, swap the first and last axis and take a look
    """
    pth_image = Path(pth_image)
    if pth_image.exists() is False:
        raise FileNotFoundError(f"File {pth_image} does not exist.")

    phantom_obj = BrachyPhantom(pth_phantom_file=pth_image)

    # swap the first and last axis
    img_array = phantom_obj.get_image_array()
    img_array = img_array.swapaxes(0, 2)
    img_array = img_array.swapaxes(1, 2)
    phantom_obj.set_image_array(img_array)

    phantom_obj.write_image_to_nrrd(pth_output)

if __name__ == "__main__":
    pth_sample_image = Path("../data_test/registration/prostate_mr_us/train_mr_image_case000000.nii.gz")
    pth_out = Path("../data_test/test_export_plan/prostate/corrected_mr_image.nrrd")
    fix_one_image(pth_sample_image, pth_out)