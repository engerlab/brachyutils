from pathlib import Path
import numpy as np
from glob import glob
from brachyutils.geometry_utils import BrachyPhantom
import asyncio
from concurrent.futures import ThreadPoolExecutor

def fix_one_image_structure(
    pth_image: Path | str,
    pth_structure: Path | str,
    pth_output: Path | str
    ):
    r"""
    Purpose:
        - load the phantom image and structure, swap the first and last axis save it as nrrd file.
        The files are unique to the micro-reg challenge where the ijktoLPS transform was not written correctly.
    Inputs:
        - pth_image: path to the image file
        - pth_structure: path to the structure file
        - pth_output: path to the output file
    """
    pth_image = Path(pth_image)
    pth_structure = Path(pth_structure)
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
    img_array = np.flip(img_array, axis=0)
    phantom_obj.set_image_array(img_array)

    # swap the first and last axis for structures
    structure_set = phantom_obj.get_structure_mask(
        phantom_obj.structure_names,
        np.ndarray
        )
    final_structure_set = {}
    for struc in structure_set:
        struc_array = structure_set[struc]
        # if struc_array is None:
        #     continue
        struc_array = struc_array.swapaxes(0, 2)
        struc_array = struc_array.swapaxes(1, 2)
        if "mr_label" in str(pth_structure):
            # pass
            struc_array = np.flip(struc_array, axis=0)
            # struc_array = np.flip(struc_array, axis=1)
            # struc_array = np.flip(struc_array, axis=2)
        final_structure_set[struc] = struc_array
    phantom_obj.set_structure_set(final_structure_set)
    phantom_obj.rename_structures(
        {
            "Segment1_Name": "Prostate",
        }
        )

    phantom_obj.export_to(dir_nrrd_out=pth_output)

def test_fix_one_image_structure():
    pth_sample_image = Path("../data_test/registration/prostate_mr_us/train_mr_image_case000000.nii.gz")
    pth_out = Path("../data_test/test_export_plan/prostate/corrected_mr_image.nrrd")
    pth_sample_structure = Path("../data_test/registration/prostate_mr_us/train_mr_label_case000000.nii.gz")

    fix_one_image_structure(pth_sample_image, pth_sample_structure, pth_out)

def fix_all_prostate_images(dir_img, dir_structure, dir_out, multi_thread: bool = False):
    dir_img = Path(dir_img)
    dir_structure = Path(dir_structure)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)
    
    all_imgs = glob(str(dir_img.joinpath("*.nii.gz")))
    all_structures = glob(str(dir_structure.joinpath("*.nii.gz")))
    if multi_thread:
        async def run_in_executor(executor, img, structure, dir_out):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, fix_one_image_structure, img, structure, dir_out)

        async def main():
            with ThreadPoolExecutor() as executor:
                tasks = []
                for img in all_imgs:
                    img_name = Path(img).name
                    structure = [s for s in all_structures if img_name in s]
                    if len(structure) == 0:
                        raise FileNotFoundError(f"No structure file found for {img_name}")
                    structure = structure[0]
                    tasks.append(run_in_executor(executor, img, structure, dir_out))
                await asyncio.gather(*tasks)

        asyncio.run(main())
    else:
        for img in all_imgs:
            img_name = Path(img).name
            structure = [s for s in all_structures if img_name in s]
            if len(structure) == 0:
                raise FileNotFoundError(f"No structure file found for {img_name}")
            structure = structure[0]
            fix_one_image_structure(img, structure, dir_out)
            return

def test_read_nrrd():
    dir_nrrd = Path("../data_test/test_export_plan/prostate")
    all_nrrd = glob(str(dir_nrrd.joinpath("*.nrrd")))
    for pth in all_nrrd:
        if pth.endswith(".seg.nrrd"):
            structure_nrrd = Path(pth)
        else:
            img_nrrd = Path(pth)
    phantom_obj = BrachyPhantom(
        pth_phantom_file=img_nrrd,
        pth_structures_file=structure_nrrd
    )
    phantom_obj.info()

if __name__ == "__main__":
    test_fix_one_image_structure()
    # # fix the mr images and structures for the prostate
    # dir_img = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/mr_images")
    # dir_structure = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/mr_labels")
    # dir_out = Path("../temp_data/registration/micro-reg/mr-train")
    # fix_all_prostate_images(dir_img, dir_structure, dir_out, False)

    # # fix the ultrasound images and structures for the prostate
    # dir_img = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/us_images")
    # dir_structure = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/us_labels")
    # dir_out = Path("../temp_data/registration/micro-reg/us-train")
    # fix_all_prostate_images(dir_img, dir_structure, dir_out, True)