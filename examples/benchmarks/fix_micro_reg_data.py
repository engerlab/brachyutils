from pathlib import Path
import numpy as np
from glob import glob
from brachyutils import BrachyPhantom
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def fix_axis(phantom_obj: BrachyPhantom):
    r"""
    Purpose:
        - fix the axis of the phantom object.
        The files are unique to the micro-reg challenge where the ijktoLPS transform was not written correctly.
    Inputs:
        - phantom_obj: BrachyPhantom object
    """
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
        if "mr_label" in str(phantom_obj.pth_structures):
            # pass
            struc_array = np.flip(struc_array, axis=0)
            # struc_array = np.flip(struc_array, axis=1)
            # struc_array = np.flip(struc_array, axis=2)
        final_structure_set[struc] = struc_array
    phantom_obj.set_structure_set(final_structure_set)

def remove_body_structure(phantom_obj: BrachyPhantom):
    r"""
    Purpose:
        - remove the body structure from the phantom object.
        The files are unique to the micro-reg challenge where the body structure was not written correctly.
    Inputs:
        - phantom_obj: BrachyPhantom object
    """
    structure_set = phantom_obj.get_structure_mask(
        phantom_obj.structure_names,
        np.ndarray
        )
    for structure in structure_set:
        if np.sum(structure_set.get(structure)) / structure_set.get(structure).size > 0.90: 
            print(f"Removing structure: {structure} from phantom object.")
            phantom_obj.remove_structure(structure)

def rename_structures(phantom_obj: BrachyPhantom):
    r"""
    Purpose:
        - rename the structures in the phantom object.
        Segment1 is labelled as Prostate, and the rest are labeled as Biopsy_0, Biopsy_1, etc.
    Inputs:
        - phantom_obj: BrachyPhantom object
    """
    new_name_mapping = {}
    for i, name in enumerate(phantom_obj.structure_names):
        if name == "Segment1_Name":
            new_name_mapping[name] = "Prostate"
        else:
            new_name_mapping[name] = f"Biopsy_{i-1}"

    phantom_obj.rename_structures(
        new_name_mapping
    )

def fix_one_image_structure(
    pth_mr_image: Path | str,
    pth_us_image: Path | str,
    dir_output: Path | str
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
    pth_mr_image = Path(pth_mr_image)
    pth_us_image = Path(pth_us_image)
    dir_output = Path(dir_output)

    pth_mr_seg = pth_mr_image.with_suffix(".seg.nrrd")
    pth_us_seg = pth_us_image.with_suffix(".seg.nrrd")
    mr_phantom = BrachyPhantom(
        pth_phantom_file=pth_mr_image,
        pth_structures_file=pth_mr_seg
    )
    us_phantom = BrachyPhantom(
        pth_phantom_file=pth_us_image,
        pth_structures_file=pth_us_seg
    )
    
    remove_body_structure(mr_phantom)
    remove_body_structure(us_phantom)

    rename_structures(mr_phantom)
    rename_structures(us_phantom)
    
    # fix_axis(mr_phantom)
    # fix_axis(us_phantom)

    mr_phantom.export_to(
        dir_nrrd_out=dir_output,)
    us_phantom.export_to(
        dir_nrrd_out=dir_output,)

def test_fix_one_image_structure():
    dir_micro_reg_challenge = Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train")
    pth_mr_image = dir_micro_reg_challenge / "nrrd-format/mr_case000023.nrrd"
    pth_us_image = dir_micro_reg_challenge / "nrrd-format/us_case000023.nrrd"
    dir_out = Path("data_test/test_export_plan/prostate")

    fix_one_image_structure(
        pth_mr_image=pth_mr_image,
        pth_us_image=pth_us_image,
        dir_output=dir_out
        )

def fix_all_prostate_images(dir_img, dir_structure, dir_out, multi_thread: bool = False):
    pass

def test_read_nrrd():
    dir_nrrd = Path("data_test/test_export_plan/prostate")
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


def convert_microreg_to_nrrd():
    r"""
    ### Purpose: To convert MR and US images and structures to NRRD format and organize
    them for the prostate micro-reg challenge.
    micro-reg challenge organization is as follows:
        - dir_mr_img_in: directory containing MR images in NIfTI format
        - dir_mr_seg_in: directory containing MR segmentation masks in NIfTI format
        - dir_us_img_in: directory containing US images in NIfTI format
        - dir_us_seg_in: directory containing US segmentation masks in NIfTI format
    We want to convert them to the following structure:
        - dir_out: directory containing the output NRRD files with the name format:
            - mr_case#.nrrd for MR images
            - mr_case#.seg.nrrd for MR segmentation masks
            - us_case#.nrrd for US images
            - us_case#.seg.nrrd for US segmentation masks
    """
    dir_micro_reg_challenge = Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train")
    dir_mr_img_in = dir_micro_reg_challenge / "mr_images"
    dir_mr_seg_in = dir_micro_reg_challenge / "mr_labels"
    dir_us_img_in = dir_micro_reg_challenge / "us_images"
    dir_us_seg_in = dir_micro_reg_challenge / "us_labels"
    dir_mr_out = dir_micro_reg_challenge / "nrrd-format"
    
    # get the files in the directories
    data_paths = {"mr_images": {}, "mr_labels": {}, "us_images": {}, "us_labels": {}}
    for dir_in, data_path in zip(
        [dir_mr_img_in, dir_mr_seg_in, dir_us_img_in, dir_us_seg_in],
        data_paths):
        if not dir_in.exists():
            raise FileNotFoundError(f"Directory {dir_in} does not exist.")
        data_paths[data_path] = list(dir_in.glob("*.nii.gz"))

    # load the image and structure files
    # Define a worker function for processing a single case
    def process_case(i):
        # get the image and structure paths
        pth_mr_image = data_paths["mr_images"][i]
        pth_mr_structure = data_paths["mr_labels"][i]
        pth_us_image = data_paths["us_images"][i]
        pth_us_structure = data_paths["us_labels"][i]

        mr_phantom = BrachyPhantom(
            pth_phantom_file=pth_mr_image,
            pth_structures_file=pth_mr_structure
        )
        mr_phantom.image_obj.name = f"mr_case{i:06d}.nrrd"
        mr_phantom.pth_image = dir_mr_out / mr_phantom.image_obj.name
        us_phantom = BrachyPhantom(
            pth_phantom_file=pth_us_image,
            pth_structures_file=pth_us_structure
        )
        
        us_phantom.image_obj.name = f"us_case{i:06d}.nrrd"
        us_phantom.pth_image = dir_mr_out / us_phantom.image_obj.name

        mr_phantom.export_to(dir_nrrd_out=dir_mr_out)
        us_phantom.export_to(dir_nrrd_out=dir_mr_out)

    # Use multiprocessing to process all cases in parallel
    num_cases = len(data_paths["mr_images"])
    indices = list(range(num_cases))
    
    # Create the output directory if it doesn't exist
    dir_mr_out.mkdir(parents=True, exist_ok=True)
    
    # Process all cases using multiprocessing
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_case, indices), total=num_cases, desc="Processing cases"))

if __name__ == "__main__":
    # # fix the mr images and structures for the prostate
    # dir_img = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/mr_images")
    # dir_structure = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/mr_labels")
    # dir_out = Path("temp_data/registration/micro-reg/mr-train")
    # fix_all_prostate_images(dir_img, dir_structure, dir_out, False)

    # # fix the ultrasound images and structures for the prostate
    # dir_img = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/us_images")
    # dir_structure = Path("/root/YourLocalHome/Data/registration/prostate_us_mri/train/us_labels")
    # dir_out = Path("temp_data/registration/micro-reg/us-train")
    # fix_all_prostate_images(dir_img, dir_structure, dir_out, True)

    # convert_microreg_to_nrrd()
    test_fix_one_image_structure()