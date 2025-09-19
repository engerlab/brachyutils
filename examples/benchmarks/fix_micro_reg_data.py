from pathlib import Path
import numpy as np
from glob import glob
from brachyutils import BrachyPhantom
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from queue import Queue
import os

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
    # swap the first and last axis for structures
    structure_set = phantom_obj.get_structure_mask(
        phantom_obj.structure_names,
        np.ndarray
        )
    final_structure_set = {}
    for struc in structure_set:
        struc_array = structure_set[struc]
        struc_array = struc_array.swapaxes(0, 2)
        struc_array = struc_array.swapaxes(1, 2)
        struc_array = np.flip(struc_array, axis=0)
        final_structure_set[struc] = struc_array

    phantom_obj.set_image_array(img_array)
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
            segment_number = int(name.split("_")[0].replace("Segment", ""))
            new_name_mapping[name] = f"Biopsy_{segment_number-1}"

    # sort the new name mapping by the numerical order of the biopsy number
    new_name_mapping = dict(
        sorted(
            new_name_mapping.items(),
            key=lambda item: int(item[1].split("_")[1]) if "Biopsy" in item[1] else -1)
        )

    phantom_obj.rename_structures(
        new_name_mapping
    )
    phantom_obj.sort_structures_by_name(
        sorted_names=list(new_name_mapping.values())
    )

def fix_one_phantom(
    pth_mr_image: Path | str,
    pth_us_image: Path | str,
    dir_output: Path | str
    ):
    r"""
    Purpose:
        - load the phantom image and structure, remove multiple body contours, rename the structures 
        and correct the axis. Then, save the phantom as nrrd file.
        The files are unique to the micro-reg challenge where the ijktoLPS transform was not written correctly.
    Inputs:
        - pth_image: path to the image file ending with .nrrd
        - pth_structure: path to the structure file ending with .seg.nrrd
        - dir_output: path to the output directory
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
    
    fix_axis(mr_phantom)
    fix_axis(us_phantom)

    mr_phantom.export_to(
        dir_nrrd_out=dir_output,)
    us_phantom.export_to(
        dir_nrrd_out=dir_output,)

def test_fix_one_phantom():
    dir_micro_reg_challenge = Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train")
    pth_mr_image = dir_micro_reg_challenge / "original-nrrd/mr_case000008.nrrd"
    pth_us_image = dir_micro_reg_challenge / "original-nrrd/us_case000008.nrrd"
    dir_out = Path("data_test/test_export_plan/prostate")

    fix_one_phantom(
        pth_mr_image=pth_mr_image,
        pth_us_image=pth_us_image,
        dir_output=dir_out
        )

def fix_all_prostate_images(dir_img_in:Path | str, dir_out, multi_thread: bool = False):
    
    if isinstance(dir_img_in, str):
        dir_img_in = Path(dir_img_in)
    mr_data_list = dir_img_in.glob("mr*.nrrd")
    mr_data_list = [pth for pth in mr_data_list if not ".seg.nrrd" in str(pth)]
    us_data_list = dir_img_in.glob("us*.nrrd")
    us_data_list = [pth for pth in us_data_list if not ".seg.nrrd" in str(pth)]
    partial_fix_phantom = partial(fix_one_phantom, dir_output=dir_out)
    if multi_thread:
        # Use a thread-safe list to store failed files
        failed_files = Queue()
        
        # Function to process files with error handling
        def process_with_error_handling(mr_path, us_path):
            try:
                partial_fix_phantom(mr_path, us_path)
            except Exception as e:
                failed_files.put((mr_path, us_path, str(e)))
                return False
            return True

        # Use CPU count to determine optimal number of workers
        max_workers = os.cpu_count() or 4  # Default to 4 if cpu_count returns None

        # Process files with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_with_error_handling, mr_data_list, us_data_list),
                total=len(mr_data_list),
                desc="Fixing MR and US images and Segmentations"
            ))

        # Collect and display failures
        failed_list = []
        while not failed_files.empty():
            failed_list.append(failed_files.get())
        
        if failed_list:
            print(f"\nFailed to process {len(failed_list)} files:")
            for mr_path, us_path, error in failed_list:
                print(f"  - MR: {mr_path.name}, US: {us_path.name}")
                # print(f"    Error: {error}")
    else:
        for mr_data, us_data in zip(mr_data_list, us_data_list):
            fix_one_phantom(mr_data, us_data, dir_out)
            # for testing
            break

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
    dir_mr_out = dir_micro_reg_challenge / "original-nrrd"
    
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
        mr_phantom.image_obj.name = "mr_"+pth_mr_image.name.split(".")[0]
        mr_phantom.pth_image = dir_mr_out / mr_phantom.image_obj.name
        us_phantom = BrachyPhantom(
            pth_phantom_file=pth_us_image,
            pth_structures_file=pth_us_structure
        )

        us_phantom.image_obj.name = "us_"+pth_us_image.name.split(".")[0]
        us_phantom.pth_image = dir_mr_out / us_phantom.image_obj.name

        mr_phantom.export_to(dir_nrrd_out=dir_mr_out)
        us_phantom.export_to(dir_nrrd_out=dir_mr_out)

    # Use multiprocessing to process all cases in parallel
    num_cases = len(data_paths["mr_images"])
    indices = list(range(num_cases))
    
    # Create the output directory if it doesn't exist
    dir_mr_out.mkdir(parents=True, exist_ok=True)
        # Process all cases sequentially
    for i in tqdm(indices, desc="Processing cases"):
        process_case(i)
        # break

def test_convert_one_phantom():
    dir_micro_reg_challenge = Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train")
    pth_us_image = dir_micro_reg_challenge / "us_images" / "case000025.nii.gz"
    pth_us_seg = dir_micro_reg_challenge / "us_labels" / "case000025.nii.gz"
    dir_out = Path("data_test/test_export_plan/prostate")
    us_phantom = BrachyPhantom(
        pth_phantom_file=pth_us_image,
        pth_structures_file=pth_us_seg
    )
    us_phantom.export_to(dir_nrrd_out=dir_out)
    us_phantom.write_structures_to_dicom(dir_output=dir_out)

if __name__ == "__main__":
    # test_convert_one_phantom()
    # test_fix_one_phantom()

    convert_microreg_to_nrrd()
    fix_all_prostate_images(
        dir_img_in=Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train/original-nrrd"),
        dir_out=Path("/home/ubuntu/YourLocalHome/Data/registration/micro-reg-prostate_us_mri/train/fixed-nrrd"),
        multi_thread=True
    )