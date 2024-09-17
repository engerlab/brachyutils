import os
import json
import glob
from random import random, seed
from pathlib import Path
import sys
from typing import List

sys.path.append("/home/sebquet/EngerLab/Breast_OAR_Segmentation")

import numpy as np
import SimpleITK as sitk

from breast_oar_seg.catheter.contour_creator import CatheterContourCreator
from breast_oar_seg.preprocessing.cropping import crop_volumes_background
from breast_oar_seg.preprocessing.dicom_to_sitk import conver_patient_to_nii
from breast_oar_seg.preprocessing.integrity_dcm import IntegrityManager
from breast_oar_seg.preprocessing.utils import (
    multilabel_to_one_channel,
    get_label_combinations,
    filter_multilabels,
    edit_label_metadata,
)


class PatientPreprocessor:

    def __init__(
        self,
        patient_folder,
        labels_file,
        name_preproc_folder="processed",
        overwrite=False,
    ):
        self.patient_folder = patient_folder
        self.labels_file = labels_file
        self.overwrite = overwrite
        with open(
            os.path.join(
                Path(__file__).parents[1], "utils", "biggest_boundaries_skin.json"
            ),
            "r",
        ) as j:
            self.boundaries = json.loads(j.read())
        self.patient_id = os.path.basename(patient_folder).split("_")[0]

        self.setup_data(name_preproc_folder)

        self.ref_CT = sitk.ReadImage(self.ct_path)

        self.contour_names = {
            file.split(".")[0]: os.path.join(self.processed_folder, file)
            for file in os.listdir(self.processed_folder)
            if not "CT" in file
        }

        self.get_mapping_dicts()

    def setup_data(self, name_preproc_folder):

        self.fix_anonymization()
        self.processed_folder = os.path.join(self.fixed_anon_path, name_preproc_folder)
        self.ct_path = os.path.join(self.processed_folder, "CT.nrrd")
        self.create_nii_masks()

    def fix_anonymization(self):

        fixed_anon_path = self.patient_folder + "_temp_processing"
        integrity_manager = IntegrityManager(self.patient_folder, dest=fixed_anon_path)
        # Check if the dcm folder is working and if the nii folder is not already present
        if integrity_manager.is_dcmfolder_working and (
            not os.path.exists(fixed_anon_path) or self.overwrite
        ):
            fixed_anon_path = integrity_manager.get_correct_dcm_dir()
        self.fixed_anon_path = fixed_anon_path

    def create_nii_masks(self):
        if not os.path.exists(self.ct_path) or self.overwrite:
            conver_patient_to_nii((self.fixed_anon_path, self.ct_path))

    def get_catheters_mask(self):
        """
        All other contours are created manually by the Radiation Oncologist.
        Catheter contours are created automatically from the digitization points.
        """
        catheter_mask_path = os.path.join(self.processed_folder, "catheters.seg.nrrd")
        if not os.path.exists(catheter_mask_path) and not self.overwrite:
            dwell_file_path = glob.glob(os.path.join(self.patient_folder, "RP*"))[0]
            creator = CatheterContourCreator(
                patient_path=None,
                dwell_file_path=dwell_file_path,
                patient_volume_path=self.processed_CT_path,
            )
            catheter_contour = creator.create_catheter_contour()
            ref_spacing = sitk.ReadImage(self.processed_CT_path).GetSpacing()
            if catheter_contour.GetSpacing() != ref_spacing:
                catheter_contour = creator.resample_volume(
                    catheter_contour, sitk.sitkNearestNeighbor, ref_spacing
                )
            sitk.WriteImage(catheter_contour, catheter_mask_path, useCompression=True)
        else:
            catheter_contour = sitk.ReadImage(catheter_mask_path)

        return sitk.GetArrayFromImage(catheter_contour)

    def get_mapping_dicts(self):
        """
        Getting useful dictionaries to map labels to names and vice versa.
        """
        with open(self.labels_file, "r") as j:
            metadata = json.loads(j.read())
            self.names_to_labels_dict = metadata["original_labels"]
            impossible_pairs_names = metadata["impossible_overlaps"]

        self.labels_to_names_dict = {v: k for k, v in self.names_to_labels_dict.items()}
        self.multilabel_to_singlelabel = get_label_combinations(
            len(list(self.names_to_labels_dict.keys()))
        )
        impossible_pairs_indexes = [
            (self.names_to_labels_dict[i[0]], self.names_to_labels_dict[i[1]])
            for i in impossible_pairs_names
        ]

        self.multilabel_to_singlelabel = filter_multilabels(
            self.multilabel_to_singlelabel, impossible_pairs_indexes
        )

        self.singlelabel_tomultinames = {}
        for k, v in self.multilabel_to_singlelabel.items():
            self.singlelabel_tomultinames[v] = [self.labels_to_names_dict[i] for i in k]

    def check_for_multiple_lung_contour(self, contour_names: List[str]):
        """
        This function was needed at the beginning of the project since sometimes I had a contour for one lung,
        sometimes for the other, sometimes for both, sometimes None.

        Args:
            contour_names (List[str]): List of the contour names in the folder of nifti segmentation masks.

        """
        multiple_lungs = [
            c_name
            for c_name in contour_names.keys()
            if ("lung" in c_name and not "both" in c_name)
        ]
        print("contour names ", contour_names.keys())
        if len(multiple_lungs) > 1:
            lung_contour_name = self.check_lung_side_distance_ptv(contour_names.keys())
            for lung_name in multiple_lungs:
                if lung_name != lung_contour_name:
                    print("poping ", lung_name, "from dict")
                    contour_names.pop(lung_name)
        elif len(multiple_lungs) == 1:
            lung_contour_name = multiple_lungs[0]
        else:
            lung_contour_name = None
        return lung_contour_name

    def check_lung_side_distance_ptv(self, rois):
        """
        At the beginning of the project, as the lungs contours were not consistent, we only considered the lung
        contour that was the closest to the PTV contour.
        """
        potential_lungs = []
        for roi in rois:
            if "lung" in roi:
                potential_lungs.append(roi.replace(" ", "_"))
            if "ptv" in roi:
                ptv = roi
        if len(potential_lungs) > 1:
            pth = os.path.join(self.processed_folder, ptv + ".seg.nrrd")
            ptv = sitk.GetArrayFromImage(sitk.ReadImage(pth))
            assert np.sum(ptv) != 0, "no ptv voxel in contour"
            x_indices_ptv = np.where(ptv > 0)[2]
            max_x_indices_ptv = np.max(x_indices_ptv)
            min_x_indices_ptv = np.min(x_indices_ptv)
            closest = None
            smallest_distance = np.inf

            for p_lung in potential_lungs:
                if "both" in p_lung:
                    continue
                path = os.path.join(self.processed_folder, p_lung + ".seg.nrrd")
                lung = sitk.GetArrayFromImage(sitk.ReadImage(path))
                if np.sum(lung) == 0:
                    continue

                x_indices_lung = np.where(lung > 0)[2]
                max_x_indices_lung = np.max(x_indices_lung)
                min_x_indices_lung = np.min(x_indices_lung)

                smallest_distance_lung = min(
                    abs(max_x_indices_lung - min_x_indices_ptv),
                    abs(min_x_indices_lung - min_x_indices_ptv),
                    abs(max_x_indices_lung - max_x_indices_ptv),
                    abs(min_x_indices_lung - max_x_indices_ptv),
                )

                if smallest_distance_lung < smallest_distance:
                    closest = p_lung
                    smallest_distance = smallest_distance_lung

            return closest
        elif len(potential_lungs) == 1:
            assert (
                not "both" in potential_lungs[0]
            ), "if there is only one lung contour, it should not be both_lungs"
            return potential_lungs[0]
        else:
            print("NO LUNG FOUND IN ROIS")
            return None

    @staticmethod
    def get_mask_name(contour_name, lung_contour_name):
        if contour_name == lung_contour_name:
            mask_name = "lung"
        else:
            mask_name = contour_name
        if "heart" in contour_name:
            mask_name = "heart"
        # elif "chest" in contour_name:
        #     print("in simple condition")
        #     print("contour name ", contour_name)
        elif ("chest" in contour_name) and not ("skin" in contour_name):
            print("you are in chest wall")
            mask_name = "chest wall"
        return mask_name

    def get_master_mask_regions(self):
        """
        Prepare a master mask from the contours in the folder.
        Choices are made on how to handle overlap between classes.
        """

        lung_contour_name = self.check_for_multiple_lung_contour(self.contour_names)

        shape_master_maks = [len(list(self.names_to_labels_dict.keys()))]
        for i in range(3):
            # 2-i and not i since numpy array is x,y,z whereas sitk is z,y,x
            shape_master_maks.append(self.ref_CT.GetSize()[2 - i])
        master_mask = np.zeros(shape_master_maks, dtype=int)

        contour_count = 1  # background is already present
        for contour_idx, contour_name in enumerate(self.contour_names.keys()):
            mask_name = self.get_mask_name(contour_name, lung_contour_name)
            if mask_name in self.names_to_labels_dict.keys():
                im = sitk.GetArrayFromImage(
                    sitk.ReadImage(
                        os.path.join(self.processed_folder, contour_name + ".seg.nrrd")
                    )
                )
                master_mask[self.names_to_labels_dict[mask_name]] = im
                contour_count += 1
            else:
                print(f"skipping {contour_name} because it is not in labels dict")

        master_mask[self.names_to_labels_dict["catheters"]] = self.get_catheters_mask()

        master_mask_region = multilabel_to_one_channel(
            master_mask, self.multilabel_to_singlelabel
        )

        all_contours_present = contour_count == len(self.names_to_labels_dict.keys())

        return master_mask_region, all_contours_present

    def get_master_mask_choice_for_overlap(self):
        """
        Prepare a master mask from the contours in the folder.When two classes (or more)
        are co-located, the mask will have a unique label for each combination of classes.
        """
        lung_contour_name = self.check_for_multiple_lung_contour(self.contour_names)

        master_mask = np.zeros_like(sitk.GetArrayFromImage(self.ref_CT))

        contour_count = 1  # background is already present
        for contour_name in self.contour_names.keys():
            mask_name = self.get_mask_name(contour_name, lung_contour_name)
            if mask_name in self.names_to_labels_dict.keys():
                im = sitk.GetArrayFromImage(
                    sitk.ReadImage(
                        os.path.join(self.processed_folder, contour_name + ".seg.nrrd")
                    )
                )
                if np.sum(im) == 0:
                    print(f"skipping {contour_name} because it is empty")
                    continue
                print(f"adding {contour_name} to master mask")
                # Chest wall contour overlaps with lung contour and ctv contour overlaps with ptv contour
                if mask_name == "chest wall":
                    master_mask[
                        (im != 0) & (master_mask != self.names_to_labels_dict["lung"])
                    ] = self.names_to_labels_dict[mask_name]
                elif mask_name == "ptv":
                    master_mask[
                        (im != 0) & (master_mask != self.names_to_labels_dict["ctv"])
                    ] = self.names_to_labels_dict[mask_name]
                else:
                    master_mask[im != 0] = self.names_to_labels_dict[mask_name]
                contour_count += 1
            else:
                print(f"skipping {contour_name} because it is not in labels dict")

        all_contours_present = contour_count == len(self.names_to_labels_dict.keys())

        return master_mask, all_contours_present

    def prepare_master_mask(
        self, regions=True, write=True, edit_label=False, mapping_names=None
    ):
        if regions:
            master_mask_npy, allcontours = self.get_master_mask_regions()
            mapping_names = self.singlelabel_tomultinames
        else:
            print("getting master mask with overlap")
            master_mask_npy, allcontours = self.get_master_mask_choice_for_overlap()
            mapping_names = self.labels_to_names_dict

        master_mask = sitk.GetImageFromArray(master_mask_npy)
        master_mask.CopyInformation(self.ref_CT)

        if edit_label:
            assert type(master_mask) == sitk.Image, "master mask should be a sitk image"
            edit_label_metadata(master_mask, mapping_names)

        if write:
            print("writing master mask")
            print(self.processed_folder)
            sitk.WriteImage(
                master_mask,
                os.path.join(self.processed_folder, "master_mask.seg.nrrd"),
                useCompression=True,
            )
        return master_mask, allcontours

    def move_to_nnunetraw(self, nnunet_raw_folder, dataset_name):
        pass


def preprocess_patient(args):
    patient_folder, dataset_name, nnunet_raw_folder, labels_file = args
    preprocessor = PatientPreprocessor(patient_folder, labels_file)

    # Saving labels
    mask, to_write = preprocessor.prepare_master_mask(
        regions=False, write=True, edit_label=True
    )
    if to_write:
        p = random()
        if p > 0.25:
            img_dest_path = os.path.join(nnunet_raw_folder, dataset_name, "imagesTr")
            label_dest_path = os.path.join(nnunet_raw_folder, dataset_name, "labelsTr")
        else:
            img_dest_path = os.path.join(
                nnunet_raw_folder.replace("nnUNet_raw", "nnUNet_test"),
                dataset_name,
                "imagesTr",
            )
            label_dest_path = os.path.join(
                nnunet_raw_folder.replace("nnUNet_raw", "nnUNet_test"),
                dataset_name,
                "labelsTs",
            )

        cropped_ct, cropped_mask = crop_volumes_background(
            preprocessor.ref_CT,
            mask=mask,
            boundaries_mm=preprocessor.boundaries,
            margin=10,
        )

        os.makedirs(label_dest_path, exist_ok=True)
        # Saving labels
        sitk.WriteImage(
            cropped_mask,
            os.path.join(label_dest_path, f"case_{preprocessor.patient_id}.seg.nrrd"),
            useCompression=True,
        )

        # Saving CT
        os.makedirs(img_dest_path, exist_ok=True)
        sitk.WriteImage(
            cropped_ct,
            os.path.join(img_dest_path, f"case_{preprocessor.patient_id}_0000.nrrd"),
            useCompression=True,
        )
    else:
        print(
            "Not all labels are present in the patient, skipping patient: ",
            os.path.basename(patient_folder),
        )


if __name__ == "__main__":
    import tqdm
    from multiprocessing import Pool

    seed(42)
    multiprocessing = False

    nnunet_raw_folder = "/home/sebquet/EngerLab/Breast_OAR_Segmentation/nnUNet_raw"
    dataset_name = "Dataset001_heartforall_region_based"
    # patient_folders = glob.glob(
    #     "/home/sebquet/EngerLab/Data/original_all_Breast_Patients/*Anon"
    # )
    patient_folders = ["/home/sebquet/EngerLab/Data/Hamed_breastCancer_patient"]

    labels_file = os.path.join(os.path.dirname(__file__), "classes_of_interest.json")
    args = [
        (patient_folder, dataset_name, nnunet_raw_folder, labels_file)
        for patient_folder in patient_folders
    ]

    if multiprocessing:

        with Pool(8) as p:
            r = list(
                tqdm.tqdm(
                    p.imap(preprocess_patient, args),
                    total=len(args),
                    desc="Preprocessing patients",
                )
            )
    else:
        for arg in tqdm.tqdm(args, total=len(args), desc="Preprocessing patients"):
            patient_fold, _, _, _ = arg
            # if not patient_fold.split("/")[-1] == "6515_Anon":
            #     continue
            preprocess_patient(arg)
            break
        print("one patient preprocessed")
