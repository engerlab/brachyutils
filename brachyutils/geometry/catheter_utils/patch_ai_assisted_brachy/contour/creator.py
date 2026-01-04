import os
import sys
import json
import tqdm
import multiprocessing
from pathlib import Path
# sys.path.append(str(Path(__file__).parents[2]))
from typing import List

import numpy as np
import pydicom as dicom
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial.distance import cdist

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.falsifier import NeedleFalsifier
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.pw_linear_interpolator import extrapolate_point
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import find_extremal_points_a
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import CatheterSetUp
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils.dicom_to_sitk import (
    convert_dicom_images_folder_to_nii,
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import get_slicer_marker_pt_dict
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import sitk_crop
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import resample_volume

class CatheterContourCreator:
    def __init__(
            self,
            patient_path = None,
            catheter_setup:CatheterSetUp=None,
            patient_volume_path=None,
            processed_folder=None,
            dilation=0, 
            add_tip_marker_contour:bool=True,
            extend_catheters_to_body:bool=False,
            body_contour_mask:sitk.Image=None,
            catheter_diameter:float=2.0,
            ):
        """
        
        Args:
        - patient_path (str): The path to the patient folder, containing DICOM files, will be used to create CatheterSetUp class instance.
        - catheter_setup (CatheterSetUp): Optional, The CatheterSetUp class instance, if None, will be created from the patient_path.
        - patient_volume_path (str): The path to the created patient CT volume in nrrd format. 
        - processed_folder (str): Optional , The folder where the processed files will be saved, used for nnunet raw dataset creation.
        - dilation (int): The number of times the catheter will be dilated.
        - add_tip_marker_contour (bool): Whether to add a tip marker contour to the catheter contour.
        - extend_catheters_to_body (bool): Whether to extend the catheters to the body contour.
        - body_contour_mask (sitk.Image): The body contour mask, needed if extend_catheters_to_body is True.
        - catheter_diameter (float): The diameter of the catheter.
        """
        if (
            (patient_path is None and catheter_setup is None) or
            (patient_path is not None and catheter_setup is not None)
            ):
            raise ValueError("Either patient_path or catheter_setup must be provided.")
        self.patient_path = patient_path if catheter_setup is None else str(catheter_setup.CT_folder)
        self.processed_folder = processed_folder
        self.plan_file_path = None
        if patient_volume_path is None:
            self.patient_volume_path = os.path.join(self.patient_path, "ct.nrrd")
        else:
            self.patient_volume_path = patient_volume_path
        self.plan_file_path = None
        self.channel_length = None
        # Defines plan_file_path and channel_length
        self.digitization_points, self.tips = self.get_digitization_points_and_tips(catheter_setup)
        self.dilation = dilation
        self.catheter_diameter = catheter_diameter
        self.add_tip_marker_contour = add_tip_marker_contour
        self.extend_catheters_to_body = extend_catheters_to_body
        self.body_contour_mask = body_contour_mask
        if not os.path.exists(self.patient_volume_path):
            print(f"The CT did not exist in nrrd format for patient {os.path.basename(self.patient_path)}, we convert from DICOM.")
            convert_dicom_images_folder_to_nii(
                (self.patient_path, self.patient_volume_path)
            )
        
        self.extra_points_input_button = None
        self.extra_pt_tip_marker = None
        self.extra_pt_before_tip_marker = None
        self.etra_pts_extended_catheters = None

        # # DEPRECATED attributes:
        # self.patient_id = os.path.basename(patient_path)
        # - add_catheter_input_and_button (bool): DEPRECATED Whether to add catheter input and button.
        # - extrapt_cat_input_dist (int): DEPRECATED The distance to the catheter input.
        # - extrapt_cat_button_dist (int): DEPRECATED The distance to the catheter button.
        self.add_catheter_input_and_button = False
        self.extrapt_cat_input_dist = 20
        self.extrapt_cat_button_dist = 7.
        
        self.add_pairs_of_points_for_contour_based_on_knowledge()
    
    def add_pairs_of_points_for_contour_based_on_knowledge(self):
        """
        This fuctions adds points to create contours. For the tip, we are getting consistent
        tip at the most distal part of the tip marker from catheter_setup class.
        """

        vol = sitk.ReadImage(self.patient_volume_path)
        if self.add_catheter_input_and_button:
            # DEPRECATED, was mostly for exploration on the contours we could create
            assert not self.add_tip_marker_contour, "Cannot add tip marker contour when adding catheter input and button"
            if self.extrapt_cat_button_dist > 0:
                self.extra_points_input_button = self.add_extra_digipt(
                    dist_to_extrapt=self.extrapt_cat_button_dist, from_tip=True, 
                    sitk_vol=vol)
            if self.extrapt_cat_input_dist > 0:
                self.extra_points_input_button = self.add_extra_digipt(
                    dist_to_extrapt=self.extrapt_cat_input_dist, sitk_vol=vol)

        if self.add_tip_marker_contour:
            assert not self.add_catheter_input_and_button, "Cannot add tip marker contour when adding catheter input and button"
            # The tip we considered is consistenly at the most distal part of the tip maker.
            self.extra_pt_tip_marker = self.add_extra_digipt(
                dist_to_extrapt=3, from_tip=True, sitk_vol=vol, 
                reverse=True)
            temp_extra_pt_before_tip_marker = self.add_extra_digipt(
                dist_to_extrapt=13, from_tip=True, sitk_vol=vol, 
                reverse=True)
            # Removing the tip marker distance from entry tip marker points
            self.extra_pt_before_tip_marker = {
                # List of list to match with digitization points format
                k: [[self.extra_pt_tip_marker[k][0][1], temp_extra_pt_before_tip_marker[k][0][1]]]
                for k in self.extra_pt_tip_marker.keys()
            }
            
        if self.extend_catheters_to_body:
            self.etra_pts_extended_catheters = {}
            self.etra_pts_extended_catheters = self.add_extra_digipt(
                    dist_to_extrapt=1, from_tip=False, sitk_vol=vol, 
                    extend_to_contour=True, contour_mask=self.body_contour_mask)
        

    def get_digitization_points_and_tips(self, catheter_setup:CatheterSetUp=None):
        if catheter_setup is None:
            catheter_setup = CatheterSetUp(self.patient_path)
        self.plan_file_path = catheter_setup.CT_folder
        self.channel_length = catheter_setup.channel_length
        self.offset = catheter_setup.offset
        tip_end_of_tip_marker, consistent_digi_pts = catheter_setup.get_consistent_tip_at_end_of_tip_marker(return_list=False)
        return consistent_digi_pts, tip_end_of_tip_marker

    def add_extra_digipt(
            self, dist_to_extrapt:float, from_tip:bool=False, 
            sitk_vol:sitk.Image=None, reverse:bool=False, 
            extend_to_contour:bool=False,
            contour_mask:sitk.Image=None
            ):
        """
        Add extra digitization points to the existing digitization points in order to 
        contour this last part as the "catheter input" to be able to locate the tip afterwards.
        The tip would then be opposite to this "catheter input".
        """
        extra_points:dict={}
        for tip_key, tip_coord in self.tips.items():
            dist_for_extrapt = dist_to_extrapt
            digipts = self.digitization_points[tip_key]
            if len(digipts) < 2:
                print(f"WARNING: Needle {tip_key} has less than 2 points, skipping")
                continue
            # Finding the digtization points closest to the tip index 0 or -1 since
            # digtization points are ordered.
            distances = cdist([tip_coord], digipts)
            if not from_tip:
                # farthest from tip is the "catheter button" in the patient body.
                farthest_idx = np.argmax(distances)
                # second farthest index
                second_farthest_idx = np.argsort(distances[0])[-2]
            else:
                # closest to tip is the "catheter button" in the patient body.
                farthest_idx = np.argmin(distances)
                # second closest index
                second_farthest_idx = np.argsort(distances[0])[1]

            pt1 = digipts[second_farthest_idx]
            pt2 = digipts[farthest_idx]

            extra_point_for_input = extrapolate_point(
                point1=pt1, point2=pt2, 
                distance=dist_for_extrapt, reverse=reverse)
            

            # Checking if the extra point is in the body contour
            if extend_to_contour:
                coord_in_vol = np.array(sitk_vol.TransformPhysicalPointToIndex(extra_point_for_input))
                extra_point_for_input = None
                contour_numpy = np.swapaxes(sitk.GetArrayFromImage(contour_mask), 0, 2)
                extra_point_outside_contour = contour_numpy[coord_in_vol[0], coord_in_vol[1], coord_in_vol[2]] == 0
                extra_point_in_volume = not(np.any(coord_in_vol < 0) or 
                                                np.any(coord_in_vol > np.array(sitk_vol.GetSize())-1))
                contour_extended = False
                while extra_point_in_volume and not extra_point_outside_contour:
                    # End point is not already outside contour, we need to extend
                    dist_for_extrapt += 1
                    extra_point_for_input_temp = extrapolate_point(
                        point1=pt1, point2=pt2, 
                        distance=dist_for_extrapt, reverse=reverse)
                    coord_in_vol = np.array(sitk_vol.TransformPhysicalPointToIndex(extra_point_for_input_temp))
                    extra_point_outside_contour = contour_numpy[coord_in_vol[0], coord_in_vol[1], coord_in_vol[2]] == 0
                    extra_point_in_volume = not(np.any(coord_in_vol < 0) or 
                                                np.any(coord_in_vol > np.array(sitk_vol.GetSize())-1))
                    if extra_point_in_volume and not extra_point_outside_contour:
                        extra_point_for_input = extra_point_for_input_temp
                        contour_extended = True
                if contour_extended:
                    print(f"Extending catheter {tip_key} to body contour by {dist_for_extrapt}mm")

            if not extra_point_for_input is None:
                # Checking if this extra point is in the CT scan volume
                coord_in_vol = np.array(sitk_vol.TransformPhysicalPointToIndex(extra_point_for_input))
                extra_point_in_volume = not(np.any(coord_in_vol <0) or np.any(coord_in_vol > np.array(sitk_vol.GetSize())-1))
                # In extreme cases where the CT scan stops at the end/beginning of a needle, we might need to reduce the 
                # distance for extrapolation back into the CT scan volume for the rest of the pipeline to work.
                while not extra_point_in_volume:
                    dist_for_extrapt -= 1
                    extra_point_for_input = extrapolate_point(
                        point1=pt1, point2=pt2, 
                        distance=dist_for_extrapt, reverse=reverse)
                    coord_in_vol = np.array(sitk_vol.TransformPhysicalPointToIndex(extra_point_for_input))
                    extra_point_in_volume = not(np.any(coord_in_vol < 0) or 
                                                np.any(coord_in_vol > np.array(sitk_vol.GetSize())-1))
                
                if tip_key in extra_points.keys():
                    extra_points[tip_key].append([pt2, extra_point_for_input])
                else:
                    extra_points[tip_key] = [[pt2, extra_point_for_input]]
            else:
                extra_points[tip_key] = []
        return extra_points    

    def create_slicer_markup_points(self, point_list, needle_nb):
        slicer_dict, markup_dict, ctrl_pt_dict = get_slicer_marker_pt_dict()
        slicer_dict["markups"].append(markup_dict)
        for pt_idx, pt in enumerate(point_list):
            temp_ctrl_pt_dict = ctrl_pt_dict.copy()
            temp_ctrl_pt_dict["id"] = str(pt_idx + 1)
            temp_ctrl_pt_dict["position"] = pt
            temp_ctrl_pt_dict["label"] = f"digitization_points_needle_{needle_nb}-{pt_idx+1}"
            slicer_dict["markups"][0]["controlPoints"].append(
                {
                    "id": str(pt_idx + 1),
                    "label": f"digitization_points_needle_{needle_nb}-{pt_idx+1}",
                    "description": "",
                    "associatedNodeID": "vtkMRMLScalarVolumeNode32",
                    "position": pt,
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": True,
                    "locked": False,
                    "visibility": True,
                    "positionStatus": "defined",
                }
            )
        if self.processed_folder is not None:
            dest = os.path.join(self.patient_path, self.processed_folder)
        else:
            dest = self.patient_path
        with open(
            os.path.join(dest, f"digitization_points_needle_{needle_nb}.mrk.json"),
            "w",
        ) as f:
            json.dump(slicer_dict, f, indent=4)

        return slicer_dict

    def process_needle(self, points, needle_key, patient_volume_sitk):
        if len(points) < 2:
            print(f"WARNING: Needle {needle_key} has less than 2 points, skipping")
            return np.zeros_like(patient_volume_sitk)

        self.create_slicer_markup_points(points, needle_key)
        current_needle_mask = np.zeros(patient_volume_sitk.GetSize(), dtype=bool)

        for pt_idx in range(len(points) - 1):
            fake_piece_of_needle = self._create_catheter_part(
                points[pt_idx], points[pt_idx + 1], patient_volume_sitk
            )
            current_needle_mask = np.logical_or(
                fake_piece_of_needle.volume, current_needle_mask
            )

        if self.extend_catheters_to_body:
            if len(self.etra_pts_extended_catheters[needle_key]) != 0:
                extra_pt = self.etra_pts_extended_catheters[needle_key][0]
                my_extra_fake_needle = self._create_catheter_part(
                    extra_pt[0], extra_pt[1], patient_volume_sitk
                )
                current_needle_mask = np.logical_or(
                    my_extra_fake_needle.volume, current_needle_mask
                )

        # When using add_line_from_voxel_indexes, dilation is done in the falsifier
        # class for each linear piece of the catheter. Since 
        # add_line_from_voxel_indexes is deprecated for add_line_from_voxel_coordinates
        # to build the needle from catheter diameter, we keep the possibility to dilate.
        # Even when created catheter is supposed to already be of the correct size.
        dilation_at_once = self.dilation != 0
        if dilation_at_once:
            current_needle_mask = self._dilate_with_endpt_correction(current_needle_mask, patient_volume_sitk)
    
        # Creating a tip marker contour: the non CT-marked part (entry tip) along with marked point at the tip location
        if self.add_tip_marker_contour:
            assert len(self.extra_pt_tip_marker[needle_key]) == 1, "Only one extra point is allowed for tip marker contour"
            assert len(self.extra_pt_before_tip_marker[needle_key]) == 1, "Only one extra point is allowed for tip marker contour"
            extra_masks = []
            for extra_pt_idx, extra_pt in enumerate([self.extra_pt_before_tip_marker[needle_key][0], self.extra_pt_tip_marker[needle_key][0]]):
                my_extra_fake_needle = self._create_catheter_part(
                    extra_pt[0], extra_pt[1], patient_volume_sitk
                )
                extra_mask = my_extra_fake_needle.volume.astype(int)
                if dilation_at_once:
                    correction = True
                    if extra_pt_idx == 1:
                        # Not correcting for tip marker dilation since itis a square 
                        # and it would mess up the whole contour
                        correction = False
                    extra_mask = self._dilate_with_endpt_correction(extra_mask, patient_volume_sitk, correction=correction)
                
                extra_masks.append(extra_mask)
            final_mask = current_needle_mask.astype(int) 
            # 1 for the needle, 2 for the entry tip marker before the tip, 3 for the tip marker 
            for extra_mask, value_mask in zip(extra_masks, [2,3]):
                final_mask_without_extra_mask = np.where(np.logical_and(final_mask !=0, extra_mask.astype(int)==0), final_mask, 0)
                final_mask = final_mask_without_extra_mask + extra_mask.astype(int) * value_mask

        # Creating extra contour for the catheter input in the body, most distal to the tip of the catheter.
        elif self.add_catheter_input_and_button:
            # First extra point is catheter button (end of catheter, just after tip), and second is catheter input.
            extra_pts = self.extra_points_input_button[needle_key]
            extra_masks = []
            for extra_pt_idx, extra_pt in enumerate(extra_pts):
                my_extra_fake_needle = self._create_catheter_part(
                    extra_pt[0], extra_pt[1], patient_volume_sitk
                )
                extra_mask = my_extra_fake_needle.volume.astype(int)
                if dilation_at_once or (extra_pt_idx == 0 and self.dist_to_extrapt_cat_button>0):
                    # Dialting the input of the catheter since it is bigger thant the catheter itself
                    if dilation_at_once:
                        ran = self.dilation
                    else:
                        # Dilating of 1 the button in all cases 
                        ran = 1
                    extra_mask = self._dilate_with_endpt_correction(
                        extra_mask, patient_volume_sitk, rang=ran)
                   
                extra_mask_and_current_needle = np.logical_or(
                    extra_mask.astype(bool), current_needle_mask.astype(bool)
                )
                extra_masks.append((extra_mask_and_current_needle.astype(int) - current_needle_mask.astype(int)) * (extra_pt_idx + 2))
            
            final_mask = current_needle_mask.astype(int) 
            for extra_mask in extra_masks:
                final_mask = final_mask + extra_mask
        else:
            final_mask = current_needle_mask.astype(int)
          
        return final_mask

    def _create_catheter_part(self, pt1:List[float], pt2:List[float], patient_volume_sitk:sitk.Image):
        fake_needle = NeedleFalsifier(
                    pt1,
                    pt2,
                    sitk_volume=patient_volume_sitk,
                    demo=False,
                    # dilation_nb_times was only used by add_line_from_voxel_indexes
                    dilation_nb_times=0,
                )
        # add_line_from_voxel_coordinates is a more precise version
        # of add_line_from_voxel_indexes to create the catheter
        fake_needle.add_line_from_voxel_coordinates(diameter=self.catheter_diameter)
        return fake_needle
    
    def _dilate_with_endpt_correction(self, needle_mask:np.ndarray, patient_volume_sitk:sitk.Image, rang:int=None, 
                                      correction:bool=True):
        needle_mask = needle_mask.astype(int)
        ref_endpoint_coord, _, _ = self.get_endpoint_line(
            needle_mask, patient_volume_sitk
        )
        if rang is None:
            rang = self.dilation
        for _ in range(rang):
            needle_mask = ndimage.binary_dilation(needle_mask).astype(
                needle_mask.dtype
            )
            if correction:
                needle_mask = self.correct_endpoint_dilation(
                    needle_mask, ref_endpoint_coord, patient_volume_sitk
                )
        return needle_mask

    def crop_around_digi_points(self, patient_volume_sitk, margin=5):
        """
        Crop the volume around the digitization points.
        """
        # Getting the extrema of the digitization points
        list_digi_pts = []
        for needle_nb, digi_pts in self.digitization_points.items():
            list_digi_pts.extend(digi_pts)
        x_max = np.max([pt[0] for pt in list_digi_pts])
        x_min = np.min([pt[0] for pt in list_digi_pts])
        y_max = np.max([pt[1] for pt in list_digi_pts])
        y_min = np.min([pt[1] for pt in list_digi_pts])
        z_max = np.max([pt[2] for pt in list_digi_pts])
        z_min = np.min([pt[2] for pt in list_digi_pts])

        for extra_pt_d in [self.extra_points_input_button, self.extra_pt_tip_marker, 
                           self.extra_pt_before_tip_marker, self.etra_pts_extended_catheters]:
            if extra_pt_d is not None:
                for needle_nb, extra_pts in extra_pt_d.items():
                    for extra_pt in extra_pts:
                        # extra_pt[0] is already in digi_pts
                        x_max = max(x_max, extra_pt[1][0])
                        x_min = min(x_min, extra_pt[1][0])
                        y_max = max(y_max, extra_pt[1][1])
                        y_min = min(y_min, extra_pt[1][1])
                        z_max = max(z_max, extra_pt[1][2])
                        z_min = min(z_min, extra_pt[1][2])
        
        first_end_point = patient_volume_sitk.TransformPhysicalPointToIndex([x_min, y_min, z_min])
        last_end_point = patient_volume_sitk.TransformPhysicalPointToIndex([x_max, y_max, z_max])

        # adding up the margin
        first_end_point = [max(0, pt - margin) for pt in first_end_point]
        # Here we keep sz and not sz - 1 since it is used as lenght rather than index in the bounding box
        last_end_point = [min(sz, pt + margin) for sz, pt in zip(patient_volume_sitk.GetSize(), last_end_point)]

        # creating the bounding box
        bounding_box = []
        for i in range(3):
            bounding_box.append(first_end_point[i])
        for i in range(3,6):
            bounding_box.append(last_end_point[i-3] - first_end_point[i-3])

        return sitk_crop(patient_volume_sitk, bounding_box), bounding_box

    def pad_mask(self, mask_placeholder, bb, patient_volume_sitk):
        """
        Pad the mask to the original size of the volume.
        """
        padding = []
        for i in range(3):
            padding.append((bb[i], patient_volume_sitk.GetSize()[i] - bb[i] - bb[i+3]))

        padded_npy_mask = np.pad(mask_placeholder, padding, mode="constant", constant_values=0)
        return padded_npy_mask

    def create_catheter_contour(
            self, multiprocess=True, use_1_mm_isotropic_spacing:bool=True, 
            write=False, out_path=None, write_ct:bool=False):

        if not os.path.exists(self.patient_volume_path):
            convert_dicom_images_folder_to_nii(
                (self.patient_path, self.patient_volume_path)
            )

        patient_volume_sitk = sitk.ReadImage(self.patient_volume_path)
        if use_1_mm_isotropic_spacing:
            wanted_spacing = [1.0, 1.0, 1.0]
            if not np.allclose(
                patient_volume_sitk.GetSpacing(), 
                wanted_spacing,
                atol=1e-5):
                patient_volume_sitk = resample_volume(
                    patient_volume_sitk,
                    new_spacing=wanted_spacing,
                )

        if write_ct:
            bn = os.path.basename(self.patient_volume_path)
            sitk.WriteImage(
                patient_volume_sitk,
                str(self.patient_volume_path).replace(
                    bn, 
                    bn.split(".")[0] +"_resampled.nrrd"
                ),
                useCompression=True,
            )

        # Cropping around digitization points to save computation time since
        # all around digitization points is going to be all 0s anyway.
        cropped_patient_volume, bb = self.crop_around_digi_points(patient_volume_sitk)
        mask_placeholder = np.zeros(cropped_patient_volume.GetSize(), dtype=int)


        if self.add_tip_marker_contour:
            max_class = 3
        else:
            max_class = 1

        if multiprocess:
            num_processes = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=num_processes)
            results = []

            for needle_key, digi_pts in self.digitization_points.items():

                if len(digi_pts) < 2:
                    print(
                        f"WARNING: Needle {needle_key} has less than 2 points, skipping"
                    )
                    continue

                if write:
                    self.create_slicer_markup_points(digi_pts, needle_key)

                results.append(
                    pool.apply_async(
                        self.process_needle,
                        args=(
                            digi_pts,
                            needle_key,
                            cropped_patient_volume,
                        ),
                    )
                )

            pool.close()
            pool.join()

            for result in tqdm.tqdm(
                results,
                desc="Creating contours for needles",
                total=len(self.digitization_points.values()),
            ):
                for i in range(1, max_class + 1):
                    mask_placeholder = np.where(np.logical_or(result.get()== i, mask_placeholder == i), i, mask_placeholder)

        else:
            for needle_key, digi_pts in tqdm.tqdm(
                    self.digitization_points.items(),
                    desc="Creating contours for needles",
                    total=len(self.digitization_points.values()),
                ):  
                if len(digi_pts) < 2:
                    print(
                        f"WARNING: Needle {needle_key} has less than 2 points, skipping"
                    )
                    continue

                if write:
                    self.create_slicer_markup_points(digi_pts, needle_key)

                for i in range(1, max_class + 1):
                    new_needle = self.process_needle(
                        digi_pts, needle_key, cropped_patient_volume
                    )
                    mask_placeholder = np.where(np.logical_or(new_needle== i, mask_placeholder == i), i, mask_placeholder)

        # Padding back to the original size of the volume with 0s.
        # Necessary from the previous cropping around digitization points.
        padded_mask_placeholder = self.pad_mask(mask_placeholder, bb, patient_volume_sitk)
        assert np.all(padded_mask_placeholder.shape == patient_volume_sitk.GetSize())

        if self.add_tip_marker_contour:
            assert np.any(padded_mask_placeholder == 3), "There is no tip marker contour"
            assert np.any(padded_mask_placeholder == 2), "There is no tip marker before tip"

        if self.extend_catheters_to_body:
            body_contour_npy = sitk.GetArrayFromImage(self.body_contour_mask).transpose(2, 1, 0)
            padded_mask_placeholder = np.where(
                np.logical_and(body_contour_npy == 1, padded_mask_placeholder != 0), 
                padded_mask_placeholder, 
                0
                )
            
        mask_image = sitk.GetImageFromArray(
            padded_mask_placeholder.transpose(2, 1, 0).astype(int)
        )
        mask_image.CopyInformation(patient_volume_sitk)
        if write:
            if out_path is not None:
                out_name = out_path
            else:
                out_name = self.patient_volume_path.replace(
                    bn,
                    "catheter_contour.seg.nrrd",
                )
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            sitk.WriteImage(
                    mask_image,
                    out_name,
                    useCompression=True,
                )
        return mask_image

    def get_endpoint_line(self, mask_placeholder, patient_volume_sitk):
        points = np.argwhere(mask_placeholder == 1)
        positions = np.array(
            # TransformIndexToPhysicalPoint only takes int
            [
                patient_volume_sitk.TransformIndexToPhysicalPoint(
                    [int(idx) for idx in pt]
                )
                for pt in points
            ]
        )
        endpoints, max_dist = find_extremal_points_a(positions)
        endpoints = endpoints[0]
        endpoints_coord = [
            np.array(patient_volume_sitk.TransformPhysicalPointToIndex(pt))
            for pt in endpoints
        ]
        return endpoints_coord, points, positions

    def correct_endpoint_dilation(
        self,
        mask_placeholder,
        ref_endpoints_coord,
        patient_volume_sitk,
        remove_similar_distance_dilated_pt=True,
    ):
        """
        Correcting for when the dilation changed the endpoints of the catheter.
        It is fine to dilate the width since we just created a line between points
        and the real catheter has a width, but it is not fine to change the length.
        """
        # Saving endpoints before dilation
        self.vox_tip_coord = ref_endpoints_coord[0]
        self.vox_end_coord = ref_endpoints_coord[1]

        # Getting the new endpoints from dilated line
        endpoints_coord, points, positions = self.get_endpoint_line(
            mask_placeholder, patient_volume_sitk
        )

        # Removing endpoints that have been dilated that are not the tip or the end
        while not (
            np.any(np.all(np.array(self.vox_tip_coord) == endpoints_coord, axis=1))
            and np.any(np.all(np.array(self.vox_end_coord) == endpoints_coord, axis=1))
        ):
            for endpt in endpoints_coord:
                not_begining = np.any(endpt != self.vox_tip_coord)
                not_end = np.any(endpt != self.vox_end_coord)
                if not_begining and not_end:
                    assert np.any(endpt != self.vox_tip_coord) and np.any(
                        endpt != self.vox_end_coord
                    ), "This point should not be an endpoint"
                    mask_placeholder[endpt[0], endpt[1], endpt[2]] = 0
                    for pt_idx, pt in enumerate(points):
                        if np.all(pt == endpt):
                            break

                    points = np.delete(points, pt_idx, axis=0)
                    positions = np.delete(positions, pt_idx, axis=0)

            endpoints, max_dist = find_extremal_points_a(positions)
            endpoints = endpoints[0]
            endpoints_coord = [
                np.array(patient_volume_sitk.TransformPhysicalPointToIndex(pt))
                for pt in endpoints
            ]

        if remove_similar_distance_dilated_pt:
            endpoints_copy = endpoints_coord.copy()
            # if the dilated points are changing the endpoints (including tip),
            # we need to remove the points that are at the same distance from the tip.
            if len(endpoints_coord) > 2:
                while len(endpoints_coord) > 2:
                    for pt_idx, pt in enumerate(endpoints_copy):
                        is_tip = np.all(np.array(self.vox_tip_coord) == pt)
                        is_end = np.all(np.array(self.vox_end_coord) == pt)
                        if not (is_tip or is_end):
                            assert np.any(pt != self.vox_tip_coord) and np.any(
                                pt != self.vox_end_coord
                            ), "This point should not be an endpoint"
                            mask_placeholder[pt[0], pt[1], pt[2]] = 0
                            break
                    endpoints_coord = np.delete(endpoints_coord, pt_idx, axis=0)

        return mask_placeholder


if __name__ == "__main__":
    patient_nb = "52193"
    patient_path = f"/home/sebquet/EngerLab/Data/export_seb/patients/{patient_nb}/"

    patient_volume_path = (
        os.path.join(patient_path, "processed", "ct.nrrd")
    )
    if not os.path.exists(patient_volume_path):
        convert_dicom_images_folder_to_nii(
            (patient_path, patient_volume_path)
        )

    creator = CatheterContourCreator(patient_path, patient_volume_path)
    catheter_contour = creator.create_catheter_contour(write=True, out_path=os.path.join(patient_path, "processed", "catheters.seg.nrrd"))
