import os 
from typing import List
import tqdm 
import warnings

import numpy as np
from scipy.spatial.distance import jensenshannon, cdist
import SimpleITK as sitk

from brachyutils.geometry.catheter_utils.utils import sitk_crop
from brachyutils.geometry.catheter_utils.utils import create_slicer_markup_points
from brachyutils.geometry.catheter_utils.digitization.spline_interpolator import NeedleSplineCreator
from brachyutils.geometry.catheter_utils.utils import get_physical_coord_for_needle
from brachyutils.geometry.catheter_utils.contour.separator import (
    ContourSeparator, ContourExpander, extend_catheter_contour_on_both_sides, 
    get_bounds_for_step, get_segment_endpoints_and_t)
from brachyutils.geometry.catheter_utils.log import Logger


def identify_points_too_far_from_a_group(pts_coords:List[List[float]]):
    """
    Identify points that are too far from a group of points.

    Args:
        pts_coords (List[List[float]]): List of point coordinates.

    Returns:
        List[int]: Indices of points that are too far from the group.
    """
    # Compute pairwise distances
    distances = cdist(pts_coords, pts_coords)
    
    # For each corner, find the minimum distance to any *other* corner
    min_dists = np.min(np.where(np.eye(len(distances)), np.inf, distances), axis=1)

    # Compute how different each is from the others
    mean_min = np.mean(min_dists)
    std_min = np.std(min_dists)

    # Flag any corner whose min distance is very different (e.g., >2 std away)
    outlier_indices = np.where(np.abs(min_dists - mean_min) > 2 * std_min)[0]

    if len(outlier_indices) > 0:
        return outlier_indices
    else:
        return None

class CatheterPostProcessor:
    def __init__(
            self, reference_ct:sitk.Image, catheter_marker_class:int=1, 
            tip_marker_class:int=3, entry_tip_marker_class:int=2,
            catheter_diameter:float=2.0, entry_tip_size:float=10.0, 
            tip_marker_size:float=3.0, save_details:bool=False, 
            save_details_path:str=None, contour_dilation:int=False, 
            log_path:str=None):
    
        self.reference_ct = reference_ct
        self.catheter_marker_class = catheter_marker_class
        self.tip_marker_class = tip_marker_class
        self.entry_tip_marker_class = entry_tip_marker_class
        self.multi_class = True
        self.catheter_diameter = catheter_diameter
        self.entry_tip_size = entry_tip_size
        self.tip_marker_size = tip_marker_size
        self.connected_components = None
        self.connected_comp_num_labels = None
        self.dilated_components = None
        self.bounding_box = None
        self.save_details = save_details
        self.save_details_path = save_details_path
        # Did you dilate contours for your training? If yes, we should 
        # consider it when post processing the contours.
        self.contour_dilation = int(contour_dilation)
        if self.contour_dilation >1:
            raise NotImplementedError("This code is designed to work contours created using a dilation of 0 or 1.") 
        
        # Big margin because we will potentially increase the catheter size of up to ~ 15 voxels, entryp tip size + tip size is 13mm
        # or even add part to the catheter to check merging of the catheters. 
        self.margin_for_cropping = 50
        if self.save_details_path:
            os.makedirs(self.save_details_path, exist_ok=True)

        assert log_path is not None, "You need to provide a path to save the logs."
        self.logger = Logger(log_path, "post_processing_logs.txt")
        self.average_tip_position  = None

        # Saving the postprocessing info as an attribute to be able to access it later
        self.postprocessed_infos = {
            "Only the tip was added to the catheter": "only_tip_added",
            "The tip was completed to full length": "tip_completed",
            "The entry tip was completed to full length and the tip was added to the catheter": "entrytip_completed_and_tip_added",
            "The entry tip and the tip were added to the catheter": "entrytip_and_tip_added",
        }

    def postprocess_catheters(self, catheters_contour_path:str, user_input_to_check_catheters:List[int]=None):
        
        catheters_contour = sitk.ReadImage(catheters_contour_path)

        if np.any(sitk.GetArrayFromImage(catheters_contour) == self.tip_marker_class):
            self.average_tip_position = np.mean(get_physical_coord_for_needle(catheters_contour, needle_idx=self.tip_marker_class), axis=0)
            if self.save_details:
                create_slicer_markup_points(
                        os.path.join(self.save_details_path, "average_created_tip_position.mrk.json"), [self.average_tip_position.tolist()]
                        )

        separator = ContourSeparator(
            reference_ct=self.reference_ct, catheters_contour_path=catheters_contour_path, catheter_marker_class=self.catheter_marker_class, 
            catheter_diameter=self.catheter_diameter, save_details=self.save_details, save_details_path=(self.save_details_path.replace(
                os.path.dirname(self.save_details_path), os.path.dirname(self.save_details_path)+"_separator") if self.save_details else None), 
            margin_for_cropping = self.margin_for_cropping, multiprocess=True, log_path=self.logger.log_folder)
        catheter_contours = separator.separate_catheters()
        self.bounding_box = separator.bounding_box
        self.connected_components = separator.modified_connected_components
        self.connected_comp_num_labels = separator.modified_connected_comp_num_labels
        self.dilated_components = separator.dilated_og_components

        # Catheter contour is cropped directly by the separator
        croppped_catheters_contour = separator.catheters_contour
        separator_infos = {"Merged catheters":separator.count_merged_catheters,
                           "Colliding catheters":separator.num_overlaps}
        
        # Getting end coordinates of bounding box of each catheter to check distance from 
        # each other later if ever there is a mismatch between user input catheter number
        # and number of created catheters.
        bb_start_coords = []
        bb_end_coords = []
        
        postprocessed_infos = {}
        postprocessed_catheters = []
        processed_catheters = np.zeros(sitk.GetArrayFromImage(croppped_catheters_contour).shape, dtype=np.uint8)
        for cat_idx, catheter_contour in tqdm.tqdm(enumerate(catheter_contours), total=len(catheter_contours), desc="Postprocessing catheters.."):

            catheter_core_voxels = np.sum(sitk.GetArrayFromImage(catheter_contour) == self.catheter_marker_class)
            if catheter_core_voxels < 150:
                # If the catheter core class is not big enough in the connected component we don't consider it.
                continue
            processed_catheter, postprocessed_info = self._postprocess_catheter(catheter_contour, croppped_catheters_contour, cat_idx)
            if processed_catheter is None:
                # If catheter core class was not present in the connected component we don't consider it.
                continue
            if self.save_details:
                sitk.WriteImage(
                    processed_catheter,
                    os.path.join(self.save_details_path, f"processed_catheter_{cat_idx}.seg.nrrd"),
                    useCompression=True)
            # If  ever there is a problem_in_catheter_count
            tmp_cat_array = sitk.GetArrayFromImage(processed_catheter)
            catheter_coords_index = np.array(np.where(tmp_cat_array > 0))
            min_x_index = int(np.min(catheter_coords_index[2, :]))
            max_x_index = int(np.max(catheter_coords_index[2, :]))
            min_y_index = int(np.min(catheter_coords_index[1, :]))
            max_y_index = int(np.max(catheter_coords_index[1, :]))
            min_z_index = int(np.min(catheter_coords_index[0, :]))
            max_z_index = int(np.max(catheter_coords_index[0, :]))
            ## Converting to physical coordinates
            min_coords = processed_catheter.TransformIndexToPhysicalPoint((min_x_index, min_y_index, min_z_index))
            bb_start_coords.append(min_coords)
            max_coords = processed_catheter.TransformIndexToPhysicalPoint((max_x_index, max_y_index, max_z_index))
            bb_end_coords.append(max_coords)

            postprocessed_catheters.append(processed_catheter)
            processed_catheter_array = sitk.GetArrayFromImage(processed_catheter).astype(np.uint8)
            processed_catheters += processed_catheter_array
            
            # Saving what was done during postprocessing
            self.logger.log([f"Postprocessed catheter {cat_idx}: {postprocessed_info} \n"])
            postprocessed_infos[cat_idx] = postprocessed_info
        
        if self.multi_class:
            assert np.max(processed_catheters) <= self.tip_marker_class, """
            Your post processing messed up the contours, 
            there should be only 3 classes in the catheter contour"""
        else:
            assert np.max(processed_catheters) == 1, """
            Your post processing messed up the contours, 
            there should be only 1 class in the catheter contour"""
        
        problem_in_catheter_count = len(postprocessed_catheters) != sum(user_input_to_check_catheters)
        if problem_in_catheter_count:
            self.logger.log([
                    "WARNING: The number of catheters found does not match user input.\n"
                ])
            if len(postprocessed_catheters) < sum(user_input_to_check_catheters):
                self.logger.log([
                    f"Number of catheters found ({len(postprocessed_catheters)}) is less than user input ({sum(user_input_to_check_catheters)}).\n"
                    "This could be due to catheters being too close to each other or merged during segmentation.\n"
                ])
                diff_catheter_number = sum(user_input_to_check_catheters) - len(postprocessed_catheters)
            else:
                self.logger.log([
                    f"Number of catheters found ({len(postprocessed_catheters)}) is more than user input ({sum(user_input_to_check_catheters)}).\n"
                    "This could be due to spurious small connected components being detected as catheters.\n"
                ])
                diff_catheter_number = len(postprocessed_catheters) - sum(user_input_to_check_catheters)
            problem_infos = {
                "Difference in catheter number": diff_catheter_number,
                "Catheters created": len(postprocessed_catheters),
                "User input catheters": sum(user_input_to_check_catheters)
            }
        else:
            problem_infos = None

        if problem_in_catheter_count:
            if len(postprocessed_catheters) == sum(user_input_to_check_catheters) + 1:
                # Might be solvable if ever one catheter is super far away from the others
                # Then it is the one we remove. 
                is_outlier, outlier_idx = self.study_spread_of_catheters(bb_start_coords, bb_end_coords)

                if is_outlier:
                    # We remove the outlier catheter
                    self.logger.log([
                        f"Catheter {outlier_idx} is an outlier in space and is removed to match user input catheter number.\n"
                    ])
                    problem_infos["Problem catheter count solved"] = True
                    problem_infos["Outlier catheter removed"] = outlier_idx
                    processed_catheters -= sitk.GetArrayFromImage(postprocessed_catheters[outlier_idx]).astype(np.uint8)
                else:
                    problem_infos["Problem catheter count solved"] = False
                    self.logger.log([
                        "No catheter is an outlier in space, we do not attempt to fix it automatically.\n"
                    ])
            else:
                problem_infos["Problem catheter count solved"] = False
                self.logger.log([
                    "More than one catheter difference between user input and found catheters, "
                    "we do not attempt to fix it automatically.\n"
                ])

        final_array = self.pad_mask(processed_catheters.transpose(2,1,0), catheters_contour)
        final_catheters = sitk.GetImageFromArray(np.swapaxes(final_array, 0, 2))
        final_catheters.CopyInformation(catheters_contour)
        return final_catheters, postprocessed_catheters, postprocessed_infos, separator_infos, problem_infos

    
    def study_spread_of_catheters(self, bb_start_coords:List[List[float]], bb_end_coords:List[List[float]]):
        """
        Study the spread of the catheters based on their bounding box coordinates.
        Compute distance from every catheter to every other catheter and see if one catheter is
        an outlier in space.
        """
        
        outlier_indices_start = identify_points_too_far_from_a_group(bb_start_coords)
        outlier_indices_end = identify_points_too_far_from_a_group(bb_end_coords)

        found_problem = False
        index = None
        if len(outlier_indices_start) == 1 and len(outlier_indices_end) == 1:
            if outlier_indices_start[0] == outlier_indices_end[0]:
                index = int(outlier_indices_start[0])
                found_problem = True
        return found_problem, index

    def pad_mask(self, mask_placeholder, patient_volume_sitk):
        """
        Pad the mask to the original size of the volume.
        """
        padding = []
        for i in range(3):
            padding.append((self.bounding_box[i], patient_volume_sitk.GetSize()[i] - self.bounding_box[i] - self.bounding_box[i+3]))

        padded_npy_mask = np.pad(mask_placeholder, padding, mode="constant", constant_values=0)
        return padded_npy_mask
    
    def _postprocess_catheter(
            self, catheter_contour:sitk.Image, catheter_contours:sitk.Image=None, 
            cat_idx:int=None, correct_tip_if_present:bool=True):
        """
        catheter_contour contains one needle
        catheter_contours conains all needles and is only used in the case the segmentation did
        not contain entry tip and tip marker classes and will be used to get average HUs for those classes.
        """

        all_classes = [0, 1]
        if self.multi_class:
            all_classes.extend([self.tip_marker_class, self.entry_tip_marker_class])

        all_classes_array = sitk.GetArrayFromImage(catheter_contour).astype(np.uint8)
        catheter_array = np.where(all_classes_array > 0, 1, 0).astype(np.uint8)
        all_catheter_contours_array = sitk.GetArrayFromImage(catheter_contours).astype(np.uint8)

        # Saving what was done during postprocessing
        postprocessed_info = None

        # Manually finishing catheters that are not complete
        if self.multi_class:
            unique_classes_in_needle = np.concatenate(
                [np.array([0]), 
                np.unique(catheter_array[catheter_array > 0] * all_classes_array[catheter_array > 0])]
            )

            # If all classes are present, we just check if the tip length is good 
            if np.all(np.isin(all_classes, unique_classes_in_needle)):
                if correct_tip_if_present:
                    (all_classes_array, 
                    _, 
                    _, 
                    _,
                    _,
                    _,
                    postprocessed_info
                    ) =self.potential_complete_part(
                        catheter_contour, all_classes_array, catheter_contours, 
                        class_idx=self.tip_marker_class, part_size=self.tip_marker_size,
                        # Half voxel on each side but dilated tip contour groud truth
                        # => -1 + 1 = 0 offset
                        offset=-1 + self.contour_dilation, cat_idx=cat_idx, grab_solo_components=False)
                
            elif not (self.catheter_marker_class in unique_classes_in_needle):
                # If no catheter core is present in the contour, we don't 
                # consider this part as main catheter and do not process it.
                if self.save_details:
                    print("No catheter is present for component {}".format(cat_idx))
                return None, None

            # If catheter is present and entry tip is present, we need to add the tip
            # we also potentially need to increase entry tip length if it is not 1cm.
            elif self.entry_tip_marker_class in unique_classes_in_needle and (
                not self.tip_marker_class in unique_classes_in_needle):
                if self.save_details:
                    print("Core and head are present for catheter {}. Checking head lenght and adding tip.".format(cat_idx))
                # Potentially completing the entry tip part if it is not big enough
                (all_classes_array, 
                 projected_new_end_entry_tip, 
                 projected_new_end_entry_tip_t, 
                 expander,
                 input_spline_creator,
                 t_solo_components,
                 postprocessed_info
                 ) =self.potential_complete_part(
                     catheter_contour, all_classes_array, catheter_contours, 
                     class_idx=self.entry_tip_marker_class, part_size=self.entry_tip_size,
                     offset= -1, cat_idx=cat_idx, grab_solo_components=True)

                ## Now that entry tip part is okay, we deal with the tip marker
                addon_tip, projected_end_tip, projected_end_tip_t = expander.add_part(
                    t_start=projected_new_end_entry_tip_t, start_pt_coords=projected_new_end_entry_tip,
                    # Finding the endpoint for the tip marker, 3mm from the end of the entry tip marker
                    step=(self.tip_marker_size - 1 + self.contour_dilation), 
                    dilation_add_on=self.contour_dilation, spline=True
                )
                if np.any(np.logical_and(addon_tip !=0, all_catheter_contours_array == self.catheter_marker_class)):
                    # Overlap should not happen at the tip.
                    warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
                else:
                    all_classes_array = np.where(addon_tip, self.tip_marker_class, all_classes_array)

                # After adding tip and entry tip we check that the solo component detected are part of 
                # the created contour
                if len(t_solo_components) > 0:
                    cather_w_solo_comps = sitk.GetArrayFromImage(input_spline_creator)
                    if np.any(np.logical_and(cather_w_solo_comps !=0, all_catheter_contours_array == self.catheter_marker_class)):
                        # Overlap should not happen at the tip.
                        warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
                    else:
                        all_classes_array = np.where(cather_w_solo_comps > 0, cather_w_solo_comps, all_classes_array)

                if self.save_details:
                    # Saving markers used to postprocess
                    create_slicer_markup_points(
                        os.path.join(self.save_details_path, f"new_tip_enpoints_catheter_{cat_idx}.mrk.json"), 
                        [projected_new_end_entry_tip, projected_end_tip], 
                    )
                    
                # If there was a solo component that was added to the catheter for the 
                # spline fitting, sanity check if it is on the same side as the tip and 
                # entry tip added.
                if len(t_solo_components) > 0:
                    assert len(t_solo_components) == 1, """
                    There should be only one t for the found solo components, on one side of the 
                    catheter. Because big components have been merged in the ContourSeparator class. 
                    This is just in case there is a tip or entry tip that is not connected with a 
                    catheter core.
                    """
                    # Extra solo component was added for spline fitting
                    if abs(t_solo_components[0]- 1) < abs(t_solo_components[0]- 0):
                        # solo components have been found around edge t==1 of the spline
                        assert abs(projected_end_tip_t- 1) < abs(projected_end_tip_t- 0)
                    else:
                        # solo components have been found around edge t==0 of the spline
                        assert abs(projected_end_tip_t- 0) < abs(projected_end_tip_t- 1)

            # If only the catheter is present, we need to add the tip and entry tip
            else:
                assert (not self.tip_marker_class in unique_classes_in_needle) and (
                not self.entry_tip_marker_class in unique_classes_in_needle)
                # Add the tip and entry tip on both side of catheter end and 
                # check distribution of the created tips and entry tips HU values
                # to identify whcih side is the correct one.
                if self.save_details:
                    print("Core is present for catheter {}. Adding head and tip.".format(cat_idx))
                # Create the spline from the contour we have
                catheter_contour_for_spline, _, t_solo_components = self._grab_isolated_components(
                    catheter_contour, catheter_contours, cat_idx, 
                    # We add 2 to the tip size to in case there is an angle between the catheter core and
                    # the tip that could prevent the add on part o find solo components.
                    search_space=self.entry_tip_size + self.tip_marker_size + 2, 
                    overlapping_classes=[self.entry_tip_marker_class, self.tip_marker_class])
                
                expander = ContourExpander(
                    catheter_contour=catheter_contour, catheter_diameter=self.catheter_diameter, 
                    multi_class=self.multi_class, save_details=self.save_details, 
                    save_details_path=self.save_details_path)

                # Identifying ts of the spline to know which side to process the catheter for.
                extremum0, t0_spline, extremum1, t1_spline = get_segment_endpoints_and_t(
                    catheter_contour, expander.needle_spline_creator)

                ### Adding the entry tip at t0_spline
                all_classes_array, addon_tip_t0, addon_entry_tip_t0 =  self._add_tip_entry_tip(
                    expander, t0_spline, extremum0, all_classes_array=all_classes_array, 
                    all_catheter_contours_array=all_catheter_contours_array,
                    dilation_add_on=self.contour_dilation, spline=True, cat_idx=cat_idx)
    
             
                ### Adding the entry tip at t1_spline
                all_classes_array, addon_tip_t1, addon_entry_tip_t1 =  self._add_tip_entry_tip(
                    expander, t1_spline, extremum1, all_classes_array=all_classes_array, 
                    all_catheter_contours_array=all_catheter_contours_array,
                    dilation_add_on=self.contour_dilation, spline=True, cat_idx=cat_idx)
               

                #### Choosing between the two added tips 
                # Choosing based on the found solo components 
                if len(t_solo_components) > 0:
                    ## Removing wrong tip based on side of found solo component. Solo component
                    # contains necessarily class entry tip or tip.
                    assert len(t_solo_components) == 1, """
                    There should be only one t for the found solo components, on one side of the 
                    catheter. Because big components have been merged in the ContourSeparator class. 
                    This is just in case there is a tip or entry tip that is not connected with a 
                    catheter core.
                    """
                    # solo components containing part of the tip have been found around t0_spline
                    if abs(t_solo_components[0] - t0_spline) < abs(t_solo_components[0] - t1_spline):
                        all_classes_array = np.where(addon_tip_t1 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t1 == 1, 0, all_classes_array)
                    # solo components containing part of the tip have been found around t0_spline
                    else:
                        all_classes_array = np.where(addon_tip_t0 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t0 == 1, 0, all_classes_array)
                
                # Choosing based on distance to the average tip position
                elif self.average_tip_position is not None:
                    _, projected_avg_tip_t, _ = expander.needle_spline_creator.project_on_spline(self.average_tip_position)
                    if abs(projected_avg_tip_t - t0_spline) < abs(projected_avg_tip_t - t1_spline):
                        all_classes_array = np.where(addon_tip_t1 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t1 == 1, 0, all_classes_array)
                    else:
                        all_classes_array = np.where(addon_tip_t0 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t0 == 1, 0, all_classes_array)
             
                # Choosing based on HUs
                else:
                    
                    # Getting ditribution of normal HUs for the tip and entry tip contours.
                    all_catheters_segmented  = sitk.GetArrayFromImage(catheter_contours)
                    assert self.reference_ct is not None, "Reference CT is needed to get HUs"
                    ct = sitk.GetArrayFromImage(sitk_crop(self.reference_ct, self.bounding_box))
                    assert np.all(all_catheters_segmented.shape == ct.shape), (
                        "CT and catheter contours should have the same shape but have {} VS {}".format(
                            all_catheters_segmented.shape, ct.shape)
                            )
                    
                    # Reference entry tip HU distribution in the patient predictions.
                    entry_tip_dist = ct[all_catheters_segmented == self.entry_tip_marker_class]           
                    fakeentrytip1_dist = ct[addon_entry_tip_t1 == 1]
                    fakeentrytip0_dist = ct[addon_entry_tip_t0 == 1]
                    
                    density_entry_tip_dist = self._create_density_distribution(entry_tip_dist)
                    density_fakeentrytip0_dist = self._create_density_distribution(fakeentrytip0_dist)
                    density_fakeentrytip1_dist = self._create_density_distribution(fakeentrytip1_dist)
                    jh0 = jensenshannon(density_entry_tip_dist, density_fakeentrytip0_dist)
                    jh1 = jensenshannon(density_entry_tip_dist, density_fakeentrytip1_dist)
                    # Removing wrong tip based on dffference of HU distribution with reference.
                    if jh0 < jh1:
                        all_classes_array = np.where(addon_tip_t1 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t1 == 1, 0, all_classes_array)
                    else:
                        all_classes_array = np.where(addon_tip_t0 == 1, 0, all_classes_array)
                        all_classes_array = np.where(addon_entry_tip_t0 == 1, 0, all_classes_array)

                # After adding tip and entry tip we check that the solo component detected are part of 
                # the created contour
                if len(t_solo_components) > 0:
                    cather_w_solo_comps = sitk.GetArrayFromImage(catheter_contour_for_spline)
                    if np.any(np.logical_and(cather_w_solo_comps > 0, all_catheter_contours_array == self.catheter_marker_class)):
                        # Overlap should not happen at the tip.
                        warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
                    else:
                        all_classes_array = np.where(cather_w_solo_comps > 0, cather_w_solo_comps, all_classes_array)

                # Saving what was done during postprocessing
                postprocessed_info = self.postprocessed_infos["The entry tip and the tip were added to the catheter"]
                self.logger.log([f"Catheter {cat_idx}: {postprocessed_info} \n"])

        processed_catheter = sitk.GetImageFromArray(all_classes_array)
        processed_catheter.CopyInformation(catheter_contour)
        return processed_catheter, postprocessed_info

    def potential_complete_part(
            self, catheter_contour:sitk.Image, all_classes_array:np.ndarray, 
            catheter_contours:sitk.Image, class_idx:int, part_size:float, 
            offset:float=0.0, cat_idx:int=0, grab_solo_components:bool=False):
        """
        Given a part index and a size, we check the parts size and if it 
        is not big enough we complete it tot he given size.
        """

        assert class_idx in [self.tip_marker_class, self.entry_tip_marker_class], """
        class_idx should be either the tip marker class or the entry tip marker class.
        We don't know the legnth of a catheter core. (ie catheter core class can be of 
        any size). """

        if not self.multi_class:
            raise NotImplementedError(
                """This function is only implemented for multi class catheter 
                segmentation and makes no sense for single catheter core class
                since we need to know the size of the part to be completed.
                """)
        all_catheter_contours_array = sitk.GetArrayFromImage(catheter_contours)
        if grab_solo_components:
            catheter_contour_for_spline, _, t_solo_components = self._grab_isolated_components(
                catheter_contour, catheter_contours, cat_idx, 
                # We add 2 to the tip size to in case there is an angle between the catheter core and
                # the tip that could prevent the add on part o find solo components.
                search_space=self.entry_tip_size + self.tip_marker_size + 2, 
                overlapping_classes=[self.entry_tip_marker_class, self.tip_marker_class])
            input_spline_creator = catheter_contour_for_spline
        else:
            input_spline_creator = catheter_contour
            t_solo_components = []

        # Create the spline from the contour we have
        expander = ContourExpander(
            catheter_contour=input_spline_creator, catheter_diameter=self.catheter_diameter, 
            multi_class=self.multi_class, save_details=self.save_details, 
            save_details_path=self.save_details_path)
        needle_spline_creator = expander.needle_spline_creator

        if self.save_details:
            # Saving markers used to postprocess
            create_slicer_markup_points(
                os.path.join(self.save_details_path, f"spline_fitting_center_points_catheter_{cat_idx}.mrk.json"), 
                needle_spline_creator.original_central_points, 
            )
            create_slicer_markup_points(
                os.path.join(self.save_details_path, f"rotated_center_points_{cat_idx}.mrk.json"), 
                needle_spline_creator.rotated_center_points
                )
            create_slicer_markup_points(
                os.path.join(self.save_details_path, f"og_catheter_pts_rotated_{cat_idx}.mrk.json"), 
                needle_spline_creator.original_catheter_pts_rotated
                )
            sitk.WriteImage(
                input_spline_creator, 
                os.path.join(self.save_details_path, f"Input_to_spline_creator_{cat_idx}.seg.nrrd"),
                useCompression=True
            )
            sitk.WriteImage(
                needle_spline_creator.rotated_needle, 
                os.path.join(self.save_details_path, f"rotated_needle_{cat_idx}.seg.nrrd"),
                useCompression=True
            )
            sitk.WriteImage(
                needle_spline_creator.needle_contour, 
                os.path.join(self.save_details_path, f"needle_used_to_fit_spline_{cat_idx}.seg.nrrd"),
                useCompression=True
            )
            if self.reference_ct is not None:
                ct_cropped_around_needle = sitk_crop(sitk_crop(self.reference_ct, self.bounding_box), 
                                                    needle_spline_creator.bounding_box)
                sitk.WriteImage(
                    ct_cropped_around_needle,
                    os.path.join(self.save_details_path, f"ct_cropped_around_needle_{cat_idx}.nrrd"),
                    useCompression=True
                )
                sitk.WriteImage(
                    needle_spline_creator.rotate_volume(ct_cropped_around_needle, interpolator=sitk.sitkLinear),
                    os.path.join(self.save_details_path, f"rotated_ct_needle_{cat_idx}.nrrd"),
                    useCompression=True
                )
        # Identifying ts of the spline to know which side to process the catheter for.
        extremum0, t0_spline, extremum1, t1_spline = get_segment_endpoints_and_t(
            catheter_contour, needle_spline_creator)

        # Project the class closest and distal point on the spline
        extremum_added_part0, t0_added_part, extremum_added_part1, t1_added_part = get_segment_endpoints_and_t(
            catheter_contour, needle_spline_creator, needle_idx=class_idx, 
            longitudinal=(False if class_idx == self.tip_marker_class else True))
        
        # Getting the side of the class: which t is it the closest to 0 or 1
        # to know in which direction extend the class marker

        # Check for distance between the two points
        length_existing_part = needle_spline_creator.distance_on_spline(extremum_added_part0, extremum_added_part1)

        ## offset
        # If the distance is 1cm or more, we are good for the class.
        # 9mm since there are half a voxel on each end voxel
        # and we compare center voxels to each other.
        # if we are completing the tip on the other end the ground truth contours
        # have been dilated so they are probably more than 3mm.
        if length_existing_part >= part_size + offset:
            if np.abs(t0_spline - t0_added_part) < np.abs(t1_spline - t0_added_part): 
                # t0_spline is closer to the class than t1_spline
                if np.abs(t0_spline - t0_added_part) < np.abs(t0_spline - t1_added_part):
                    # t0_added_part is closer t0_spline than t1_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = extremum_added_part0, t0_added_part
                else:
                    # t1_added_part is closer t0_spline than t0_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = extremum_added_part1, t1_added_part
            else:
                # t1_spline is closer to the class than t0_spline
                if np.abs(t1_spline - t0_added_part) < np.abs(t1_spline - t1_added_part):
                    # t0_added_part is closer t1_spline than t1_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = extremum_added_part0, t0_added_part
                else:
                    # t1_added_part is closer t1_spline than t0_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = extremum_added_part1, t1_added_part
            
            # Saving what was done during postprocessing
            postprocessed_info = None

        # If the distance is less than 1cm, we need to increase the length of the class
        else:
            if self.save_details:
                co = "tip" if class_idx == self.tip_marker_class else "entry tip"
                print("Correcting {} for catheter {}".format(co, cat_idx))
            # offset can be useful since we dilate the groundtruth contour or to make up 
            # for half voxel lost on each side of the voxel.
            size_part_to_add = part_size + offset - length_existing_part

            # If the class is closer to the first point of the catheter, we need to extend the class
            # in the direction of the first point of the catheter
            if np.abs(t0_spline - t0_added_part) < np.abs(t1_spline - t0_added_part):
                # t0_spline is closer to the class than t1_spline
                if np.abs(t0_spline - t0_added_part) < np.abs(t0_spline - t1_added_part):
                    # t0_added_part is closer t0_spline than t1_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = self._get_segment_coords_on_spline(
                        extremum_added_part0, t0_added_part, size_part_to_add, needle_spline_creator)
                    old_end_added_part = extremum_added_part1
                else:
                    # t1_added_part is closer t0_spline than t0_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = self._get_segment_coords_on_spline(    
                        extremum_added_part1, t1_added_part, size_part_to_add, needle_spline_creator)
                    old_end_added_part = extremum_added_part0
            else:
                # t1_spline is closer to the class than t0_spline
                if np.abs(t1_spline - t0_added_part) < np.abs(t1_spline - t1_added_part):
                    # t0_added_part is closer t1_spline than t1_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = self._get_segment_coords_on_spline(
                        extremum_added_part0, t0_added_part, size_part_to_add, needle_spline_creator)
                    old_end_added_part = extremum_added_part1
                else:
                    # t1_added_part is closer t1_spline than t0_added_part
                    projected_new_end_added_part, projected_new_end_added_part_t = self._get_segment_coords_on_spline(
                        extremum_added_part1, t1_added_part, size_part_to_add, needle_spline_creator)
                    old_end_added_part = extremum_added_part0

            # We create the contour between the two defined points
            addon_added_part = expander._create_addon(
                point1=projected_new_end_added_part, point2=old_end_added_part, 
                dilation_add_on=self.contour_dilation, spline=True)
            if np.any(np.logical_and(addon_added_part > 0, all_catheter_contours_array == self.catheter_marker_class)):
                # Overlap should not happen at the tip.
                warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
            else:
                all_classes_array = np.where(addon_added_part, class_idx, all_classes_array)

            if self.save_details:
                create_slicer_markup_points(
                    os.path.join(self.save_details_path, f"new_added_part_enpoints_catheter_{cat_idx}.mrk.json"), 
                    [projected_new_end_added_part, old_end_added_part], 
                )
            # Saving what was done during postprocessing
            if class_idx == self.entry_tip_marker_class:
                postprocessed_info = self.postprocessed_infos["The entry tip was completed to full length and the tip was added to the catheter"]
            else:
                postprocessed_info = self.postprocessed_infos["The tip was completed to full length"]
            self.logger.log([f"Catheter {cat_idx}: {postprocessed_info} \n"])

        return (all_classes_array, projected_new_end_added_part, projected_new_end_added_part_t, 
                expander, input_spline_creator, t_solo_components, postprocessed_info)

    def _add_tip_entry_tip(self, expander:ContourExpander, t_start:float, 
        start_pt_coords:np.ndarray, all_classes_array:np.ndarray, 
        all_catheter_contours_array:np.ndarray,
        spline:bool=False, dilation_add_on:int=0, cat_idx:int=0):
        # Adding the entry tip at t_start
        addon_entry_tip, projected_end_entry_tip, projected_end_entry_tip_t = expander.add_part(
                t_start=t_start, start_pt_coords=start_pt_coords,
                # step is entrytip size even if dilated due to the way the ground truth 
                # contours are created
                step=self.entry_tip_size, dilation_add_on=dilation_add_on, 
                spline=spline
            )
        if np.any(np.logical_and(addon_entry_tip!=0, all_catheter_contours_array == self.catheter_marker_class)):
            # Overlap should not happen at the tip.
            warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
        else:
            all_classes_array = np.where(addon_entry_tip, self.entry_tip_marker_class, all_classes_array)
        
        # Adding the tip at t_start
        addon_tip, _, _ = expander.add_part(
            t_start=projected_end_entry_tip_t, start_pt_coords=projected_end_entry_tip,
            # Finding the endpoint for the tip marker, 3mm from the end of the entry tip marker
            step=(self.tip_marker_size - 1 + self.contour_dilation), 
            dilation_add_on=dilation_add_on, spline=spline)
        if np.any(np.logical_and(addon_tip!=0, all_catheter_contours_array == self.catheter_marker_class)):
            # Overlap should not happen at the tip.
            warnings.warn(f"What you want to add overlaps with existing catheter {cat_idx}, we don't add it.")
        else:
            all_classes_array = np.where(addon_tip, self.tip_marker_class, all_classes_array)

        return all_classes_array, addon_tip, addon_entry_tip

    def _get_segment_coords_on_spline(
            self, point_coords:np.ndarray, point_t:float, 
            step:float, needle_spline_creator:NeedleSplineCreator):
        bmin , bmax = get_bounds_for_step(point_t)
        projected_end, projected_end_t = needle_spline_creator.step_in_spline(
            point_coords,
            step=step,
            bound_min=bmin,
            bound_max=bmax,
            arc=True
        )
        return projected_end, projected_end_t
    
    @staticmethod
    def _create_density_distribution(data:np.ndarray):
        # Define the bins for histogram
        bound_min = -1000
        bound_max = 1000
        step = 25
        bins = np.linspace(bound_min, bound_max, int((bound_max - bound_min)/step) + 1)

        # Compute histograms
        hist, _ = np.histogram(data, bins=bins, density=True)
        epsilon = 1e-10
        hist += epsilon

        hist /= np.sum(hist)
        return hist

    def _get_connected_components(self, contour_array:np.ndarray, dilated:bool=False):
        return self.connected_components, self.connected_comp_num_labels

    def _grab_isolated_components(
            self, catheter_contour:sitk.Image, catheter_contours:sitk.Image=None, 
            cat_idx:int=None, dilation_add_on:int=3, overlapping_classes:List[int]=None, 
            search_space:float=None):
        """
        We create a first spline with the pincipal components of the catheter.
        We extend this spline a bit on each side.
        If the extended part overlap with another connected component, we include it in the 
        catheter contour.
        """

        all_classes_array = sitk.GetArrayFromImage(catheter_contour).astype(np.uint8)

        addon_entry_tip_t0, addon_entry_tip_t1, t0_spline, t1_spline = extend_catheter_contour_on_both_sides(
            catheter_contour=catheter_contour, catheter_diameter=self.catheter_diameter, 
            dilation_add_on=dilation_add_on, search_space=search_space, multi_class=self.multi_class)

        # Reaching out for isolated connected components 
        t_of_found_solo_components = []
        found_components_connected = []
        catheter_contours_array = sitk.GetArrayFromImage(catheter_contours).astype(np.uint8)
        connected_comp, connected_comp_num_labels = self._get_connected_components(contour_array=None)
        for connected_comp_idx in range(1, connected_comp_num_labels + 1):
            if self.dilated_components:
                # 1s at the non dilated contour and at the connected component
                needle_array = np.array(
                    np.logical_and(
                        connected_comp == connected_comp_idx, catheter_contours_array != 0 
                    ),
                    dtype=np.uint8,
                )
            else:
                needle_array = np.array(
                    connected_comp == connected_comp_idx, dtype=np.uint8
                )

            # Check for overlap with the extended part of the catheter
            for overlapping_cl in overlapping_classes:
                # If extension found a solo component
                if np.any(np.logical_and(needle_array, addon_entry_tip_t0)):
                    if np.any(addon_entry_tip_t0 * catheter_contours_array == overlapping_cl):
                        # The solo component class was 2 or 3 so the tip is here
                        print("FOUND CONNECTED COMPONENT ON T0 SIDE BELONGING TO TIP")
                        all_classes_array = np.where(needle_array, catheter_contours_array, all_classes_array)
                        if not t0_spline in t_of_found_solo_components:
                            t_of_found_solo_components.append(t0_spline)
                        found_components_connected.append(connected_comp_idx)

                if np.any(np.logical_and(needle_array, addon_entry_tip_t1)):
                    if np.any(addon_entry_tip_t1 * catheter_contours_array == overlapping_cl):
                        # The solo component class was 2 or 3 so the tip is here
                        print("FOUND CONNECTED COMPONENT ON T1 SIDE")
                        all_classes_array = np.where(needle_array, catheter_contours_array, all_classes_array)
                        if not t1_spline in t_of_found_solo_components:
                            t_of_found_solo_components.append(t1_spline)
                        found_components_connected.append(connected_comp_idx)
                      
        # This new contour is only 1s but this is just for spline creation so we don't care.
        new_contour_w_isolated_components = sitk.GetImageFromArray(all_classes_array)
        new_contour_w_isolated_components.CopyInformation(catheter_contour)
        if self.save_details:
            sitk.WriteImage(
                new_contour_w_isolated_components,
                os.path.join(self.save_details_path, f"processed_catheter_{cat_idx}_w_isolated_components.seg.nrrd"),
                useCompression=True)
        return new_contour_w_isolated_components, found_components_connected, t_of_found_solo_components

if __name__ == "__main__":
    import time
    import os

    dataset_name = "Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1" # "Dataset004_catheters_and_tip_markers" #"Dataset006_catheters_and_tip_makers_diameter_2.0_dilation_1"
    if "dilation" in dataset_name:
        contour_dilation = int(dataset_name.split("dilation_")[1].split("_")[0])
    else:
        contour_dilation = 0
    bs8 = True
    if bs8:
        bs8_suffix = '_bs8_threshold4'
    else:
        bs8_suffix = ''
    benchmark = "test_fold01234" # "test_fold01234" # "val_benchmark_fold0" # "train_benchmark_fold01234" # "val_benchmark"
    data_folder = f"/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/{dataset_name+bs8_suffix}/{benchmark}"
    patient_id = "971621" # "45474" "79206" "119127" "771143" "259984" "1289500" 
    catheters_contour_path = os.path.join(data_folder, patient_id, "ai_generated_catheters.seg.nrrd")
    reference_ct = None #sitk.ReadImage(os.path.join(data_folder, patient_id,"ct.nrrd"))
    t0 = time.time()
    post_processor = CatheterPostProcessor(
        reference_ct=reference_ct, save_details=True, 
        save_details_path=os.path.join(data_folder, patient_id, "post_processed_catheters/"), 
        contour_dilation=contour_dilation, log_path=os.path.dirname(catheters_contour_path),
        )
    processed_contour, _, post_processed_infos,separator_infos = post_processor.postprocess_catheters(catheters_contour_path=catheters_contour_path)
    print("It takes {} seconds to post process a contour seeebb.".format(time.time() - t0))
    print("writing the image here ", os.path.join(data_folder, patient_id, "ai_generated_catheters_postprocessed_new2.seg.nrrd"))
    print("post_processed_infos", post_processed_infos)
    print("separator_infos", separator_infos)
    assert np.any(sitk.GetArrayFromImage(processed_contour) == post_processor.catheter_marker_class), (
        "The processed contour should contain catheter markers"
    )
    for c in np.unique(sitk.GetArrayFromImage(processed_contour)):
        print("Class ", c, "has ", np.sum(sitk.GetArrayFromImage(processed_contour) == c), "voxels.")
    sitk.WriteImage(
        processed_contour, 
        os.path.join(data_folder, patient_id, "ai_generated_catheters_postprocessed_new2.seg.nrrd"),
        useCompression=True)
