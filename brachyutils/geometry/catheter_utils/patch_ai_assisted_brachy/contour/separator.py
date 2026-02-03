import os
import copy
from itertools import combinations
import time
import tqdm
from typing import List, Tuple
import warnings

import SimpleITK as sitk
import multiprocessing as mp
import numpy as np
from scipy import ndimage

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.log import Logger
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.creator import extrapolate_point
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.pw_linear_interpolator import Segment
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.falsifier import NeedleFalsifier
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.spline_interpolator import NeedleSplineCreator
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import get_physical_coord_for_needle, distance, describe_array
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import sitk_crop, crop_around_mask
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import create_slicer_markup_points

### These functions are out of any class to enable efficient multiprocessing
def fit_spline(neelde_contour: sitk.Image, multiclass:bool=True):
    needle_spline_creator = NeedleSplineCreator(neelde_contour, multiclass=multiclass)
    needle_spline_creator.interpolate_spline()
    return needle_spline_creator


def get_distances_to_spline(neelde_contour: sitk.Image, multiclass:bool=True):
    """
    Fits a spline from a contour, then compute the distance from
    every voxel of the contour to the spline.
    """
    needle_spline_creator = fit_spline(neelde_contour, multiclass)
    contour_voxel_positions = get_physical_coord_for_needle(neelde_contour)
    distances_to_spline = []
    for voxel_pos in contour_voxel_positions:
        _, _, distance_to_spline = needle_spline_creator.project_on_spline(voxel_pos)
        distances_to_spline.append(distance_to_spline)
    return distances_to_spline


def detect_overlapping_catheter_areas(
    component:np.ndarray, catheters_contour:sitk.Image, 
    catheter_shaft_threshold:float=4.0, break_on_find:bool=False, 
    multiclass:bool=False):
    """
    After fitting a spline on the solo component, if the spline fitting is not
    inside the contour of the component, that means there are two separate parts
    on the component. Since the component is connected, these two parts join 
    at some point. Volume is expected to be smooth (AI generated and not 
    piece wise linear mnually created).
    """

    component_img = sitk.GetImageFromArray(component)
    component_img.CopyInformation(catheters_contour)

    # We need to find the points where the spline is not inside the contour
    # of the component. We can do this by checking the distance between the
    # spline and the contour. If the distance is larger than a certain threshold
    # we consider that the spline is not inside the contour.
    
    overlapping = False

    # Creating the spline
    needle_spline_creator = fit_spline(component_img, multiclass)

    # Getting the physical coordinates of the points segmented as a catheter
    contour_voxel_positions = get_physical_coord_for_needle(component_img)

    # Projecting those points on the spline and checking distance between the
    # projected point and the original point.
    distances_to_spline = []
    point_witin_catheter_shaft = []
    for voxel_pos in contour_voxel_positions: 
        _, _, distance_to_spline = needle_spline_creator.project_on_spline(voxel_pos)
        distances_to_spline.append(distance_to_spline)
        if distance_to_spline > catheter_shaft_threshold:
            # We have an overlapping catheter
            # We need to separate the catheter in two parts
            overlapping = True
            if break_on_find:
                break
        else:
            point_witin_catheter_shaft.append(voxel_pos)

    return overlapping, point_witin_catheter_shaft, needle_spline_creator, distances_to_spline

def detect_overlap_task(args:Tuple[int, np.ndarray, sitk.Image, bool]):
    connected_comp_idx, labeled_mask, catheters_contour, multiclass = args 
    component_array = np.array(
                labeled_mask == connected_comp_idx, dtype=np.uint8
            )
    overlapping, overlapping_pts, needle_spline_creator, distances_to_spline = detect_overlapping_catheter_areas(
        component_array, catheters_contour, break_on_find=True, multiclass=multiclass
    )
    return overlapping, overlapping_pts, needle_spline_creator, distances_to_spline, connected_comp_idx

def extend_catheter_contour_on_both_sides(
        catheter_contour:sitk.Image, 
        catheter_diameter:float, dilation_add_on:int=3,
        search_space:float=None, spline:bool=False, 
        multi_class:bool=True):
    expander = ContourExpander(catheter_contour=catheter_contour, catheter_diameter=catheter_diameter, multi_class=multi_class)
    addon_entry_tip_t0, addon_entry_tip_t1, t0_spline, t1_spline = expander._extend_catheter_contour_on_both_sides(
        dilation_add_on, search_space, spline
    )
    return addon_entry_tip_t0, addon_entry_tip_t1, t0_spline, t1_spline

def extend_catheter_contour_on_both_sides_wrapper( args:Tuple[sitk.Image, int, int, float, bool]):
    needle, diameter, old_comp_idx, dilation, search_space, spline, multi_class = args
    addon_entry_tip_t0, addon_entry_tip_t1, _, _ = extend_catheter_contour_on_both_sides(
            catheter_contour=needle, catheter_diameter=diameter, dilation_add_on=dilation, 
            search_space=search_space, spline=spline, multi_class=multi_class
    )
    dilated_add_on = addon_entry_tip_t0 + addon_entry_tip_t1
    return dilated_add_on, old_comp_idx, needle

def get_bounds_for_step(t_used:float, margin:float=0.5):
    """
    t_used represents the t value of the spline
    """
    if abs(t_used - 0) < abs(t_used - 1):
        bound_min = t_used - margin
        bound_max = t_used 
    else:
        bound_min = t_used 
        bound_max = t_used + margin
    return bound_min, bound_max

def get_segment_endpoints_and_t(
            catheter_contour:sitk.Image, needle_spline_creator:NeedleSplineCreator, 
            needle_idx:int=None, longitudinal:bool=True):
    """
    Getting the points at boths ends of a part in the catheter (needle_idx class).
    If that part is longitudinal like the catheter core, or entry tip, getting the 
    farthest voxels is okay to project on the spline and get their corresponding ts. 
    However if the class is another shape like a square-ish for the tip marker, 
    then the farthest voxels of the class might not correspond to the farthest 
    away projected points on the spline. In that case we just get all the ts and take 
    the farthest away ts and corresponding projected points.        
    """
    catheter_pt_coords = get_physical_coord_for_needle(catheter_contour, needle_idx=needle_idx)
    assert len(catheter_pt_coords) > 0, "No points found in the catheter contour"
    if longitudinal:
        seg = Segment(
            catheter_pt_coords,
            ref_slice_coord=None,
            interslice_ax=2,
            init_line=False,
            init_2D=False,
        )
        extremum_catheter_pt_coords = seg.extremum_points
        extremum_catheter_pt_coords_on_spline = [
            needle_spline_creator.project_on_spline(pt) for pt in extremum_catheter_pt_coords]
        point0 = extremum_catheter_pt_coords_on_spline[0][0]
        point1 = extremum_catheter_pt_coords_on_spline[1][0]
        t0_spline = extremum_catheter_pt_coords_on_spline[0][1]
        t1_spline = extremum_catheter_pt_coords_on_spline[1][1]
    else:
        all_t = []
        all_projected_points = []
        for pt in catheter_pt_coords:
            projected_pt, t, _ = needle_spline_creator.project_on_spline(pt)
            all_t.append(t)
            all_projected_points.append(projected_pt)
        point0 = all_projected_points[np.argmin(all_t)]
        point1 = all_projected_points[np.argmax(all_t)]
        t0_spline = np.min(all_t)
        t1_spline = np.max(all_t)

    return point0, t0_spline, point1, t1_spline


class ContourSeparator:

    """
    This class is here to separate the catheters segmentation. The catheters are gathetered in a single
    class but to digitize the catheters we need only specifics catheters one at a time.
    This class also detects overlap in catheters.
    """

    def __init__(
        self, catheters_contour_path:str, reference_ct:sitk.Image=None,
        catheter_marker_class:int=1, catheter_diameter:float=2.0, 
        margin_for_cropping:int=40, save_details:bool=False, 
        save_details_path:str=None,
        multiprocess:bool=True, log_path:str=None, 
        log_file_name:str='separator_logs.txt'):

        self.reference_ct = reference_ct
        self.cropped_ref_ct = None
        self.catheters_contour_path = catheters_contour_path
        self.catheter_marker_class = catheter_marker_class
        self.multi_class = True
        self.catheter_diameter = catheter_diameter
        self.og_connected_components = None
        self.modified_connected_components = None
        self.og_connected_comp_num_labels = None
        self.modified_connected_comp_num_labels = None
        self.dilated_og_components = None
        self.found_connected_comp_pairs = None
        self.bounding_box = None
        self.save_details = save_details
        if save_details:
            os.makedirs(save_details_path, exist_ok=True)
        self.save_details_path = save_details_path
        self.margin_for_cropping = margin_for_cropping
        self.multiprocess = multiprocess
        self.count_merged_catheters = 0
        self.num_overlaps = 0

        if self.save_details_path:
            os.makedirs(self.save_details_path, exist_ok=True)

        catheters_contour = sitk.ReadImage(self.catheters_contour_path)
        self.catheters_contour, self.bounding_box = crop_around_mask(catheters_contour, margin_mm=self.margin_for_cropping)
        self.contour_array = sitk.GetArrayFromImage(self.catheters_contour).astype(np.uint8)
        self.modified_contour_array = None
        if self.reference_ct is not None:
            self.cropped_ref_ct = sitk_crop(self.reference_ct, self.bounding_box)
            if save_details:
                sitk.WriteImage(sitk_crop(self.reference_ct, self.bounding_box),
                                os.path.join(self.save_details_path, "ct_cropped_around_catheters.nrrd"),
                                useCompression=True)
        
        assert log_path is not None, "Please provide a path to save the logs."
        self.logger = Logger(log_path, log_file_name)
    

    def _get_connected_components(self, contour_array:np.ndarray, dilated:bool=False):
        # Label connected components in the mask
        if self.og_connected_components is None:
            s = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
            if dilated:
                dilated_contour_array = ndimage.binary_dilation(contour_array).astype(
                    contour_array.dtype
                )
                labeled_mask, num_labels = ndimage.label(dilated_contour_array, structure=s)
            else:
                labeled_mask, num_labels = ndimage.label(contour_array, structure=s)
            self.og_connected_components = labeled_mask
            self.og_connected_comp_num_labels = num_labels
            self.dilated_og_components = dilated

        return self.og_connected_components, self.og_connected_comp_num_labels
    
    def separate_catheters(
            self, nb_needles: int = None, 
            dilated: bool = False, all_class_necessary:bool=False, 
            check_classes_in_comp:bool=False, save_solo_components:bool=False, 
            potential_merge:bool=True, min_size:int=5, verbose:bool=False):
        """
        Separate the needles from the contour file.

        Parameters
        ----------
        sitk_needles : SimpleITK.Image
            SimpleITK image containing the needles.

        Returns
        -------
        list
            List of SimpleITK images, each containing a single needle.
        """
        # If the contour provided is made of different classes, we need to binarize it 
        # to separate the different catheters.
        if self.multi_class:
            # Every class becomes 1.
            contour_array = np.where(self.contour_array > 0, 1, 0).astype(np.uint8)
            if check_classes_in_comp:
                all_classes = np.unique(self.contour_array)
        else:
            contour_array = self.contour_array

        assert np.any(contour_array), "There should be needle voxels in the mask"
        # Label connected components in the mask

        labeled_mask, num_labels = self._get_connected_components(
            contour_array, dilated=dilated)

        if self.save_details:
            img = sitk.GetImageFromArray(labeled_mask)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(self.save_details_path, "og_connected_components_map_from_contour.seg.nrrd"),
                True
                )

        ### Correcting for potential overlapping catheters
        overlap, overlapping_comps, _ = self._detect_overlap(
            labeled_mask, num_labels, self.catheters_contour, min_size=min_size)
        self.num_overlaps = len(overlapping_comps)

        if overlap:
            t0 = time.time()
            if self.save_details:
                overlap_comps = ("_").join([str(comp) for comp in overlapping_comps])
                img = sitk.GetImageFromArray(labeled_mask)
                img.CopyInformation(self.catheters_contour)
                sitk.WriteImage(
                    img, 
                    os.path.join(self.save_details_path, f"connected_comps_whose_comp{overlap_comps}_contains_overlapping_catheters.seg.nrrd"),
                    True
                    )
                
            print("Overlaps detected for contour : ", self.catheters_contour_path, "and components: " , overlapping_comps)
            self.logger.log([f"{self.num_overlaps} overlaps detected for components: {overlapping_comps} \n"])
            for overlap_idx, overlapping_comp in enumerate(overlapping_comps):
                # Overwriting labeled_mask, num_labels
                labeled_mask, num_labels = self._solve_potential_overlap(
                    labeled_mask, num_labels, overlapping_comp)
                if self.save_details:
                    img = sitk.GetImageFromArray(labeled_mask)
                    img.CopyInformation(self.catheters_contour)
                    sitk.WriteImage(
                        img, 
                        os.path.join(self.save_details_path, f"connected_comps_after_solving_{overlap_idx+1}_overlaps.seg.nrrd"),
                        True
                        )
            print("Overlaps solved in ", time.time() - t0, "s.")
        else:
            if self.save_details:
                print("No overlap detected for contour : ", self.catheters_contour_path)

        assert num_labels != 0, "There should be needle voxels in the mask"
        if nb_needles is not None:
            assert (
                num_labels == nb_needles
            ), "There should be only one connected component per needle"
        assert (
            num_labels < 100
        ), "There cannot be more than 100 needles inside the patients, your contours might be represented by too many connected components"
        
        needles = []
        needles_comp_idx = []
        if save_solo_components:
            solo_components_array = np.zeros_like(self.contour_array, dtype=np.uint8)
        
        for connected_comp_idx in range(1, num_labels + 1):
            if dilated:
                # 1s at the non dilated contour and at the connected component
                needle_array = np.array(
                    np.logical_and(
                        labeled_mask == connected_comp_idx, contour_array != 0
                    ),
                    dtype=np.uint8,
                )
            else:
                needle_array = np.array(
                    labeled_mask == connected_comp_idx, dtype=np.uint8
                )
            if save_solo_components:
                component_array = connected_comp_idx * needle_array
                solo_components_array += component_array.astype(np.uint8)

            # Removing needles that are not complete
            if check_classes_in_comp:
                unique_classes_in_needle = np.concatenate(
                    [np.array([0]), 
                    np.unique(needle_array[needle_array > 0] * self.contour_array[needle_array > 0])]
                    )
                if all_class_necessary:
                    if not np.all(np.isin(all_classes, unique_classes_in_needle)):
                        print(f"""Needle {connected_comp_idx}, with {np.sum(needle_array)} voxels, does not include all components 
                              of a needle but only {unique_classes_in_needle}, not considered.""")
                        continue
                # Need at least class 1 and 2
                else:
                    if not (1 in unique_classes_in_needle and 2 in unique_classes_in_needle):
                        print(f"""Needle {connected_comp_idx}, with {np.sum(needle_array)} voxels, does not include both 
                              tip and catheter but only {unique_classes_in_needle}, not considered.""")
                        continue
                    else:
                        if verbose:
                            print(f"We keep needle {connected_comp_idx}, with {np.sum(needle_array)} voxels")

            # Removing needles that are too small
            if np.sum(needle_array) < min_size:
                # In general a needle should have between 500 to 2000 voxels
                print(f"Needle {connected_comp_idx}, with {np.sum(needle_array)} voxels, is too small, not considered.")
                continue
            else:
                if self.multi_class:
                    needle_array = needle_array * self.contour_array
                needle = sitk.GetImageFromArray(needle_array)
                needle.CopyInformation(self.catheters_contour)
                needles.append(needle)
                needles_comp_idx.append(connected_comp_idx)
                
        if save_solo_components:
            components_img = sitk.GetImageFromArray(solo_components_array)
            components_img.CopyInformation(self.catheters_contour)

        # Adapting the labeled mask to the removal of small components
        labeled_mask, num_labels = self._handle_small_components_removal(labeled_mask, needles_comp_idx)
        if potential_merge:
            ### Mering two part of the same catheters in case some have been separated
            # Overwriting labeled_mask, num_labels
            merger = ContourMerger(labeled_mask, num_labels, self.catheters_contour, 
                                   needles, self.catheter_diameter, self.catheter_marker_class,
                                   self.margin_for_cropping, self.multi_class, self.multiprocess,
                                   self.save_details, (self.save_details_path.replace("post_processed_catheters", "preprocessing_merger")
                                                       if self.save_details else None))
            merger.merge_catheters(search_space=self.margin_for_cropping)
            labeled_mask, num_labels = merger.get_labeled_mask()
            needles = merger.get_catheters_list()
            self.count_merged_catheters = merger.get_count_merged_catheters()
            if self.count_merged_catheters > 0:
                print(f"{self.count_merged_catheters} catheters have been merged.")
                self.logger.log([f"{self.count_merged_catheters} catheters have been merged. \n"])
            self.modified_contour_array = merger.get_modified_contour_array()

        # Saving the modified version of connected components map as attributes
        self.modified_connected_components = labeled_mask
        self.modified_connected_comp_num_labels = num_labels

        if save_solo_components:
            return needles, components_img
        else:
            return needles

    def _handle_small_components_removal(self, labeled_mask:np.ndarray, needles_comp_idx:List[int]):
        """
        Remove small components from the connected components map.
        """
        new_labeled_mask = np.zeros_like(labeled_mask)
        for new_lab, old_lab in zip(range(1, len(needles_comp_idx)+1), needles_comp_idx):
            new_labeled_mask = np.where(labeled_mask == old_lab, new_lab, new_labeled_mask)
            if self.save_details:
                print(f"Needle {old_lab} has been renumbered to {new_lab}")
        new_num_labels = len(needles_comp_idx)
        if self.save_details:
            img = sitk.GetImageFromArray(new_labeled_mask)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(self.save_details_path, "labeled_mask_after_removal_of_small_components.seg.nrrd"),
                True
                )
        return new_labeled_mask, new_num_labels


    def _detect_overlap(self, labeled_mask:np.ndarray, num_labels:int, catheters_contour:sitk.Image, min_size:int=50):
        """
        Detect potential overlapping catheters. Only works with realistic contours (
        can be AI) and not piece wise linear pseudo contours.
        """
        any_overlap = False
        overlapping_comp = []
        distances_to_splines = []
        if self.multiprocess:
            args = []
            for lab in range(1, num_labels + 1):
                if np.sum(labeled_mask == lab) <= min_size:
                    continue
                args.append(
                    (lab, labeled_mask, catheters_contour, self.multi_class)
                )
            
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    message="The required storage space exceeds the available storage space.",
                    category=RuntimeWarning, module="scipy")
                with mp.Pool(8) as p:
                    r = list(
                        tqdm.tqdm(
                            p.imap(
                                detect_overlap_task,
                                args,
                            ),
                            total=num_labels,
                            desc="Checking for overlap...",
                        )
                    )

            for overlap, _, _, distances_to_spline, connected_comp_idx in r:
                if overlap:
                    any_overlap = True
                    overlapping_comp.append(connected_comp_idx)
                    distances_to_splines.append(distances_to_spline)

        else:
            for lab in tqdm.tqdm(range(1, num_labels + 1), total=num_labels, desc="Checking for overlap..."):
                if np.sum(labeled_mask == lab) <= min_size:
                    continue
                overlapping, _, _, distances_to_spline, _ = detect_overlap_task(
                    (lab, labeled_mask, catheters_contour, self.multi_class)
                    )
                
                # If overlaping then we need to separate the catheters
                if overlapping:
                    any_overlap = True
                    overlapping_comp.append(lab)
                    distances_to_splines.append(distances_to_spline)

        return any_overlap, overlapping_comp, distances_to_splines

    def _solve_potential_overlap(
            self, labeled_mask:np.ndarray, num_labels:int, component_with_catheter_overlap:int, 
            threshold_increment:float=0.1):
        """
        Solve potential overlapping catheters. Check distance between the contour voxels and the spline fitted
        in the midle of the two catheters. Remove the overlapping voxels which are the ones closer to the 
        spline. The distance to the spline threshold for the removal is gradually increased until the 
        overlapping area is remove from the component and the component is separated in different parts.
        This different parts are then merged based on the best combination of parts that results in the 
        smallest average distance between the voxels belonging to a catheter and the corresponding 
        created splines.
        """
        ### Creating sitk image of the overlapping catheters contour
        overlapping_comp_array = np.array(labeled_mask == component_with_catheter_overlap, dtype=np.uint8)
        temp_overlapping_comp_array = np.copy(overlapping_comp_array)
        separated_components = []
        only_single_catheter_parts = False
        steps = 0
        needle_spline_creator = None
        if self.save_details:
            img = sitk.GetImageFromArray(overlapping_comp_array)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(
                    self.save_details_path, 
                    f"connected_comp{component_with_catheter_overlap}_before_trying_to_solve_overlap.seg.nrrd"
                    ),
                True
                )
        
        start_distance_to_spline_threshold = threshold_increment
        ### Separating the group of catheters into multiple parts.
        while not only_single_catheter_parts:
            only_single_catheter_parts = True
            components, num_lab, needle_spline_creator, final_threshold_distance_to_spline = self._remove_catheter_shaft_until_separation(
                temp_overlapping_comp_array, start_distance_to_spline_threshold=start_distance_to_spline_threshold, 
                threshold_increment=threshold_increment,
                # We reuse the spline created from the og pair of catheters to separate. 
                # Important not to refit a plsine every time.
                needle_spline_creator=needle_spline_creator
                )
            # Resetting the components to separate
            temp_overlapping_comp_array = np.zeros_like(temp_overlapping_comp_array)
            # For each component we check if this is only a single catheter or not
            count_overlaps = 0
            for lab in range(1, num_lab + 1):
                component = np.array(components == lab, dtype=np.uint8)
                overlapping, _, _, distances_to_spline = detect_overlapping_catheter_areas(
                    component, self.catheters_contour, break_on_find=True, multiclass=False)
                if overlapping:
                    # We will continue in the while loop until all components are catheter parts
                    only_single_catheter_parts = False
                    # Keeping this component which remains to be separated.
                    temp_overlapping_comp_array += component
                    count_overlaps += 1
                else:
                    # We save this component for later
                    separated_components.append(component)

            # Parts have been separated into separately overlapping parts
            # The separation must not be complete, we keep increasing 
            # the threshold until all parts are separated correctly.
            if count_overlaps == num_lab:
                start_distance_to_spline_threshold = final_threshold_distance_to_spline
            else:
                start_distance_to_spline_threshold = threshold_increment

            steps += 1

            if self.save_details:
                img = sitk.GetImageFromArray(components)
                img.CopyInformation(self.catheters_contour)
                sitk.WriteImage(
                    img, 
                    os.path.join(
                        self.save_details_path, 
                        f"connected_comp{component_with_catheter_overlap}_step{steps}.seg.nrrd"
                        ),
                    True
                    )
        
        # Recreate the connected components map
        new_components_map = np.zeros_like(labeled_mask)
        for lab, separated_comp in zip(range(1, len(separated_components) + 1), separated_components):
            new_components_map = np.where(separated_comp == 1, lab, new_components_map)
        if self.save_details:
            img = sitk.GetImageFromArray(new_components_map)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(
                    self.save_details_path, 
                    "new_components_map_after_separation.seg.nrrd"
                    ),
                True
                )
            
        ### This new components map has lost lot of small components that had less voxels than min_size
        ### We get those components back.

        # We fit one spline per component.
        splines = []
        for lab in range(1, len(separated_components) + 1):
            separated_comp_img = sitk.GetImageFromArray(np.array(new_components_map == lab, dtype=np.uint8))
            separated_comp_img.CopyInformation(self.catheters_contour)
            splines.append(fit_spline(separated_comp_img, multiclass=False))
        
        # For each of the voxels that were removed from the component maps, we assign it
        # the component of th spline that is the closest to the voxel.
        # We project the voxel on each spline and check the distance.
        remaining_voxels = np.where(np.logical_and(overlapping_comp_array == 1, new_components_map == 0))
        for voxel_index in zip(*remaining_voxels):
            voxel_pos = [int(voxel_index[i]) for i in range(3)][::-1]
            distances_to_splines = []
            for spline in splines:
                _, _, distance_to_spline = spline.project_on_spline(self.catheters_contour.TransformIndexToPhysicalPoint(voxel_pos))
                distances_to_splines.append(distance_to_spline)
            closest_spline_idx = np.argmin(distances_to_splines)
            new_components_map[tuple(voxel_index)] = closest_spline_idx + 1

        if self.save_details:
            img = sitk.GetImageFromArray(new_components_map)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(
                    self.save_details_path, 
                    "new_components_map_after_separation_and_recuperation_of_lost_voxels.seg.nrrd"
                    ),
                True
                )

        # At this point the component that was originally identified as having overlapping catheters
        # is divided into a set of parts. These parts either belong to catheter 1 or catheter 2.
        # We make the assumption that there cannot be 3 catheters overlapping at the same time.
        # This was never seen in the dataset.
        # We need to assign these different parts to a catheter. As the buttons and the applicator 
        # connectors are separated on the breast, there should not be overlapping on the extremities 
        # of the catheters and thus we should never see an uneven number of parts here. 6 would in 
        # theory be possible even though higly unlikely sinc eit would mean the catheter bends in 
        # different directions...


        # One could: fit a spline between every potential group of catheters parts and see which one 
        # has the smallest average distance from voxel contour to spline. But we can also use our
        # merger class which check for the best merge and is probably more robust.
        use_merger = True
        if use_merger:
            merger = ContourMerger(
                new_components_map, np.max(new_components_map), self.catheters_contour, 
                None, self.catheter_diameter, self.catheter_marker_class,
                self.margin_for_cropping, self.multi_class, False,
                save_details=self.save_details, save_details_path=(self.save_details_path.replace(
                    "post_processed_catheters", 
                    f"overlap_merger_comp_{component_with_catheter_overlap}") if self.save_details
                    else None)
                    )
            # Here we merge catheter parts that are already touching each other=> no need for a big search space.
            # We also do not want to alter the contour here so no fill_in_between_parts
            merger.merge_catheters(search_space=2, fill_in_between_parts=False, must_be_catheter_core=False)
            final_components_map, _ = merger.get_labeled_mask()
            rangemax = np.max(final_components_map)
        else:
            potential_2groups = self.create_potential_2groups(classes=range(1, len(separated_components) + 1))

            splines = []
            metric_for_goodness_of_match = []
            for potential_2group in potential_2groups:
                # Fitting the spline on the first group of components
                potnetial_group1_array = np.zeros_like(new_components_map)
                for group in potential_2group[0]:
                    potnetial_group1_array = np.where(new_components_map == group, 1, potnetial_group1_array)
                potential_group1_img = sitk.GetImageFromArray(potnetial_group1_array)
                potential_group1_img.CopyInformation(self.catheters_contour)
                distances1 = get_distances_to_spline(potential_group1_img, multiclass=False)
                # Fitting the spline on the second group of components
                potential_group2_array = np.zeros_like(new_components_map)
                for group in potential_2group[1]:
                    potential_group2_array = np.where(new_components_map == group, 1, potential_group2_array)
                potential_group2_img = sitk.GetImageFromArray(potential_group2_array)
                
                potential_group2_img.CopyInformation(self.catheters_contour)
                distances2 = get_distances_to_spline(potential_group2_img, multiclass=False)
                # We take the average distance here but in theory the max distance should also work.
                metric_for_goodness_of_match.append(np.mean(distances1 + distances2))

            best_2groups = potential_2groups[np.argmin(metric_for_goodness_of_match)]

            # We assign the components to the 2 catheters
            final_components_map = np.zeros_like(new_components_map)
            for final_comp_idx, group in zip(range(1, len(best_2groups) + 1), best_2groups):
                for element in group:
                    final_components_map = np.where(
                        new_components_map == element, final_comp_idx, final_components_map
                    )

            rangemax = len(best_2groups)

        if self.save_details:
            img = sitk.GetImageFromArray(final_components_map)
            img.CopyInformation(self.catheters_contour)
            sitk.WriteImage(
                img, 
                os.path.join(
                    self.save_details_path, 
                    "final_components_map_after_separation_and_recuperation_of_lost_voxels.seg.nrrd"
                    ),
                True
                )
        
        # Merging this separated components with the og connected component maps
        new_labeled_mask = np.copy(labeled_mask)
        new_num_labels = num_labels
        for separated_comp_idx in range(1, rangemax + 1):
            if separated_comp_idx==1:
                # The first separated component can keep the idx of the og 
                # catheters-overlapping connected component
                new_labeled_mask = np.where(
                    final_components_map == separated_comp_idx, component_with_catheter_overlap, new_labeled_mask
                )
            else:
                # We add a new connected component
                new_labeled_mask = np.where(
                    final_components_map == separated_comp_idx, num_labels + separated_comp_idx - 1, new_labeled_mask
                )
                new_num_labels += 1

        return new_labeled_mask, new_num_labels
    
    @staticmethod
    def create_potential_2groups(classes:List[int]):
        """
        Create all the potential pairs of 2 groups that contain all the different classes.
        If number of classes is even, we ensure the same number of classes in each group:
        i.e. we assume the overlap can only be between two catheters and not more.
        If number of classes is unven, we allow uneven groups whose minimum number of 
        components should be the number of classes // 2.
        """
        n = len(classes)
        potential_2groups = []

        min_group_size = 1
        for group_a_size in range(1, (n // 2) + 2):  # Include up to n//2 + 1
            for group_a in combinations(classes, group_a_size):
                group_b = [cls for cls in classes if cls not in group_a]
                
                if len(group_a) < min_group_size or len(group_b) < min_group_size:
                    # Here we again assume we are only dealing with two catheters 
                    # and not more.
                    continue
                
                # Ensure lexicographical order to be able to remove duplicates with set()
                # Smaller group always comes first
                if len(list(group_a)) < len(group_b):
                    groups = (list(group_a), group_b)
                else:
                    groups = (group_b, list(group_a))
                
                if not groups in potential_2groups:
                    potential_2groups.append(groups)

        # Remove duplicates
        # Making the groups tuples since list is unhashable
        tuples_potential_2groups = []
        for potential_2group in potential_2groups:
            tuples_potential_2groups.append(
                tuple([tuple(potential_2group[0]), tuple(potential_2group[1])])
            )
        potential_2groups = list(set(tuples_potential_2groups))
        return potential_2groups 
    

    def _remove_catheter_shaft_until_separation(
            self, overlapping_comp_array:np.ndarray, min_size:int=10, 
            start_distance_to_spline_threshold:float=None,
            threshold_increment:float=0.1, needle_spline_creator:NeedleSplineCreator=None, 
            crop_to_process:bool=True):
        """
        Remove the catheter shaft until the catheter is separated in two parts.
        """

        ### Creating sitk image of the overllapping catheters contour
        overlapping_component_img = sitk.GetImageFromArray(overlapping_comp_array)
        overlapping_component_img.CopyInformation(self.catheters_contour)

        ### Cropping the array for faster array indexing and computation later on
        if crop_to_process:
            og_shape = overlapping_comp_array.shape
            overlapping_component_img, bounding_box = crop_around_mask(overlapping_component_img)
            overlapping_comp_array = self.crop_array_from_sitk_bbox(overlapping_comp_array, bounding_box)
            assert np.all(overlapping_comp_array.shape[::-1] == overlapping_component_img.GetSize()), (
                "The cropping did not work. We have shapes: ", overlapping_comp_array.shape, overlapping_component_img.GetSize()
            )
            assert np.all(overlapping_comp_array == sitk.GetArrayFromImage(overlapping_component_img)), (
                "The cropping did not work. The arrays are not the same."
            )

        ### Create the spline between the two catheters
        if needle_spline_creator is None:
            needle_spline_creator = fit_spline(overlapping_component_img, self.multi_class)

        ### Map every voxel to their distance to the spline.
        # Getting the physical coordinates of the points segmented as a catheter
        contour_voxel_positions, contour_voxel_indexes = get_physical_coord_for_needle(
            overlapping_component_img, return_indexes=True)        
        
        # Projecting those points on the spline and checking distance between the
        # projected point and the original point.
        contour_voxel_distances_to_spline = []
        for voxel_pos in contour_voxel_positions: 
            _, _, distance_to_spline = needle_spline_creator.project_on_spline(voxel_pos)
            contour_voxel_distances_to_spline.append(distance_to_spline)

        ### Gradually remove the voxels that are closer to the spline than the threshold
        modified_overlapping_comp_array = np.copy(overlapping_comp_array)

        if start_distance_to_spline_threshold is not None:
            threshold_distance_to_spline = start_distance_to_spline_threshold
        else:
            threshold_distance_to_spline = threshold_increment
        max_dist = 0
        indexes_already_removed = []
        separated = False
        while not separated:
            validated_components = []
            for voxel_index, dist_to_spline in zip(contour_voxel_indexes, contour_voxel_distances_to_spline):
                if dist_to_spline > max_dist:
                    max_dist = dist_to_spline
                if dist_to_spline < threshold_distance_to_spline and not(voxel_index in indexes_already_removed):
                    assert modified_overlapping_comp_array[tuple(voxel_index[::-1])] == 1, (
                        "The voxel you are trying to 0out is not part of the component."
                    )
                    indexes_already_removed.append(voxel_index)
                    modified_overlapping_comp_array[tuple(voxel_index[::-1])] = 0
            if self.save_details:
                img = sitk.GetImageFromArray(modified_overlapping_comp_array)
                img.CopyInformation(overlapping_component_img)
                sitk.WriteImage(img, os.path.join(self.save_details_path, f"modified_connected_comp_after_threshold_{threshold_distance_to_spline}.seg.nrrd"), True)
            labeled_mask, num_labels = ndimage.label(modified_overlapping_comp_array)
            for lab in range(1, num_labels + 1):
                if np.sum(labeled_mask == lab) > min_size:
                    validated_components.append(lab)
                else:
                    modified_overlapping_comp_array[labeled_mask == lab] = 0
                    points_indexes = np.where(labeled_mask == lab)
                    for point_index in zip(*points_indexes):
                        # Inverting the indexes because SimpleITK uspiecewise_lineares (z, y, x) and numpy uses (x, y, z)
                        int_pos = [int(point_index[i]) for i in range(3)][::-1]
                        indexes_already_removed.append(int_pos)
            if self.save_details:
                img = sitk.GetImageFromArray(modified_overlapping_comp_array)
                img.CopyInformation(overlapping_component_img)
                sitk.WriteImage(img, os.path.join(self.save_details_path, f"modified_connected_comp_after_threshold_{threshold_distance_to_spline}_and_removal_of_small_components.seg.nrrd"), True)
            separated = len(validated_components) > 1 or len(validated_components) == 0
            threshold_distance_to_spline += threshold_increment

        new_components = np.zeros_like(overlapping_comp_array)
        for comp_idx, validated_comp in zip( range(1, len(validated_components) + 1), validated_components):
            new_components = np.where(labeled_mask == validated_comp, comp_idx, new_components)


        ### Padding back to original shape
        if crop_to_process:
            new_components = self.pad_mask(new_components, bounding_box)
            assert np.all(new_components.shape == og_shape), (
                "The new components shape is not the same as the original shape.", new_components.shape, og_shape
            )
        return new_components, len(validated_components), needle_spline_creator, threshold_distance_to_spline

    def crop_array_from_sitk_bbox(self, array:np.ndarray, sitk_bounding_box:List[int]):
        """
        Crop the array to the bounding box of the sitk image.
        """
        return array[
            sitk_bounding_box[2]:sitk_bounding_box[2] + sitk_bounding_box[5],
            sitk_bounding_box[1]:sitk_bounding_box[1] + sitk_bounding_box[4],
            sitk_bounding_box[0]:sitk_bounding_box[0] + sitk_bounding_box[3]
        ]

    def pad_mask(self, array:np.ndarray, bounding_box:List[int]):
        """
        Pad the mask to the original size of the volume: self.catheters_contour
        """
        padding = []
        for i in range(3):
            padding.append((bounding_box[i], self.catheters_contour.GetSize()[i] - bounding_box[i] - bounding_box[i+3]))
        padded_array = np.pad(array, padding[::-1], mode="constant", constant_values=0)
        return padded_array

class ContourMerger:
    def __init__(self, labeled_mask:np.ndarray, num_labels:int, catheters_contour:sitk.Image,
                 catheters:List[sitk.Image], catheter_diameter:float=2.0, catheter_marker_class:int=1,
                 margin_for_cropping:int=20, multi_class:bool=True, multiprocess:bool=True, 
                 save_details:bool=False, save_details_path:str=None):
        
        self.labeled_mask = labeled_mask
        self.num_labels = num_labels
        self.catheters_contour = catheters_contour
        if catheters is None:
            catheters = []
            for i in range(1, num_labels+1):
                catheters.append(sitk.GetImageFromArray(np.array(labeled_mask == i, dtype=np.uint8)))
                catheters[-1].CopyInformation(catheters_contour)
        self.catheters = catheters
        self.catheter_diameter = catheter_diameter
        self.save_details = save_details
        if save_details:
            os.makedirs(save_details_path, exist_ok=True)
        self.save_details_path = save_details_path
        self.catheter_marker_class = catheter_marker_class
        self.margin_for_cropping = margin_for_cropping
        self.multi_class = multi_class
        self.multiprocess = multiprocess


        ## Initialization of the "state" of the contour merger
        self.map_component_to_merge = []
        self.validated_pairs_to_merge = []
        self.pairs_interfering_with_other_pairs = []
        self.goodness_of_merge = []
        self.pairs_to_potentially_merge = []
        self.spline_interpolators = []
        self.count_merged_catheters = 0

    def creating_add_ons_for_all_catheters(
            self, search_space:float=20, dilation_for_search:int=3, iteration_nb:int=1,
            must_be_catheter_core:bool=True):
        """
        Extend all the catheters on both sides.
        """
        add_ons = {}

        assert len(self.catheters) == self.num_labels, (
                "The number of catheters should be the same as the number of labels."
            )
        # Creating add ons for all the needles
        if self.multiprocess:
            args = []
            for needle, lab in zip(self.catheters, range(1, self.num_labels +1)):
                if must_be_catheter_core:
                    if not np.any(sitk.GetArrayFromImage(needle) == self.catheter_marker_class):
                        continue
                else:
                    if not np.any(sitk.GetArrayFromImage(needle) > 0):
                        continue
                args.append(
                    (needle, self.catheter_diameter, lab, dilation_for_search, search_space, False, self.multi_class)
                )
    
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    message="The required storage space exceeds the available storage space.",
                    category=RuntimeWarning, module="scipy")

                with mp.Pool(8) as p:
                    r = list(
                        tqdm.tqdm(
                            p.imap(
                                extend_catheter_contour_on_both_sides_wrapper,
                                args,
                            ),
                            total=self.num_labels,
                            desc="Creating add ons for potential merge..",
                        )
                    )
            for dilated_add_on, old_comp_idx, needle in r:
                add_ons[str(old_comp_idx)] = (dilated_add_on, needle)
                if self.save_details:
                    add_on_img = sitk.GetImageFromArray(dilated_add_on)
                    add_on_img.CopyInformation(needle)
                    sitk.WriteImage(
                        add_on_img,
                        os.path.join(self.save_details_path, f"addons_iteration_{iteration_nb}_merge_comp{old_comp_idx}.seg.nrrd"),
                        useCompression=True
                    )

        else:
            for needle, lab  in tqdm.tqdm(
                zip(self.catheters, range(1, self.num_labels +1)), 
                total=self.num_labels, 
                desc="Creating add ons for potential merge.."):

                if must_be_catheter_core:
                    if not np.any(sitk.GetArrayFromImage(needle) == self.catheter_marker_class):
                        continue
                else:
                    if not np.any(sitk.GetArrayFromImage(needle) > 0):
                        continue

                addon_entry_tip_t0, addon_entry_tip_t1, _, _ = extend_catheter_contour_on_both_sides(
                    catheter_contour=needle, catheter_diameter=self.catheter_diameter, dilation_add_on=0, 
                    search_space=search_space, spline=False, multi_class=self.multi_class
                )
                
                # Dilating all at once instead of dilating to part and t1 part since dilation takes time.
                if dilation_for_search >0:
                    dilated_add_on = ndimage.binary_dilation(
                        addon_entry_tip_t0 + addon_entry_tip_t1, 
                        iterations=dilation_for_search).astype(addon_entry_tip_t1.dtype)
                else:
                    dilated_add_on = addon_entry_tip_t0 + addon_entry_tip_t1

                if self.save_details:
                    add_on_img = sitk.GetImageFromArray(dilated_add_on)
                    add_on_img.CopyInformation(needle)
                    sitk.WriteImage(
                        add_on_img,
                        os.path.join(self.save_details_path, f"addons_iteration_{iteration_nb}_merge_comp{lab}.seg.nrrd"),
                        useCompression=True
                    )

                add_ons[str(lab)] = (dilated_add_on, needle)
        
        return add_ons

    def _evaluate_goodness_of_merge(self, pairs_to_potentially_merge:List[List[int]]):
        """
        From a set of pairs of components to merge, return pairs without overlap caheter 
        overlap along with their goodness merge: average distance from voxel to spline.
        """

        goodness_of_merge = []
        pairs_to_merge = []
        spline_interpolators = []
        for potential_pair_to_merge in tqdm.tqdm(pairs_to_potentially_merge, 
                                                 total=len(pairs_to_potentially_merge), 
                                                 desc="Evaluating goodness of merge.."):
            # We create the merged component
            merged_array = np.logical_or(
                self.labeled_mask == potential_pair_to_merge[0], 
                self.labeled_mask == potential_pair_to_merge[1]
                ).astype(np.uint8)

            assert np.sum(merged_array) > 0, "The merged component should have some voxels."
            # Check for overlapping catheters in merged components 
            overlap, _, spline_interpolator, distances_to_spline = detect_overlapping_catheter_areas(
                component=merged_array, catheters_contour=self.catheters_contour, 
                catheter_shaft_threshold=4.0, break_on_find=True, multiclass=self.multi_class
                )
            
            if self.save_details:
                spline_pts = spline_interpolator.get_spline_points()
                create_slicer_markup_points(
                    os.path.join(self.save_details_path, f"spline_{potential_pair_to_merge[0]}_{potential_pair_to_merge[1]}.mrk.json"), spline_pts
                )

            if overlap:
                if self.save_details:
                    print("When trying to merge components ", potential_pair_to_merge, "we have an overlap.")
                # We know this merge proposal is bad we filter it out of potential merges.
                continue
            else:
                if self.save_details:
                    print("Between components ", potential_pair_to_merge, 
                          "we have distances to spline",describe_array(distances_to_spline), 
                          "with median", np.median(distances_to_spline))
                pairs_to_merge.append(potential_pair_to_merge)
                goodness_of_merge.append(np.std(distances_to_spline))
                spline_interpolators.append(spline_interpolator)

        return goodness_of_merge, pairs_to_merge, spline_interpolators

    @staticmethod
    def _validate_pairs_to_merge(pairs_to_potentially_merge:List[List[int]]):
        """
        Check if a pair of components can be merged without interfering with any 
        other potential merge: i.e. if the two elements of the tuple of the pair 
        do not appear in any other pair.
        """
        validated_pairs_to_merge_idxs = []
        pairs_interfering_with_other_pairs_idxs = []
        for pair_idx1, pair_to_merge in enumerate(pairs_to_potentially_merge):
            is_valid = True
            for pair_idx2, other_pair in enumerate(pairs_to_potentially_merge):
                if pair_idx1 == pair_idx2:
                    continue
                if pair_to_merge[0] in other_pair or pair_to_merge[1] in other_pair:
                    is_valid = False
                    pairs_interfering_with_other_pairs_idxs.append(pair_idx1)
                    break
            if is_valid:
                validated_pairs_to_merge_idxs.append(pair_idx1)
        return validated_pairs_to_merge_idxs, pairs_interfering_with_other_pairs_idxs
    

    def merge_two_parts(
            self, pair_to_merge:Tuple[int, int], spline:NeedleSplineCreator, 
            dilation_of_catheter_shaft:int=1, fill_in_between_parts:bool=True):
        """
        Merge two parts of the same catheter. Modifies the labeled mask and num labels accordingly.
        """
        if self.save_details:
            print(f"Merging components {pair_to_merge}.")
        staying_comp = min(pair_to_merge)
        going_comp = max(pair_to_merge)

        # Joing the two parts of the components by fitting a spline on the two components and expanding it
        assert np.any(self.labeled_mask == pair_to_merge[0]), "The component to merge should have some voxels."
        pair0_img = sitk.GetImageFromArray(np.array(self.labeled_mask == pair_to_merge[0]).astype(np.uint8))
        pair0_img.CopyInformation(self.catheters_contour)
        point0_0, t0_spline_0, point1_0, t1_spline_0 = get_segment_endpoints_and_t(
            catheter_contour=pair0_img, needle_spline_creator=spline)
        assert np.any(self.labeled_mask == pair_to_merge[1]), "The component to merge should have some voxels."
        pair1_img = sitk.GetImageFromArray(np.array(self.labeled_mask == pair_to_merge[1]).astype(np.uint8))
        pair1_img.CopyInformation(self.catheters_contour)
        point0_1, t0_spline_1, point1_1, t1_spline_1 = get_segment_endpoints_and_t(
            catheter_contour=pair1_img, needle_spline_creator=spline)
        if abs(t0_spline_0 - t0_spline_1) < abs(t1_spline_0 - t0_spline_1):
            t0_spline = t0_spline_0
            start_pt = point0_0
        else:
            t0_spline = t1_spline_0
            start_pt = point1_0
        if abs(t0_spline - t0_spline_1) < abs(t0_spline - t1_spline_1):
            t1_spline = t0_spline_1
            end_pt = point0_1
        else:
            t1_spline = t1_spline_1
            end_pt = point1_1
        if self.save_details:
            create_slicer_markup_points(
                os.path.join(self.save_details_path, f'pts_for_spline_to_complete_comp_{staying_comp}.mrk.json'),
                [start_pt, end_pt], 
                color=[0.5,0.,0.5]
                )
            spl_pts = spline.get_spline_points()
            create_slicer_markup_points(
                os.path.join(self.save_details_path, f'spline_comp_{staying_comp}_pts.mrk.json'),
                spl_pts, 
                color=[0.5,0.,0.5]
                )
        
        # Updating the labeled mask
        # Giving the two components the same comp idx
        self.labeled_mask = np.where(self.labeled_mask==going_comp, staying_comp, self.labeled_mask)

        if fill_in_between_parts:
            # Completing betweent the two parts
            merging_component = spline.create_spline_from_voxel_coordinates(
                start_pt, end_pt, diameter=self.catheter_diameter
            )
            if dilation_of_catheter_shaft > 0:
                merging_component = ndimage.binary_dilation(merging_component, iterations=dilation_of_catheter_shaft).astype(merging_component.dtype)

            # Updating the labeled mask where it is 0
            self.labeled_mask = np.where(
                # We only condsider parts of the labeled mask that are 0 to add contour
                # otherwise it might affect the subsequent merging.
                np.logical_and(self.labeled_mask == 0, 
                            merging_component.transpose(2,1,0)==1), 
                staying_comp, 
                self.labeled_mask)
        
        # Adapting the labeled mask to the missing component
        self.labeled_mask = np.where(self.labeled_mask > going_comp, self.labeled_mask - 1, self.labeled_mask)
        self.num_labels -= 1
        self.count_merged_catheters += 1

    
    def catheter_matching(
            self, add_ons:Tuple[np.ndarray, sitk.Image], addon_idx1:str, addon_idx2:str):
        """
        Check if the two catheters extend to each other.
        """
        if addon_idx1 not in add_ons or addon_idx2 not in add_ons:
            return False
        addon1, catheter1 = add_ons[addon_idx1]
        addon2, catheter2 = add_ons[addon_idx2]

        overlap_of_extensions = (
            # Either the add ons overlap
            np.any(np.logical_and(addon1, addon2)) or
            # Or the add ons overlap with the catheter
            np.any(np.logical_and(addon1, sitk.GetArrayFromImage(catheter2))) or
            np.any(np.logical_and(addon2, sitk.GetArrayFromImage(catheter1)))
        )

        return overlap_of_extensions
    
    def _get_potential_merges(self,search_space:float, dilation_for_search:int, 
                              iteration_nb:int, must_be_catheter_core:bool):
        # Extending all catheters
        add_ons = self.creating_add_ons_for_all_catheters(
            search_space, dilation_for_search, iteration_nb, must_be_catheter_core)
        
        # Checking if the extensins overlap ie if there should be a potential 
        # merge between two parts.
        pairs_of_comp = list(combinations(range(1, self.num_labels+1), 2))
        map_component_to_any_merge = {}
        for pair_of_comp in pairs_of_comp:
            catheter_extensions_overlap = self.catheter_matching(
                add_ons, str(pair_of_comp[0]), str(pair_of_comp[1]))
            
            if catheter_extensions_overlap:
                if self.save_details:
                    print("overlap of catheter extensions between ", pair_of_comp)
                if pair_of_comp[0] in map_component_to_any_merge:
                    map_component_to_any_merge[pair_of_comp[0]].append(pair_of_comp[1])
                else:
                    map_component_to_any_merge[pair_of_comp[0]] = [pair_of_comp[1]]

        # Here one component can be paired to a list of components
        # We transform this dictionnary to a set of pairs.
        pairs_to_potentially_merge = []
        for key, values in map_component_to_any_merge.items():
            for value in values:
                pairs_to_potentially_merge.append((key, value))
        # Removing duplicates
        pairs_to_potentially_merge = [list(tup) for tup in list(set(pairs_to_potentially_merge))]
        return pairs_to_potentially_merge


    def _merge_iteration(self, search_space:float, dilation_for_search:int, dilation_of_catheter_shaft:int, 
                         iteration_nb:int, fill_in_between_parts:bool=True, must_be_catheter_core:bool=True, 
                         one_hesitating_merge_per_iteration:bool=True):
        """
        This function performs one iteration of the merge process.
        It extends all current catheter parts and checks if there are potential merges.
        It first merges the catheter parts that only overlap between them selves.
        Then it performs the most probable merge between the pairs of components that
        interfere with other pairs (potentially could merge with other parts).

        one_hesitating_merge_per_iteration should always be True. 
        If we perform one merge and then fit all the splines again it is more precise than 
        performing all potnetial merges at each iteration.
        """
        
        pairs_to_potentially_merge = self._get_potential_merges(
            search_space=search_space, dilation_for_search=dilation_for_search, 
            iteration_nb=iteration_nb, must_be_catheter_core=must_be_catheter_core)
        assert len([tuple(i) for i in pairs_to_potentially_merge]) == len(set([tuple(i) for i in pairs_to_potentially_merge])), (
            f"The map_component_to_merge should not have duplicates but you have {pairs_to_potentially_merge}"
        )

        if len(pairs_to_potentially_merge) != 0:
            # First we test the potential merge: does it result in overlap?
            # If not is it a good fit? metric= average distance from voxel to spline.
            self.goodness_of_merge, self.pairs_to_potentially_merge, self.spline_interpolators = self._evaluate_goodness_of_merge(pairs_to_potentially_merge)
    
            # Check if a pair of components can be merged without interfering with any 
            # other potential merge: i.e. if the two elements of the tuple of the pair 
            # do not appear in any other pair.
            validated_pairs_idx, pairs_interfering_with_other_pairs_idx = self._validate_pairs_to_merge(self.pairs_to_potentially_merge)
            self.validated_pairs_to_merge = [self.pairs_to_potentially_merge[idx] for idx in validated_pairs_idx]

            assert np.all(np.sort(validated_pairs_idx) == np.array(validated_pairs_idx)), (
                "The validated pairs indexes are not sorted. Below loop will not work."
            ) 
            for already_removed_counter, validated_pair_idx in enumerate(validated_pairs_idx):
                validated_pair_to_merge = self.pairs_to_potentially_merge[validated_pair_idx-1*already_removed_counter]
                spline_interp = self.spline_interpolators[validated_pair_idx-1*already_removed_counter]
                # merge_two_parts method updates labeled mask and num labels attributes
                self.merge_two_parts(
                    validated_pair_to_merge, spline_interp, dilation_of_catheter_shaft, fill_in_between_parts
                    )
                # _update_state updates validated_pairs_to_merge, pairs_interfering_with_other_pairs
                # and map_component_to_merge class attributes.
                self._update_state(validated_pair_to_merge, validated_pair_idx-1*already_removed_counter)

                if self.save_details:
                    img = sitk.GetImageFromArray(self.labeled_mask)
                    img.CopyInformation(self.catheters_contour)
                    sitk.WriteImage(
                        img,
                        os.path.join(self.save_details_path, f"merged_connected_components_{self.num_labels}_labels.seg.nrrd"),
                        useCompression=True)
                    
                # Updating pairs_interfering_with_other_pairs_idx regarding pairs that have been
                # removed since they were validated.
                pairs_interfering_with_other_pairs_idx = [idx if idx < validated_pair_idx-1*already_removed_counter else idx - 1
                                                          for idx in pairs_interfering_with_other_pairs_idx]
                if self.save_details:
                    print("New pairs_interfering_with_other_pairs_idx", pairs_interfering_with_other_pairs_idx)
                    print("self.pairs_to_potentially_merge", self.pairs_to_potentially_merge)

            if self.save_details:
                self.print_state(f"after iteration {iteration_nb} easy merge update")

            self.pairs_interfering_with_other_pairs = [self.pairs_to_potentially_merge[idx] for idx in pairs_interfering_with_other_pairs_idx]
            
            if len(self.pairs_interfering_with_other_pairs) != 0:
                
                if one_hesitating_merge_per_iteration:
                    # We take the most probable fit for the merge of the pairs_interfering_with_other_pairs
                    # we do this for the first pairs as long as the pairs are not interfering with each other.
                    most_likely = np.argmin(self.goodness_of_merge)
                    self.merge_two_parts(
                            self.pairs_interfering_with_other_pairs[most_likely], 
                            self.spline_interpolators[most_likely], dilation_of_catheter_shaft, 
                            fill_in_between_parts
                            )
                    self._update_state(self.pairs_interfering_with_other_pairs[most_likely], most_likely)
                    if self.save_details:
                        self.print_state(f"after iteration {iteration_nb} most likely merge from inerferring parts update")

                else:
                    # Testing to see if we can merge the pairs that don't interfere with other pairs by taking all the most
                    # likely merges from the pairs_interfering_with_other_pairs.
                    goodness_of_merge_interfering_pairs = [self.goodness_of_merge[idx] for idx in pairs_interfering_with_other_pairs_idx]
                    order_of_likelyness = np.argsort(goodness_of_merge_interfering_pairs)
                    assert len(order_of_likelyness) == len(self.pairs_interfering_with_other_pairs), (
                        "The order of likelyness should be the same as the pairs_interfering_with_other_pairs"
                    )

                    # From the list of pairs that interfere with other pairs we keep the first ones
                    # that do not interfere with each other.
                    sorted_pairs_interfering_with_other_pairs = [self.pairs_interfering_with_other_pairs[i] for i in order_of_likelyness]
                    first_occurence_pairs_interfering_with_other_pairs = []
                    first_occurence_interfering_pairs_idxs = []
                    for pair in sorted_pairs_interfering_with_other_pairs:
                        if not any([pair[0] in pair_to_merge or pair[1] in pair_to_merge 
                                    for pair_to_merge in first_occurence_pairs_interfering_with_other_pairs]):
                            first_occurence_pairs_interfering_with_other_pairs.append(pair)
                            first_occurence_interfering_pairs_idxs.append(self.pairs_to_potentially_merge.index(pair))
                    first_occurence_interfering_pairs_idxs = sorted(first_occurence_interfering_pairs_idxs)
                    assert np.all(np.sort(first_occurence_interfering_pairs_idxs) == np.array(first_occurence_interfering_pairs_idxs)), (
                        "The validated pairs indexes are not sorted. Below loop will not work."
                    ) 
                    for already_removed_counter, first_occurence_interfering_pairs_idx in enumerate(first_occurence_interfering_pairs_idxs):
                        validated_pair_to_merge = self.pairs_to_potentially_merge[first_occurence_interfering_pairs_idx-1*already_removed_counter]
                        spline_interp = self.spline_interpolators[first_occurence_interfering_pairs_idx-1*already_removed_counter]
                        # merge_two_parts method updates labeled mask and num labels attributes
                        self.merge_two_parts(
                            validated_pair_to_merge, spline_interp, dilation_of_catheter_shaft, fill_in_between_parts
                            )
                        # _update_state updates validated_pairs_to_merge, pairs_interfering_with_other_pairs
                        # and map_component_to_merge class attributes.
                        self._update_state(validated_pair_to_merge, first_occurence_interfering_pairs_idx-1*already_removed_counter)
                        if self.save_details:
                            self.print_state(f"after iteration {iteration_nb} most likely merge from interfering parts update {already_removed_counter}")

            # Create the catheters list
            new_set_needles = []
            catheters_array = sitk.GetArrayFromImage(self.catheters_contour)
            for connected_comp in range(1, self.num_labels+1):
                # Already existing catheter
                specific_catheter = np.array(self.labeled_mask == connected_comp).astype(np.uint8) * catheters_array
                # We add the parts created during merge
                specific_catheter = np.where(
                    np.logical_and(
                        self.labeled_mask == connected_comp, 
                        catheters_array == 0),
                    self.catheter_marker_class, specific_catheter)
                img = sitk.GetImageFromArray(specific_catheter)
                img.CopyInformation(self.catheters_contour)
                new_set_needles.append(img)
            self.catheters = new_set_needles
        else:
            self.pairs_interfering_with_other_pairs = []
            self.pairs_to_potentially_merge = []

    def merge_catheters(
            self, search_space:float=20, dilation_for_search:int=3, 
            dilation_of_catheter_shaft:int=1, fill_in_between_parts:bool=True, 
            must_be_catheter_core:bool=False):
        """
        Going through every components (pars of catheter), extending them on both sides to see if there
        are other main catheter parts that could belong to the same catheter.
        Potentially merging the catheter parts together.
        """
        
        assert self.margin_for_cropping >= search_space, (
            """The margin for cropping should be greater than the search space for merging catheters.search_space.
            Otherwise we are going to create addons outside the volume.
            """
        )
        if self.save_details:
            self.print_state("before any merge")
        iteration_count = 1
        self._merge_iteration(search_space, dilation_for_search, dilation_of_catheter_shaft, 
                              iteration_count, fill_in_between_parts, must_be_catheter_core)
        
        # If there was already no doubt with a ctheter that could potnetially merge with many parts we are done.
        # Otherwise we go for more likely merge and then repeat the process.
        while not (len(self.pairs_interfering_with_other_pairs) == 0 and len(self.pairs_to_potentially_merge) == 0):
            iteration_count += 1
            self._merge_iteration(search_space, dilation_for_search, dilation_of_catheter_shaft, 
                                  iteration_count, fill_in_between_parts, must_be_catheter_core)

    @staticmethod
    def _update_list_of_pairs(old_to_new_values_mapping:dict, pair_to_remove:List[int], pairs_to_filter:List[List[int]]):
        filtered_pairs = copy.deepcopy(pairs_to_filter)
        if pair_to_remove in filtered_pairs:
            filtered_pairs.remove(pair_to_remove)
        for idx in range(len(filtered_pairs)):
            filtered_pairs[idx] = [
                old_to_new_values_mapping[filtered_pairs[idx][0]], 
                old_to_new_values_mapping[filtered_pairs[idx][1]]]    
        # Removing duplicates
        # filtered_pairs = [list(tup) for tup in set([tuple(i) for i in filtered_pairs])]
        return filtered_pairs

    def _update_state(self, pair_used_to_merge:Tuple[int, int], pair_idx:int):
        """
        When we modified the labeled mask and num labels because of a 
        merge of two parts of caheter, the pairs of components previously 
        identified are not valid anymore, we need to change the indexes 
        of the components accordingly.
        pair_idx is the index in the map_component_to_merge list.
        """

        # This component disappeared during the merge
        max_pair = max(pair_used_to_merge)
        min_pair = min(pair_used_to_merge)
        assert len(pair_used_to_merge) == 2, f"The pair should have two elements. but is {pair_used_to_merge}"

        # Create the mapping
        old_to_new_values = {}
        # slef.num_labels should be updated before using this function
        # i.e. you should run merge_two_parts() before using this function.
        # We add +1 to num_labels because it is already updated in merge_two_parts
        for i in range(1, self.num_labels+1 +1):
            if i == max_pair:
                old_to_new_values[i] = min_pair
            elif i > max_pair:
                old_to_new_values[i] = i-1
            else:
                old_to_new_values[i] = i

        self.validated_pairs_to_merge = self._update_list_of_pairs(
            old_to_new_values, pair_used_to_merge, self.validated_pairs_to_merge)
        self.pairs_interfering_with_other_pairs = self._update_list_of_pairs(
            old_to_new_values, pair_used_to_merge, self.pairs_interfering_with_other_pairs)
        self.pairs_to_potentially_merge = self._update_list_of_pairs(
            old_to_new_values, pair_used_to_merge, self.pairs_to_potentially_merge)
        
        del self.goodness_of_merge[pair_idx]
        del self.spline_interpolators[pair_idx]

    def print_state(self, when:str=""):
        print(f"======================State of the Merger {when}:")
        print("validated_pairs_to_merge", self.validated_pairs_to_merge)
        print("pairs_interfering_with_other_pairs", self.pairs_interfering_with_other_pairs)
        print("goodness_of_merge", self.goodness_of_merge)
        print("pairs_to_potentially_merge", self.pairs_to_potentially_merge)
        print("Number of merged catheters ", self.count_merged_catheters)
        print("Number of labels ", self.num_labels)
        for i in range(1, self.num_labels+1):
            print("Number of voxels in catheter ", i, ":", np.sum(self.labeled_mask == i))
        print("=====================================================")
        img = sitk.GetImageFromArray(self.labeled_mask)
        img.CopyInformation(self.catheters_contour)
        sitk.WriteImage(
            img,
            os.path.join(self.save_details_path, f"merged_connected_comps_{('_').join(when.split(' '))}.seg.nrrd"),
            useCompression=True)

    def get_labeled_mask(self):
        return self.labeled_mask, self.num_labels
    
    def get_catheters_list(self):
        return self.catheters
    
    def get_count_merged_catheters(self):
        return self.count_merged_catheters

    def get_modified_contour_array(self):
        """
        Reconstructs the contour array with all the added catheter parts 
        between two merged catheter parts.
        """
        contour_array = sitk.GetArrayFromImage(self.catheters_contour)
        modified_contour_array = np.copy(contour_array)

        modified_contour_array = np.where(
            np.logical_and(
                self.labeled_mask!=0, 
                contour_array ==0),
            self.catheter_marker_class, modified_contour_array)
        return modified_contour_array

class ContourExpander:

    """
    We package the expansion of the contours in this class.
    """
    def __init__(self, catheter_contour:sitk.Image, catheter_diameter:float=2.0, 
                 multi_class:bool=True, save_details:bool=False, save_details_path:str=None):
        
        self.catheter_contour = catheter_contour
        self.all_classes_array = sitk.GetArrayFromImage(catheter_contour).astype(np.uint8)
        self.needle_spline_creator = NeedleSplineCreator(catheter_contour, multiclass=multi_class)
        self.needle_spline_creator.interpolate_spline()
        self.catheter_diameter = catheter_diameter
        self.multi_class = multi_class
        self.save_details = save_details
        if save_details:
            os.makedirs(save_details_path, exist_ok=True)
        self.save_details_path = save_details_path


    def _extend_catheter_contour_on_both_sides(
            self, dilation_add_on:int=3,
            search_space:float=None, spline:bool=False):
        
        # Identifying ts of the spline to know which side to process the catheter for.
        extremum0, t0_spline, extremum1, t1_spline = get_segment_endpoints_and_t(
            catheter_contour=self.catheter_contour, needle_spline_creator=self.needle_spline_creator)
        
        ### Adding the entry tip + tip length on t0_spline side
        addon_entry_tip_t0, _, _ = self.add_part(
            t_start=t0_spline, start_pt_coords=extremum0,
            step=search_space, dilation_add_on=dilation_add_on, 
            spline=spline
        )

        ### Adding the entry tip + tip length on t1_spline side
        addon_entry_tip_t1, _, _ = self.add_part( 
            t_start=t1_spline, start_pt_coords=extremum1,
            step=search_space, dilation_add_on=dilation_add_on, 
            spline=spline
        )

        return addon_entry_tip_t0, addon_entry_tip_t1, t0_spline, t1_spline
    
    def add_part(self, t_start:float, start_pt_coords:np.ndarray, 
                  step:float, dilation_add_on:int=0, spline:bool=False):

        # Adding the part from t_start

        # Getting the bounds coordinates for the part on the spline
        projected_end, projected_end_t = self._get_segment_coords_on_spline(
            start_pt_coords, t_start, step)

        # Creating the segment from the bounds
        addon = self._create_addon(
            point1=start_pt_coords, point2=projected_end, 
            dilation_add_on=dilation_add_on, spline=spline)

        return addon, projected_end, projected_end_t

    def _get_segment_coords_on_spline(
            self, point_coords:np.ndarray, point_t:float, 
            step:float):
        bmin , bmax = get_bounds_for_step(point_t)
        projected_end, projected_end_t = self.needle_spline_creator.step_in_spline(
            point_coords,
            step=step,
            bound_min=bmin,
            bound_max=bmax,
            # No need to compute perfect projection since we are adding a fake contour here.
            # Faster no to use arc length.
            arc=False
        )
        return projected_end, projected_end_t
        
    
    def _create_addon(
            self, point1:np.ndarray, point2:np.ndarray, 
            dilation_add_on:int=0, spline:bool=False):
        
        can_proceed, point1, point2 = self._check_pts_in_contour(point1, point2)

        if can_proceed:
            if spline:
                ### Creating the contour by extending a spline (points on the spline)
                assert self.needle_spline_creator is not None, "Needle spline creator is needed for spline creation"
                assert np.all(
                    sitk.GetArrayFromImage(self.needle_spline_creator.input_catheter_contour).shape == 
                    self.all_classes_array.shape)
                addon = self.needle_spline_creator.create_spline_from_voxel_coordinates(
                    point1, point2, diameter=self.catheter_diameter)
                if np.sum(addon) > 0:
                    if dilation_add_on > 0:
                        addon = self.needle_spline_creator.dilate(
                            volume=addon, dilation_nb_times=dilation_add_on, 
                            point1=point1, point2=point2, from_raw_volume=True)
                    addon = np.where(np.logical_and(
                        self.all_classes_array==0, 
                        addon.transpose(2, 1, 0) == 1),
                        1, 0).astype(np.uint8)
                else:
                    addon = np.zeros_like(self.all_classes_array)
            else:
                # Creating the contour as a segment (points on a linear function)
                fake_needle = NeedleFalsifier(
                        point1,
                        point2,
                        sitk_volume=self.catheter_contour,
                        demo=False,
                        # dilation_nb_times was only used by add_line_from_voxel_indexes
                        dilation_nb_times=0,
                    )
                fake_needle.add_line_from_voxel_coordinates(diameter=self.catheter_diameter)
                if np.sum(fake_needle.volume) > 0:
                    if dilation_add_on > 0:
                        fake_needle.dilate(dilation_add_on)
                    addon = np.where(np.logical_and(
                        self.all_classes_array==0, 
                        fake_needle.volume.transpose(2, 1, 0) == 1),
                        1, 0).astype(np.uint8)
                else:
                    addon = np.zeros_like(self.all_classes_array)
        else:
            addon = np.zeros_like(self.all_classes_array)
        return addon

    def _check_pts_in_contour(self, point1:np.ndarray, point2:np.ndarray):
        """
        Check if the points are in the patient volume. 
        If both of them are int the patient volume we  can use them to build 
        contour.
        If one of them is not, we create a segment between the two points 
        and return the farthest point in the segment within the patient volume. 
        If both of them are not in the patient volume, we don't create any addon.
        """
        if not self._is_in_volume(point1) and not self._is_in_volume(point2):
            return False, None, None
        elif not self._is_in_volume(point1):
            og_distance = distance(point1, point2)
            new_point = np.copy(point1)
            while not self._is_in_volume(new_point):
                # End point is not already outside contour, we need to extend
                og_distance -= 0.1
                new_point = extrapolate_point(
                    point1=point2, point2=point1, 
                    distance=og_distance, reverse=False)
            return True, new_point, point2
        elif not self._is_in_volume(point2):
            og_distance = distance(point1, point2)
            new_point = np.copy(point2)
            while not self._is_in_volume(new_point):
                # End point is not already outside contour, we need to extend
                og_distance -= 0.1
                new_point = extrapolate_point(
                    point1=point1, point2=point2, 
                    distance=og_distance, reverse=False)
            return True, point1, new_point
        else:
            return True, point1, point2
    
    def _is_in_volume(self, point:np.ndarray):
        """
        Check if the point is in the patient volume.
        """
        coord_in_vol = np.array(self.catheter_contour.TransformPhysicalPointToIndex(point))
        return not(np.any(coord_in_vol < 0) or 
                   np.any(coord_in_vol > np.array(self.catheter_contour.GetSize())-1))

        
if __name__ == "__main__":
    import time
    import os
     # /home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1_bs8_threshold4_minsize5_merge_evennocore_stru_s_for_concomp_mergeextansion50/test_fold01234/277561
    # /home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1_bs8_threshold4_minsize5_merge_evennocore_stru_s_for_concomp/277561
    # dataset_name = 'Dataset006_catheters_and_tip_makers_diameter_2.0_dilation_1'
    dataset_name = 'Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1_bs8_threshold4_minsize5_merge_evennocore_stru_s_for_concomp_mergeextansion50'
    if "dilation" in dataset_name:
        contour_dilation = int(dataset_name.split("dilation_")[1].split("_")[0])
    else:
        contour_dilation = 0
    data_folder = f"/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/{dataset_name}/test_fold01234" #/val_benchmark/"
    patient_id = "91249" # "971621" #"811894" # "277561" # "390607" # "277561"
    catheters_contour_path = os.path.join(data_folder, patient_id, "ai_generated_catheters.seg.nrrd")
    reference_ct = sitk.ReadImage(os.path.join(data_folder, patient_id,"ct.nrrd"))
    t0 = time.time()
    separator = ContourSeparator(
        reference_ct=reference_ct, catheters_contour_path=catheters_contour_path, save_details=True, 
        save_details_path=os.path.join(data_folder, patient_id, "post_processed_catheters/"),
        multiprocess=True, log_path = os.path.join(data_folder, patient_id, "log.txt"),
        )
    catheters = separator.separate_catheters()
    print("It takes {} seconds to consolidate a contour.".format(time.time() - t0))
    idx = 0
    # sitk.WriteImage(
    #     catheters[idx], 
    #     os.path.join(data_folder, patient_id, f"ai_generated_catheters_{idx}.seg.nrrd"),
    #     useCompression=True)
    