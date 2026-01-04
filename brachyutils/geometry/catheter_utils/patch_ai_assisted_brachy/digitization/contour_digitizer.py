import os
import tqdm
from typing import List, Dict, Any
import warnings

import SimpleITK as sitk
import numpy as np
from scipy.interpolate import splev
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import CatheterSetUp, get_rotation_from_position, dilate_mask_in_mm
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.separator import ContourSeparator
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.pw_linear_interpolator import (
    create_segments_by_slice,
    PiecewiseLinear3D,
    Segment,
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.spline_interpolator import NeedleSplineCreator
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import (
    x_y_z_to_list,
    list_to_x_y_z,
    reorder_points_tsp,
    fit_segment,
    linear_func,
    fit_line,
    fit_spline,
    distance,
    get_physical_coord_for_needle,
    min_dist_two_list,
    avg_dist_closest_pts_two_lists
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.labelling import CatheterIdentificator


def remove_dp_inside_mask(created_needle_dict:Dict[str,Any], mask:sitk.Image, margin_mm: float = 0.0) -> None:
    r"""
    ### Purpose:
    - To filter out the dwell positions that are inside a given mask.

    ### Inputs:
    - self := the CatheterTable object.
    - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

    ### Outputs:
    - None
    """
    if margin_mm > 0.0:
        mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)
    filtered_dp = keep_dwell_positions(created_needle_dict["Dwell positions"], mask, condition="outside")
    created_needle_dict["Dwell positions"] = filtered_dp
    return created_needle_dict
    
def remove_dp_outside_mask(created_needle_dict:Dict[str,Any], mask:sitk.Image, margin_mm: float = 0.0) -> None:
    r"""
    ### Purpose:
    - To filter out the dwell positions that are outside a given mask.

    ### Inputs:
    - self := the CatheterTable object.
    - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

    ### Outputs:
    - None
    """
    if margin_mm > 0.0:
        mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)
    filtered_dp = keep_dwell_positions(created_needle_dict["Dwell positions"], mask, condition="inside")
    created_needle_dict["Dwell positions"] = filtered_dp
    return created_needle_dict

def keep_dwell_positions(dwell_positions:Dict[str,List[float]], mask:sitk.Image, condition:str="inside") -> None:
    r"""
    ### Purpose:
    - To keep the dwell positions that are inside or outside a given mask.
    ### Inputs:
    - mask:sitk.Image := the mask to filter the dwell positions.
    - condition:str := "inside" or "outside" to keep the dwell positions inside or outside the mask.
    ### Outputs:
    - None
    """
    assert len(dwell_positions) > 0, "Please provide dwell positions first."

    if condition == "inside":
        condition_checker = lambda x: x > 0
    elif condition == "outside":
        condition_checker = lambda x: x == 0
    else:
        raise ValueError(f"Condition {condition} not recognized. Please use 'inside' or 'outside'.")
    
    ### Update the dwell positions
    filtered_dp = {}
    for k, dp in dwell_positions.items():
        filtered_dp[k] = []
        for pt in dp:
            # pt_sitk = sitk.Point(pt)
            pt_index = mask.TransformPhysicalPointToIndex(pt)
            if condition_checker(mask.GetPixel(pt_index)):
                filtered_dp[k].append(pt)
    dwell_positions = filtered_dp
    return dwell_positions


class DwellPositionCreator:

    def __init__(
        self,
        sitk_needles_contour_path:str,
        fit_function:str="spline",
        tip_distal:bool=True, 
        multi_class:bool=True,  
        tip_class_idx:int=3,
        entry_tip_class_idx:int=2,
        catheter_core_class_idx:int=1, 
        contour_dilation:int=0, 
        treatment_site:str="breast",
        ):
        self.sitk_needles_contour_path = sitk_needles_contour_path
        assert fit_function in [
            "spline",
            "linear",
            "piecewise_linear",
        ], "fit_function should be either 'spline', 'linear' or 'piecewise_linear'"
        self.fit_function_name = fit_function
        if self.fit_function_name == "spline":
            self.fit_function = self.fit_spline
        elif self.fit_function_name == "linear":
            self.fit_function = fit_line
        elif self.fit_function_name == "piecewise_linear":
            self.fit_function = self.fit_piecewise_linear
        self.tip_distal = tip_distal
        self.multi_class = multi_class
        self.catheter_core_class_idx = catheter_core_class_idx
        self.tip_class_idx = tip_class_idx
        self.entry_tip_class_idx = entry_tip_class_idx
        self.t_margin_around_curve = 0.1
        self.contour_dilation = contour_dilation
        self.treatment_site = treatment_site
        assert self.treatment_site == "breast", (
            "Catheter labelling is not implemented for other sites for which I ignore convetions."
        )

    def preprocess_contour(self):

        separator = ContourSeparator(
            catheters_contour_path=self.sitk_needles_contour_path,reference_ct=None,
            catheter_marker_class=self.catheter_core_class_idx, catheter_diameter=2.0, 
            save_details=False, save_details_path=None, 
            log_path=os.path.dirname(self.sitk_needles_contour_path),log_file_name='digitizer_separator_logs.txt')
        contoured_needles, components_img = separator.separate_catheters(
            save_solo_components=True, 
            # The catheters either have been created manually or postprocessed 
            # if AI generated, so there should be only full catheters.
            potential_merge=False)
        
        return contoured_needles, components_img
    
    def create_points_from_contours(self, step_dwell_pos:float, for_viz:bool=False, 
                                    save_labelling_details_folder:str=None, save_files_labelling:bool=True, 
                                    ct_vol_path_for_breast_side_check:str=None, 
                                    nb_catheters_per_row_for_labelling:List[int]=None, 
                                    auto_labelling:bool=True):
        """
        Create a list of points from a contour file.

        Parameters
        ----------
        sitk_contour_path : str
            Path to the contour file.
            needle_contour_array should contain masks with a different integer assigned to each needle

        Returns
        -------
        list
            List of points.
        """

        contoured_needles, components_img = self.preprocess_contour()

        contoured_points = {}
        fitted_params = {}
        needle_points = {}
        dwell_positions = {}
        tips = {}
        distal_tips = {}
        t0pts = {}
        t1pts = {}
        for contoured_needle_idx, contoured_needle in enumerate(
            tqdm.tqdm(contoured_needles,desc="Digitizing needles...", total=len(contoured_needles))):

            if not self.is_complete_catheter(contoured_needle):
                continue

            dict_key = f"Contour_needle_{contoured_needle_idx}"

            ### Getting the physical coordinates of the points segmented as a catheter
            points_needle_contour = get_physical_coord_for_needle(contoured_needle)
            contoured_points[dict_key] = points_needle_contour
            fit_func_input = np.array(points_needle_contour)
            if self.fit_function == self.fit_spline:
                fit_func_input = contoured_needle

            ### Fitting a function to represent the catheter
            fitted_func_param = self.fit_function(fit_func_input)
            fitted_params[dict_key] = fitted_func_param

            ### Generate tip point(s)
            tips_pair, distal_tip, t0pt, t1pt = self.get_tips(
                points_from_segmentation=points_needle_contour, 
                needle_spline_creator=fitted_func_param, 
                contoured_needle=contoured_needle,
                tip_marker_distal=self.tip_distal)
            if self.multi_class:
                # We identified the tip so we can only give one set of dwell positions
                tips[dict_key] = tips_pair[0]
            else:
                # We don't know which end of the catheter contour is the tip so
                # we give two sets of dwell positions, starting from both ends.
                tips[dict_key] = tips_pair
            distal_tips[dict_key] = distal_tip
            t0pts[dict_key] = [t0pt]
            t1pts[dict_key] = [t1pt]

            # Optional
            if for_viz:
                ### Generate points from fitted function for visualization purposes
                points_needle_x_y_z = self.generate_points_from_fitted_function(fitted_func_param)
                points_needle = x_y_z_to_list(points_needle_x_y_z[0], points_needle_x_y_z[1], points_needle_x_y_z[2])
            else:
                points_needle = None
            needle_points[dict_key] = points_needle

            ### Generate usable dwell positions
            dwell_pos = self.generate_dwell_positions(
            fitted_func_param, tips_pair, step_dwell_pos=step_dwell_pos
            )
            if self.multi_class:
                # We identified the tip so we can only give one set of dwell positions
                dwell_positions[dict_key] = dwell_pos
            else:
                # We don't know which end of the catheter contour is the tip so
                # we give two sets of dwell positions, starting from both ends.
                dwell_positions_from_start, dwell_positions_from_end = dwell_pos
                dwell_positions[dict_key] = {
                    "From start": dwell_positions_from_start,
                    "From end": dwell_positions_from_end,
                }

        created_needle = {
            "Points from segmentation": contoured_points,
            "Generated needle": needle_points,
            "Fitted function": self.fit_function_name,
            "Fitted function params": fitted_params,
            "Dwell positions": dwell_positions,
            "Tips": tips,
            "Distal tips": distal_tips,
            "T0 points": t0pts,
            "T1 points": t1pts,
            "Step size": step_dwell_pos
        }
        # return created_needle, components_img

        # # We order the channel using the convention of our institution
        # # The top right corner catheter as channel 1, then the one on the left as 
        # channel 2, etc until the row is done and then we go to next row. 
        if auto_labelling:
            # If we did not create a number of catheter matching the user input then there
            # is no point in trying to order them by convention.
            total_nb_catheters = sum(nb_catheters_per_row_for_labelling)
            if total_nb_catheters != len(created_needle['Dwell positions'].keys()):
                return self.order_randomly(created_needle), components_img
            else:
                created_needle_ordered_channel = self.order_catheter_by_JGH_breast_convention(
                    created_needle,
                    ct_vol_path_for_breast_side_check, 
                    user_input=nb_catheters_per_row_for_labelling,
                    save_details_folder=save_labelling_details_folder, 
                    save_files=save_files_labelling)
        else:
            created_needle_ordered_channel = self.order_randomly(created_needle)
        return created_needle_ordered_channel, components_img


    def order_randomly(self, created_needle:Dict[str,Any]):
        """
        Order the catheter randomly.
        This is not used in practice but can be useful for testing purposes.
        """
        assert self.treatment_site == "breast", (
            "The top right to left bottom convention is used at the JGH for breast patient.",
            "This function is made for this site and should likely be adapted for other sites."
        )
        catheter_identificator = CatheterIdentificator(
            self.sitk_needles_contour_path,
            save_details_folder=None)
        created_needle_ordered_channel = catheter_identificator.order_randomly(created_needle)
        return created_needle_ordered_channel

    def order_catheter_by_JGH_breast_convention(
            self, 
            created_needle:Dict[str,Any], 
            ct_vol_path_for_breast_side_check:str=None, 
            user_input:List[int]=None,
            save_details_folder:str=None, 
            save_files:bool=True):
        """
        Order the catheter by convention.
        The top right corner catheter as channel 1, then the one on the left as 
        channel 2, etc until the row is done and then we go to below rows.

        To do this, we first rotate the catheter volume to have all the insertion in 
        the axial plane and then we order the catheter by the x and y coordinates.
        This function will fail if not all catheters have been manually inserted in 
        the same direction. 
        """
        assert self.treatment_site == "breast", (
            "The top right to left bottom convention is used at the JGH for breast patient.",
            "This function is made for this site and should likely be adapted for other sites."
        )
        catheter_identificator = CatheterIdentificator(
            ct_volume_path=ct_vol_path_for_breast_side_check,
            catheters_contour_path=self.sitk_needles_contour_path,
            save_details_folder=save_details_folder)
        created_needle_ordered_channel, identified_n_rows = catheter_identificator.order_catheter_by_JGH_breast_convention(
            created_needle, user_input=user_input, save_files=save_files)
        return created_needle_ordered_channel
       

    def is_complete_catheter(self, catheter:sitk.Image):
        """
        At this point the catheter has been postprocessed if AI generated and is complete if 
        analytically created. The only way to have an incomplete catheter at this point is 
        if an overlap was detected in the catheter and solved by the ContourSeparator but 
        that would mean  the curvature of the AI genrated catheter was super high and then 
        most likely this catheter was wrong anyway. This case happened for case 310728 alone.
        """
        catheter_array = sitk.GetArrayFromImage(catheter)
        unique_classes_in_catheter = np.concatenate(
            # Faster computation ith masking > 0 voxels.
            [np.array([0]), np.unique(catheter_array[catheter_array > 0])]
            )
        if self.multi_class:
            all_classes = np.array([self.catheter_core_class_idx, self.tip_class_idx, self.entry_tip_class_idx])
        else:
            all_classes = np.array([self.catheter_core_class_idx])
        if not np.all(np.isin(all_classes, unique_classes_in_catheter)):
            return False
        else:
            return True
    
    def get_tips(self, 
                 points_from_segmentation:List[List[float]], 
                 needle_spline_creator:NeedleSplineCreator, 
                 contoured_needle:sitk.Image,
                 tip_marker_distal:bool=True, 
                 tip_marker_size:float=3.):
        """
        Getting the tip of the catheter. If tip-class-idx is provided, we use this class to 
        identify the tip, if it is None we create two tips from the endpoints of the sgmented 
        contour. Either way, we take the tip voxel coordinates, and project them on the spline
        we fitted.
        """
        # Getting extremum points from the predicted needle
        # In any case, the points_from_segmentation contain all the voxels
        # representing the catheter, no matter it the catheter is divided in 
        # multiple class or not.
        seg = Segment(
            points_from_segmentation,
            ref_slice_coord=None,
            interslice_ax=2,
            init_line=False,
            init_2D=False,
        )
        # Getting both tips since we don't know which one has been used
        # as a starting point during digitization
        # These are the voxel coordinates of the two extremum contoured points.
        potential_seg_tip_voxel_coords = seg.extremum_points

        # Creating a single tip specific to the tip class index
        if self.multi_class:

            needle_array = sitk.GetArrayFromImage(contoured_needle)

            if np.max(needle_array) == self.tip_class_idx:
                # Tip was found we proceed with with tip identification knowing tip marker info
                most_distal_identified_part = self.tip_class_idx
            else:
                assert np.any(needle_array==self.entry_tip_class_idx), (
                    "Neither tip class nor entry tip class found in needle array."
                    )
                # Identifying the extremum close to the tip.
                most_distal_identified_part = self.entry_tip_class_idx

            # Identifying the extremum close to the tip or entry tip.
            all_tip_pts = np.where(needle_array==most_distal_identified_part)
            random_tip_pt = all_tip_pts[0][0], all_tip_pts[1][0], all_tip_pts[2][0]
            random_tip_pt_coords = contoured_needle.TransformIndexToPhysicalPoint(
                [int(ax_idx) for ax_idx in random_tip_pt][::-1]) 
            
            # Checking with any two endpoints of tip marker is fine to identify
            # the correct catheter contour endpoint for the tip position. 
            if distance(random_tip_pt_coords, potential_seg_tip_voxel_coords[0]) < distance(
                random_tip_pt_coords, potential_seg_tip_voxel_coords[1]):
                distal_tip_seg_voxel_coords = potential_seg_tip_voxel_coords[0]
                other_catheter_seg_endpoint = potential_seg_tip_voxel_coords[1]
            else:
                distal_tip_seg_voxel_coords = potential_seg_tip_voxel_coords[1]
                other_catheter_seg_endpoint = potential_seg_tip_voxel_coords[0]

            # At this point we identified from the whole cateter segmented, which endpoint 
            # corresponds to the tip or entry tip. Now we can make the decision on wether 
            # we place the tip coordinate point (1st dwel position) at most distal or at 
            # the beginning of the tip marker (class tip_class_idx) or 0.3cm from the entry 
            # tip.
            projected_distal_tip_point, projected_distal_tip_t, _ = needle_spline_creator.project_on_spline(
                    distal_tip_seg_voxel_coords)

            assert (projected_distal_tip_t < 1.1 and projected_distal_tip_t > 0.9) or (
                projected_distal_tip_t < 0.1 and projected_distal_tip_t > -0.1), (
                    "Projected distal tip t should be around 1.0 or 0.0, not "
                    f"{projected_distal_tip_t}."
                )
            
            if self.contour_dilation > 0:

                print("88888888888888888888888888888888888888888888888888888")
                print("Movidng the tip from dilation of contour")
                print("88888888888888888888888888888888888888888888888888888")
                # The contour was dilated so if we take the most distal voxel of 
                # the tip marker, it should be far from self.contour_dilation
                # of the actual tip. So we will move the tip if there was dilation
                # in the contour of self.contour_dilation * spacing (in mm).
                spacings = np.array(contoured_needle.GetSpacing())
                assert np.allclose(spacings, 1.), (
                    "Contour dilation should only be implemented for isotropic spacing."
                    )
                if abs(projected_distal_tip_t-0.0) < abs(projected_distal_tip_t-1.0):
                    # projected_distal_tip_t is around 0.0, bounds [0.0, 0.1]
                    bound_min = projected_distal_tip_t 
                    bound_max = projected_distal_tip_t + 0.1
                else:
                    # projected_distal_tip_t is around 1.0, bounds [0.9, 1.0]
                    bound_min = projected_distal_tip_t - 0.1
                    bound_max = projected_distal_tip_t 
            
                projected_distal_tip_point, projected_distal_tip_t = needle_spline_creator.step_in_spline(
                        projected_distal_tip_point,
                        step=float(self.contour_dilation) * spacings[0],
                        bound_min=bound_min,
                        bound_max=bound_max,
                        arc=True
                    )
            
            if tip_marker_distal:
                # Most distal part of tip marker. Also most distal part of catheter contour.
                if most_distal_identified_part == self.tip_class_idx:
                    identified_tip_point = projected_distal_tip_point
                    tip_t = projected_distal_tip_t
                else:
                    # Entry of tip marker distal part is the projected_distal_tip_t. 
                    # We place the tip 0.3cm from the entry of the tip marker.
                    if abs(projected_distal_tip_t-0.0) < abs(projected_distal_tip_t-1.0):
                        # projected_distal_tip_t is around 0.0, bounds [-0.1, 0.0]
                        bound_min = projected_distal_tip_t - 0.1
                        bound_max = projected_distal_tip_t
                    else:
                        # projected_distal_tip_t is around 1.0, bounds [1.0, 1.1]
                        bound_min = projected_distal_tip_t
                        bound_max = projected_distal_tip_t + 0.1
                
                    identified_tip_point, tip_t = needle_spline_creator.step_in_spline(
                            projected_distal_tip_point,
                            step=tip_marker_size,
                            bound_min=bound_min,
                            bound_max=bound_max,
                            arc=True
                        )

            else:
                if most_distal_identified_part == self.tip_class_idx:
                    # Start of tip marker. Not most distal part of catheter contour.
                    if abs(projected_distal_tip_t-0.0) < abs(projected_distal_tip_t-1.0):
                        bound_min = projected_distal_tip_t
                        bound_max = 1. + self.t_margin_around_curve
                    else:
                        bound_min = -self.t_margin_around_curve,
                        bound_max = projected_distal_tip_t
                    identified_tip_point, tip_t = needle_spline_creator.step_in_spline(
                            projected_distal_tip_point,
                            step=tip_marker_size,
                            bound_min=bound_min,
                            bound_max=bound_max,
                            arc=True
                        )
                else:
                    # Start of tip marker can be assimilated to distal point of entry tip marker.
                    identified_tip_point = projected_distal_tip_point
                    tip_t = projected_distal_tip_t

            other_end_point, endpt_t, _ = needle_spline_creator.project_on_spline(
                        other_catheter_seg_endpoint)
            
            # First tuple is the tip point, second is the other endpoint.
            return [(identified_tip_point, tip_t), (other_end_point, endpt_t)], projected_distal_tip_point, needle_spline_creator.get_point_from_spline(0), needle_spline_creator.get_point_from_spline(1)
            
        # Getting two potential tips from the contour
        else:
            # Getting both tips since we don't know which one has been used
            # as a starting point during digitization
            # These are the voxel coordinates of the two extremum contoured points.

            # We have to project now the tip points on the catheter
            tip_pts_on_catheter = []
            # t should be between -self.t_margin_around_curve, and 1. + self.t_margin_around_curve maximum because the spline was interpolated
            # within a t range of [0,1]
            previous_t = 9.0
            for tip_voxel_coord in potential_seg_tip_voxel_coords:
                point, t = needle_spline_creator.step_in_spline(
                    tip_voxel_coord,
                    step=0.0,
                    bound_min=-self.t_margin_around_curve,
                    bound_max=1. + self.t_margin_around_curve,
                    arc=True
                )

                # Ordering the tip points, the one with the smallest t (around 0) is the
                # first in the list and the one with the largest t (around 1) is the last.
                if t < previous_t:
                    previous_t = t
                    tip_pts_on_catheter.insert(0, (point, t))
                else:
                    tip_pts_on_catheter.append((point, t))

            ### Ordering by Z axis. Be careful with the cosine matrix/coordinate system!!
            coord_sys_z_direction = sitk.ReadImage(
                self.sitk_needles_contour_path).GetDirection()[-1]
            if coord_sys_z_direction < 0:
                # The z axis is inverted, so we want to keep the tip point with the largest z coordinate
                # We want to keep the tip point with the largest z coordinate
                if tip_pts_on_catheter[0][0][2] < tip_pts_on_catheter[1][0][2]:
                    # The first point is the one with the smallest z coordinate
                    # We want to keep the tip point with the largest z coordinate
                    tip_pts_on_catheter.reverse()
            else:
                # The z axis is not inverted, so we want to keep the tip point with the smallest z coordinate
                # We want to keep the tip point with the smallest z coordinate
                if tip_pts_on_catheter[0][0][2]  > tip_pts_on_catheter[1][0][2]:
                    # The first point is the one with the largest z coordinate
                    # We want to keep the tip point with the smallest z coordinate
                    tip_pts_on_catheter.reverse()
            return tip_pts_on_catheter, None, None, None

    def generate_dwell_positions(self, fitted_func_param:NeedleSplineCreator, 
                                 tips:List[tuple[List[float], float]], step_dwell_pos:float=2.5):
        """
        Generate dwell positions from a fitted function.
        """
        if self.fit_function == self.fit_spline:
            if not self.multi_class:
                ### Creating dwell positions from starting point
                dwell_positions_from_start = self._generate_dwellpos_between_two_pts(
                    fitted_func_param, tips[0], tips[1], step_dwell_pos
                )
                ### Creating dwell positions from ending point
                dwell_positions_from_end = self._generate_dwellpos_between_two_pts(
                    fitted_func_param, tips[1], tips[0], step_dwell_pos
                )
                return dwell_positions_from_start, dwell_positions_from_end
            else:
                # There is a specific class for the tip that allowed to identify the tip.
                # Only one tip => one set of dwell positions. 
                dwell_positions = self._generate_dwellpos_between_two_pts(
                    # Here we know for sure tips[0] is the identified tip and tips[1] is the other end point.
                    fitted_func_param, tips[0], tips[1], step_dwell_pos
                )
                return dwell_positions
        else:
            raise NotImplementedError(
                """Only implemented for spline for now, 
                since this way of creating the catheter is the most true to ground truth.""")
            

    def _generate_dwellpos_between_two_pts(
            self, fitted_func_param:NeedleSplineCreator, start_pt:tuple[List[float], float],
            end_pt:tuple[List[float], float], step_dwell_pos:float):
        """
        Generate dwell positions between two points, from start_pt to end_pt.
        The curve is defined between t=0.0 and t=1.0, which is how we know how to define the 
        bounds for the optimization.
        """
        start_pt_coord, startpt_t = start_pt
        end_pt_coord, endpt_t = end_pt

        needle_spline_creator = fitted_func_param
        dwell_positions = []
        previous_pt = start_pt_coord
        distance_to_last = distance(start_pt_coord, end_pt_coord)
        dwell_positions.append(list(previous_pt))
        
        t_used = startpt_t

        if abs(t_used - 0) < abs(t_used - 1):
            # The starting point is closer to the beginning of the spline
            # T used around 0.0, not necessarily ==0.0 since we can place the 
            # tip not at the most distal part of the marker but at the entry
            # of the tip marker.
            bound_min = t_used
            # If bound max is endpt_t, the last point will be the end_tip
            # this is what the bounded optimization finds
            # but the distance to the last point will be less than step_dwell_pos
            # This is a decision to be made.
            # I considered that the most important is one of the tip points.
            # But once the tip point is fixed (only one of them), the subsequently
            # created dwell positions are derived from this only tip point and the last
            # dwell position can potentially be after the second (unused) tip point.
            bound_max = endpt_t + self.t_margin_around_curve
            start_pt_close_to_t0 = True
        else:
            # The starting point is closer to the end of the spline
            # T used around 1.0, not necessarily ==1.0 since we can place the 
            # tip not at the most distal part of the marker but at the entry
            # of the tip marker.
            # Same choice here for bound_min
            bound_min = endpt_t - self.t_margin_around_curve
            bound_max = t_used
            start_pt_close_to_t0 = False

        while distance_to_last > step_dwell_pos / 2:
            point, t = needle_spline_creator.step_in_spline(
                previous_pt,
                step_dwell_pos,
                bound_min=bound_min,
                bound_max=bound_max,
                arc=True
            )

            dwell_positions.append(point)
            previous_pt = point
            t_used = t
            distance_to_last = needle_spline_creator.distance_on_spline(point, end_pt_coord)
            if start_pt_close_to_t0:
                bound_min = t_used
            else:
                bound_max = t_used

        return dwell_positions
        

    def generate_points_from_fitted_function(self, fitted_func_param, nb_points=1000):
        """
        Generate points from a fitted function. More for visualization purposes.
        """
        t = np.linspace(-0.5, 1.5, nb_points)
        if self.fit_function == self.fit_spline_raw:
            tck, u = fitted_func_param
            x_coord, y_coord, z_coord = splev(t, tck)

        if self.fit_function == self.fit_spline:
            needle_spline_creator = fitted_func_param
            pts = [needle_spline_creator.get_point_from_spline(t_val) for t_val in t]
            x_coord, y_coord, z_coord = list_to_x_y_z(pts)

        elif self.fit_function == fit_line:
            mean, direction = fitted_func_param
            points = mean + t[:, np.newaxis] * direction
            x_coord, y_coord, z_coord = points[:, 0], points[:, 1], points[:, 2]

        elif self.fit_function == self.fit_piecewise_linear:
            x_coord, y_coord, z_coord = [], [], []
            evaluator = fitted_func_param
            range_sampling = t[-1] - t[0]
            for t_val in t:
                # Scaling back the t to [0,1] because the range is alread modified
                # in the fit_piecewise_linear get_points function
                x, y, z = evaluator.evaluate((t_val - t[0]) / range_sampling)
                x_coord.append(x)
                y_coord.append(y)
                z_coord.append(z)

        elif self.fit_function == self.fit_piecewise_linear_raw:
            change_points, fitted_params = fitted_func_param
            x_coord, y_coord, z_coord = [], [], []
            for segment_index in range(len(change_points) - 1):
                t = np.linspace(0, 1, 100)
                params_x, params_y, params_z = fitted_params[segment_index]
                x = linear_func(t, *params_x)
                y = linear_func(t, *params_y)
                z = linear_func(t, *params_z)
                x_coord.extend(x)
                y_coord.extend(y)
                z_coord.extend(z)
        return x_coord, y_coord, z_coord
    
    def fit_spline(self, neelde_contour: sitk.Image):
        warnings.warn("TEST SPLINE HYPER PARAMETERS")
        needle_spline_creator = NeedleSplineCreator(neelde_contour, multiclass=self.multi_class)
        needle_spline_creator.interpolate_spline()
        return needle_spline_creator

    @staticmethod
    def fit_spline_raw(points):
        return fit_spline(points, s=5, k=2, nest=0)

    @staticmethod
    def fit_piecewise_linear(points):

        # We first split the points into segments by slice
        segments_by_slice = create_segments_by_slice(points)

        # Then we let the SegmentMerger merge the segments
        # and create a line per segment
        piecewise_linear_func = PiecewiseLinear3D(segments_by_slice)
        return piecewise_linear_func

    @staticmethod
    def fit_piecewise_linear_raw(points, topk=True):
        # Reorder points based on nearest neighbor
        ordered_points = reorder_points_tsp(points)  # reorder_points_nneighbor(points)
        print("length of ordered_points", len(ordered_points))
        # Calculate differences between consecutive points in the reordered array
        diffs = np.diff(ordered_points, axis=0)

        # Calculate norms of differences
        norms = np.linalg.norm(diffs, axis=1)

        # Detect change points based on changes in direction
        all_peaks, _ = find_peaks(norms, height=np.mean(norms))
        if topk:
            # Sort peaks by the norm values in descending order
            sorted_peaks = np.argsort(norms[all_peaks])[::-1]
            print("sorted_peaks", sorted_peaks)
            # Interslice should be 1mm
            print(
                "WARNING: Interslice should be 1mm to select top cahnges in direction."
            )
            sorted_peaks = [peak for peak in sorted_peaks if norms[peak] >= 1.0]
            print("filtered sorted_peaks", sorted_peaks)
            all_peaks = np.sort(all_peaks[sorted_peaks])
            # print("k first norm values", np.sort(norms[all_peaks])[::-1][:topk])
            # # Select the top k peaks (or fewer if there are less than topk)
            # all_peaks = np.sort(all_peaks[:topk])

        # Add the start and end of the sequence
        change_points = np.insert(all_peaks, 0, 0)

        change_points = np.append(change_points, len(ordered_points) - 1)
        print("change_points", change_points)
        # Segments are between change points
        segments = [
            ordered_points[change_points[i] : change_points[i + 1] + 1]
            for i in range(len(change_points) - 1)
        ]
        # print("segments", segments)
        fitted_params = [fit_segment(segment) for segment in segments]

        return change_points, fitted_params


def select_best_needle_from_list(
    dp_lists_to_select_from:Dict[str,List[List[float]]],
    reference_single_needle_dwell_positions: List[float],
    key: str = None,
    return_min:bool=False
):
    """
    Finding the matching needles in a dictionary that is closest to the provided
    list of dwell positions.
    """
    avg_distance = np.inf
    closest_needle_idx = None
    for needle_idx, dp in dp_lists_to_select_from.items():
        if key is not None:
            to_compare = dp[key]
        else:
            to_compare = dp
        avg_distance_needle = avg_dist_closest_pts_two_lists(
            to_compare,
            reference_single_needle_dwell_positions
            )

        if avg_distance_needle < avg_distance:
            avg_distance = avg_distance_needle
            closest_needle_idx = needle_idx
    if return_min:
        return closest_needle_idx, avg_distance
    else:
        return closest_needle_idx

class CatheterTableTimesFiller:
    def __init__(self, created_needle_dict:Dict[str,Any], altered:bool=False):
        """
        Class made to create a catheter table that matches brachyutils catheter table format and 
        assign dwell times to the created dwell positions. 
        Args:
            created_needle_dict (dict): Dictionary with the created dwell positions from DwellPositionCreator class.
        """
        self.created_needle_dict = created_needle_dict
        if isinstance(self.created_needle_dict["Dwell positions"]["Channel_1"], dict):
            # We have two sets od dwell positions, "From start" and "From end"
            self.oneset = False
        else:
            assert isinstance(self.created_needle_dict["Dwell positions"]["Channel_1"], list), (
                "Dwell positions should be a list of dwell positions or a dictionary with two lists of dwell positions."
            )
            # We have only one set of dwell positions because we could identify
            # the tip from a specific class.
            self.oneset = True
        self.catheter_table = None
        
        # Altered means that dwell positions were created and then altered
        # by removing some dwell positions that were outside or inside a contour.
        self.altered = altered
        self.zerosec_table = self.create_zerosec_brachyutils_catheter_table(self.created_needle_dict["Dwell positions"], self.altered)
        for catheter_idx, catheter in enumerate(self.zerosec_table):
            point_list = [dp["position"].tolist() for dp in catheter["dwells"]]
            if len(point_list) == 0:
                print("No points found for catheter ", catheter_idx+1)
                assert self.altered, (
                    "No dwell positions found for catheter "f"{catheter_idx+1} but altered is False."
                )

    def create_zerosec_brachyutils_catheter_table(self, template_catheter_dwell_pos:dict, altered:bool=False):
        """
        Create a catheter table with 0 sec dwell time for each dwell position.
        The catheter table matches brachyutils catheter table format.
        Recomputes angle for the dwell positions.
        """
        temp_gen_needles = template_catheter_dwell_pos.copy()
        zerosec_catheter_table = []
        if self.oneset:
            total_count_dp = 0
        else:
            total_count_dp = {
                "From start": 0, 
                "From end": 0
            }

        for needle_idx, needle in temp_gen_needles.items():
            if self.oneset:
                zerosec_catheter_table.append({
                    "channel_number": int(needle_idx.split("_")[-1]),
                    # Why is "points" useful in brachyutils? TODO: ask Hossein
                    "points": [],
                    # Channel total time updated below
                    "channel_total_time": 0.,
                    "dwells": [],
                })
                count_dp = 0
                dwells = []
                if len(needle) == 0:
                    assert self.altered, (
                        "No dwell positions found for needle "f"{needle_idx} but altered is False."
                    )
                    # If masked to be inside the PTV it can happen that a catheter has no dwell positions left.
                    continue
                step_size = float(np.round(cdist([needle[1]],[needle[0]])[0][0], 2))
                for dp_idx, dp in enumerate(needle):
                    if dp_idx > 0:
                        d = cdist([dp],[needle[dp_idx-1]])
                        # Checking only at 0.1mm precision since the dwell positions have
                        # been placed using distance on spline (arc length) and not distance
                        # which can vary.
                        if altered:
                            # Weaker condition since some dwell positions might have been removed
                            # Allow d to be close to any multiple of step_size within 0.1
                            multiples = np.round(d / step_size)
                            closest_multiple = multiples * step_size
                            assert np.isclose(d, closest_multiple, atol=1e-1), (
                                f"Dwell positions are not ordered, distance {d} is not close to a multiple of step size {step_size}"
                            )
                        else:
                            # Stronger condition
                            assert np.isclose(d, step_size, atol=1e-1), (
                                f"Dwell positions are not ordered, distance {d} VS step size {step_size}"
                            )
                    dwells.append(
                        {
                            "index": float(dp_idx),
                            # We are not using any shielding
                            "angle": 0.0,
                            "position": np.array(dp, dtype=np.float32),
                            # Relative position https://dicom.innolitics.com/ciods/rt-plan/rt-brachy-application-setups/300a0230/300a0280/300a02d0/300a02d2
                            "relativePos": dp_idx * step_size,
                            # Rotation updated below
                            "rotation": 999,
                            "time": 0.0,
                            "weight": 0.0,
                        }
                    )
                    count_dp += 1
                # Compute rotation from the different dwell positions 
                for i in range(len(dwells)):
                    dwells[i]["rotation"] = get_rotation_from_position(i, dwells)

                zerosec_catheter_table[-1][f"dwells"] = dwells
                
                total_count_dp += count_dp
                # 1 sec for each dwell position
                zerosec_catheter_table[-1][f"channel_total_time"] = 0.
            
            else:
                zerosec_catheter_table.append({
                    "channel_number": int(needle_idx.split("_")[-1]),
                    "points": [],
                    "channel_total_time_from_start": 999,
                    "channel_total_time_from_end": 999,
                    "dwells_from_start": [],
                    "dwells_from_end": [],
                })

                for set_dell_pos in ["From start", "From end"]:
                    count_dp = 0
                    dwells = []
                    step_size = float(np.round(cdist([needle[set_dell_pos][1]],[needle[set_dell_pos][0]])[0][0], 2))
                    for dp_idx, dp in enumerate(needle[set_dell_pos]):
                        if dp_idx > 0:
                            assert np.allclose(cdist([dp],[needle[set_dell_pos][dp_idx-1]]), step_size, atol=1e-3), (
                                f"Dwell positions are not ordered"
                            )
                        dwells.append(
                            {
                                "index": float(dp_idx),
                                # We are not using any shielding
                                "angle": 0.0,
                                "position": np.array(dp, dtype=np.float32),
                                # Relative position https://dicom.innolitics.com/ciods/rt-plan/rt-brachy-application-setups/300a0230/300a0280/300a02d0/300a02d2
                                "relativePos": dp_idx * step_size,
                                # Rotation updated below
                                "rotation": 999,
                                "time": 0.0,
                                "weight": 0.0,
                            }
                        )
                        count_dp += 1
                    # Compute rotation from the different dwell positions 
                    for i in range(len(dwells)):
                        dwells[i]["rotation"] = get_rotation_from_position(i, dwells)

                    zerosec_catheter_table[-1][f"dwells_{('_').join(set_dell_pos.lower().split(' '))}"] = dwells
                    
                    total_count_dp[set_dell_pos] += count_dp
                    # 1 sec for each dwell position
                    zerosec_catheter_table[-1][f"channel_total_time_{('_').join(set_dell_pos.lower().split(' '))}"] = 0.
                
        return zerosec_catheter_table

    def get_catheter_table_mapping(
            self, treatment_catheter_table:Dict[str,Any], 
            catheter_table_used_for_mapping:Dict[str,Any]):
        """
        Get mapping between treatment catheter table and created catheter table.
        Returns:
            catheter_table_used_for_mapping: Catheter table used for mapping with treatment catheter table.
            treatment_to_new: Dictionary mapping treatment catheter index to created catheter index.
        """

        treatment_to_new = {}

        # Match needle from brachyutils BrachyPlan and our created needles
        for i in range(len(treatment_catheter_table)):

            treatment_needle_dp = [dp["position"] for dp in treatment_catheter_table[i]["dwells"]]
            if len(treatment_needle_dp) == 0:
                print(f"Needle {i} has no dwell positions in the treatment plan.")
                continue
            if self.oneset:
                dwellpos_to_select = "dwells"
            else:
                dwellpos_to_select = "dwells_from_start"
            gen_key, avg_min_dw_to_dw_dist = select_best_needle_from_list(
                dp_lists_to_select_from={
                    needle["channel_number"]:[dp["position"] for dp in needle[dwellpos_to_select]]
                    for needle in catheter_table_used_for_mapping
                }, 
                reference_single_needle_dwell_positions=treatment_needle_dp, 
                return_min=True
                )
            # if avg_min_dw_to_dw_dist > 10:
            #     print(f"Needle {i} has no match with a created needle.")
            #     continue
            print(f"Treatment needle {i} is matched with created needle {gen_key}")
            idx_created_needle= np.where([needle["channel_number"] == gen_key for needle in catheter_table_used_for_mapping])[0][0]
            treatment_to_new[treatment_catheter_table[i]["channel_number"]] = int(idx_created_needle)

        assert len(set(treatment_to_new.values())) == len(treatment_to_new.values()), (
                f"Multiple treatment needles are matched with the same created needle. {treatment_to_new}"
            )
        return treatment_to_new
    
    def get_catheter_table_with_treatment_table_times(
            self, treatment_catheter_table:Dict[str,Any], 
            catheter_table_used_for_mapping:Dict[str,Any]=None, 
            mode: str ="closest"):
        """
        Assign created dwell positions the time of the closest real dwell positions.
        """
        if catheter_table_used_for_mapping is None:
            catheter_table_used_for_mapping = self.zerosec_table.copy()
        
        treatment_to_new = self.get_catheter_table_mapping(
            treatment_catheter_table, 
            catheter_table_used_for_mapping
            )
        
        new_catheter_table = self.zerosec_table.copy()
        # Resetting dwell time and weight that will be updated
        for needle in new_catheter_table:
            if self.oneset:
                for dwell in needle["dwells"]:
                    dwell["time"] = 0.0
                    dwell["weight"] = 0.0
            else:
                for dwell in needle["dwells_from_start"]:
                    dwell["time"] = 0.0
                    dwell["weight"] = 0.0
                for dwell in needle["dwells_from_end"]:
                    dwell["time"] = 0.0
                    dwell["weight"] = 0.0

        
        # For each needle, assign dwell times to new dwell positions. 
        # For now we only assign dwell times to corresponding closest dwell psoition. 
        if mode == "closest":
            if not self.oneset:
                fs_distances = []
                fe_distances = []
            for ref_needle in treatment_catheter_table:
                # If treatment catheter table is empty (no dwell positions) we move to next one
                if len(ref_needle["dwells"]) == 0:
                    continue
                # We just modify the dwell positions
                # We have two sets of created dwell positions, we assign dwell positions
                # for both. After we will only use the closest set.
                # if ref_needle['channel_number'] not in treatment_to_new:
                #     print(f"Needle {ref_needle['channel_number']} has no match with a created needle.")
                #     continue
                print(f"Handling Needle {ref_needle['channel_number']} matched with {treatment_to_new[ref_needle['channel_number']]}")

                # This function modifies the dwell times
                distances_set = self.assign_original_plan_dwelltimes_to_created_dwellpos(
                    ref_needle,
                    new_catheter_table[treatment_to_new[ref_needle['channel_number']]]
                    )
                if not self.oneset:
                    fs_distances.append(distances_set["dwells_from_start"])
                    fe_distances.append(distances_set["dwells_from_end"])

            if not self.oneset:
                # Get closest set of dwell positions between "From start" and "From end".
                # Only for one needle is sufficient since tip should be on the same side for
                # every needle.
                # Tip is still undefined so we have one set of dwell positions from each side
                # of the needle.
                if np.mean(np.concatenate(fs_distances, axis=0)) < np.mean(np.concatenate(fe_distances, axis=0)):
                    print("Chosen set of dwell positions are the set from start.")
                    to_assign = "From start"
                else:
                    print("Chosen set of dwell positions are the set from end.")
                    to_assign = "From end"

            for treatment_idx in range(len(treatment_catheter_table)):
                # Needle treatment_idx has no dwell positions in the treatment plan.
                if not treatment_catheter_table[treatment_idx]["channel_number"] in treatment_to_new:
                    continue
                created_idx = treatment_to_new[treatment_catheter_table[treatment_idx]["channel_number"]]
                if self.oneset:
                    new_catheter_table[created_idx]["channel_total_time"] = sum(
                        [dp["time"] for dp in new_catheter_table[created_idx]["dwells"]])
                    assert np.isclose(
                        new_catheter_table[created_idx]["channel_total_time"], 
                        treatment_catheter_table[treatment_idx]["channel_total_time"],
                        ), "{} != {} for new catheter {} VS treatment catheter {}".format(
                            new_catheter_table[created_idx]["channel_total_time"],
                            treatment_catheter_table[treatment_idx]["channel_total_time"],
                            created_idx,
                            treatment_idx   
                        )
                else:
                    new_catheter_table[created_idx]["channel_total_time_from_start"] = sum(
                        [dp["time"] for dp in new_catheter_table[created_idx]["dwells_from_start"]])
                    new_catheter_table[created_idx]["channel_total_time_from_end"] = sum(
                        [dp["time"] for dp in new_catheter_table[created_idx]["dwells_from_end"]])
                    assert np.isclose(
                        new_catheter_table[created_idx]["channel_total_time_from_start"], 
                        new_catheter_table[created_idx]["channel_total_time_from_end"]
                        )
                    assert np.isclose(
                        new_catheter_table[created_idx]["channel_total_time_from_start"], 
                        treatment_catheter_table[treatment_idx]["channel_total_time"]
                        )
                    new_catheter_table[created_idx]["channel_total_time"] = treatment_catheter_table[treatment_idx]["channel_total_time"]

                    # Choose correct dwell positions
                    if to_assign == "From start":
                        new_catheter_table[created_idx]["dwells"] = new_catheter_table[created_idx]["dwells_from_start"].copy()
                    else:
                        new_catheter_table[created_idx]["dwells"] = new_catheter_table[created_idx]["dwells_from_end"].copy()
        
                    del new_catheter_table[created_idx]["dwells_from_start"]
                    del new_catheter_table[created_idx]["dwells_from_end"]
                    del new_catheter_table[created_idx]["channel_total_time_from_start"]
                    del new_catheter_table[created_idx]["channel_total_time_from_end"]

            
            if self.oneset:
                for catheter_idx in range(len(new_catheter_table)):
                    # If channel has not been updated acoording to the treatment plan
                    # it means it has not been used => dwell_times = 0.0
                    if new_catheter_table[catheter_idx]["channel_total_time"] == len(new_catheter_table[catheter_idx]["dwells"]):
                        new_catheter_table[catheter_idx]['channel_total_time'] = 0.0
                        new_dwell_positions = []
                        for dp in new_catheter_table[catheter_idx]["dwells"]:
                            new_dwell_positions.append(
                                {
                                    "index": dp["index"],
                                    "angle": dp["angle"],
                                    "position": dp["position"],
                                    "relativePos": dp["relativePos"],
                                    "rotation": dp["rotation"],
                                    "time": 0.0,
                                    "weight": 0.0,
                                }
                            )
                        new_catheter_table[catheter_idx]["dwells"] = new_dwell_positions

            else: 
                # Renaming channel_total_time_from_start to channel_total_time for unsued catheters
                for catheter_idx in range(len(new_catheter_table)):
                    if "dwells_from_start" in new_catheter_table[catheter_idx]:
                        new_catheter_table[catheter_idx] = {
                            'channel_number': new_catheter_table[catheter_idx]['channel_number'], 
                            'points': [], 
                            'channel_total_time': 0.0, 
                            'dwells': []
                            }
                    
            # Assign correct weights
            total_treatment_time = sum([catheter["channel_total_time"] for catheter in new_catheter_table])
            for catheter in new_catheter_table:
                for dp in catheter["dwells"]:
                    dp["weight"] = dp["time"] / total_treatment_time
            return new_catheter_table
        else:
            raise NotImplementedError("Only closest mode is implemented for now")

    def assign_original_plan_dwelltimes_to_created_dwellpos(self, ref_dwell_positions:List[dict], created_dwell_positions:List[dict]):
        """
        Assign dwell times from the reference dwell positions to the closest created dwell positions.
        """
        if self.oneset:

            all_distances = []
            for ref_dwell_pos in ref_dwell_positions["dwells"]:
                ref_dps = [ref_dwell_pos["position"]]
                created_dps = [dp["position"] for dp in created_dwell_positions["dwells"]]
                # Distance from one ref dwell pos to all created dwell positions
                distances = cdist(ref_dps, created_dps, metric="euclidean")
                all_distances.append(np.min(distances))
                # Get the index of the closest dwell position
                closest_dp_index = np.argmin(distances, axis=1)[0]

                # Ensure RelativePosition is within a reasonnable error
                step_size = float(np.round(cdist(
                    [created_dwell_positions["dwells"][1]["position"]],[created_dwell_positions["dwells"][0]["position"]]
                    )[0][0], 2))
                diff_relativ_pos = created_dwell_positions["dwells"][closest_dp_index]["relativePos"] - ref_dwell_pos["relativePos"]
                # Here we assume that the biggest error we can do from the tip is 5 times the step size
                # assert np.abs(diff_relativ_pos) <= 5 * step_size, (
                #     "Relative position is not the same. {} VS {}.".format(
                #         created_dwell_positions["dwells"][closest_dp_index]["relativePos"],
                #         ref_dwell_pos["relativePos"]
                #     )
                # )
                if not created_dwell_positions["dwells"][closest_dp_index]["time"] == 0.0:
                    print('In created ', ref_dwell_pos)
                    print("This dwell already had a non 0s dwell time: ", created_dwell_positions["dwells"][closest_dp_index])
                    print("And we will add ", ref_dwell_pos["time"], "s. from ", ref_dwell_pos)
                # Assign dwell time to the closest dwell position
                created_dwell_positions["dwells"][closest_dp_index]["time"] += ref_dwell_pos["time"]

            return all_distances
        else:
            all_distances_set = {}

            for set_dwell_pos in ["dwells_from_start", "dwells_from_end"]:
                all_distances = []
                for ref_dwell_pos in ref_dwell_positions["dwells"]:
                    ref_dps = [ref_dwell_pos["position"]]
                    created_dps = [dp["position"] for dp in created_dwell_positions[set_dwell_pos]]
                    # Distance from one ref dwell pos to all created dwell positions
                    distances = cdist(ref_dps, created_dps, metric="euclidean")
                    all_distances.append(np.min(distances))
                    # Get the index of the closest dwell position
                    closest_dp_index = np.argmin(distances, axis=1)[0]

                    if not created_dwell_positions[set_dwell_pos][closest_dp_index]["time"] == 0.0:
                        print('In created ', set_dwell_pos)
                        print("This dwell already had a non 0s dwell time: ", created_dwell_positions[set_dwell_pos][closest_dp_index])
                        print("And we will add ", ref_dwell_pos["time"], "s. from ", ref_dwell_pos)
                    # Assign dwell time to the closest dwell position
                    created_dwell_positions[set_dwell_pos][closest_dp_index]["time"] += ref_dwell_pos["time"]
                all_distances_set[set_dwell_pos] = all_distances

            return all_distances_set

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('tkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.creator import CatheterContourCreator
    # from brachyutils.dicom_utils import get_catheter_table_and_source_info_from_dicom


    patient_path = "/home/sebquet/EngerLab/Data/Hamed_breastCancer_patient/"
    patient_nb = "6515" #"1331978"
    patient_path = f"/home/sebquet/EngerLab/Data/export_seb/patients/{patient_nb}/"

    dataset_name = "Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1_bs8_threshold4"

    benchmark = "test_fold01234" #"train_benchmark_fold01234" # "test"
    contour = "ai_generated_catheters_postprocessed" #"analytically_generated_catheters" #"ai_generated_catheters_postprocessed_new2" # "analytically_generated_catheters" # ai_generated_catheters
    # sitk_needles_contour_path = f"/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/{dataset_name}/{benchmark}/{patient_nb}/{contour}.seg.nrrd"
    # sitk_needles_contour_path = os.path.join(patient_path, "processed", "catheters.seg.nrrd")
    sitk_needles_contour_path = "/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/nnUNet_raw/Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1/labelsTr/case_6515.nrrd"

    catheter_contour_dilation = (0 if not "dilation" in dataset_name else
                int(dataset_name.split("dilation_")[-1].split("_")[0]))
    if not os.path.exists(sitk_needles_contour_path):
        creator = CatheterContourCreator(
            patient_path,
            patient_volume_path=os.path.join(patient_path, "processed", "CT.nrrd"),
            # patient_volume_path="/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/nnUNet_raw/Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1/imagesTr/case_6515_0000.nrrd",
            dilation=catheter_contour_dilation,
        )
        catheter_contour = creator.create_catheter_contour(write=True, out_path=sitk_needles_contour_path)

    patient_plan = CatheterSetUp(patient_path)
    dwellpos_dict = patient_plan.dwell_positions
    step_dwell_pos = patient_plan.get_step_size()
    nb_needles = len(patient_plan.dwell_positions)
    print(f"There is supposed to be {nb_needles} for this patient")

    dwell_pos_creator = DwellPositionCreator(
        sitk_needles_contour_path,
        fit_function="spline",
        # CatheterEvaluator takes consistent tip at the most distal part
        # of tip marker now => tip_distal always True.
        tip_distal=True
    )
    created_needle_dict, solo_components = dwell_pos_creator.create_points_from_contours(patient_plan.step_size, for_viz=True)
    if not os.path.exists(os.path.join(patient_path, "processed")):
        os.makedirs(os.path.join(patient_path, "processed"))
    sitk.WriteImage(
        solo_components, 
        os.path.join(patient_path, "processed", "solocompents_from_digitization.seg.nrrd"), 
        True)
    # print("=================================")
    # print("created_needle_dict ", created_needle_dict)
    # print("=================================")
    # treatment_catheter_table = get_catheter_table_and_source_info_from_dicom(patient_plan.plan_file_path)[0]
    # # print("treatment_catheter_table", treatment_catheter_table)    
    # catheter_table_creator = CatheterTableTimesFiller(created_needle_dict)
    # table = catheter_table_creator.get_catheter_table_with_treatment_table_times(treatment_catheter_table)
    # # print(table)


    if np.max(sitk.GetArrayFromImage(sitk.ReadImage(sitk_needles_contour_path))) > 1:
        multiclass = True
    else:
        multiclass = False

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 17
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # print("OG dwell positions KEYS ", patient_plan.dwell_positions.keys())
    # print("dwell positions KEYS ", patient_plan.dwell_positions.keys())

    lengths = []

    plot_segmentation_pts = False
    plot_central_pts = False
    plot_created_dwell_pos = True
    real_needle_of_interests = ["Channel_1"] #["Channel_12", "Channel_13"] #  None # "Channel_13" #  
    plot_needle = True
    extra_pts = False
    one_legend = True
    extra_pat_coords = [96.0, -41.56, 136.16]
    if real_needle_of_interests is not None:
        created_needle_of_interests = []
        for real_needle_of_interest in real_needle_of_interests:
            # We have two sets of dwell positions
            if "From start" in created_needle_dict["Dwell positions"]:
                created_needle_of_interests.append(select_best_needle_from_list(
                        dp_lists_to_select_from=created_needle_dict["Dwell positions"], 
                        reference_single_needle_dwell_positions=patient_plan.dwell_positions[real_needle_of_interest], 
                        key="From start"
                        ))
            # Only one set of dwell positions
            else:
                created_needle_of_interests.append(select_best_needle_from_list(
                        dp_lists_to_select_from=created_needle_dict["Dwell positions"], 
                        reference_single_needle_dwell_positions=patient_plan.dwell_positions[real_needle_of_interest]
                        ))
    else:
        created_needle_of_interests = None
    # Plot existing dwell positions
    legend_added = False
    for needle_idx, dwell_positions in patient_plan.dwell_positions.items():
    # for needle_idx, dwell_positions in patient_plan.non_zero_dwell_positions.items():
        if real_needle_of_interests is not None and needle_idx not in real_needle_of_interests:
            continue
        x, y, z = zip(*dwell_positions)
        if one_legend:
            if not legend_added:
                label = f"Clinical treatment dwell positions"
                legend_added = True
            else:
                label = None
        else:
            label = f"Clinical treatment dwell positions {needle_idx}"
        ax.scatter(x, y, z, label=label, c="blue")
        # ax.scatter(x, y, z, label=f"Existing Non-0s dwell positions {needle_idx}") # , c="blue"
        lengths.append(len(dwell_positions))
        
    print("lengths real dwell positions", lengths)

    # Plot created dwell positions
    if plot_created_dwell_pos:
        legend_added = False
        for needle_idx, dwell_positions in created_needle_dict["Dwell positions"].items():
            if created_needle_of_interests is not None and needle_idx not in created_needle_of_interests:
                continue
            if multiclass:
                dwell_pos = dwell_positions
                x, y, z = zip(*dwell_pos)
                if one_legend:
                    if not legend_added:
                        label = f"Created dwell positions"
                        legend_added = True
                    else:
                        label = None
                else:
                    label = f"Created dwell positions from contour Needle {needle_idx}"
                ax.scatter(x, y, z, c="red", label=label)
                lengths.append(len(dwell_pos))
            else:
                dwell_pos = dwell_positions["From start"]
                x, y, z = zip(*dwell_pos)
                ax.scatter(x, y, z, c="red", label=f"DwellPos from start Needle {needle_idx}")
                lengths.append(len(dwell_pos))

                dwell_pos = dwell_positions["From end"]
                x, y, z = zip(*dwell_pos)
                ax.scatter(x, y, z, c="green", label=f"DwellPos from end Needle {needle_idx}")

    # Plot segmentation points
    if plot_segmentation_pts:
        for needle_idx, dwell_positions in created_needle_dict[
            "Points from segmentation"
        ].items():
            if created_needle_of_interests is not None and needle_idx not in created_needle_of_interests:
                continue
            dwell_pos = dwell_positions
            x, y, z = zip(*dwell_pos)
            ax.scatter(x, y, z, c="purple", label=f"Pts from segmentation Needle {needle_idx}")
    
    # Plot central points used for spline fitting
    if plot_central_pts:
        for needle_idx, func_params in created_needle_dict[
            "Fitted function params"
        ].items():
            if created_needle_of_interests is not None and needle_idx not in created_needle_of_interests:
                continue
            central_pts = func_params.original_central_points
            x, y, z = zip(*central_pts)
            ax.scatter(x, y, z, c="orange", label=f"Central points {needle_idx}")
    
    # Plot the spline fitted on contour points
    legend_added = False
    if plot_needle:
        for needle_idx, needle_points in created_needle_dict["Generated needle"].items():
            if created_needle_of_interests is not None and needle_idx not in created_needle_of_interests:
                continue
            x, y, z = zip(*needle_points)
            if one_legend:
                if not legend_added:
                    label = f"Created catheter spline representations"
                    legend_added = True
                else:
                    label = None
            else:
                label = f"Created catheter spline representation Spline Needle {needle_idx}"
            ax.plot(x, y, z, label=label, c="black")

    if extra_pts:
        created_needle_of_interest = created_needle_of_interests[0]
        assert not real_needle_of_interest is None, "Need to specify a needle of interest to plot extra points"
        ax.scatter(extra_pat_coords[0], extra_pat_coords[1], extra_pat_coords[2], c="black", label="Extra points", marker="*", s=200)
        real_dp = patient_plan.dwell_positions[real_needle_of_interest]
        # Getting closest point
        min_dist, min_dist_pt = min_dist_two_list([extra_pat_coords], real_dp, return_points=True)
        for pair in min_dist_pt:
            print("Distance between extra point (black) that receive an extra dwell time, with the closest real dp: ", min_dist)
            print(pair)
            assert np.all (pair[0] == extra_pat_coords), "Extra point should be the same as the one used for distance calculation"
            ax.scatter(pair[1][0], pair[1][1], pair[1][2], c="brown", label="Real dp closest to extra point", marker="*", s=200)
        # Getting second closest point t
        # removing first closest point from the list
        real_dp.remove(pair[1])
        min_dist, min_dist_pt = min_dist_two_list([extra_pat_coords], real_dp, return_points=True)
        for pair in min_dist_pt:
            print("Distance between extra point (black) that receive an extra dwell time, with the second closest real dp: ", min_dist)
            print(pair)
            assert np.all (pair[0] == extra_pat_coords), "Extra point should be the same as the one used for distance calculation"
            ax.scatter(pair[1][0], pair[1][1], pair[1][2], c="grey", label="Second Real dp closest to extra point", marker="*", s=200)
        second_closest_pt = pair[1]
        
        # Getting closest point to the point cloest to the extra point t
        # From start or From end to manually choose 
        created_dp = created_needle_dict["Dwell positions"][created_needle_of_interest]["From end"]

        min_dist, min_dist_pt = min_dist_two_list(created_dp, [second_closest_pt], return_points=True)
        for pair in min_dist_pt:
            print("Distance between the real point the 2nd closest to extra point to any other created point except the extra point:", min_dist)
            print(pair)
            ax.scatter(pair[0][0], pair[0][1], pair[0][2], c="blue", label="Created dp closest to second real dp closest to extra pt", marker="*", s=200)

    print("lengths created dwell positions", lengths)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    exit()
