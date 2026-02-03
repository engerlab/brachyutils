import copy
import os
import json
from pathlib import Path

from typing import List, Dict, Any, Union, Tuple

import SimpleITK as sitk
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from scipy.interpolate import splev
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import (
    check_corners_convex_hull,
    fit_line, 
    fit_spline,
    find_four_corners_convex_hull,
    get_angle_between_two_vectors,
    get_catheter_directions_from_dwell_positions,
    compute_surface_normal,
    compute_surface_normal_4_points,
    project_point_to_line, 
    project_point_to_spline,
    distance, 
    describe_array, 
    get_binning_metric_for_n_clusters,
    get_nb_rows_from_dbscan,
    create_group_from_labels,
    save_grps_3Dslicer,
    create_slicer_markup_points, 
    get_potential_mean_directions_oriented_rows,
    project_on_z_coord
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.rotation import (
    calculate_rotation_matrix,
    create_rotation_transform,
    rotate_volume
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import (
    create_marker_pts_from_catheter_dict, 
    create_slicer_markup_points
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import get_non_zeros_bounds, sitk_crop, crop_around_mask
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import determine_breast_side



class GridViewSelector:
    """
    Class to select between two insertion grid views based on the catheters insertion scheme.
    """
    def __init__(self, grid_view_1:List[List[float]], grid_view_2:List[List[float]], save_details_folder:str|Path=None):
        self.grid_view_1 = grid_view_1
        self.grid_view_2 = grid_view_2
        self.save_details_folder = save_details_folder

    def select_grid_view(self, clustering_method:str="dbscan", save_files:bool=False, print_details:bool=False) -> List[List[float]]:
        """
        Select the appropriate grid view based on a clustering of the catheter insertion grid.
        If all the clusters of points in one view can be fitted to a line that is parallel to the
        x axis, we select that view. If the cluster are not good enough (not representing rows),
        they probably cannot be fitted to lines parallel to the x axis, and we return both views.
        """

        # If the difference is not clear enough, we will use a clustering method to identify rows
        # and see which grid has rows more parallel to the x axis.
        if clustering_method == "binning":
            best_metric_grid1, best_labels_grid1, best_n_cluster_grid1 = get_binning_metric_for_n_clusters(
                self.grid_view_1, n_clusters=10
            )
            best_metric_grid2, best_labels_grid2, best_n_cluster_grid2 = get_binning_metric_for_n_clusters(
                self.grid_view_2, n_clusters=10
            )
            if print_details:
                print(f"Grid View 1 - Best n_clusters: {best_n_cluster_grid1}, Best metric: {best_metric_grid1}")
                print(f"Grid View 2 - Best n_clusters: {best_n_cluster_grid2}, Best metric: {best_metric_grid2}")
        elif clustering_method == "dbscan":
            best_metric_grid1, best_nb_rows_grid1, best_labels_grid1 = get_nb_rows_from_dbscan(self.grid_view_1)
            best_metric_grid2, best_nb_rows_grid2, best_labels_grid2 = get_nb_rows_from_dbscan(self.grid_view_2)
            if print_details:
                print(f"Grid View 1 - Best n_rows: {best_nb_rows_grid1}, Best metric: {best_metric_grid1}")
                print(f"Grid View 2 - Best n_rows: {best_nb_rows_grid2}, Best metric: {best_metric_grid2}")
        else:
            raise ValueError("Unsupported clustering method. Use 'binning' or 'dbscan'.")
        
        if min(best_metric_grid1, best_metric_grid2) < 20.:
            #  If the mean angle of the clustered rows is less than 20 degrees,
            # we consider the grid view as valid, and surely representing the
            # insertion grid in the correct orientation. We only return one view.
            if best_metric_grid1 < best_metric_grid2:
                if print_details:
                    print("Grid View 1 is selected.")
                grid_view_to_choose = [self.grid_view_1]
                labels_to_save = best_labels_grid1
            else:
                if print_details:
                    print("Grid View 2 is selected.")
                grid_view_to_choose = [self.grid_view_2]
                labels_to_save = best_labels_grid2
            if save_files:
                assert self.save_details_folder is not None, (
                    "Save details folder must be provided to save the grid view."
                )
                save_grps_3Dslicer(
                    create_group_from_labels(grid_view_to_choose[0], labels_to_save),
                    self.save_details_folder,
                    key=f"{clustering_method}_grid_view_{1 if grid_view_to_choose[0] == self.grid_view_1 else 2}_"
                )
        else:
            # If the mean angle of the clustered rows is greater than 20 degrees,
            # we consider the grid view as not valid, and we return both views.
            if print_details:
                print("Both grid views are selected.")
            grid_view_to_choose = [self.grid_view_1, self.grid_view_2]

        return grid_view_to_choose

class InsertionGridViewer:

    def  __init__(self, ct_volume_path:str|Path|sitk.Image=None, catheters_contour_path:str|Path|sitk.Image=None, 
                 save_details_folder:str|Path=None, dwell_positions:dict=None, 
                 name_dwell_pos:str="clinical", crop_around_catheters:bool=True, margin_around_catheters_mm:float=5.0, 
                 breast_side:str=None, button_view:bool=True):
        """
        Class made to rotate the ct volume in such a way to face the insertion grid with
        rows parallel to the x axis to be able to understand the catheters insertion scheme.
        """
        assert ct_volume_path is not None or catheters_contour_path is not None, (
            "At least one volume must be provided to create transforms in this class."
        )
        if ct_volume_path is None:
            self.ct_volume = None
        else:
            if isinstance(ct_volume_path, sitk.Image):
                self.ct_volume = ct_volume_path
            else:
                self.ct_volume = sitk.ReadImage(ct_volume_path)
        
        if catheters_contour_path is None:
            self.catheters_contour = None
        else:
            if isinstance(catheters_contour_path, sitk.Image):
                self.catheters_contour = catheters_contour_path
            else:
                self.catheters_contour = sitk.ReadImage(catheters_contour_path)

        self.original_ct_volume = copy.deepcopy(self.ct_volume)
        self.original_catheters_contour = copy.deepcopy(self.catheters_contour)

        if crop_around_catheters:
            assert self.catheters_contour is not None, (
                "Catheters contour must be provided to crop around the catheters."
            )
            self.catheters_contour, bb = crop_around_mask(
                self.catheters_contour, margin_mm=margin_around_catheters_mm, use_sitk=True)
            if not(self.ct_volume is None):
                self.ct_volume = sitk_crop(self.ct_volume, bb)

        self.save_details_folder = save_details_folder
        self.dwell_positions = dwell_positions
        self.name_dwell_pos = name_dwell_pos

        self.current_dwell_positions = dwell_positions
        self.current_ct_volume = self.ct_volume
        self.button_view = button_view

        ### Depending on the treated breast side, the reference points to look at when 
        # orienting the insertion grid will changes. We assume that the Dr. is always facing 
        # the patient, hence if the treated breast if the right one (we see it on the right on 
        # an axial slice but it is called left by Dr when doing the OAR contours), the Dr. 
        # should see the insertion grid with the most left catheter(s) inserted closer to the 
        # CT top left corner. If the treated breast is the left one, (we see it on the left 
        # on an axial slice but all related structures are called "right"), the Dr. should 
        # see the insertion grid with the most right catheter(s) inserted the closer to the 
        # CT top right corner.
        self.reference_point = None
        if breast_side is None:
            assert self.original_ct_volume is not None, (
                "Breast side must be provided if no ct volume is provided."
            )
            dp_list = []
            for channel_key in self.current_dwell_positions.keys():
                assert isinstance(self.current_dwell_positions[channel_key], list), (
                    f"Dwell positions for channel {channel_key} must be a list."
                )
                dp_list.extend(self.current_dwell_positions[channel_key])
            self.breast_side = determine_breast_side(
                self.original_ct_volume, dp_list)
        else:
            self.breast_side = breast_side

        self.current_closer_to_og_ref_points = None
        self.insertion_parallel_to_body_length = self.determine_insertion_orientation()
        self.get_condition_to_maintain()

        self.current_catheters_contour = self.catheters_contour
        self.current_grid = None
        self.current_grid_real_zs = None
        self.transforms = []


    def determine_insertion_orientation(self):
        """
        Determine the insertion orientation based on the current dwell positions and breast side.
        
        """
        assert np.allclose(self.original_ct_volume.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), atol=1e-4), (
            "CT volume direction must be identity. The code is not implemented for other directions."
            "Probably just need to resample the volume with identity direction."
            f" Current direction: {self.original_ct_volume.GetDirection()}"
        )
        body_length_direction_towards_head = np.array([0, 0, 1])

        catheter_directions = get_catheter_directions_from_dwell_positions(
            self.dwell_positions, button_first=True
        )

        # We could compare the angle between the catheter directions and the body length 
        # direction and the angle between the catheter directions and the side insertion 
        # direction. The side insertion direction depends on the breast side. But just 
        # comparing these two angles is not robust. Instead we just see if the mean angle 
        # between the catheter directions and the body length direction is smaller than
        # a threshold.
        # ### DEPRECATED
        # if self.breast_side == "left":
        #     # For left breast, the insertion should be from left to right
        #     side_insertion_direction = np.array([1, 0, 0])
            
        # else:
        #     # For right breast, the insertion should be from right to left
        #     side_insertion_direction = np.array([-1, 0, 0])
        
        angles_considering_body_length_insertion = []
        for catheter_direction in catheter_directions:
            # Compute the angle between the catheter direction and the body length direction
            angle_to_body_length = get_angle_between_two_vectors(
                catheter_direction, body_length_direction_towards_head)
            angles_considering_body_length_insertion.append(angle_to_body_length)

        # 30 degrees arbitrarily set but for those patient generally it is lower e.g. 15
        if np.mean(angles_considering_body_length_insertion) < 30.0:
            # If the mean angle to body length insertion is smaller, we consider the insertion
            # is parallel to the body length.
            insertion_parallel_to_body_length = True
        else:
            # If the mean angle to side insertion is smaller, we consider the insertion
            # is perpendicular to body length.
            insertion_parallel_to_body_length = False

        return insertion_parallel_to_body_length

    def get_condition_to_maintain(self)-> None:
        """
        Get the condition to maintain to have the insertion grid
        with rows parallel to the x axis.
        """
        assert np.allclose(self.original_ct_volume.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), atol=1e-4), (
            "CT volume direction must be identity. The code is not implemented for other directions."
            "Probably just need to resample the volume with identity direction."
            f" Current direction: {self.original_ct_volume.GetDirection()}"
        )

        origin = self.original_ct_volume.GetOrigin()
        shape = self.original_ct_volume.GetSize()
        spacing = self.original_ct_volume.GetSpacing()
        avg_xs_tips = np.mean([self.dwell_positions[channel_key][0][0] 
                    for channel_key in self.dwell_positions.keys()])
        avg_ys_tips = np.mean([self.dwell_positions[channel_key][0][1] 
                               for channel_key in self.dwell_positions.keys()])
        avg_zs_tips = np.mean([self.dwell_positions[channel_key][0][2] 
                               for channel_key in self.dwell_positions.keys()])
        if not self.insertion_parallel_to_body_length:
            # Most common case, the insertion is perpendicular to the body length.
            if self.breast_side == "left":
                # The insertion grid should be oriented with the top left catheter(s) inserted
                # closer to the CT top left corner.
                ref_pt_left_of_dr = [
                    # At the left of the axial slice, left of Dr's head.
                    avg_xs_tips, # More robust than origin[0] , 
                    # Origin is already above the patient's head.
                    avg_ys_tips, # More robust than origin[1], 
                    # Towards the head of the patient, not their feet.
                    origin[2] + (shape[2]-1) * spacing[2]
                    ]  # Top left corner of the CT volume
                self.reference_point = ref_pt_left_of_dr
            else:
                assert self.breast_side == "right", (
                    "Breast side must be left if not right."
                )
                ref_pt_right_of_dr = [
                    # At the right of the axial slice, right of Dr's head.
                    avg_xs_tips, # More robust than origin[0] + shape[0] * spacing[0], 
                    # Origin is already above the patient's head.
                    avg_ys_tips, # More robust than origin[1], 
                    # Towards the head of the patient, not their feet.
                    origin[2] + (shape[2]-1) * spacing[2]
                    ]  # Top right corner of the CT volume
                self.reference_point = ref_pt_right_of_dr
        else:
            # The insertion is parallel to the body length, hence the insertion grid should be oriented
            # differently.
            if self.breast_side == "left":
                ref_pt_left_of_dr = [
                    # At the left of the axial slice, left of Dr's head.
                    origin[0] if self.button_view else origin[0] + shape[0] * spacing[0], 
                    # Origin is already above the patient's head.
                    avg_ys_tips, # More robust than origin[1], 
                    # Towards the feet of the patient, not their head.
                    avg_zs_tips  # Bottom left corner of the CT volume
                    ]  
                self.reference_point = ref_pt_left_of_dr
            else:
                ref_pt_right_of_dr = [
                    # At the right of the axial slice, right of Dr's head.
                    origin[0] + shape[0] * spacing[0] if self.button_view else origin[0], 
                    # Origin is already above the patient's head.
                    avg_ys_tips, # More robust than origin[1], 
                    # Towards the feet of the patient, not their head.
                    avg_zs_tips  # Bottom right corner of the CT volume
                    ]  
                self.reference_point = ref_pt_right_of_dr
        ### Getting dwell positions at the extermities
        assert self.dwell_positions is not None, (
            "Dwell positions must be provided to get the condition to maintain during grid visualization."
        )
        closest_pt_to_ref = None
        farthest_pt_to_ref = None
        min_dist_to_ref = np.inf
        max_dist_to_ref = 0
        if self.button_view:
            idx_dp = 0
        else:
            raise Warning("We do not take -1 because we were testing with analytical contours "
                         "For these contours the last dwell positions can be outside the patient body since the "
                         "pseudo contours are created form the digitization points. "
                         "However, outside the patient body the catheters quickly mingle, making it hard to keep a"
                         "insertion grid with proper rows.") 
            idx_dp = -5 # -1
        for ch in self.dwell_positions.keys():
            dp = np.array(self.dwell_positions[ch][idx_dp])
            dp_dist_to_ref = distance(dp, self.reference_point)
            if dp_dist_to_ref < min_dist_to_ref:
                min_dist_to_ref = dp_dist_to_ref
                closest_pt_to_ref = dp
            if dp_dist_to_ref > max_dist_to_ref:
                max_dist_to_ref = dp_dist_to_ref
                farthest_pt_to_ref = dp
    
        self.current_closer_to_og_ref_points = [
            # First point should be closer to ref, second further.
            # This condition should be maintained during the visualization.
            # The distance between these rotated points and the rotated reference points
            # is never going to change. However, the reference point will change.
            # Since we are interested in the Dr's view, the new reference point will
            # be the top left for right breast brachy (left Dr's view) and top right 
            # for left breast brachy (right Dr's view).
            closest_pt_to_ref.tolist(),
            farthest_pt_to_ref.tolist()
        ]
        
        if not (self.save_details_folder is None):
            for i in range(len(self.current_closer_to_og_ref_points)):
                create_slicer_markup_points(
                        os.path.join(self.save_details_folder, f"points_closer_to_og_ref_{i}.mrk.json"), 
                        [self.current_closer_to_og_ref_points[i]], 
                        color=[0.5,0.75*i,0.],  
                    )
            create_slicer_markup_points(
                os.path.join(self.save_details_folder, "ref_point.mrk.json"), 
                [self.reference_point], 
                color=[0.8,0.8,0.8],  
            )

    def check_condition(self, input_dict:Dict=None, inplace:bool=True, save_files:bool=False,
                        state_name="parallel_to_plane") -> None:
        """
        Once the grid is rotated and we face the insertion points, we check if the condition
        to maintain is respected. But now the reference point changes based on the breast side.
        The condition is that the original closest dwell position to the reference point 
        should remain so with the new reference point. The new reference point is top left for 
        for right breast brachy (left Dr's view) and top right for left breast brachy (right 
        Dr's view).
        """
        assert self.reference_point is not None, (
            "Reference point must be set before checking the condition."
        )
        if input_dict is not None:
            assert not inplace, (
                "Input dictionary can only be used when inplace is False."
            )
        (
            current_ct_volume,
            current_catheters_contour,
            current_dwell_positions,
            current_grid,
            current_grid_real_zs,
            current_closer_to_og_ref_points,
            current_transforms
        ) = self.unwrap_state(input_dict=input_dict)

        assert not(current_ct_volume is None) or not(current_catheters_contour is None), (
            "Current CT volume or catheters contour must be set to check the condition."
        )
        
        new_ref_pt, ref_vol = self.get_corner_view_ref_point_for_volume(
            current_ct_volume, current_grid=current_grid, 
            closer_to_og_ref_points=current_closer_to_og_ref_points)
        
        if save_files:
            save_dir = os.path.join(self.save_details_folder, state_name)
            create_slicer_markup_points(
                    os.path.join(save_dir, f"{state_name}_ref_point.mrk.json"), 
                    [new_ref_pt], 
                    color=[0.8,0.8,0.8],  
                )

        condition_maintained = self.is_condition_maintained(new_ref_pt, current_closer_to_og_ref_points)

        # Rotate the volume/grid to maintain the condition.
        rotation_matrix_180_degres = [
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.]
        ]
        state = {
            "ct_volume": current_ct_volume,
            "catheters_contour": current_catheters_contour,
            "dwell_positions": current_dwell_positions,
            "grid": current_grid,
            "grid_real_zs": current_grid_real_zs,
            "closer_to_og_ref_points": current_closer_to_og_ref_points,
            "transforms": current_transforms
        }
        if not condition_maintained:
            clockwise_transform = create_rotation_transform(
                        # ref_vol, rotation_matrix_90_degres
                        ref_vol, rotation_matrix_180_degres
                    )
            state = self.transform_state(
                clockwise_transform, input_dict=state, inplace=False)
            if save_files:
                save_dir = os.path.join(self.save_details_folder, 
                                        f"clockwise_rotation_{state_name}")
                self.save_state(save_dir, state_name=state_name, input_dict=state)
            new_ref_pt, ref_vol = self.get_corner_view_ref_point_for_volume(
                state["ct_volume"], current_grid=state["grid"], 
                closer_to_og_ref_points=state["closer_to_og_ref_points"])
            if save_files:
                create_slicer_markup_points(
                        os.path.join(save_dir, f"{state_name}_ref_point.mrk.json"), 
                        [new_ref_pt], 
                        color=[0.8,0.8,0.8],  
                    )
            condition_maintained = InsertionGridViewer.is_condition_maintained(
                new_ref_pt, state["closer_to_og_ref_points"])
            assert condition_maintained, (
                "Condition should be maintained after the rotation."
            )
        
        if inplace:
            self.current_ct_volume = state["ct_volume"]
            self.current_catheters_contour = state["catheters_contour"]
            self.current_dwell_positions = state["dwell_positions"]
            self.current_grid = state["grid"]
            self.current_grid_real_zs = state["grid_real_zs"]
            self.current_closer_to_og_ref_points = state["closer_to_og_ref_points"]
            self.transforms = state["transforms"]
        else:
            # If not inplace, we return the state.
            return state
                

    def get_corner_view_ref_point_for_volume(
            self, ref_vol:sitk.Image, current_grid:Dict[str,List[List[float]]]=None, 
            closer_to_og_ref_points:List[List[float]]=None) -> List[float]:
        """
        Get the reference point for the corner view of the volume.
        This is used to visualize the insertion grid in 3D Slicer.
        The reference point is the top left corner of the volume.
        """
        if current_grid is not None:
            current_grid = self.current_grid
            
        current_origin = ref_vol.GetOrigin()
        current_spacing = ref_vol.GetSpacing()
        current_shape = ref_vol.GetSize()

        avg_ys_closer_pts = np.mean([closer_to_og_ref_points[0][1], 
                                     closer_to_og_ref_points[1][1]])
        avg_zs_closer_pts = np.mean([closer_to_og_ref_points[0][2], 
                                     closer_to_og_ref_points[1][2]])
        if self.breast_side == "left":
            new_ref_point_top_left = [
                # At the left of the axial slice, left of Dr's head.
                current_origin[0],
                avg_ys_closer_pts, 
                avg_zs_closer_pts
            ]
            new_ref_pt = new_ref_point_top_left
        else:
            assert self.breast_side == "right", (
                "Invalid breast side. Should be either 'left' or 'right'."
            )
            new_ref_point_top_right = [
                # At the right of the axial slice, right of Dr's head.
                current_origin[0] + current_shape[0] * current_spacing[0],
                avg_ys_closer_pts, 
                avg_zs_closer_pts 
            ] 
            new_ref_pt = new_ref_point_top_right
        return new_ref_pt, ref_vol

    @staticmethod
    def is_condition_maintained(ref_pt: List[float], current_closer_to_og_ref_points: List[List[float]]) -> bool:
        """
        Check if the condition to maintain is respected.
        """
        condition_maintained = distance(
            current_closer_to_og_ref_points[0], ref_pt) < distance(
                current_closer_to_og_ref_points[1], ref_pt)
        return condition_maintained

    def save_state(self, save_dir:str, state_name:str="original", input_dict:Dict=None, save_ref_pt:bool=False):
        """
        Save the current state of the class to a folder.
        The state includes the current grid, the current ct volume and the current dwell positions.
        The state is saved in a folder with the name of the state.
        """
        if input_dict is not None:
            current_grid = input_dict["grid"]
            current_grid_real_zs = input_dict["grid_real_zs"]
            current_ct_volume = input_dict["ct_volume"]
            current_dwell_positions = input_dict["dwell_positions"]
            current_closer_to_og_ref_points = input_dict["closer_to_og_ref_points"]
        else:
            current_grid = self.current_grid
            current_grid_real_zs = self.current_grid_real_zs
            current_ct_volume = self.current_ct_volume
            current_dwell_positions = self.current_dwell_positions
            current_closer_to_og_ref_points = self.current_closer_to_og_ref_points

        os.makedirs(save_dir, exist_ok=True)
        current_grid_points_l = [l_pts[0] for l_pts in current_grid.values()]

        create_slicer_markup_points(
            os.path.join(save_dir,  
                         f"{state_name}_all_current_grid_points_l.mrk.json"),
            current_grid_points_l, [0.1,0.1,0.3])

        self.save_points(
            current_grid, os.path.join(
                save_dir, f"{state_name}_grid_points.mrk.json"), [0.1,0.1,0.3])
        self.save_points(
            current_grid_real_zs, os.path.join(
                save_dir, f"{state_name}_grid_real_zs.mrk.json"), 
                [0.01568627450980392,0.0784313725490196,0.5019607843137255])
        
        if current_ct_volume is not None:
            sitk.WriteImage(
                current_ct_volume,
                os.path.join(
                    save_dir, 
                    f"ct_{state_name}_rotated_patient_for_catheter_mapping.nrrd"),
                True
            )
        if current_dwell_positions is not None:
            for channel_key in current_dwell_positions.keys():
                dwell_p = current_dwell_positions[channel_key]
                create_slicer_markup_points(
                    os.path.join(save_dir, f"{state_name}_{self.name_dwell_pos}_dwell_positions_channel_{channel_key}.mrk.json"), 
                    dwell_p, 
                    color=[0.1333333,0.7254901960784313,0.9803921568627451],
                )
        if current_closer_to_og_ref_points is not None:
            for i in range(len(current_closer_to_og_ref_points)):
                create_slicer_markup_points(
                        os.path.join(save_dir, f"{state_name}_closer_to_og_ref_points_{i}.mrk.json"), 
                        [current_closer_to_og_ref_points[i]], 
                        color=[0.5,0.75*i,0.],  
                    )
        
        if save_ref_pt:
            new_ref_pt, ref_vol = self.get_corner_view_ref_point_for_volume(
                current_ct_volume, current_grid=current_grid, 
                closer_to_og_ref_points=current_closer_to_og_ref_points
                )
            create_slicer_markup_points(
                    os.path.join(save_dir, f"{state_name}_ref_point.mrk.json"), 
                    [new_ref_pt], 
                    color=[0.8,0.8,0.8],  
                )
            
    @staticmethod
    def transform_points(points:List[List[float]], transform:sitk.Transform):
        """
        Transform a list of points using a SimpleITK transform.
        Points should be in the form of a list of lists, where each inner list is a point [x, y, z].
        """
        transformed_points = []
        for pt in points:
            transformed_pt = transform.TransformPoint(pt)
            transformed_points.append(list(transformed_pt))
        return transformed_points

    @staticmethod
    def transform_pts_dict(pts_dict:dict, transform:sitk.Transform):
        """
        Transform a dictionary of points using a SimpleITK transform.
        The dictionary should have keys as strings and values as lists of points.
        """
        transformed_dict = {}
        for key in pts_dict.keys():
            transformed_dict[key] = InsertionGridViewer.transform_points(pts_dict[key], transform)
        return transformed_dict
    
    def transform_dp_and_grid(self, transform:sitk.Transform, current_dwell_positions:Dict[str,List[List[float]]]=None):
        """
        Transform the insertion grid using a SimpleITK transform.
        The grid is a dictionary of points, where each key is a channel and the value is a list of points.
        """

        ## Transform the dwell positions
        current_dwell_positions = InsertionGridViewer.transform_pts_dict(current_dwell_positions, transform)
        
        ## Transform the insertion grid
        # Create a grid of points based on the tips and their average z coord.
        if self.button_view:
            # The first dwell posiiton of each channel is the tip position.
            idx_dp = 0
        else:
            # We are interested in the last dwell position.
            raise Warning("We do not take -1 because we were testing with analytical contours "
                         "For these contours the last dwell positions can be outside the patient body since the "
                         "pseudo contours are created form the digitization points. "
                         "However, outside the patient body the catheters quickly mingle, making it hard to keep a"
                         "insertion grid with proper rows.") 
            idx_dp = -5 # -1
        grid_pts = {}
        for key in current_dwell_positions.keys():
            tip = copy.deepcopy(current_dwell_positions[key][idx_dp])
            grid_pts[key] = [tip]
        current_grid_real_zs = copy.deepcopy(grid_pts)
        # Get mean z for all tips
        avg_z = float(np.mean([grid_pts[key][0][2] for key in grid_pts.keys()]))
        # Assign this z to all points
        for k in grid_pts.keys():
            grid_pts[k][0][2] = avg_z
        current_grid = grid_pts
        return current_dwell_positions, current_grid, current_grid_real_zs

    def transform_state(
            self, transform:sitk.Transform, crop_after_transform:bool=True, inplace:bool=True, 
            input_dict:Dict=None):
        """
        Transform the current state: rotate the volume, the dwell positions
        and insertion grid.
        """
        if input_dict is not None:
            assert not inplace, (
                "Input dictionary can only be used when inplace is False."
            )
        (
            current_ct_volume,
            current_catheters_contour,
            current_dwell_positions,
            current_grid,
            current_grid_real_zs,
            current_closer_to_og_ref_points,
            current_transforms
        ) = self.unwrap_state(input_dict=input_dict)

        current_transforms.append(transform)

        if self.ct_volume is not None:
            current_ct_volume = rotate_volume(
                current_ct_volume,
                transform,
                interpolator=sitk.sitkLinear)
            if crop_after_transform:
                # Crop the volume to remove air around the patient added by rotation
                non_zero_bounds = get_non_zeros_bounds(
                    current_ct_volume
                )
                current_ct_volume = sitk_crop(
                    current_ct_volume, 
                    non_zero_bounds
                )
            else:
                non_zero_bounds = None
        else:
            non_zero_bounds = None
            
        if self.catheters_contour is not None:
            current_catheters_contour = rotate_volume(
                current_catheters_contour,
                transform,
                interpolator=sitk.sitkLinear)
            if crop_after_transform:
                assert non_zero_bounds is not None, (
                    "Non-zero bounds must be defined from ct_volume to crop the catheters contour."
                )
                current_catheters_contour = sitk_crop(
                    current_catheters_contour,
                    non_zero_bounds
                )

       
        current_dwell_positions, current_grid, current_grid_real_zs = self.transform_dp_and_grid(
            transform, current_dwell_positions)

        ## Transform the reference points
        if self.current_closer_to_og_ref_points is not None:
            current_closer_to_og_ref_points = InsertionGridViewer.transform_points(
                current_closer_to_og_ref_points, transform)
        
        if inplace:
            if self.ct_volume is not None:
                self.current_ct_volume = current_ct_volume
            if self.catheters_contour is not None:
                self.current_catheters_contour = current_catheters_contour
            self.current_dwell_positions = current_dwell_positions
            self.current_grid = current_grid
            self.current_grid_real_zs = current_grid_real_zs
            if self.current_closer_to_og_ref_points is not None:
                self.current_closer_to_og_ref_points = current_closer_to_og_ref_points
            self.transforms = current_transforms
        else:
            return {
                "ct_volume": current_ct_volume,
                "catheters_contour": current_catheters_contour,
                "dwell_positions": current_dwell_positions,
                "grid": current_grid,
                "grid_real_zs": current_grid_real_zs,
                "closer_to_og_ref_points": current_closer_to_og_ref_points,
                "transforms": current_transforms
            }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the class.
        The state includes the current grid, the current ct volume and the current dwell positions.
        """
        return {
            "ct_volume": self.current_ct_volume,
            "catheters_contour": self.current_catheters_contour,
            "dwell_positions": self.current_dwell_positions,
            "grid": self.current_grid,
            "grid_real_zs": self.current_grid_real_zs,
            "closer_to_og_ref_points": self.current_closer_to_og_ref_points,
            "transforms": self.transforms
        }

    def get_insertion_grid_as_rows(self, save_files:bool=False, print_details:bool=False, 
                                   project_onto_plane:bool=False) -> List[Dict[str, Any]]:
        """
        Get the insertion grid as rows.
        The function will rotate the volume to have all the insertion in the axial plane
        and then rotate the axial plane to have the rows parallel to the x axis.
        """
        if save_files:
            assert self.save_details_folder is not None, "self.save_details_folder must be set to save the points."
        if print_details:
            print("FIRST ROTATION...")
        # Create rotation to have all the insertion in the axial plane
        transform = self.rotate_volume_to_have_catheters_perpendicular_to_axial_slices(self.current_dwell_positions)
        # Rotate the volume to have all the insertion in the axial plane
        if print_details:
            print("FIRST TRANSFORM...")
        self.transform_state(transform)

        if save_files:
            save_dir = os.path.join(self.save_details_folder, "first_rotation")
            self.save_state(save_dir, state_name="parallel_to_insertion")
            os.makedirs(save_dir, exist_ok=True)

        if print_details:
            print("SECOND ROTATION...")
        transform1 = self.rotate_volume_to_have_all_insertion_points_on_one_axial_slice(
            self.current_grid_real_zs, save_files=save_files)
        if print_details:
            print("SECOND TRANSFORM...")
        self.transform_state(transform1)

        if save_files:
            save_dir = os.path.join(self.save_details_folder, "firstbis_rotation")
            self.save_state(save_dir, state_name="parralel_to_plane")
            os.makedirs(save_dir, exist_ok=True)

        if print_details:
            print("THIRD ROTATION...")
        # The axial slice contains the insertion side of the catheter but
        # we need to have the rows on ~ the same y coordinate. 
        transform2 = self.rotate_xy_to_have_insertion_rows_parallel_to_x_axis(
            self.current_grid, self.current_ct_volume, save_files=save_files)
        if print_details:
            print("THIRD TRANSFORM...")
        # Rotate the volume to have all the insertion in the axial plane
        if isinstance(transform2, sitk.Transform):
            self.transform_state(transform2)
            if save_files:
                save_dir = os.path.join(self.save_details_folder, "second_rotation")
                self.save_state(save_dir, state_name="oriented_rows", save_ref_pt=True)
                os.makedirs(save_dir, exist_ok=True)
            
            self.check_condition(save_files=save_files)
            if project_onto_plane:
                self.project_on_plane()
            return [self.get_state()]
        else:
            assert isinstance(transform2, list), (
                "Transform must be a SimpleITK Transform or a list of transforms."
            )
            assert len(transform2) == 2, (
                "Two transforms should have been identified with the two potential views."
            )
            state1 = self.transform_state(
                transform2[0], inplace=False, input_dict=self.get_state()
            )
            state2 = self.transform_state(
                transform2[1], inplace=False, input_dict=self.get_state()
            )
            state1 = self.check_condition(
                input_dict=state1, inplace=False, save_files=save_files, 
                state_name="second_rotation_testdirection1")
            state2 = self.check_condition(
                input_dict=state2, inplace=False, save_files=save_files, 
                state_name="second_rotation_testdirection2")
            if project_onto_plane:
                state1 = self.project_on_plane(
                    input_dict=state1, inplace=False)
                state2 = self.project_on_plane(
                    input_dict=state2, inplace=False)
            return [state1, state2]

    def rotate_volume_to_have_catheters_perpendicular_to_axial_slices(
            self, dwell_positions:dict):

        catheter_directions = get_catheter_directions_from_dwell_positions(
            dwell_positions, button_first=self.button_view)

        # Get a mean direction vector for all the catheters
        mean_catheter_direction = np.mean(catheter_directions, axis=0)

        rotation_matrix = calculate_rotation_matrix(
            mean_catheter_direction, target=np.array([0.,0.,1.]), prt=False)

        if self.ct_volume is not None:
            sitk_volume = self.ct_volume
        else:
            sitk_volume = self.catheters_contour
        transform = create_rotation_transform(
            sitk_volume, rotation_matrix
        )
        return transform
    
    def rotate_volume_to_have_all_insertion_points_on_one_axial_slice(
            self, point_d:dict, save_files:bool=False):
        """
        Create a rotation to have the point grid plane on the axial plane. 
        """
        points_3D = np.array([
            point_d[key][0] for key in point_d.keys()           
        ])

        if points_3D.shape[0] < 4:
            # If there is only one row and less than 4 points, we cannot find four corners.
            # If there are 3 points (3 catheters implanted), we can use them to define the plane.
            assert points_3D.shape[0] == 3, (
                "If there is only one row, we did not implement the case with less than 3 "
                "catheters because it never ocurred in our dataset."
            )
            normal_to_surface = compute_surface_normal(points_3D[0], points_3D[1], points_3D[2])
        else:
            # Find out a plane for your transform
        
            temp_pts_3D = copy.deepcopy(points_3D)
            sure_about_the_plane = False
            while not sure_about_the_plane:
                points_2D = temp_pts_3D[:, :2]  # Take only the x and y coordinates

                corners_hull = find_four_corners_convex_hull(points_2D, order_pca=False).tolist()
                corners_hull_3D = []
                for corner_pt in corners_hull:
                    # Find back the z coordinate of the point
                    for pt_idx, pt in enumerate(points_2D):
                        if pt[0] == corner_pt[0] and pt[1] == corner_pt[1]:
                            corners_hull_3D.append(points_3D[pt_idx].tolist())
                            break
                
                sure_about_the_plane, problematic_pt_3D = check_corners_convex_hull(
                    corners_hull_3D, threshold_mm=20)
                if not sure_about_the_plane:
                    with open(os.path.join(
                        self.save_details_folder, "first_rotation", 
                        "problematic_point_not_sure_about_plane.txt"), 'w') as f:
                        f.write(
                            "The following point was considered problematic to define the plane:\n"
                            f"{problematic_pt_3D}\n"
                        )
                sure_about_the_plane = True
                if not sure_about_the_plane:
                    raise NotImplementedError(
                        " This function is under developement."
                        " We will project the plane onto the defined plane"
                    )
                    # If the corners are not sure, we remove the point that is farthest from the plane
                    # defined by the other points.
                    print("Removing point", problematic_pt_3D)
                    temp_pts_3D = np.delete(temp_pts_3D, np.where(
                        (temp_pts_3D == problematic_pt_3D).all(axis=1)), axis=0)
                    points_3D = temp_pts_3D
                    if points_3D.shape[0] < 4:
                        # We cannot find four corners anymore.
                        break
    
            assert len(corners_hull_3D) == 4, (
                "The number of corners is not 4. The equality condition probably did not work."
            )
            if save_files:
                create_slicer_markup_points(
                    os.path.join(self.save_details_folder, "first_rotation", "corners_used_to_define_plane.mrk.json"),
                    corners_hull_3D,
                    color=[0.1, 0.1, 0.3]
                )
            normal_to_surface = compute_surface_normal_4_points(*[np.array(c) for c in corners_hull_3D])

        # Checking normal and z axis are not in opposite directions
        if np.dot(normal_to_surface, np.array([0.,0.,1.])) < 0:
            normal_to_surface = -normal_to_surface

        rotation_matrix = calculate_rotation_matrix(
            direction=normal_to_surface, target=np.array([0.,0.,1.]), prt=False)

        if self.ct_volume is not None:
            sitk_volume = self.ct_volume
        else:
            sitk_volume = self.catheters_contour
        transform = create_rotation_transform(
            sitk_volume, rotation_matrix
        )
        return transform
    
    def rotate_xy_to_have_insertion_rows_parallel_to_x_axis(
            self, points_d:dict, volume:sitk.Image, save_files:bool=False, 
            print_details:bool=False) -> Union[sitk.Transform, List[sitk.Transform]]:
        """
        Create a rotation to have the points average direction (line fitter 
        to the points) parallel to the x axis.
        """
        # Second rotation
        # Step 1: Your 2D points in the XY plane (Z is constant, say z = 5)
        points = np.array([ 
            points_d[key][0] for key in points_d.keys()           
        ])  # (N, 3) array

        if points.shape[0] < 4:
            # We cannot compute the four corners of the convex hull.
            m, d = fit_line(points)
            directions = [d]
        else:
            mean, directions = get_potential_mean_directions_oriented_rows(
                points, save_files=save_files,
                save_dir = os.path.join(self.save_details_folder, "firstbis_rotation")
            )
        def direction_to_transform(direction:np.ndarray):
            if direction[0] < 0:
                direction *= -1

            # Step 3: Get angle between the direction vector and the x-axis
            theta = np.arctan2(direction[1], direction[0])  # angle in radians
            # To get the angle in degrees: angle_degrees = np.degrees(theta)

            # Step 4: Build 3x3 rotation matrix around Z-axis
            rotation_matrix_2 = np.array([
                [np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta),  np.cos(-theta), 0],
                [0,               0,              1]
            ])

            transform2 = create_rotation_transform(
                volume, rotation_matrix_2
            )
            return transform2
        if len(directions) == 1:
            # Ensure the direction points rightward (positive x), to avoid Y reversal
            transform2 = direction_to_transform(directions[0])
        else:
            # There are multiple potential directions for this rotation.
            # We will try to find the best one by checking the spread of the points
            # after the rotation.
            pot_tr1 = direction_to_transform(directions[0])
            pot_tr2 = direction_to_transform(directions[1])

            # Saving data
            save_current_ct_volume = copy.deepcopy(self.current_ct_volume)
            save_current_catheters_contour = copy.deepcopy(self.current_catheters_contour)
            save_current_dwell_positions = copy.deepcopy(self.current_dwell_positions)
            save_current_grid = copy.deepcopy(self.current_grid)
            save_current_grid_real_zs = copy.deepcopy(self.current_grid_real_zs)
            save_current_closer_to_og_ref_points = copy.deepcopy(self.current_closer_to_og_ref_points)

            transformed_data1 = self.transform_state(pot_tr1, inplace=False)
            self.transform_state(pot_tr1, inplace=True)
            if save_files:
                save_dir = os.path.join(self.save_details_folder, "second_rotation_testdirection1")
                self.save_state(save_dir, state_name="oriented_rows", save_ref_pt=True)

            #Restore state
            self.transforms.pop()  # Remove the last transform
            self.current_ct_volume = save_current_ct_volume
            self.current_catheters_contour = save_current_catheters_contour
            self.current_dwell_positions = save_current_dwell_positions
            self.current_grid = save_current_grid
            self.current_grid_real_zs = save_current_grid_real_zs
            self.current_closer_to_og_ref_points = save_current_closer_to_og_ref_points

            transformed_data2 = self.transform_state(pot_tr2, inplace=False)
            self.transform_state(pot_tr2, inplace=True)
            if save_files:
                save_dir = os.path.join(self.save_details_folder, "second_rotation_testdirection2")
                self.save_state(save_dir, state_name="oriented_rows", save_ref_pt=True)
            #Restore state
            self.transforms.pop()  # Remove the last transform
            self.current_ct_volume = save_current_ct_volume
            self.current_catheters_contour = save_current_catheters_contour
            self.current_dwell_positions = save_current_dwell_positions
            self.current_grid = save_current_grid
            self.current_grid_real_zs = save_current_grid_real_zs
            self.current_closer_to_og_ref_points = save_current_closer_to_og_ref_points

            pts_grid1 = [
                transformed_data1["grid"][key][0] for key in transformed_data1["grid"].keys()
            ]
            pts_grid2 = [
                transformed_data2["grid"][key][0] for key in transformed_data2["grid"].keys()
            ]
            
            selector = GridViewSelector(
                pts_grid1, pts_grid2,
                save_details_folder=os.path.join(self.save_details_folder, "second_rotation")
            )
            grid = selector.select_grid_view(save_files=save_files, print_details=False)
            
            if len(grid) == 1:
                # If only one grid is returned by the detector, we are sure about which view to use.
                if np.allclose(grid[0][0], pts_grid1[0], atol=1e-3):
                    if print_details:
                        print("Using first transformation.")
                    transform2 = pot_tr1
                else:
                    if print_details:
                        print("Using second transformation.")
                    transform2 = pot_tr2
            else:
                assert len(grid) == 2, (
                    "The grid view selector should return either 1 or 2 grids."
                )
                if print_details:
                    print("WARNING: The grid view selector could not decide which transformation to use.")
                    print("Trying both transformations.")
                transform2 = [pot_tr1, pot_tr2]

        return transform2
                
    def unwrap_state(self, input_dict:Dict=None) -> Dict[str, Any]:
        """
        Unwrap the state of the class.
        This is useful to save the state of the class in a file.
        """
        if input_dict is not None:
            assert isinstance(input_dict, dict), (
                "Input dictionary must be a dictionary."
            )
            current_ct_volume = input_dict["ct_volume"]
            current_catheters_contour = input_dict["catheters_contour"]
            current_dwell_positions = copy.deepcopy(input_dict["dwell_positions"])
            current_grid = copy.deepcopy(input_dict["grid"])
            current_grid_real_zs = copy.deepcopy(input_dict["grid_real_zs"])
            current_closer_to_og_ref_points = copy.deepcopy(input_dict["closer_to_og_ref_points"])
            current_transforms = copy.deepcopy(input_dict["transforms"])
        else:
            current_ct_volume = self.current_ct_volume
            current_catheters_contour = self.current_catheters_contour
            current_dwell_positions = copy.deepcopy(self.current_dwell_positions)
            current_grid = copy.deepcopy(self.current_grid)
            current_grid_real_zs = copy.deepcopy(self.current_grid_real_zs)
            current_closer_to_og_ref_points = copy.deepcopy(self.current_closer_to_og_ref_points)
            current_transforms = copy.deepcopy(self.transforms)

        return (current_ct_volume,
                current_catheters_contour,
                current_dwell_positions,
                current_grid,
                current_grid_real_zs,
                current_closer_to_og_ref_points,
                current_transforms) 
    
    def project_on_plane(self, input_dict:Dict=None, inplace:bool=True, projection:str="spline") -> Dict[str, Any]:
        """
        Project the function defined by a list of points (dwell positions), onto a plane.
        Function fitted can be a line or a spline. The plane is simply defined by a z
        coordinate. This is assuming the catheters have already been rotated in a way
        that the insertion rows are parallel to the x axis and catheters ~ perpendicular
        to the z axis.
        This function is useful if ever one applicator is not fully inserted until the end
        of one catheter.
        """
        assert projection in ["spline", "line"], (
            "Projection must be either 'spline' for spline or 'line' for line."
        )
        if input_dict is not None:
            assert not inplace, (
                "Input dictionary can only be used when inplace is False."
            )
        (
            current_ct_volume,
            current_catheters_contour,
            current_dwell_positions,
            current_grid,
            current_grid_real_zs,
            current_closer_to_og_ref_points,
            current_transforms
        ) = self.unwrap_state(input_dict=input_dict)

        # Projecting on min z coordinate of the points
        z_coord = np.min([current_grid_real_zs[channel][0][2] 
                          for channel in current_grid_real_zs.keys()])

        new_grid = {}
        for channel in current_grid.keys():
            new_grid[channel] = []
            pt, _, _ = project_on_z_coord(current_dwell_positions[channel], z_coord, projection)
            new_grid[channel].append(pt)

        if inplace:
            self.current_grid = new_grid
            self.current_grid_real_zs = current_grid_real_zs
            self.current_dwell_positions = current_dwell_positions
            self.current_ct_volume = current_ct_volume
            self.current_catheters_contour = current_catheters_contour
            self.current_closer_to_og_ref_points = current_closer_to_og_ref_points
            self.transforms = current_transforms
        else:
            return {
                "ct_volume": current_ct_volume,
                "catheters_contour": current_catheters_contour,
                "dwell_positions": current_dwell_positions,
                "grid": new_grid,
                "grid_real_zs": current_grid_real_zs,
                "closer_to_og_ref_points": current_closer_to_og_ref_points,
                "transforms": current_transforms
            }

    def get_potential_nb_catheters_per_row(
            self, grid_points:Dict[str, List[List[float]]], 
            y_step:float=10., save_files:bool=False, 
            save_sub_folder:str="second_rotation_testdirection1") -> int:

        """
        We cut the grid int 2 cm chunks in the x axis, check the spread in the y axis.
        We do this because directly taking y_step = (y_max - y_min) / n_rows will not work
        properly if the instertion rows are bended a lot.
        Finding the (y_max - y_min) of each chunk
        Assumption, rows are at least 1cm appart, which should be the case if the Dr uses a grid
        to insert the catheters.
        """
        raise NotImplementedError(
            "This function does not work but I just want to keep it to "
            "show we tried that approach and it does not work."
        )
        points = np.array([
            grid_points[key][0][:2] for key in grid_points.keys()
        ])

        xs = points[:,0]
        ys = points[:,1]

        y_min, y_max = np.min(ys), np.max(ys)
        nb_catheters_per_row = []
        angles_between_every_two_pts_in_row_and_x_axis = []
        for y_bound1 in np.arange(y_min, y_max, y_step):
            y_bound2 = y_bound1 + y_step
            idx_pts_to_consider = np.where((ys > y_bound1) & (ys < y_bound2))
            corresponding_xs = xs[idx_pts_to_consider]
            corresponding_ys = ys[idx_pts_to_consider]
            # Sort pts by x coordinate
            sorted_indices = np.argsort(corresponding_xs)
            corresponding_xs_sorted = corresponding_xs[sorted_indices]
            corresponding_ys_sorted = corresponding_ys[sorted_indices]
            for i in range(len(idx_pts_to_consider[0]) - 1):
                pt1 = (corresponding_xs_sorted[i], corresponding_ys_sorted[i])
                pt2 = (corresponding_xs_sorted[i+1], corresponding_ys_sorted[i+1])
                angle = np.abs(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
                angles_between_every_two_pts_in_row_and_x_axis.append(angle)
            if save_files:
                mx = np.mean(corresponding_xs)
                pts = [
                    [mx, y_bound1, points[0][2]],
                    [mx, y_bound2, points[0][2]],
                ]
                create_slicer_markup_points(
                    os.path.join(self.save_details_folder, save_sub_folder, f"interval_{y_bound1}.mrk.json"), 
                    pts, 
                    color=[0.6,0.6,0.3]
                    )
            nb_catheters_per_row.append(len(corresponding_xs))
        
        return nb_catheters_per_row, angles_between_every_two_pts_in_row_and_x_axis

    @staticmethod
    def save_points(points:List[List[float]], file_path:str, color:List[float]=[1.0,0.8,0.7]):
        """
        Save the points to a json file.
        """
        assert file_path.endswith(".mrk.json"), "File name must end with .mrk.json to be used in 3D Slicer."
        create_marker_pts_from_catheter_dict(
                file_path,
                points, 
                color=color
                )
    
class CatheterIdentificator:
    def __init__(self, ct_volume_path:str|Path=None, catheters_contour_path:str|Path=None, 
                 save_details_folder:str|Path=None, name_dwell_pos:str="clinical"):
        """
        Class made to label the catheter numbers in the volume.
        """
        assert ct_volume_path is not None or catheters_contour_path is not None, (
            "At least one volume must be provided to create transforms in this class."
        )
        self.ct_volume_path = ct_volume_path
        self.catheters_contour_path = catheters_contour_path
        if ct_volume_path is None:
            self.ct_volume = None
        else:
            self.ct_volume = sitk.ReadImage(ct_volume_path)

        if catheters_contour_path is None:
            self.catheters_contour = None
        else:
            self.catheters_contour = sitk.ReadImage(catheters_contour_path)

        self.save_details_folder = save_details_folder
        self.rotated_dp = None
        self.mapping_to_catheter_channel = None
        # If we have a number of catheters that is a square, we might
        # have to rotate the volume 90 degrees more to have the catheter
        # correctly aligned with the x axis.
        self.rotating_90_more = False

        self.name_dwell_pos = name_dwell_pos

        self.grid_viewer = None
        self.breast_side = None
        self.correct_view_state = None

    def order_randomly(self, created_needle:dict):
        """
        Order the catheter randomly.
        This function is used when the catheter are not inserted in a grid
        and we cannot order them by convention.
        """
        mapping = {}
        counter_channel = 0
        for key in created_needle["Dwell positions"].keys():
            counter_channel += 1
            mapping[key] = "Channel_" + str(counter_channel)
        
        return self.replace_keys_from_mapping(mapping, created_needle)

    @staticmethod
    def replace_keys_from_mapping(mapping, created_needle:dict):
        # Remaming all the keys in the dictionnaries
        new_created_needle_d = {}
        for key in created_needle.keys():
            new_created_needle_d[key] = {}
            if isinstance(created_needle[key], dict):
                for sub_k in created_needle[key].keys():
                    new_key = mapping[sub_k]
                    assert new_key not in new_created_needle_d[key].keys(), (
                        "The key {} already exists in the new created needle dict. {} ".format(new_key, new_created_needle_d[key].keys())
                    )
                    new_created_needle_d[key][new_key] = copy.deepcopy(created_needle[key][sub_k])
            else:
                new_created_needle_d[key] = copy.deepcopy(created_needle[key])

        return new_created_needle_d
    
    def order_catheter_by_JGH_breast_convention(
            self, created_needle:dict, user_input:List[int]=None, save_files:bool=False):
        """
        Order the catheter by convention.
        The top right corner catheter as channel 1, then the one on the left as
        channel 2, etc until the row is done and then we move to next row.
        To do this, we first rotate the catheter volume to have all the insertion in
        the axial plane and then we order the catheter by the x and y coordinates.
        This function will fail if not all catheters have been manually inserted in
        the same direction. 
        """
        if save_files:
            assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
        
        self.grid_viewer = InsertionGridViewer(
            ct_volume_path=(self.ct_volume_path if self.ct_volume_path is not None else None),
            catheters_contour_path=(self.catheters_contour_path if self.catheters_contour_path is not None else None),
            save_details_folder=self.save_details_folder,
            dwell_positions=created_needle['Dwell positions'],
            name_dwell_pos="created_analytical"
        )
        viewer_states = self.grid_viewer.get_insertion_grid_as_rows(save_files=save_files)
        self.breast_side = self.grid_viewer.breast_side

        if user_input is not None:
            # User input is a list of integers indicating the number of catheters per row
            # We assume that the user input is correct and that the number of catheters
            # is equal to the sum of the user input.
            self.mapping_to_catheter_channel, identified_n_rows = self.order_right_to_left_descending_w_input(
                viewer_states, user_input=user_input, save_files=save_files)
        else:
            # We will try to automatically find the best number of rows based on the points
            # and the closeness to a line metric.
            # This function is in progress and does not work for all patients.
            self.mapping_to_catheter_channel, identified_n_rows = self.order_right_to_left_descending_auto(
                viewer_states, print_details=save_files, save_files=save_files)
        
        # Remnaming all the keys in the dictionnaries
        new_created_needle_d = {}
        for key in created_needle.keys():
            new_created_needle_d[key] = {}
            if isinstance(created_needle[key], dict):
                for sub_k in created_needle[key].keys():
                    # Grid should have a standard "Channel_X" naming starting at 1
                    # This is the case for dictionnaries coming from the CatheterSetUp class
                    # but this is not the case for digitized AI or analytical contours
                    # which have a naming like "Contour_needle_X" starting at 0
                    if not sub_k.startswith("Channel_"):
                        assert sub_k.startswith("Contour_needle_"), (
                            "The catheter keys should either start with 'Channel_' for the CatheterSetUp class "
                            "or 'Contour_needle_' for digitized AI or analytical contours."
                        )
                        channel_index = int(sub_k.split("_")[-1])  # Get the index of the catheter
                        target_sub_k = "Channel_" + str(channel_index + 1)
                    else:
                        target_sub_k = sub_k
                    new_key = self.mapping_to_catheter_channel[target_sub_k]
                    assert new_key not in new_created_needle_d[key].keys(), (
                        "The key {} already exists in the new created needle dict. {} ".format(new_key, new_created_needle_d[key].keys())
                    )
                    new_created_needle_d[key][new_key] = copy.deepcopy(created_needle[key][sub_k])
            else:
                new_created_needle_d[key] = copy.deepcopy(created_needle[key])

        return new_created_needle_d, identified_n_rows
    
    def compute_cost_of_assignement(
            self, points:List[List[float]], our_insertion_grid_points:Dict[str, List[float]], 
            x_min:float, x_max:float, x_step:float, y_step:float, n_rows:int, max_cols:int, 
            user_input:List[int], user_grid_y_min:float, align:str = "left"):

        assert align in ["right", "center", "left"], (
            "The alignment must be one of: right, center, left"
        )   
        user_input_grid_pts = {}
        channel_counter = 0
        for i in range(n_rows):
            user_grid_x_min = x_min
            user_grid_x_max = x_max
            # Centering the catheter insertion points in the middle of the grid
            # So that x_gap between catheters is always the same
            diff_nb_caheters = max_cols - user_input[i]
            shift_on_both_sides = diff_nb_caheters * x_step / 2
            if align == "center":
                # Center alignment
                user_grid_x_min += shift_on_both_sides
                user_grid_x_max -= shift_on_both_sides
            elif align == "left":
                # Left alignment
                user_grid_x_max -= shift_on_both_sides * 2
            else:
                # Right alignment
                user_grid_x_min += shift_on_both_sides * 2
            # Is the new rows that we created separated by the same x_steps?
            assert np.isclose((user_grid_x_max - user_grid_x_min), x_step * (user_input[i]-1), atol=1e-4), (
                "The new rows are not separated by the same x_steps. "
                f"Range in x you determined {(user_grid_x_max - user_grid_x_min)} is not equal to "
                f"the range that will be used for the grid : {x_step * (user_input[i]-1)}"
            )
            row_coord = user_grid_y_min + (i * y_step)
            row_pts = np.linspace(user_grid_x_min, user_grid_x_max, user_input[i])
            if self.breast_side == "right":
                # For right breast brachy, we want the catheter rows from bottom to top
                if self.grid_viewer.button_view:
                    row_pts = row_pts[::-1]  # Reverse the order of the points in the row
            else:
                assert self.breast_side == "left", (
                    "Invalid breast side. Should be either 'left' or 'right'."
                )
                # For left breast brachy, we want the catheter rows from top to bottom
                if not self.grid_viewer.button_view:
                    row_pts = row_pts[::-1]  # Reverse the order of the points in the row

            for row_pt in row_pts:
                channel_counter += 1
                # We take the first point of the catheter as the insertion point
                user_input_grid_pts[f"Channel_{channel_counter}"] = [
                        # x defined by number of catheter per row
                        float(row_pt), 
                        # y defined by the row number
                        float(row_coord),
                        # z is always the same, just as the insertion grid points we created
                        our_insertion_grid_points[list(our_insertion_grid_points.keys())[0]][2]
                    ]
                
        user_input_grid_pts_l = list(user_input_grid_pts.values())
        our_insertion_grid_points_l = list(our_insertion_grid_points.values())

        # Creating the cost/distance from our insertion grid points 
        # to the user input grid points
        cost_matrix = np.empty((len(our_insertion_grid_points_l), len(user_input_grid_pts_l)))
        for i in range(len(our_insertion_grid_points_l)):
            for j, pt in enumerate(points):
                cost_matrix[j, i] = distance(pt, user_input_grid_pts_l[i])

        # Apply the assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        all_assign_costs = cost_matrix[row_ind, col_ind]

        total_cost = all_assign_costs.sum()
        assignments = [(i, j) for i, j in zip(row_ind, col_ind)]

        return user_input_grid_pts, assignments, total_cost, all_assign_costs

    def order_right_to_left_descending_w_input(
            self, viewer_states:List[Dict[str,Any]], 
            user_input:List[int], save_files:bool=True, 
            print_details:bool=False) -> Tuple[Dict[str, str], int]:
        """
        Order the catheter by convention.
        The top right corner catheter as channel 1, then the one on the left as
        channel 2, etc until the row is done and then we move to next row.
        Inputs:
        - d: a dictionary containing the catheter insertion points
        - user_input: a list of integers indicating the number of catheters per row,
        ordered from the top row to the bottom row.
        Returns:
        - mapping: a dictionary mapping the catheter insertion points to the channel number
        - identified_n_rows: the number of rows identified based on the user input.
        """
        assert isinstance(user_input, list), "user_input must be a list of integers."
        assert len(user_input) > 0, "user_input must contain at least one integer."
        
        if save_files:
            assert self.save_details_folder is not None, (
                "self.save_details_folder must be set to save the points."
            )
            # Save the user input points
            with open(os.path.join(self.save_details_folder, "user_input.txt"), "w") as f:
                f.write("User input: {}\n".format(user_input))
        
        if self.save_details_folder is not None:
            save_dir = os.path.join(self.save_details_folder, "second_rotation")
            os.makedirs(save_dir, exist_ok=True)

        min_cost = float("inf")
        for i, state in enumerate(viewer_states):
            d = state["grid"]
            if print_details:
                print(f"Processing viewer state {i+1}/{len(viewer_states)}")
            
            (temp_min_cost, 
             temp_user_input_grid_pts, 
             temp_our_insertion_grid_points, 
             temp_assignments, 
             temp_align, 
             temp_all_costs, 
             y_step, 
             x_step, 
             y_min,
             y_max) = self.compute_assignement_cost_to_grid(
                user_input, d, save_files=save_files, state_idx=i
            )
            if print_details:
                print(f"Cost for viewer state {i}: {temp_min_cost:.2f}")
            if temp_min_cost < min_cost:
                min_cost = temp_min_cost
                user_input_grid_pts = temp_user_input_grid_pts
                our_insertion_grid_points = temp_our_insertion_grid_points
                assignments = temp_assignments
                align = temp_align
                index_correct_view = i
                self.rotated_dp = state["dwell_positions"]
                self.correct_view_state = state
                all_costs = temp_all_costs
        if print_details:
            print(f"Using viewer state {index_correct_view} with minimum cost: {min_cost:.2f}") 

        if save_files:
            our_insertion_grid_points_l = list(our_insertion_grid_points.values())
            user_input_grid_pts_l = list(user_input_grid_pts.values())
            save_dir = os.path.join(self.save_details_folder, "user_input_catheter_assignement")
            os.makedirs(save_dir, exist_ok=True)
            self.grid_viewer.save_state(
                save_dir, state_name=f"correct_view", input_dict=viewer_states[index_correct_view], save_ref_pt=True
            )

            for k in our_insertion_grid_points.keys():
                create_slicer_markup_points(
                    os.path.join(save_dir,  f"our_insertion_grid_points_{k}.mrk.json"),
                    [our_insertion_grid_points[k]], [0.5,0.5,0.9])
            for k in user_input_grid_pts.keys():
                create_slicer_markup_points(
                    os.path.join(save_dir,  f"user_input_grid_pts_{k}_{align}_alignement.mrk.json"),
                    [user_input_grid_pts[k]], [1.,0.7,0.8])

            create_slicer_markup_points(
                os.path.join(save_dir,  f"all_our_insertion_grid_points_l.mrk.json"),
                our_insertion_grid_points_l, [0.5,0.5,0.9])
            create_slicer_markup_points(
                os.path.join(save_dir,  f"all_user_input_grid_pts_l_{align}_alignement.mrk.json"),
                user_input_grid_pts_l, [1.,0.7,0.8])
            with open(os.path.join(save_dir, "all_assignment_costs.json"), "w") as f:
                assign_to_cost = {str(assign[0])+"to"+str(assign[1]): float(cost) for assign, cost in zip(assignments, all_costs)}
                metadata = {
                    "assign_to_cost": assign_to_cost,
                    "y_step": float(y_step),
                    "x_step": float(x_step),
                    "y_min": float(y_min),
                    "y_max": float(y_max)
                }
                json.dump(metadata, f, indent=4)

        ## Based on this assignement we create the mapping dictionnary
        mapping = {}
        for i, j in assignments:
            # i is the index of the point in our insertion grid points
            # j is the index of the point in the user input grid points
            catheter_key = list(our_insertion_grid_points.keys())[i]
            channel_key = list(user_input_grid_pts.keys())[j]
            mapping[catheter_key] = channel_key

        if print_details:
            print("created mapping from fitted grid", mapping)

        # Saving the mapped insertion grid points
        if save_files:
            reversed_mapping = {v: k for k, v in mapping.items()}
            save_dir = os.path.join(self.save_details_folder, "user_input_catheter_assignement")
            os.makedirs(save_dir, exist_ok=True)
            for k in reversed_mapping.keys():
                create_slicer_markup_points(
                    os.path.join(save_dir, f"mapped_insertion_grid_points_{k}.mrk.json"),
                    [our_insertion_grid_points[reversed_mapping[k]]], [0.5,0.4,0.7])
            # Saving the mapping as a json file
            with open(os.path.join(save_dir, "mapping.json"), "w") as f:
                json.dump(mapping, f)

        return mapping, len(user_input)

    def compute_assignement_cost_to_grid(
            self, user_input:List[int], d:Dict[str, List[List[float]]], 
            print_details:bool=False, save_files:bool=False, state_idx:int=0):
        """
        Compute the assignment cost of the user input to the grid points.
        There is two grids: the one created by the viewer which is more or less
        the tip of each catheter and the one created based on the user input.
        The way we create the user input grid points is highly important
        and make the algorithm works or fail. We make the grid based on 
        the min and max x and y coordinates of the insertion points and
        the number of rows and we find out the y step manually.
        We also tried shifting the user grid far from the insertion grid
        but it does not work. The best solution is to superimpose them the best
        we can.

        When a user input tells us that a row does not contain the same number of 
        catheters than other rows, we try different alignments (left, center, right)
        and take the one with the lowest assignment cost between the two grids.
        """
        points = [d[k][0] for k in d.keys()]
        n_rows = len(user_input)

        # Get the min and max x and y coordinates
        xs = np.array([pt[0] for pt in points])
        ys = np.array([pt[1] for pt in points])
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        # Create a grid of points corresponding to the user input
        # Grid should have a standard "Channel_X" naming starting at 1
        # This is the case for dictionnaries coming from the CatheterSetUp class
        # but this is not the case for digitized AI or analytical contours
        # which have a naming like "Contour_needle_X" starting at 0
        renamed_d = {}
        for k in d.keys():
            if not k.startswith("Channel_"):
                assert k.startswith("Contour_needle_"), (
                    "The catheter keys should either start with 'Channel_' for the CatheterSetUp class "
                    "or 'Contour_needle_' for digitized AI or analytical contours."
                )
                channel_index = int(k.split("_")[-1])  # Get the index of the catheter
                new_name = "Channel_" + str(channel_index+1)
            else:
                new_name = k
            renamed_d[new_name] = d[k]
        our_insertion_grid_points = {
            k: renamed_d[k][0] for k in renamed_d.keys()
        }

        max_cols = max(user_input)
        x_step = (x_max - x_min) / (max_cols - 1)

        ## Finding the step for y
        
        # ### Option 1: based on x_step
        # # We assume the grid used to insert the catheters was filled at equal spacing for x and y
        # if (y_max - y_min) / n_rows > x_step:
        #     diff = (y_max - y_min) - (n_rows - 1) * x_step
        #     print("diff bro1", diff)
        #     user_grid_y_min = y_min + diff / 2
        # else:
        #     diff = (n_rows - 1) * x_step - (y_max - y_min)
        #     print("diff bro2", diff)
        #     user_grid_y_min = y_min - diff / 2
        # y_step = x_step
        # user_grid_y_min = (
        #     y_min + 
        #     # Centering the middle row in the middle y coord of the grid
        #     (y_max - y_min) / 2 - 
        #     # Between every two row of the user grid there will be a 2 * user_grid_step gap
        #     (2 * ((n_rows-1)/2) * user_grid_step)
        #  ) # + shift_mm
        # # user_grid_y_max = y_max + shift_mm

        ### Option 2: we find out potential y_step manually
        # We cut the grid int 2 cm chunks in the x axis, check the spread in the y axis.
        # We do this because directly taking y_step = (y_max - y_min) / n_rows will not work
        # properly if the instertion rows are bended a lot.
        # Finding the (y_max - y_min) of each chunk
        max_y_range = 0
        if n_rows == 1:
            y_step = 0
        else:
            for x_bound1 in np.arange(x_min, x_max, 20.):
                x_bound2 = x_bound1 + 20.
                idx_pts_to_consider = np.where((xs > x_bound1) & (xs < x_bound2))
                if len(idx_pts_to_consider[0]) > 0:
                    corresponding_ys = ys[idx_pts_to_consider]
                    range_y = np.max(corresponding_ys) - np.min(corresponding_ys)
                    if range_y > max_y_range:
                        max_y_range = range_y

            y_step = max_y_range / (n_rows -1)

        # This is the maximum range we want for the y coordinates
        # We will use this to center the user grid in the middle of the insertion grid
        diff_defined_y_range_full_y_range = (y_max - y_min) - max_y_range
        assert diff_defined_y_range_full_y_range >= 0, (
            "The defined y range is larger than the full y range. "
            "This should not happen. Check the points."
        )
        user_grid_y_min = (
            y_min +
            # Centering the defined y range in the middle of the full y range
            diff_defined_y_range_full_y_range / 2
        ) 
      
        shifting = False
        if shifting:
            shift_mm = 100.
            assert shift_mm > (x_max - x_min), (
                "The shift_mm must be larger than the x range of the points. "
                "This is to avoid the points from our insertion grid to be too close to the user input grid points."
            )
            if shift_mm > 0:
                x_min -= shift_mm
                x_max -= shift_mm
                user_grid_y_min -= shift_mm

        if print_details:
            print(f"x_min: {x_min}, x_max: {x_max}, x_step: {x_step}")
            print(f"user_grid_y_min: {user_grid_y_min}, user_grid_y_step: {y_step}")
        if len(set(user_input)) == 1:
            # All rows have the same number of catheters
            align = "center"
            user_input_grid_pts, assignments, total_cost, all_costs = self.compute_cost_of_assignement(
                points, our_insertion_grid_points, x_min, x_max, x_step, y_step, n_rows, max_cols,
                user_input, user_grid_y_min, align=align)
            if save_files:
                our_insertion_grid_points_l = list(our_insertion_grid_points.values())
                user_input_grid_pts_l = list(user_input_grid_pts.values())
                save_dir = os.path.join(self.save_details_folder, f"second_rotation_testdirection{state_idx+1}")
                os.makedirs(save_dir, exist_ok=True)
                for k in our_insertion_grid_points.keys():
                    create_slicer_markup_points(
                        os.path.join(save_dir,  f"our_insertion_grid_points_{k}_{align}_alignement.mrk.json"),
                        [our_insertion_grid_points[k]], [0.01568627450980392,0.0784313725490196,0.5019607843137255])
                for k in user_input_grid_pts.keys():
                    create_slicer_markup_points(
                        os.path.join(save_dir,  f"user_input_grid_pts_{k}_{align}_alignement.mrk.json"),
                        [user_input_grid_pts[k]], [1.,0.00784313725490196,0.00784313725490196])

                create_slicer_markup_points(
                    os.path.join(save_dir,  f"all_our_insertion_grid_points_l_{align}_alignement.mrk.json"),
                    our_insertion_grid_points_l, [0.01568627450980392,0.0784313725490196,0.5019607843137255])
                create_slicer_markup_points(
                    os.path.join(save_dir,  f"all_user_input_grid_pts_l_{align}_alignement.mrk.json"),
                    user_input_grid_pts_l, [1.,0.00784313725490196,0.00784313725490196])
                with open(os.path.join(save_dir, f"all_assignment_costs_alignement_{align}.json"), "w") as f:
                    assign_to_cost = {str(assign[0])+"to"+str(assign[1]): float(cost) for assign, cost in zip(assignments, all_costs)}
                    metadata = {
                        "assign_to_cost": assign_to_cost,
                        "y_step": float(y_step),
                        "x_step": float(x_step),
                        "y_min": float(y_min),
                        "y_max": float(y_max)
                    }
                    json.dump(metadata, f, indent=4)

            if print_details:
                print("Cost of the assignement with alignement {}: {}".format(align, total_cost))
            return total_cost, user_input_grid_pts, our_insertion_grid_points, assignments, align, all_costs, y_step, x_step, y_min, y_max
        else:
            # Rows have different number of catheters
            # We try different alignments and take the one with the lowest cost
            align = None
            user_input_grid_pts = None
            assignments = None
            min_cost = np.inf
            list_of_costs = None
            # for use_ipt in [user_input, user_input[::-1]]:
            #     # We try the user input as is and reversed
            #     print("Trying user input: {}".format(use_ipt))
            for potential_align in ["right", "center", "left"]:
                # Create the user input grid points based on the user input
                potential_user_input_grid_pts, potential_assignments, total_cost, all_costs = self.compute_cost_of_assignement(
                    points, our_insertion_grid_points, x_min, x_max, x_step, y_step, n_rows, max_cols,
                    user_input, user_grid_y_min, align=potential_align)
                if print_details:
                    print("Cost of the assignement with alignement {}: {}".format(potential_align, total_cost))
                if total_cost < min_cost:
                    min_cost = total_cost
                    user_input_grid_pts = potential_user_input_grid_pts
                    assignments = potential_assignments
                    align = potential_align
                    list_of_costs = all_costs
                if save_files:
                    our_insertion_grid_points_l = list(our_insertion_grid_points.values())
                    user_input_grid_pts_l = list(user_input_grid_pts.values())
                    save_dir = os.path.join(self.save_details_folder, f"second_rotation_testdirection{state_idx+1}")
                    os.makedirs(save_dir, exist_ok=True)
                    for k in our_insertion_grid_points.keys():
                        create_slicer_markup_points(
                            os.path.join(save_dir,  f"our_insertion_grid_points_{k}_{potential_align}_alignement.mrk.json"),
                            [our_insertion_grid_points[k]], [0.01568627450980392,0.0784313725490196,0.5019607843137255])
                    for k in user_input_grid_pts.keys():
                        create_slicer_markup_points(
                            os.path.join(save_dir,  f"user_input_grid_pts_{k}_{potential_align}_alignement.mrk.json"),
                            [user_input_grid_pts[k]], [1.,0.00784313725490196,0.00784313725490196])

                    create_slicer_markup_points(
                        os.path.join(save_dir,  f"all_our_insertion_grid_points_l_{potential_align}_alignement.mrk.json"),
                        our_insertion_grid_points_l, [0.01568627450980392,0.0784313725490196,0.5019607843137255])
                    create_slicer_markup_points(
                        os.path.join(save_dir,  f"all_user_input_grid_pts_l_{potential_align}_alignement.mrk.json"),
                        user_input_grid_pts_l, [1.,0.00784313725490196,0.00784313725490196])
                    with open(os.path.join(save_dir, f"all_assignment_costs_{potential_align}_alignement.json"), "w") as f:
                        assign_to_cost = {str(assign[0])+"to"+str(assign[1]): float(cost) for assign, cost in zip(assignments, all_costs)}
                        metadata = {
                            "assign_to_cost": assign_to_cost,
                            "y_step": float(y_step),
                            "x_step": float(x_step),
                            "y_min": float(y_min),
                            "y_max": float(y_max)
                        }
                        json.dump(metadata, f, indent=4)


            return min_cost, user_input_grid_pts, our_insertion_grid_points, assignments, align, list_of_costs, y_step, x_step, y_min, y_max

    def order_right_to_left_descending_auto(
            self, viewer_states:List[Dict[str, Any]], 
            save_files:bool=False, print_details:bool=False):
        """
        Order the catheter automatically by convention.
        The objective of this function is to separate the catheters into different
        rows. Once the rows are identified we can order the rows by y 
        coordinate and then the catheters by x coordinate. 
        Inputs:
        - d: a dictionary containing the catheter insertion points
        - save_files: a boolean indicating if we want to save the files
        """
        if save_files:
            assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
        
        d = viewer_states[0]["grid"]
        raise NotImplementedError(
            "This function is not implemented yet. It was in development until we realized it "
            "is hard to always have correct orientation for the insertion grid."
        )
        mapping = {}
        points = [d[k][0] for k in d.keys()]
        catheter_keys = list(d.keys())

        # Sorting all points based on x coord leftmost first
        xs_order = np.argsort([pt[0] for pt in points])
        sorted_points_x = [points[i] for i in xs_order]
        sorted_catheter_keys_x = [catheter_keys[i] for i in xs_order]


        # First, we check if row size is 1 (1 catheter per row)
        # i.e. one row of catheter only
        # We fit a spline to the insertion points and see if all points
        # are close to the spline. If they are we can say that the
        # catheter are in a single row. (Inspired by quality-of-fit
        # metric/detection of overlapping/colliding catheters in the 
        # catheter contour Separator class).
        
        # Testing if there is only one row of catheters. There cannot be
        # one column only since we rotated to have direction parallel to 
        # x axis.
        one_row_of_catheter = is_one_row(
            sorted_points_x, save_files=save_files, 
            save_dir=os.path.join(self.save_details_folder, "second_rotation/is_one_row"), 
            print_details=print_details)

        if one_row_of_catheter:
            best_bet_n_rows = 1
            best_match_rows = [sorted_points_x]
            best_match_labels = [0 for _ in range(len(sorted_points_x))]
        else:
            best_bet_n_rows = None
            best_match_rows = None
            best_match_labels = None
            metric = 9999

            # Assumption that there cannot be more rows than columns. 
            max_rows = int(np.ceil(np.sqrt(len(sorted_points_x)))) 
            for n_rows in range(2, max_rows+1):
                if print_details:
                    print("Trying {} rows.".format(n_rows))
                impossible_n_rows, best_match_rows, best_match_labels = self.make_groups_fitting_grid(
                    sorted_points_x, n_rows)
                if impossible_n_rows:
                    if print_details:
                        print("Impossible to create {} rows.".format(n_rows))
                    continue
                else:
                    if print_details:
                        print("Possible to create {} rows.".format(n_rows))
                    best_bet_n_rows = n_rows
                    break

                # Rows should have the same number of catheters +- 2
                potential_rows, potential_labels, conditions_met = self.make_n_groups_based_on_y(
                    sorted_points_x, n_rows, save_files=True)
                if not conditions_met:
                    if print_details:
                        print("Conditions not met for {} rows.".format(n_rows))
                    continue

                group_metric = self.compute_closness_to_line_metric(potential_rows)

                if print_details:
                    print("Grouping Metric {}.".format(group_metric))
                    print("Potential rows {}.".format(potential_rows))
                if group_metric < metric:
                    metric = group_metric
                    best_bet_n_rows = n_rows
                    print("Updating best match rows to {}.".format(potential_rows))
                    best_match_rows = potential_rows
                    best_match_labels = potential_labels
            if print_details:
                print("Best bet number of rows {}.".format(best_bet_n_rows))
                print("Grouping Metric {}.".format(metric))
                print("Best grouping rows {}.".format(best_match_rows))

            ### Last check on the angle of the different rows compared to 
            # the catheters general row direction. THey should be more or less
            # parallel so any max angle greater than 45 degrees is alarming.
            angle = self.compute_angle_metric(best_match_rows, save_files=save_files)
            if not self.rotating_90_more:
                assert angle < 45, (
                    "The angle between the different rows is too big {}.".format(angle)
                )
            else:
                assert angle % 90 < 45, (
                    "The angle between the different rows is too big {}.".format(angle)
                )


        if save_files:
            # Just to save first assignement
            _ = self.make_n_groups_based_on_y(
                sorted_points_x, best_bet_n_rows, clustering_method="custom", save_files=True)
            save_dir = os.path.join(self.save_details_folder, "second_rotation")
            for row_idx, row in enumerate(best_match_rows):
                create_slicer_markup_points(
                    os.path.join(save_dir, f"row_{row_idx}.mrk.json"),
                    row, 
                    color=[1.0,0.8,0.7] # red for created data
                )
        
        # Once we have the rows, already ordered by y, we can order the
        # catheter of each row by y coordinate.
        best_match_catheter_keys = []
        for i in range(best_bet_n_rows):
            keys = []
            for catheter_key, label in zip(sorted_catheter_keys_x, best_match_labels):
                if label == i:
                    keys.append(catheter_key)
            best_match_catheter_keys.append(keys)

        # We have rows that are not necessarily ordered by y coordinate anymore
        # Because the clustering messed up the order. Let's order them back.
        mean_y_per_row = []
        for row in best_match_rows:
            mean_y_per_row.append(np.mean([pt[1] for pt in row]))
        ys_order = np.argsort(mean_y_per_row)
        best_match_rows = [best_match_rows[i] for i in ys_order]
        best_match_catheter_keys = [best_match_catheter_keys[i] for i in ys_order]
        if print_details:
            print("Best match rows: ", best_match_rows)
            print("Best match catheter keys: ", best_match_catheter_keys)

        counter_channel = 0
        print("Assigning channels to catheters...")
        for row_idx, (row, catheter_keys) in enumerate(zip(best_match_rows, best_match_catheter_keys)):
            # Ordering the catheter within a row. Low x first i.e. higher in CT scan. 
            print("catheter_keys", catheter_keys)
            print("Row {}: {}".format(row_idx, row))
            xs_order = np.argsort([pt[0] for pt in row])
            print("xs", [pt[0] for pt in row])
            print("xs_order", xs_order)
            for x in xs_order:
                counter_channel += 1
                mapping[catheter_keys[x]] = "Channel_" + str(counter_channel)

        return mapping, best_bet_n_rows
    
    def make_row_of_same_size(self, points:List[List[float]], row_size:int):
        """
        Split a list of points into groups of size row_size.
        """
        return [points[i:i + row_size] for i in range(0, len(points), row_size)]
    
    def make_groups_fitting_grid(self, points:List[List[float]], n_rows:int):
        """
        Creates a grid and assigns the points to the grid.
        """

        impossible_grid = True
        # Get the min and max x and y coordinates
        xs = np.array([pt[0] for pt in points])
        ys = np.array([pt[1] for pt in points])
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        # Create a grid of points
        # We want exactly n_rows rows of points
        grid_y = np.linspace(y_min, y_max, n_rows + 1)
        setp_y = grid_y[1] - grid_y[0]
        grid_y += setp_y / 2
        # Squares are 10cm size.
        x_step = 10
        # # Does not work because makes too many points in the grid.
        # grid_x = np.arange(x_min, x_max+x_step, x_step )
        # To get an idea of the maximum number of catheters per
        # row, we can divide the number of catheters by the number of rows
        # This does not work since there is not always the same number of
        # catheters per row
        # max_n_catheters = int(np.ceil(len(points)/n_rows))
        # We make a first assignement, get the maximum number of catheters
        # per row and then we create the grid.
        potential_rows, potential_labels, conditions_met = self.make_n_groups_based_on_y(
            points, n_clusters=n_rows, 
            clustering_method="custom", save_files=True)
        if not conditions_met:
            print("Conditions not met for {} rows.".format(n_rows))
            return impossible_grid, None, None
        # Get the maximum number of catheters per row
        max_n_catheters = np.max([len(r) for r in potential_rows])
        grid_x = np.linspace(x_min, x_max, max_n_catheters + 1)

        # Create a grid of points
        print("grid_x", grid_x)
        print("grid_y", grid_y)
        grid_points = []
        for i in range(len(grid_x) - 1):
            for j in range(len(grid_y) - 1):
                grid_points.append([grid_x[i], grid_y[j], points[0][2]])

        if len(grid_points) < len(points):
            return impossible_grid, None, None
        else:
            save_dir = os.path.join(self.save_details_folder,"second_rotation", "grid_tests")
            os.makedirs(save_dir, exist_ok=True)
            create_slicer_markup_points(
                os.path.join(save_dir,  f"grid_points_{n_rows}_rows.mrk.json"),
                grid_points, [0.5,0.5,0.9])
            
            # Creating the cost/distance from point to grid point matrix
            # distance_matrix is len(points) x len(grid_points)
            distance_matrix = np.empty((len(points), len(grid_points)))
            for i in range(len(grid_points)):
                for j, pt in enumerate(points):
                    distance_matrix[j, i] = distance(pt, grid_points[i])
            print("len(points)", len(points))
            print("len(grid_points)", len(grid_points))
            print("nrows", n_rows)
            print("distance_matrix", distance_matrix)
            print("shape", distance_matrix.shape)
            cost_matrix = np.copy(distance_matrix)
            # Apply the assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assignments = [(i, j) for i, j in zip(row_ind, col_ind) ]

            labels = []
            for point_idx, grid_pt_idx in assignments:
                labels.append(int(np.where(grid_y == grid_points[grid_pt_idx][1])[0][0]))

            print("created labels from fitted grid", labels)
            new_grps = create_group_from_labels(points, labels)
            save_dir = os.path.join(self.save_details_folder, "second_rotation")
            for row_idx, row in enumerate(new_grps):
                create_slicer_markup_points(
                    os.path.join(save_dir,  "grid_tests", f"{n_rows}_rows", f"row_{row_idx}.mrk.json"),
                    row, 
                    color=[1.0,0.8,0.7] # red for created data
                )

            grid_refiner = GridSanityChecker(points, labels, save_details_folder=self.save_details_folder)
            possible_rows, labels, grps = grid_refiner.refine_row_proposal()
            new_grps = create_group_from_labels(points, labels)

            grid_checker = GridSanityChecker(points, labels)
            cluster_proposal_respect_conditions = grid_checker.check_grid_conditions()
            impossible_grid = not cluster_proposal_respect_conditions
            return impossible_grid, new_grps, labels

    def make_n_groups_based_on_y(
            self, points:List[List[float]], n_clusters:int, 
            clustering_method:str="custom", save_files:bool=False):
        """
        Cluster the points based on their x and y coordinates.
        """

        # Extract y-coordinates only
        y_coords = np.array(points)[:, 1].reshape(-1, 1)

        if clustering_method == "kmeans":
            # Perform KMeans clustering on y-coordinates
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(y_coords)

        elif clustering_method == "gmm":
            # Perform Gaussian Mixture Model clustering
            weight_y = True
            if weight_y:
                X_weighted = np.copy(points)
                X_weighted[:, 1] *= 2
            else:
                X_weighted = np.copy(points)
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
            labels = gmm.fit_predict(X_weighted)

        elif clustering_method == "dbscan":
            # DBSCAN clustering
            # eps is super important to tune
            # min_samples should be 2 (or 1) since in a neighborhood of 1 catheter in a row
            # there should be two catheters (1 on each side)
            clustering = DBSCAN(eps=10, min_samples=2, metric='euclidean').fit(points)
            labels = clustering.labels_

        elif clustering_method == "agglomerative":
            # Perform Agglomerative clustering 
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward').fit(points)
            labels = clustering.labels_

        elif clustering_method == "custom":
            first_assignement = "binning" # "gmm_means" # "binning"
            ### First assignment 
            if first_assignement == "gmm_means":
                mixture = GaussianMixture(n_components=n_clusters, random_state=0, init_params='k-means++').fit(y_coords)
                row_means = mixture.means_.flatten()
                print("MEANS YS GAUSSIAN MIXTURES", row_means)

                # Assign each point to closest mean y (i.e. nearest row)
                def find_nearest_mean(y, means):
                    return np.argmin(np.abs(means - y))

                labels_1st_assign = [find_nearest_mean(y, row_means) for y in y_coords]
                print("labels_1st_assign", labels_1st_assign)

            elif first_assignement == "binning":
                # Step 1: Compute y-range and bin edges
                ys = np.array(points)[:, 1]
                y_min, y_max = np.min(ys), np.max(ys)
                bin_edges = np.linspace(y_min, y_max, n_clusters + 1)

                # Step 2: Assign each point to a row/bin based on its y
                labels_1st_assign = np.digitize(ys, bin_edges) - 1
                # (digitize bins are 1-based, so subtract 1)

                # Fix edge case: y == y_max should go in last bin
                labels_1st_assign[labels_1st_assign == n_clusters] = n_clusters - 1
                print("labels_1st_assign", labels_1st_assign)
            else:
                raise NotImplementedError(
                    "First assignement method {} not implemented.".format(first_assignement)
                )
            print("len(labels_1st_assign), len(points)", len(labels_1st_assign), len(points))
            ### Second/Refined assignement
            # If after the first assignment, we fit a line on each row
            # and some point is assigned a row but is closer to the line
            # of another row, we change the assignment. You can see such cases
            # by looking at the first assignment rows saved below.

            grps = create_group_from_labels(points, labels_1st_assign)
            print("grp legnths", [len(grp) for grp in grps])
            mean_1st_assignemnt_fit = []
            direction_1st_assignement_fit = []
            for grp_idx, grp in enumerate(grps):
                mean, direction = fit_line(grp)
                # Force the direction to be from the point with higher x to point with
                # lower x in the group. 
                xs = np.array([pt[0] for pt in grp])
                arg_max_x = np.argmax(xs, axis=0)
                arg_min_x = np.argmin(xs, axis=0)

                if np.dot(direction, np.array(grp[arg_max_x]) - np.array(grp[arg_min_x])) < 0:
                    direction = -direction
                mean_1st_assignemnt_fit.append(mean)
                direction_1st_assignement_fit.append(direction)
                
            if save_files:
                assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
                save_dir = os.path.join(self.save_details_folder, "second_rotation", "first_custom_assignement", f"{n_clusters}_rows")
                save_grps_3Dslicer(grps, save_dir, key="first_a_")
                mean_x_all_points = np.mean([pt[0] for pt in points])
                mean_z_all_points = np.mean([pt[2] for pt in points])
                if first_assignement == "gmm_means":
                    # Creating a center point for each row
                    centers = []
                    for i in range(n_clusters):
                        centers.append([mean_x_all_points, row_means[i], mean_z_all_points])
                    create_slicer_markup_points(
                            os.path.join(save_dir, f"center_for_different_rows.mrk.json"), 
                            centers, 
                            color=[1.0,0.8,0.7] # red for created data
                            )
                else:
                    # Creating a bin edge point for each row separation
                    bin_pts = []
                    for i in range(len(bin_edges)):
                        bin_pts.append([mean_x_all_points, bin_edges[i], mean_z_all_points])
                    create_slicer_markup_points(
                            os.path.join(save_dir, f"bin_edges_for_different_rows.mrk.json"), 
                            bin_pts, 
                            color=[1.0,0.8,0.7] # red for created data
                            )

            labels = copy.deepcopy(labels_1st_assign)

            ### Checking if the catheter in each rows are not too close to each other
            # and if a catheter would be better assigned to another row i.e. if the 
            # angle between the newly created row and the mean catheter direction is 
            # smaller with the potential new assignement.
            grid_checker = GridSanityChecker(points, labels, save_details_folder=self.save_details_folder)
            possible_rows, labels, grps = grid_checker.refine_row_proposal()


        else:
            raise ValueError("Unknown clustering method: {}".format(clustering_method))

        # Create groups based on labels
        groups = create_group_from_labels(points, labels)
        if not possible_rows:
            cluster_proposal_respect_conditions = False
        else:
            grid_checker = GridSanityChecker(points, labels)
            cluster_proposal_respect_conditions = grid_checker.check_grid_conditions(print_details=True)
        return groups, labels, cluster_proposal_respect_conditions


    def compute_closness_to_line_metric(self, groups:List[List[List[float]]], plot:bool=False):
        """
        Compute a closness to line metric based on the rows provided.
        The metric is the mean distance of the different points to its projection on the corresponding
        line.
        """
        # Get a direction vector for each of the catheters
        distances_to_lines = []
        if plot:
            f = plt.figure(figsize=(10, 10))
        for grp_idx, grp in enumerate(groups):
            mean, direction = fit_line(grp)
            # Force the direction to be from the point with higher x to point with
            # lower x in the group. 
            xs = np.array([pt[0] for pt in grp])
            arg_max_x = np.argmax(xs, axis=0)
            arg_min_x = np.argmin(xs, axis=0)

            if np.dot(direction, np.array(grp[arg_max_x]) - np.array(grp[arg_min_x])) < 0:
                direction = -direction

            distances_to_line = []
            for pt in grp:
                projected_point = project_point_to_line(pt, mean, direction, return3D=True).tolist()
                distances_to_line.append(distance(projected_point, pt))

            segment_pts = [
                project_point_to_line(grp[arg_max_x], mean, direction, return3D=True).tolist(),
                project_point_to_line(grp[arg_min_x], mean, direction, return3D=True).tolist()
                ]
            xs = [pt[0] for pt in grp]
            ys = [pt[1] for pt in grp]
            if plot:
                plt.plot([p[0] for p in segment_pts], [p[1] for p in segment_pts], 'r-', lw=2, label=f'Line_{grp_idx}')
                plt.scatter(xs, ys, label=f'Catheters_{grp_idx}')
                print("Distances to line ", describe_array(distances_to_line))
            distances_to_lines.extend(distances_to_line)
            
        if plot:
            plt.title("Line fitting to rows for {} rows".format(len(groups)))
            plt.show()

        return np.max(distances_to_lines)
    
    def compute_closness_to_spline_metric(self, groups:List[List[List[float]]], plot:bool=False):
        """
        Compute a closness to spline metric based on the rows provided.
        The metric is the mean distance of the different points to its projection on the corresponding
        spline.
        """
        def get_spline_points(tck):
            """
            Get the points from the spline.
            """
            points = []
            u_fine = np.linspace(0, 1, 100)
            for u in u_fine:
                pt = splev(u, tck)
                points.append(pt)
            return points

        # Get a direction vector for each of the catheters
        distances_to_splines = []
        if plot:
            f = plt.figure(figsize=(10, 10))
        for grp_idx, grp in enumerate(groups):
            tck, u = fit_spline(np.array(grp), s=len(grp), k=3, nest=0)

            distances_to_spline = []
            for pt in grp:
                projected_point, t, d = project_point_to_spline(pt, tck)
                distances_to_spline.append(d)
            spline_pts = get_spline_points(tck)
            xs = [pt[0] for pt in grp]
            ys = [pt[1] for pt in grp]
            if plot:
                plt.plot([p[0] for p in spline_pts], [p[1] for p in spline_pts], 'r-', lw=2, label=f'Spline_{grp_idx}')
                plt.scatter(xs, ys, label=f'Catheters_{grp_idx}')
            distances_to_splines.extend(distances_to_spline)
            print("Distances to spline ", describe_array(distances_to_spline))
        if plot:
            plt.title("Spline fitting to rows for {} rows".format(len(groups)))
            plt.show()
        return np.mean(distances_to_splines)
    
    def compute_angle_metric(self, groups:List[List[List[float]]], save_files:bool=False, save_sub_dir:str="second_rotation"):
        """
        Compute an angle metric based on the rows provided.
        The metric is the mean angle of the different rows to the means direction of the rows.
        """

        # Get a direction vector for each of the catheters
        row_directions = []
        if save_files:
            row_means = []
        for grp_idx, grp in enumerate(groups):
            mean, direction = fit_line(grp)
            # Force the direction to be from the point with higher x to point with
            # lower x in the group. 
            xs = np.array([pt[0] for pt in grp])
            arg_max_x = np.argmax(xs, axis=0)
            arg_min_x = np.argmin(xs, axis=0)

            if np.dot(direction, np.array(grp[arg_max_x]) - np.array(grp[arg_min_x])) < 0:
                direction = -direction
            row_directions.append(direction)

            if save_files:
                row_color = np.random.uniform(low=0, high=1, size=(3)).tolist()
                save_dir = os.path.join(self.save_details_folder, save_sub_dir)
                row_means.append(mean)
                # save_segments_from_points( XXX HJ: commenting this out to avoid errors
                #     grp, mean, direction, save_dir, 
                #     file_name=f"row_{grp_idx}_direction_segments.mrk.json", 
                #     color=row_color, save_pts=True)    
                
                
        # Get a mean direction vector for all the catheters
        ## Mean directions from the different vectors
        # mean_catheter_direction = np.mean(row_directions, axis=0)
        # mean_catheter_mean =  np.mean(row_means, axis=0)
        ## Mean direction from the different points
        points = []
        for grp in groups:
            points += grp
        points = np.array(points)
        mean_catheter_mean, mean_catheter_direction = get_potential_mean_directions_oriented_rows(
            points, save_files=True, save_details_folder=save_details_folder
        )

        angles_with_reference = self.angles_to_reference(np.array(row_directions), mean_catheter_direction)
        print("Angles with reference direction: ", angles_with_reference)
        return np.max(angles_with_reference)
        

    @staticmethod
    def angles_to_reference(vectors, reference):
        # Normalize the reference vector
        ref_norm = np.linalg.norm(reference)
        if ref_norm == 0:
            raise ValueError("Reference vector cannot be zero.")
        reference = reference / ref_norm

        # Normalize the input vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Input vectors cannot contain zero vectors.")
        normalized_vectors = vectors / norms

        # Compute dot product with the reference
        dot_products = np.dot(normalized_vectors, reference)

        # Clamp values to avoid numerical errors outside the arccos domain
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Compute angles in radians
        angles_rad = np.arccos(dot_products)

        # Optionally convert to degrees
        angles_deg = np.degrees(angles_rad)

        return angles_deg


    def save_dwell_positions_by_channel(self, needle_dict:dict, save_dir:str|Path=None, dwell_pos_name:str="clinical"):
        """
        Save the dwell positions by channel.
        """
        if save_dir is not None:
            save_folder = save_dir
        else:
            assert self.save_details_folder is not None, (
                "You need to provide a save folder to save the points."
            )
            save_folder = self.save_details_folder
        
        # Get the dwell positions and rotate them
        dwell_positions = needle_dict["Dwell positions"]
        reverse_mapping = {v: k for k, v in self.mapping_to_catheter_channel.items()}
        for key in dwell_positions.keys():
            dp = dwell_positions[key]
            create_slicer_markup_points(
                os.path.join(save_folder, f"{key}_{dwell_pos_name}_dwell_positions.mrk.json"), 
                dp)
            rotated_dp = self.rotated_dp[reverse_mapping[key]]
            create_slicer_markup_points(
                os.path.join(save_folder, f"{key}_rotated_{dwell_pos_name}_dwell_positions.mrk.json"),
                rotated_dp)
            
    def apply_transforms_and_save_pts(self, dwell_positions:dict, save_dir:str|Path=None, dwell_pos_name:str="clinical"):
        """
        Apply the transforms to the dwell positions and save them.
        """
        if save_dir is not None:
            save_folder = save_dir
        else:
            assert self.save_details_folder is not None, (
                "You need to provide a save folder to save the points."
            )
            save_folder = self.save_details_folder
        
        # Transform the dwell positions
        tr_dwell_pos = copy.deepcopy(dwell_positions)
        for tr in self.correct_view_state["transforms"]:
            tr_dwell_pos = InsertionGridViewer.transform_pts_dict(tr_dwell_pos, tr)

        for key in tr_dwell_pos.keys():
            dp = tr_dwell_pos[key]
            create_slicer_markup_points(
                os.path.join(save_folder, f"{dwell_pos_name}_{key}_rotated_dwell_positions.mrk.json"),
                dp)
            
class GridSanityChecker:

    
    def __init__(
        self, original_points:List[List[float]], 
        original_labels:List[int], save_details_folder:str=None):
        """
        The aim of this class is to receive a proposal for groups of rows, 
        refine this proposal if possible and check if the proposal meets some
        conditions.
        """
        self.grps:List[List[List[float]]] = create_group_from_labels(
            original_points, original_labels)
        self.original_points = original_points  
        self.original_labels = original_labels
        self.save_details_folder = save_details_folder
        self.n_rows = len(self.grps)

        # Insertion grid is 1cm squares. Assuming we rotated the grid
        # properly so that we have the row parallel to the x axis,
        # the catheters should be more or less within 1cm of each 
        # other. However, skin moves.
        # Max distance between two catheters of the same row: 2 cm.
        self.max_dist_x_catheter = 20
        # Min distance between two catheters of the same row: 5mm.
        self.min_dist_x_catheter = 5
        self.max_dist_y_catheter = 10

    def refine_row_proposal(self, save_files:bool=True):
        """
        We assume the proposed rows have been proposed based on y coordinates.
        Since rows can be bent, the assignement of catheter points to rows is not
        necessarily correct. In the clinic, the insertion grid they use has 1 cm
        squares and no needles should have been implanted in the same square. We
        check if 2 catheters are very close in x coordinate. If that is the case,
        we try modifying the assignement so that the catheter are not too close in x.
        """

        points = copy.deepcopy(self.original_points)
        labels = copy.deepcopy(self.original_labels)
        unsolvable_issue = False
        previous_grps = copy.deepcopy(self.grps)
        new_grps = copy.deepcopy(self.grps)
        groups_created_are_evolving = True
        iteration = 0

        # Looping ensures we find the best possible config, 
        # see example for only 1 iteration of patient 1325562.
        while not unsolvable_issue and groups_created_are_evolving:
            iteration += 1

            print("Iteration ", iteration)
            print("********************REGROUPING TO CLOSEST ROW***************")

            labels, new_grps = self.regroup_pts_to_closest_row_direction(
                points, labels, new_grps, save_files=save_files, iteration=iteration)
            
            print("********************CHECKING DISTANCE X TOO CLOSE***************")

            catheters_min_x_distance_is_ok = False
            unsolvable_issue = False
            previous_problematic_pts = []
            while not catheters_min_x_distance_is_ok and not unsolvable_issue:
                unsolvable_issue, catheters_min_x_distance_is_ok, labels, new_grps, problematic_pts = self.fix_catheters_too_close_in_x(
                    new_grps, labels, points)
                # If the fix did not solve the problem, we need to check if the
                # problematic points are not the same to know if we are not doing
                # the same things over and over again. 
                if unsolvable_issue:
                    assert not catheters_min_x_distance_is_ok
                else:
                    if not catheters_min_x_distance_is_ok:
                        if problematic_pts[0] in previous_problematic_pts and problematic_pts[1] in previous_problematic_pts:
                            unsolvable_issue = True
                            print("We already tried to fix these points => Unsolvale min spread.")
                        else:
                            unsolvable_issue = False
                            print("previous_problematic_pts", previous_problematic_pts)
                        previous_problematic_pts.extend(problematic_pts)
                print("problematic_pts", problematic_pts)           
            if save_files:
                assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
                save_dir = os.path.join(self.save_details_folder, "second_rotation", "refined_custom_assignement", 
                                        f"{len(new_grps)}_rows", f"iteration_{iteration}", "closest_row_solved_x_too_close")
                save_grps_3Dslicer(new_grps, save_dir, key="second_a_")

            print("********************CHECKING DISTANCE X TOO FAR***************")
            if catheters_min_x_distance_is_ok:
                # # Now we check if catheters are not too far in x. 
                # # Mostly for last points in rows. 
                # catheters_max_x_distance_is_ok = False
                # unsolvable_issue = False
                # previous_problematic_pts = []
                # while not catheters_max_x_distance_is_ok and not unsolvable_issue:
                #     unsolvable_issue, catheters_max_x_distance_is_ok, labels, new_grps, problematic_pts = self.fix_catheters_too_far_in_x(
                #         new_grps, labels, points)
                #     # If the fix did not solve the problem, we need to check if the
                #     # problematic points are not the same to know if we are not doing
                #     # the same things over and over again. 
                #     if unsolvable_issue:
                #         assert not catheters_max_x_distance_is_ok
                #     else:
                #         if not catheters_max_x_distance_is_ok: # Now we check if catheters are not too far in x. 
                # Mostly for last points in rows. 
                catheters_max_x_distance_is_ok = False
                unsolvable_issue = False
                previous_problematic_pts = []
                while not catheters_max_x_distance_is_ok and not unsolvable_issue:
                    unsolvable_issue, catheters_max_x_distance_is_ok, labels, new_grps, problematic_pts = self.fix_catheters_too_far_in_x(
                        new_grps, labels, points)
                    # If the fix did not solve the problem, we need to check if the
                    # problematic points are not the same to know if we are not doing
                    # the same things over and over again. 
                    if unsolvable_issue:
                        assert not catheters_max_x_distance_is_ok
                    else:
                        if not catheters_max_x_distance_is_ok:
                            if problematic_pts[0] in previous_problematic_pts and problematic_pts[1] in previous_problematic_pts:
                                unsolvable_issue = True
                                print("We already tried to fix these points => Unsolvale max spread.")
                            else:
                                unsolvable_issue = False
                                print("previous_problematic_pts", previous_problematic_pts)
                            previous_problematic_pts.extend(problematic_pts)
                        else:
                            print("catheters_max_x_distance_is_ok", catheters_max_x_distance_is_ok)
                    print("problematic_pts", problematic_pts)

                if save_files:
                    assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
                    save_dir = os.path.join(self.save_details_folder, "second_rotation", "refined_custom_assignement", 
                                            f"{len(new_grps)}_rows", f"iteration_{iteration}", "closest_row_solved_x_too_close_and_far")
                    save_grps_3Dslicer(new_grps, save_dir, key="second_a_")

                #             if problematic_pts[0] in previous_problematic_pts and problematic_pts[1] in previous_problematic_pts:
                #                 unsolvable_issue = True
                #                 print("We already tried to fix these points => Unsolvale max spread.")
                #             else:
                #                 unsolvable_issue = False
                #                 print("previous_problematic_pts", previous_problematic_pts)
                #             previous_problematic_pts.extend(problematic_pts)

                #     print("problematic_pts", problematic_pts)
                # if save_files:
                #     assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
                #     save_dir = os.path.join(self.save_details_folder, "second_rotation", "refined_custom_assignement", 
                #                             f"{len(new_grps)}_rows", f"iteration_{iteration}", "closest_row_solved_x_too_close_and_far")
                #     save_grps_3Dslicer(new_grps, save_dir, key="second_a_")
                print("********************CHECKING IF GROUPS ARE EVOLVING***************")
                groups_created_are_evolving = False
                for prev_grp, grp in zip(previous_grps, new_grps):
                    if len(prev_grp) != len(grp):
                        groups_created_are_evolving = True
                        break
                    for pt in grp:
                        if pt not in prev_grp:
                            groups_created_are_evolving = True
                            break
                previous_grps = copy.deepcopy(new_grps)
                print("Groups are evolving: ", groups_created_are_evolving)
            else:
                print("Unsolvable, finishing the loop")
                assert unsolvable_issue
    
        cluster_proposal_respect_conditions = catheters_min_x_distance_is_ok # and catheters_max_x_distance_is_ok
        print("Cluster proposal respects conditions: ", cluster_proposal_respect_conditions)
        return cluster_proposal_respect_conditions, labels, new_grps
    
    def regroup_pts_to_closest_row_direction(
            self, points:List[List[float]], labels:List[int], grps:List[List[List[float]]], save_files:bool=False, 
            iteration:int=0):   
        """
        Reassign the points to the closest row direction.
        This is done by projecting the points onto the lines fitted to the rows
        and checking if the distance is smaller than the distance to the other
        lines. If it is, we reassign the point to the closest line.
        This might lead to wrong assignement which will be corrected later on by 
        fix_catheters_too_close_in_x() and fix_catheters_too_far_in_x() functions.
        """

        all_pts_are_closer_to_their_row = False
        while not all_pts_are_closer_to_their_row:
            all_pts_are_closer_to_their_row = True
            for pt, lab_old in zip(points, labels):
                ## Checking if the pt is closer to the line of another row
                # Projecting the point onto each line and check distance
                distances = []
                for grp in grps:
                    m, dir = fit_line(grp)
                    # Force the direction to be from the point with higher x to point with
                    # lower x in the group. 
                    xs = np.array([pt[0] for pt in grp])
                    arg_max_x = np.argmax(xs, axis=0)
                    arg_min_x = np.argmin(xs, axis=0)

                    if np.dot(dir, np.array(grp[arg_max_x]) - np.array(grp[arg_min_x])) < 0:
                        dir = -dir
                    projected_point = project_point_to_line(pt, m, dir, return3D=True).tolist()
                    distances.append(distance(projected_point, pt))
                # Find the closest line
                closest_line_idx = int(np.argmin(distances))
                # If the closest line is not the current line, update the label
                if closest_line_idx != lab_old:
                    labels[points.index(pt)] = closest_line_idx
                    print(f"Point {pt} changed from label {lab_old} to {closest_line_idx}")
                    all_pts_are_closer_to_their_row = False
                    grps = create_group_from_labels(points, labels)
                    break

        if save_files:
            assert self.save_details_folder is not None, "save_details_folder must be set to save the points."
            save_dir = os.path.join(self.save_details_folder, "second_rotation", "refined_custom_assignement", 
                                    f"{len(grps)}_rows", f"iteration_{iteration}", "closest_row")
            grps = create_group_from_labels(points, labels)
            save_grps_3Dslicer(grps, save_dir, key="second_a_")

        return labels, grps

    @staticmethod
    def fit_line_to_rows( grps:List[List[List[float]]]):
        """
        Fit a line to the rows.
        """
        means = []
        directions = []
        for grp in grps:
            mean, direction = fit_line(grp)
            means.append(mean)
            directions.append(direction)
        return means, directions

    def fix_catheters_too_close_in_x(
        self, grps:List[List[List[float]]], labels:List[int], points:List[List[float]],
        print_details:bool=True):
        """
        Fix the wrong row assignement based on the x coordinates.
        The idea is to check if the distance between two points x coord in a row 
        is too small and if so, we can say that they should not be in the same row.
        We then assign the best row based on distance to the projected point onto the 
        new potential row.
        """
        spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_x_min_spread(
            grps, print_details=print_details)
        
        if not spread_of_catheter_x_is_ok:
            ## Figuring out which point is in the wrong row
            new_grps = copy.deepcopy(grps)

            # Find all the points that could be impacted by the problematic points
            # We take all the points in the column of the problematic points and
            # make a square of 1cm around it (the size of the insertion grid 
            # squares).
            xs = np.array(points)[:, 0]
            problematic_pts_x = np.array(problematic_pts)[:, 0]
            side_extension = (10 - (
                np.max(problematic_pts_x) - np.min(problematic_pts_x))) / 2
            problematic_pts_column_min_x = np.min(problematic_pts_x) - side_extension
            problematic_pts_column_max_x = np.max(problematic_pts_x) + side_extension

            all_column_concerned_pt_idxs = np.where(
                (xs > problematic_pts_column_min_x) &
                (xs < problematic_pts_column_max_x)
            )[0]
            points_column_concerned = [
                points[i] for i in all_column_concerned_pt_idxs
            ]
            print("points_column_concerned", points_column_concerned)
            print("len(points_column_concerned)'", len(points_column_concerned))
            print("len(grps)", len(grps))
            if len(points_column_concerned) > len(grps):
                # We have more points in the column than rows, we cannot assign
                # the points to another row.
                print("We have more points in the column than rows, we cannot assign")
                print("the points to another row.")
                return True, spread_of_catheter_x_is_ok, labels, new_grps, problematic_pts
            else:
                # We assign each point to the closest row fitted line by minimizing
                # the sum of the distances from each point to its repsective assigned
                # line.

                # We first project the points to each row fitted line, then we find the 
                # best assignement.
                means, directions = self.fit_line_to_rows(grps)
             
                # Creating the cost/distance from point to projected point matrix
                distance_matrix = np.empty((len(points_column_concerned), len(grps)))
                for i in range(len(grps)):
                    mean, direction = means[i], directions[i]
                    projected_points = project_point_to_line(
                        points_column_concerned, mean, direction, return3D=True)
                    # projected_points = np.array(projected_points).tolist()
                    print("projected_points", projected_points)
                    for j, pt in enumerate(points_column_concerned):
                        distance_matrix[j, i] = distance(pt, projected_points[j])
                print("distance_matrix", distance_matrix)
                print("shape", distance_matrix.shape)

                # Each entry is the distance from point i to line j

                # Pad the matrix to make it square by adding a dummy row with large values
                # This way we allow one line to remain unassigned
                cost_matrix = distance_matrix.copy()
                # cost_matrix = np.vstack([cost_matrix, [1e6]*cost_matrix.shape[1]])

                # Apply the assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Filter out the dummy row assignments (i.e., row >= len(grps))
                assignments = [(i, j) for i, j in zip(row_ind, col_ind) if i < len(grps)]

                # Show the results
                for point_idx, new_row_idx in assignments:
                    print(f"Assign Point {points_column_concerned[point_idx]} from row"\
                          f"{labels[points.index(points_column_concerned[point_idx])]} "\
                            f"to row {new_row_idx}, Distance = {distance_matrix[point_idx, new_row_idx]:.2f}")  
                    labels[points.index(points_column_concerned[point_idx])] = int(new_row_idx)
                new_grps = create_group_from_labels(points, labels)
                spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_x_min_spread(
                new_grps, print_details=print_details)
                print("yooo")
                return False, spread_of_catheter_x_is_ok, labels, new_grps, problematic_pts
            
        else:
            new_grps = copy.deepcopy(grps)
            return False, spread_of_catheter_x_is_ok, labels, new_grps, problematic_pts

    def fix_catheters_too_far_in_x(
            self, grps:List[List[List[float]]], labels:List[int], points:List[List[float]],
            print_details:bool=True):
        """
        Fix the wrong row assignement based on the x coordinates.
        If a catheter belongs to a column but is not close too any other catheter of the row, 
        we try to assign it to another row. If that assignement results in ok spread of 
        catheters in x, we keep it. If not, the grid is not okay.
        """
        spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_x_max_spread(
            grps, print_details=print_details)
        if not spread_of_catheter_x_is_ok:
            # If the problematic point is the first or last point of the row, 
            # we test moving it. If not, then we cannot pick which of the two 
            # points to move, or choose a point to add since the min spread has 
            # been investigated already previously => assingement is not good.
            row_of_interest = grps[problematic_grp_idx]
            print("row_of_interest", row_of_interest)
            print("problematic_pts", problematic_pts)
            if row_of_interest[0] in problematic_pts or row_of_interest[-1] in problematic_pts:
                # Problem is neccasarily in the first or last point of the row.
                if row_of_interest[0] in problematic_pts:
                    problematic_pt = row_of_interest[0]
                else:
                    problematic_pt = row_of_interest[-1]


                # Find best fit for this point.
                # First check which row could receive this new point without compromising 
                # the spread between catheters in x.
                # Find the closest x in each row.
                x_prob_pt = problematic_pt[0]
                distances_in_x = {}
                for row_idx, grp in enumerate(grps):
                    if row_idx == problematic_grp_idx:
                        continue
                    row_xs = np.array([pt[0] for pt in grp])
                    print("row_xs", row_xs, "for row", row_idx)
                    print("problematic_pt", problematic_pt)
                    distances_to_row_xs = np.abs(row_xs - x_prob_pt)
                    print("distances_to_row_xs", distances_to_row_xs, "for row", row_idx)
                    dst_of_interest = np.min(distances_to_row_xs)
                    if dst_of_interest > self.min_dist_x_catheter and dst_of_interest < self.max_dist_x_catheter:
                        m_d = np.argmin(row_xs)
                        distances_in_x[row_idx] = distance(grp[m_d], problematic_pt)

                print("Distances between rows and problematic point", distances_in_x)
                # If we have no row that can receive the point, we cannot move it.
                if len(distances_in_x) == 0:
                    return True, spread_of_catheter_x_is_ok, labels, grps, problematic_pts
                else:
                    # We have at least one row that can receive the point.
                    # We assign the point to the closest row.
                    idx_best_row = int(np.argmin(list(distances_in_x.values())))
                    new_row_idx = int(list(distances_in_x.keys())[idx_best_row])
                    if print_details:
                        print("distances_in_x", distances_in_x)
                        print("idx_best_row", idx_best_row)
                        print("new_row_idx", new_row_idx)
                        print(f"Point {problematic_pt} changed from label {labels[points.index(problematic_pt)]} to {new_row_idx}")
                    labels[points.index(problematic_pt)] = new_row_idx
                    new_grps = create_group_from_labels(points, labels)
                    spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_x_max_spread(
                        new_grps, print_details=print_details)
                    return False, spread_of_catheter_x_is_ok, labels, new_grps, problematic_pts
            else:
                # We cannot move the point since it is not the first or last point
                # of the row.
                return True, spread_of_catheter_x_is_ok, labels, grps, problematic_pts
        else:
            new_grps = copy.deepcopy(grps)
            return False, spread_of_catheter_x_is_ok, labels, new_grps, problematic_pts
        
    def check_spread(self, grps:List[List[List[float]]], comparison:str="greater", 
                  axe:int=0, threshold:float=5.0, print_details:bool=False ):
        """
        Check if the distance in x between two adjacent points in a row is
        greater/lower than a threshold.
        """
        assert comparison in ["greater", "lower"], (
            "Condition should be either greater or lower."
        )
        assert axe in [0, 1], (
            "Axe should be either 0 or 1."
        )
        if axe == 0:
            ax_name = "x"
        else:
            ax_name = "y"
        spread_of_catheter_is_ok = True
        problematic_grp_idx = None
        problematic_pts = []
        for grp_idx, grp in enumerate(grps):
            ax_coord_grp = np.array([pt[axe] for pt in grp])
            ax_order = np.argsort(ax_coord_grp)
            sorted_points_ax = np.array([ax_coord_grp[i] for i in ax_order])
            diff_ax = np.diff(sorted_points_ax)
            # Check the distance between every two adjacent points 
            # in the row.
            print("COndition in checking spread", comparison)
            print("threshold in checking spread", threshold)
            if comparison == "greater":
                bools = diff_ax > threshold
                mess = "too far"
            else:
                bools = diff_ax < threshold
                mess = "too close"
            condition = np.any(bools)
            if condition:
                if print_details:
                    print(f"Potential missassignement since catheters are {mess} in {ax_name} within a row.")
                    print("sorted_points_ax", sorted_points_ax)
                    print("diff sorted_points_ax", diff_ax)
                idx_error_pt = int(np.where(bools)[0][0])
                if print_details:
                    print("Catheter {} and {} from row {} are {}.".format(
                        sorted_points_ax[idx_error_pt], sorted_points_ax[idx_error_pt+1], grp_idx, mess))
                problematic_pts.extend(
                    [grp[ax_order[int(idx_error_pt)]], grp[ax_order[int(idx_error_pt)+1]]])
                spread_of_catheter_is_ok = False
                problematic_grp_idx = grp_idx
                break
        return spread_of_catheter_is_ok, problematic_grp_idx, problematic_pts

    def check_x_min_spread(self, grps:List[List[List[float]]], print_details:bool=False):
        spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_spread(
            grps, comparison="lower", axe=0, threshold=self.min_dist_x_catheter, 
            print_details=print_details)
        return spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts

    def check_x_max_spread(self, grps:List[List[List[float]]], print_details:bool=False):   
        spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts = self.check_spread(
            grps, comparison="greater", threshold=self.max_dist_x_catheter, print_details=print_details)
        return spread_of_catheter_x_is_ok, problematic_grp_idx, problematic_pts

    def check_y_max_spread(self, grps:List[List[List[float]]], print_details:bool=False):   
        spread_of_catheter_y_is_ok, problematic_grp_idx, problematic_pts = self.check_spread(
            grps, comparison="greater", axe=1, threshold=self.max_dist_y_catheter, 
            print_details=print_details)
        return spread_of_catheter_y_is_ok, problematic_grp_idx, problematic_pts

    def check_grid_conditions(self, print_details:bool=False):
        group_lengths = [len(g) for g in self.grps]
        if print_details:
            print("Group lengths: ", group_lengths)

        ##########################################################
        ### CONDITIONS on the created rows
        ### Condition number of catheters
        ## One row should not have 4 catheters more than another row 
        condition_group_lengths_diff = True
        if np.max(group_lengths) - np.min(group_lengths) > 3:
            if print_details:
                print("Rows containing different number of catheters {}.".format(group_lengths))
                print("Skipping potential number of rows: {}.".format(self.n_rows))
            condition_group_lengths_diff = False
        ## One row should not be composed of only one catheter
        condition_no_1catheter_grp = True
        if np.min(group_lengths) == 1:
            if print_details:
                print("Rows containing only one catheter {}.".format(group_lengths))
                print("Skipping potential number of rows: {}.".format(self.n_rows))
            condition_no_1catheter_grp = False
        ### Condition on catheter insertion. Catheter insertion is done
        # through a grid that is glued to the breast, ensuring catheters
        # implemented are cloe to each other.
        ## Two rows should not be within 5 mm of each other
        condition_short_spread_catheter_x, _, _ = self.check_x_min_spread(self.grps, print_details=print_details)
        if not condition_short_spread_catheter_x:
            if print_details:
                print("Rows containing catheter too close in x {}.".format(group_lengths))
                print("Skipping potential number of rows: {}.".format(self.n_rows))
        condition_rows_too_close = True
        avg_y_per_group = [np.mean([pt[1] for pt in g]) for g in self.grps]
        # If two rows mean y coordinates are too close, we can say that they are
        # actually the same row.
        if np.any(np.diff(np.sort(avg_y_per_group))< 5):
            if print_details:
                print("Rows containing very close mean y {}.".format(avg_y_per_group))
                print("Skipping potential number of rows: {}.".format(self.n_rows))
                print("With diff y per group: ", np.diff(np.sort(avg_y_per_group)))
            condition_rows_too_close = False
        ## Within one row, two catheter xs should not be furhter than 2 cm. 
        # Ordering the points by x coordinate within a row
        # condition_large_spread_catheter_x = True
        # for row_idx, pot_row in enumerate(self.grps):
        #     xs_order = np.argsort([pt[0] for pt in pot_row])
        #     sorted_points_x = [pot_row[i] for i in xs_order]
        #     diff_xs = np.diff([pt[0] for pt in sorted_points_x])
        #     print("Row {}: diff_xs {}".format(row_idx, diff_xs))
        #     if np.any(diff_xs > 20):
        #         condition_large_spread_catheter_x = False
        #         if print_details:
        #             print("Catheters in row {} are too far apart {}.".format(
        #                 row_idx, diff_xs))
        #             print("Skipping potential number of rows: {}.".format(self.n_rows))
            # print("diffs only x ", )
            # 
            # for i in range(len(sorted_points_x)-1):
            #     dist = distance(sorted_points_x[i], sorted_points_x[i+1])
            #     if dist > 25:
            #         if print_details:
            #             print("Points within a row are too far apart {}.".format(dist))
            #             print("Distance between catheter {} and {} from row {} : {}".format(
            #                 i, i+1, row_idx, dist))
            #             print("Skipping potential number of rows: {}.".format(self.n_rows))
            #         condition_large_spread_catheter_x = False
        ##########################################################
        return condition_group_lengths_diff and condition_no_1catheter_grp and \
            condition_short_spread_catheter_x and condition_rows_too_close # and \
            # condition_large_spread_catheter_x

if __name__ == "__main__":
    
    import time
    import pickle
    # from ai_assisted_brachy.catheter.evaluation.evaluate_dwell_positions import DigitizationResults
    from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import CatheterSetUp
    from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.creator import CatheterContourCreator
    from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.contour_digitizer import DwellPositionCreator


    home = "/home/sebq"

    #### Trying to find pattern in assignment cost of patients that are wrongly labelled.
    for patient_id in os.listdir(f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/"):
        patient_temp_folder = f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/{patient_id}"
        assignment_cost_path = os.path.join(patient_temp_folder, "user_input_catheter_assignement/all_assignment_costs.json")

        with open(assignment_cost_path, "r") as f:
            all_assignment_costs = json.load(f)
            all_costs = list(all_assignment_costs["assign_to_cost"].values())
        max_cost = np.max(all_costs)
        diag = np.sqrt(all_assignment_costs["y_step"]**2 + all_assignment_costs["x_step"]**2)

        thr = max(all_assignment_costs["x_step"], all_assignment_costs["y_step"]) if all_assignment_costs["y_step"]!= 0 else all_assignment_costs["x_step"]
        thr = min(all_assignment_costs["x_step"], all_assignment_costs["y_step"]) if all_assignment_costs["y_step"]!= 0 else all_assignment_costs["x_step"]
        # if max_cost > thr:
        #     print("patient ", patient_id, "max cost: ", np.round(max_cost, 2), "diag: ", np.round(diag, 2), "thr: ", thr)

        
        q1 = np.percentile(all_costs, 25)
        q3 = np.percentile(all_costs, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if np.any(np.array(all_costs) > upper_bound):
            print("patient ", patient_id, "max cost: ", np.round(max_cost, 2), "diag: ", np.round(diag, 2), "thr: ", thr, "upper_bound: ", np.round(upper_bound, 2))
        
    exit()  
    
    nnunet_raw_path = f"{home}/EngerLab/AI_Assisted_Brachytherapy/nnUNet_raw/"
    dataset_name = "Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1"

    patient_ids = [
        img_file_name.split("_")[1] for img_file_name in os.listdir(
            os.path.join(nnunet_raw_path, dataset_name, "imagesTr"))
            ]
    for patient_nb in patient_ids:
        print(f"******************************* Processing patient {patient_nb} *******************************")
        # Patient 250333 does not work with custom assignement
        # 657 catheters per row instead of 666
        # if not "424067" in patient_nb: # 52193 208782 1466013
        #     continue
        patient_path = f"{home}/EngerLab/Data/export_seb/patients/{patient_nb}"

        # We want to develop this algorithm on the training set and not touch the test set.
    
        sitk_needles_contour_path = os.path.join(nnunet_raw_path, dataset_name, "labelsTr", f"case_{patient_nb}.nrrd")

        created_needle_dictpath = f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/{patient_nb}/created_needle_dict_{patient_nb}.pkl"
        create = not os.path.exists(created_needle_dictpath)
        patient_plan = CatheterSetUp(patient_path, setup=True)
        user_input = patient_plan.get_nb_catheters_per_row()
        patient_plan.save_digitization_points(
            f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/{patient_nb}"
        )
        patient_plan.save_dwell_positions(
            f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/{patient_nb}"
        )
        save_details_folder = f"{home}/EngerLab/AI_Assisted_Brachytherapy/temp/{patient_nb}"
        if create:
            dwell_pos_creator = DwellPositionCreator(
                sitk_needles_contour_path,
                fit_function="spline",
                # CatheterEvaluator takes consistent tip at the most distal part
                # of tip marker now => tip_distal always True.
                tip_distal=True
            )
            created_needle_dict, solo_components = dwell_pos_creator.create_points_from_contours(
                patient_plan.get_step_size(), for_viz=True, save_labelling_details_folder=save_details_folder+"_dp_creation", 
                nb_catheters_per_row_for_labelling=user_input,
                ct_vol_path_for_breast_side_check=sitk_needles_contour_path.replace(
                    "labelsTr", "imagesTr").replace(f"case_{patient_nb}.nrrd", f"case_{patient_nb}_0000.nrrd")
            )
            with open(created_needle_dictpath, "wb") as f:
                pickle.dump(created_needle_dict, f)
        else:
            assert os.path.exists(created_needle_dictpath), "The created needle dict path does not exist."
            with open(created_needle_dictpath, "rb") as f:
                created_needle_dict = pickle.load(f)

        contour_path = os.path.join(nnunet_raw_path, dataset_name, "labelsTr", f"case_{patient_nb}.nrrd")
        ct_volume_path = os.path.join(nnunet_raw_path, dataset_name, "imagesTr", f"case_{patient_nb}_0000.nrrd")
        
        clinical_dwell_positions = patient_plan.dwell_positions

        sitk.WriteImage(
            sitk.ReadImage(ct_volume_path), 
            os.path.join(save_details_folder, "ct.nrrd")
        )
        # print("In InsertionGridViewer..")
        # t0 = time.time()
        # grid_viewer = InsertionGridViewer(
        #     ct_volume_path=ct_volume_path,
        #     catheters_contour_path=contour_path,
        #     save_details_folder=save_details_folder, 
        #     dwell_positions=clinical_dwell_positions,
        #     )
        # print("In get_insertion_grid_as_rows()...")
        # grid_viewer.get_insertion_grid_as_rows(
        #     save_files=True
        #     )
        # t1 = time.time()
        # print("Time to get insertion grid as rows: ", t1-t0)
        # exit()

        # identificator = CatheterIdentificator(
        #     ct_volume_path=ct_volume_path,
        #     catheters_contour_path=contour_path,
        #     save_details_folder=save_details_folder
        #     )
        # new_d, identified_n_rows = identificator.order_catheter_by_JGH_breast_convention(
        #     created_needle_dict, user_input=user_input, save_files=True)
        # identificator.save_dwell_positions_by_channel(new_d, dwell_pos_name="analytical_contour")
        # identificator.apply_transforms_and_save_pts(clinical_dwell_positions)

        # ### Evaluate channel labelling positions
        # digi_results = DigitizationResults(
        #     os.path.join(save_details_folder, "channel_labelling_results"), 
        #     None, 
        #     experiments=["analytical_contour"], 
        #     load_model=False
        #     )
        # prepared_d = {
        #     "analytical_contour": new_d
        # }

        # digi_results.evaluate_one_patient(
        #     patient_path=patient_plan, create=False, path_to_created_catheters=None, created_catheters=prepared_d
        #     )
        # res = digi_results.per_patient_channel_labelling_results
        # print("res", res)
        # if res['analytical_contour'][patient_nb]['correct_percent'] != 1.0:
        #     print("***************************************************")
        #     print("===================================================")
        #     print("CHANNELS ARE NOT LABELLED CORRECTLY for patient ", patient_nb)
        #     print("===================================================")
        #     print("***************************************************")
        #     with open(os.path.join(Path(__file__).parents[0], "patient_wrong_labelization.txt" ), "a") as myfile:
        #         myfile.write(f"****************************** {patient_nb} ******************************\n")
        #         myfile.write(f"The catheter labelling is not correct for patient {patient_nb}.\n") 
        #         myfile.write(f"{identified_n_rows} rows have been identified by our algo.\n")
        #         myfile.write(f"{res['analytical_contour'][patient_nb]['correctnumber']} correct for {res['analytical_contour'][patient_nb]['wrongnumber']} wrong assingement. \n \n")
