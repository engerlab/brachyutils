import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage.draw import line_nd
import torchio as tio

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import (
    find_extremal_points_a,
    fit_line,
    project_point_to_line,
    distance,
)

"""
From 3DSlicer, catheter pixel values are between 250 and 900 (we catheter is in the axial plane, it is between 300 and 900) 
it goes down to 0 when the catheter fades because it enters another slice.
The black part at the tip HU is between 0 t0 -700.


Make the creation in isometric space and then convert to original spacing

NN interpolation for masks!

get only the CT with a Code that check for modailty in metadata.csv

"""


class NeedleFalsifier(object):

    def __init__(
        self,
        tip_coord_pt,
        end_coord_pt,
        volume_shape=None,
        volume_pos=None,
        voxel_spacing=[1, 1, 1],
        sitk_volume=None,
        dilation_nb_times=1,
        distance_to_non_marker_region=1,
        len_non_marker_region=1,
        demo=False,
    ):
        """_summary_

        :param coord_pt1: 1st point describing the straight line, coordinates in medical image space
        :type coord_pt1: array of 3 floating points
        :param coord_pt2: 2nd point describing the straight line, coordinates in medical image space
        :type coord_pt2: array of 3 floating points
        :param volume_shape: shape of medical image
        :type volume_shape: array of 3 integers
        :param volume_pos: position of medical image
        :type volume_pos: array of 3 floating points
        :param voxel_spacing: voxel size in each axis in mm, defaults to [1,1,1]
        :type voxel_spacing: list of 3 floating points, optional

        """

        assert sitk_volume is not None or (
            voxel_spacing is not None
            and volume_shape is not None
            and volume_pos is not None
        ), "Either provide a sitk volume or voxel_spacing, volume_shape and volume_pos"
        if sitk_volume is not None:
            self.sitk_volume = sitk_volume
            self.voxel_spacing = np.array(sitk_volume.GetSpacing()).astype(np.float32) 
            self.volume_shape = sitk_volume.GetSize()
            self.volume_pos = np.array(sitk_volume.GetOrigin())
        else:
            self.sitk_volume = None
            self.voxel_spacing = np.array(voxel_spacing)
            self.volume_shape = volume_shape
            self.volume_pos = np.array(volume_pos)

        self.tip_coord_pt = np.array(tip_coord_pt)
        self.end_coord_pt = np.array(end_coord_pt)
        self.segment_outside_volume = False
        self.tip_coord_pt, self.end_coord_pt = self._sanity_check_segment_in_volume()

        self.volume_end = (
            self.volume_pos + np.array(self.volume_shape) * self.voxel_spacing
        )
        self.distance_to_non_marker_region = distance_to_non_marker_region
        self.len_non_marker_region = len_non_marker_region
        self.vox_tip_coord = self.vox_from_coord(self.tip_coord_pt)
        self.vox_end_coord = self.vox_from_coord(self.end_coord_pt)
        self.demo = demo
        if self.demo:
            print("voxels of interest : ", self.vox_tip_coord, self.vox_end_coord)
        self.dilation_nb_times = dilation_nb_times
        self.volume = np.zeros(self.volume_shape, dtype=np.uint8)
        if self.demo:
            print("voilume shape : ", self.volume.shape)
        


    def _sanity_check_segment_in_volume(self, precision_sampling:float=0.1):
        # Check if the segment is in the volume
        # If not, return the part of the segment that is in the volume
        # If the segment is not in the volume at all, return None
        # If the segment is in the volume, return the segment
        tip_coord_pt_temp, end_coord_pt_temp = np.copy(self.tip_coord_pt), np.copy(self.end_coord_pt)

        voxel_in_vol_tip = np.array(self.sitk_volume.TransformPhysicalPointToIndex(tip_coord_pt_temp))
        is_tip_in_volume = not(np.any(voxel_in_vol_tip < 0) or 
                                        np.any(voxel_in_vol_tip > np.array(self.sitk_volume.GetSize())-1))
        voxel_in_vol_end = np.array(self.sitk_volume.TransformPhysicalPointToIndex(end_coord_pt_temp))
        is_end_in_volume = not(np.any(voxel_in_vol_end < 0) or
                                        np.any(voxel_in_vol_end > np.array(self.sitk_volume.GetSize())-1))
        if is_end_in_volume and is_tip_in_volume:
            return self.tip_coord_pt, self.end_coord_pt
        elif not is_end_in_volume and not is_tip_in_volume:
            print("Segment is not in the volume, we will not create anything.")
            self.segment_outside_volume = True
            return self.tip_coord_pt, self.end_coord_pt
        else:
            print("=====================================")
            print("Part of the segment you provided is not in the volume (end_coord_pt")
            print("We are reducing the segment until the full segment is in the volume")
            print("=====================================")
            points_in_segment, t_samples, mean, direction = self._get_points_in_segment(precision_sampling)
            if not is_end_in_volume:
                # End is not in the volume
                # We need to find the intersection of the segment with the volume

                t_endpoint = project_point_to_line(end_coord_pt_temp, mean, direction)
                if np.isclose(t_endpoint, t_samples[-1], atol=1e-3):
                    t_samples = t_samples[::-1]

                # End point is not in the volume we move forward in the segment to find the first point that is
                # in the contour
                for t in t_samples:
                    point = mean + t * direction
                    voxel = np.array(self.vox_from_coord(point))
                    if np.any(voxel < 0) or np.any(voxel > np.array(self.sitk_volume.GetSize())-1):
                        continue
                    else:
                        end_coord_pt_temp = point
                        break
            else:
                assert not is_tip_in_volume, "This case should have been handled before"
                # Tip is not in the volume
                # We need to find the intersection of the segment with the volume
    
                t_tippoint = project_point_to_line(tip_coord_pt_temp, mean, direction)
                if np.isclose(t_tippoint, t_samples[-1], atol=1e-3):
                    t_samples = t_samples[::-1]

                # End point is not in the volume we move forward in the segment to find the first point that is
                # in the contour
                for t in t_samples:
                    point = mean + t * direction
                    voxel = np.array(self.vox_from_coord(point))
                    if np.any(voxel < 0) or np.any(voxel > np.array(self.sitk_volume.GetSize())-1):
                        continue
                    else:
                        tip_coord_pt_temp = point
                        break

            return tip_coord_pt_temp, end_coord_pt_temp           

    def vox_from_coord(self, point):
        if self.sitk_volume is not None:
            return list(self.sitk_volume.TransformPhysicalPointToIndex(point))
        else:
            x_voxel = round(
                (point[0] - self.volume_pos[0]) / self.voxel_spacing[0]
            )  # before was int
            y_voxel = round((point[1] - self.volume_pos[1]) / self.voxel_spacing[1])
            z_voxel = round((point[2] - self.volume_pos[2]) / self.voxel_spacing[2])
            return [abs(x_voxel), abs(y_voxel), abs(z_voxel)]

    def coord_from_vox(self, voxel_idx):
        if self.sitk_volume is not None:
            return self.sitk_volume.TransformIndexToPhysicalPoint(voxel_idx)
        else:
            print("coord_from_vox manual implementation NOT TESTED")
            exit()
            x_coord = self.volume_pos[0] + voxel_idx[0] * self.voxel_spacing[0]
            y_coord = self.volume_pos[1] + voxel_idx[1] * self.voxel_spacing[1]
            z_coord = self.volume_pos[2] + voxel_idx[2] * self.voxel_spacing[2]
            return [x_coord, y_coord, z_coord]

    def add_line_from_voxel_indexes(self):
        """
        DEPREACTED
        Creating the line in the array volume from skimage functions.
        Less precise than building it from coordinates and radius of the needle.
        see: add_line_from_voxel_coordinates
        """
        line_voxels = line_nd(self.vox_tip_coord, self.vox_end_coord, endpoint=True)
        self.volume[line_voxels] = 1
        self.dilate(self.dilation_nb_times)
        
    def dilate(self, dilation_nb_times:int=1):
        for _ in range(dilation_nb_times):
            if self.demo:
                print("Dilating")
            self.volume = ndimage.binary_dilation(self.volume).astype(self.volume.dtype)
            _ = self.correct_endpoint_dilation()

    def _get_points_in_segment(self, precision_sampling:float=0.1):
        # Fitting a line to endpoints coordinates
        points_to_fit_line = np.array([self.tip_coord_pt, self.end_coord_pt])
        assert points_to_fit_line.shape == (2, 3), "Need two points to fit a line"
        mean, direction = fit_line(points_to_fit_line)

        # Create points along the line
        # We go exactly from the tip to the end: need to find which t corresponds to the enpoints of the curve.
        # Project all points onto the line to find the scalar values
        t_values = [
            project_point_to_line(point, mean, direction)
            for point in points_to_fit_line
        ]   

        # Find the scalar values corresponding to the endpoints
        t_min, t_max = min(t_values), max(t_values)

        # Generate sample points within the range defined by the endpoints
        # One point every precision_sampling mm
        nb_points = int(
            distance(self.tip_coord_pt, self.end_coord_pt) / precision_sampling
        )
        t_samples = np.linspace(t_min, t_max, nb_points)
        points_coord_in_line = mean + t_samples[:, np.newaxis] * direction
        return points_coord_in_line, t_samples, mean, direction

    def add_line_from_voxel_coordinates(self, precision_sampling:float=0.1, diameter=2):
        """
        Creating the line in the array volume. Creating a line between two digitization points.
        For each point in the line, see if the distance of the line to any voxel coordinate falls
        in the radius of the needle. If yes, set the voxel to 1."""

        if  distance(self.tip_coord_pt, self.end_coord_pt) < 1 or self.segment_outside_volume:
            print("You segment lenght is null or outside the volume, there is nothing to create.")
            return
        # Get all points in the segment
        points_coord_in_line, _, _, _ = self._get_points_in_segment(precision_sampling)
        
        # Get all points coordinates of points close to the line
        # (no need to evaluate the full volume, expensive in memory)
        # We want a margin that makes sure we will find all points potnetially
        # in the radius of the needle.
        radius = diameter / 2
        # Side note: for this operation, you better have spacing as float32
        margin_x = int(np.ceil(radius / self.voxel_spacing[0]))
        margin_y = int(np.ceil(radius / self.voxel_spacing[1]))
        margin_z = int(np.ceil(radius / self.voxel_spacing[2]))

        # Create list of points to evaluate distance to.
        # We do it in a separate step since there can be duplicates,
        # and we don't want to do unnecessary distance computations.
        points_to_evaluate_indexes = np.empty((0, 3), dtype=int)
        for pt_in_l in points_coord_in_line:
            pt_vox = self.vox_from_coord(pt_in_l)
            xs_to_evaluate = np.arange(
                max(pt_vox[0] - margin_x, 0),
                min(pt_vox[0] + margin_x, self.volume_shape[0] - 1) + 1,
            )
            ys_to_evaluate = np.arange(
                max(pt_vox[1] - margin_y, 0),
                min(pt_vox[1] + margin_y, self.volume_shape[1] - 1) + 1,
            )
            zs_to_evaluate = np.arange(
                max(pt_vox[2] - margin_z, 0),
                min(pt_vox[2] + margin_z, self.volume_shape[2] - 1) + 1,
            )
            new_pt_indexes = np.array(
                np.meshgrid(xs_to_evaluate, ys_to_evaluate, zs_to_evaluate)
            ).T.reshape(-1, 3)
            # Check if new_point is already in points_to_evaluate_indexes
            for newpt in new_pt_indexes:
                if not np.any(np.all(points_to_evaluate_indexes == newpt, axis=1)):
                    # If new_point is not in the array, append it
                    points_to_evaluate_indexes = np.vstack(
                        (points_to_evaluate_indexes, newpt)
                    )

        points_to_evaluate_coords = np.array(
            [
                self.coord_from_vox([int(idx) for idx in pt])
                for pt in points_to_evaluate_indexes
            ]
        )

        # Compute distance between points to evaluate and points in the line
        distances = cdist(points_to_evaluate_coords, points_coord_in_line)

        # Determine points belonging to catheter
        pts_belonging_to_catheter = np.unique(np.where(distances < radius)[0])
        for pt_idx in pts_belonging_to_catheter:
            pt_vox = points_to_evaluate_indexes[pt_idx]
            self.volume[pt_vox[0], pt_vox[1], pt_vox[2]] = 1

        # The fact that we fill the catheter volume around a radius should affect
        # only the shaft and not the endpoints. We correct for this by removing
        # the points that would modify the endpoints of the catheter.
        if radius >= min(self.voxel_spacing):
            _ = self.correct_endpoint_dilation()

    def get_endpoint_line(self):
        points = np.argwhere(self.volume == 1)
        positions = np.array(
            # TransformIndexToPhysicalPoint only takes int
            [self.coord_from_vox([int(idx) for idx in pt]) for pt in points]
        )
        endpoints, max_dist = find_extremal_points_a(positions)
        endpoints = endpoints[0]
        endpoints_coord = [np.array(self.vox_from_coord(pt)) for pt in endpoints]
        return endpoints_coord, points, positions

    def correct_endpoint_dilation(self, remove_similar_distance_dilated_pt=True):

        # Getting the new endpoints from dilated line
        endpoints_coord, points, positions = self.get_endpoint_line()
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
                    self.volume[endpt[0], endpt[1], endpt[2]] = 0
                    for pt_idx, pt in enumerate(points):
                        if np.all(pt == endpt):
                            break

                    points = np.delete(points, pt_idx, axis=0)
                    positions = np.delete(positions, pt_idx, axis=0)

            endpoints, max_dist = find_extremal_points_a(positions)
            # If two points are at the same distance from an enpoint
            # find_etremal pioint function can return many pairs.
            # We take the first one since all are going to be processed
            # until we find the endpoints coord.
            endpoints = endpoints[0]
            assert len(endpoints) == 2, f"There should be two endpoints but you only have {endpoints}"
            endpoints_coord = [np.array(self.vox_from_coord(pt)) for pt in endpoints]

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
                            self.volume[pt[0], pt[1], pt[2]] = 0
                            break
                    endpoints_coord = np.delete(endpoints_coord, pt_idx, axis=0)

        return endpoints_coord

    def line_equation(self, t):
        # trying to code from https://www.geeksforgeeks.org/equation-of-a-line-in-3d/
        return (self.end_coord_pt - self.tip_coord_pt) - t * self.tip_coord_pt

    def create_distance_from_tip(self):
        """
        Efficiently computes a volume of distances to the dwell position matching the input volume shape, position and spacing
        Inputs :
            shape : shape of the volume we want top compute the distance in
            spacing : spacing of x y and z axes of the volume we want to compute distance in (x,y and z TPS axis)
            img_pos : coordinates of the center of the corner voxels of this same volume ( minimum of x y and z coordinates + half spacing)
        Returns :
            Volume of distances to the sepcified dwell positions computed with distance = square root of ( (x-x1)**2 + (y-y1)**2 + (z-z1)**2 )
        """
        # x, y, z = np.meshgrid(np.arange(img_pos[0],img_pos[0]+shape[0]*spacings[0],spacings[0], dtype=np.float32),
        #                         np.arange(img_pos[1],img_pos[1]+shape[1]*spacings[1],spacings[1], dtype=np.float32),
        #                         np.arange(img_pos[2],img_pos[2]+shape[2]*spacings[2],spacings[2], dtype=np.float32),
        #                         sparse=True, indexing='ij') # get some error on the edges ==> linspace
        x, y, z = np.meshgrid(
            np.linspace(
                self.volume_pos[0],
                self.volume_end[0],
                num=self.volume_shape[0],
                endpoint=True,
                dtype=np.float32,
            ),
            np.linspace(
                self.volume_pos[1],
                self.volume_end[1],
                num=self.volume_shape[1],
                endpoint=True,
                dtype=np.float32,
            ),
            np.linspace(
                self.volume_pos[2],
                self.volume_end[2],
                num=self.volume_shape[2],
                endpoint=True,
                dtype=np.float32,
            ),
            sparse=True,
            indexing="ij",
        )

        # x, y, z = np.ogrid[ img_pos[0]:img_pos[0]+ shape[0]*spacings[0]:spacings[0],img_pos[1]:img_pos[1]+shape[1]*spacings[1]:spacings[1], img_pos[2]:img_pos[2]+shape[2]*spacings[2]:spacings[2]]
        distances = np.sqrt(
            (x - self.tip_coord_pt[0]) ** 2
            + (y - self.tip_coord_pt[1]) ** 2
            + (z - self.tip_coord_pt[2]) ** 2
        )

        return distances

    def create_no_marker_region(self):
        new_point = self.line_equation(1)
        if self.demo:
            print(" new point i created ", new_point)
        self.dist_from_tip = self.create_distance_from_tip()

        distance_masked_vol = self.volume * self.dist_from_tip
        max_dist = np.sqrt(
            (self.end_coord_pt[0] - self.tip_coord_pt[0]) ** 2
            + (self.end_coord_pt[1] - self.tip_coord_pt[1]) ** 2
            + (self.end_coord_pt[2] - self.tip_coord_pt[2]) ** 2
        )
        if self.demo:
            self.plot(
                self.dist_from_tip, vmax=max_dist, title="Distance from needle tip"
            )
            self.plot(distance_masked_vol, vmax=max_dist, title="Masked distance")
        self.non_CT_marked_needle_mask = (
            distance_masked_vol >= self.distance_to_non_marker_region
        ) & (
            distance_masked_vol
            <= self.len_non_marker_region + self.distance_to_non_marker_region
        )
        self.CT_marked_needle_mask = self.volume - self.non_CT_marked_needle_mask
        if self.demo:
            self.plot(
                self.non_CT_marked_needle_mask, vmax=1, title="non CT marked needle"
            )
            self.plot(self.CT_marked_needle_mask, vmax=1, title="CT marked needle")

    def assign_values(self, range_CT_marker, range_non_CT_marker):

        def get_uniform_params_from_range(rang):
            """_summary_

            :param rang: [min_val, max_val] that I found exploring one CT volume with Slicer.
            Problem, values tend to decrease when the edges of the needle are in the image and not the middle.
            :type rang: [int, int]
            :return: mu and sigma of uniform ditribution
            :rtype: float, float
            """
            mu = (rang[0] + rang[1]) / 2  # mean
            # Here I imagine that the upper bound is 3sigmas of my normal distribution
            sigma = (rang[1] - mu) / 3  # sigma
            return mu, sigma

        mu_CT_marked, sigma_CT_marked = get_uniform_params_from_range(range_CT_marker)
        potential_values_CT_marked = np.random.default_rng().normal(
            mu_CT_marked, sigma_CT_marked, self.volume_shape
        )
        if self.demo:
            plt.hist(potential_values_CT_marked.flatten())
            plt.title("distribution of values for CT marked needle parts")
            plt.show()
        mu_non_CT_marked, sigma_non_CT_marked = get_uniform_params_from_range(
            range_non_CT_marker
        )
        potential_values_non_CT_marked = np.random.default_rng().normal(
            mu_non_CT_marked, sigma_non_CT_marked, self.volume_shape
        )
        if self.demo:
            plt.hist(potential_values_non_CT_marked.flatten())
            plt.title("distribution of values for non CT marked needle parts")
            plt.show()
        self.needle_with_real_values = (
            self.non_CT_marked_needle_mask * potential_values_non_CT_marked
            + self.CT_marked_needle_mask * potential_values_CT_marked
        )
        if self.demo:
            self.plot(
                self.needle_with_real_values,
                vmin=-1000,
                vmax=1000,
                title="Needle with real values",
            )

        filtered_values_non_CT_marked = gaussian_filter(
            potential_values_non_CT_marked, sigma=1
        )
        filtered_values_CT_marked = gaussian_filter(potential_values_CT_marked, sigma=1)
        self.needle_with_real_values2 = (
            self.non_CT_marked_needle_mask * filtered_values_non_CT_marked
            + self.CT_marked_needle_mask * filtered_values_CT_marked
        )
        if self.demo:
            plt.hist(filtered_values_non_CT_marked.flatten())
            plt.title("distribution of values for non CT marked filtered needle parts")
            plt.show()
            plt.hist(filtered_values_CT_marked.flatten())
            plt.title("distribution of values for CT marked filtered needle parts")
            plt.show()
            self.plot(
                self.needle_with_real_values2,
                vmin=-1000,
                vmax=1000,
                title="Needle Gaussian filtered with real values",
            )

    def elastic_transform(self, max_displacement=15, num_control_points=5):
        random_elastic = tio.RandomElasticDeformation(
            max_displacement=(max_displacement, max_displacement, 0),
            num_control_points=num_control_points,
        )
        tensor_like = np.expand_dims(self.needle_with_real_values, axis=0)
        self.needle_displaced = random_elastic(tensor_like)
        if self.demo:
            self.plot(
                self.needle_displaced[0],
                vmin=-1000,
                vmax=1000,
                title="Needle displaced",
            )
        # we need this smaller volume because random elastic deformation can mess up with borders
        safety_voxel = 5  # do we need safety voxels after deformation ?
        self.smaller_vol_around_needle = self.needle_displaced[0][
            max(0, self.vox_tip_coord[0] - safety_voxel) : self.vox_end_coord[0]
            + 1
            + safety_voxel,
            max(0, self.vox_tip_coord[1] - safety_voxel) : self.vox_end_coord[1]
            + 1
            + safety_voxel,
            max(0, self.vox_tip_coord[2] - safety_voxel) : self.vox_end_coord[2]
            + 1
            + safety_voxel,
        ]
        if self.demo:
            print("self.vox_tip_coord : ", self.vox_tip_coord)
            print("self.vox_end_coord : ", self.vox_end_coord)
            print("shape smaller needle volume :", self.smaller_vol_around_needle.shape)
        self.smaller_vol_final_mask = np.array(
            self.smaller_vol_around_needle != 0, dtype=bool
        )
        if self.demo:
            print("Shape my mask : ", self.smaller_vol_final_mask.shape)
            self.plot(self.smaller_vol_final_mask, vmin=0, vmax=1, title="Final mask")

    def convert_spacing(self, dest_vol):
        ndimage.zoom()
        pass

    def insert_inside_volume(self, new_vol):
        shape_mask = self.smaller_vol_final_mask.shape

        pos_not_found = True
        while pos_not_found:

            rdn_pos = [
                random.choice(range(new_vol.shape[0])),
                random.choice(range(new_vol.shape[1])),
                random.choice(range(new_vol.shape[1])),
            ]
            if (
                rdn_pos[0] + shape_mask[0] < new_vol.shape[0]
                and rdn_pos[1] + shape_mask[1] < new_vol.shape[1]
                and rdn_pos[2] + shape_mask[2] < new_vol.shape[2]
            ):
                pos_not_found = False

        if self.demo:
            print("Your catheter will be inserted here : ", rdn_pos)

        mask_right_size = np.pad(
            self.smaller_vol_final_mask,
            (
                (rdn_pos[0], new_vol.shape[0] - shape_mask[0] - rdn_pos[0]),
                (rdn_pos[1], new_vol.shape[1] - shape_mask[1] - rdn_pos[1]),
                (rdn_pos[2], new_vol.shape[2] - shape_mask[2] - rdn_pos[2]),
            ),
            constant_values=0,
        )
        needle_right_size = np.pad(
            self.smaller_vol_around_needle,
            (
                (rdn_pos[0], new_vol.shape[0] - shape_mask[0] - rdn_pos[0]),
                (rdn_pos[1], new_vol.shape[1] - shape_mask[1] - rdn_pos[1]),
                (rdn_pos[2], new_vol.shape[2] - shape_mask[2] - rdn_pos[2]),
            ),
            constant_values=0,
        )
        if self.demo:
            print("shape needle_right_size ", needle_right_size.shape)
            print("shape new_vol :", new_vol.shape)
            self.plot(
                new_vol, vmin=-1000, vmax=1000, title="Breast volume before needle"
            )
        new_vol[mask_right_size] = 0
        if self.demo:
            self.plot(
                new_vol,
                vmin=-1000,
                vmax=1000,
                title="Breast volume with room for needle",
            )
        new_vol += needle_right_size
        if self.demo:
            self.plot(
                new_vol,
                vmin=-1000,
                vmax=1000,
                title="Breast volume with needle inserted",
            )

        return new_vol, mask_right_size

    def show_needle(self):
        print("Are all my values 0 ? :", np.all(self.volume.flatten() == 0))
        print("Are any of my values 1 ? :", np.any(self.volume.flatten() == 1))
        self.plot(self.volume, vmax=1, title="Created needle")

    # def plot(self, volume, vmin=None, vmax=None, title="my plot"):
    #     fig, ax = plt.subplots()
    #     tracker = IndexTracker(fig, ax, volume, slider = True, vmin=vmin, vmax=vmax, title=title)
    #     plt.show()
    #     plt.close()

    def show_masked_distance(self):
        masked_vol = self.volume * self.dist_from_tip
        # print("What are my mask values  ? :",set(masked_vol.flatten()))

        max_dist = np.sqrt(
            (self.end_coord_pt[0] - self.tip_coord_pt[0]) ** 2
            + (self.end_coord_pt[1] - self.tip_coord_pt[1]) ** 2
            + (self.end_coord_pt[2] - self.tip_coord_pt[2]) ** 2
        )
        self.plot(masked_vol, vmax=max_dist, title="Masked distance")
        masked_vol = (masked_vol >= self.distance_to_non_marker_region) & (
            masked_vol
            <= self.len_non_marker_region + self.distance_to_non_marker_region
        )
        self.plot(masked_vol, vmax=1, title="Remaining unmarked area in the needle")


if __name__ == "__main__":

    pt1_coord = [20.3, 15.15, 0.6]
    pt2_coord = [80.13, 80.67, 0.6]

    vol_shape = [100, 100, 100]
    vol_pos = [0, 0, 0]
    spacing = [1, 1, 1]
    distance_to_non_marker_region = 4
    len_non_marker_region = 5

    my_first_fake_needle = NeedleFalsifier(
        pt1_coord,
        pt2_coord,
        vol_shape,
        vol_pos,
        voxel_spacing=spacing,
        distance_to_non_marker_region=distance_to_non_marker_region,
        len_non_marker_region=len_non_marker_region,
        demo=False,
    )
    my_first_fake_needle.add_line()
    my_first_fake_needle.create_no_marker_region()
    # my_first_fake_needle.show_needle()
    # my_first_fake_needle.show_masked_distance()

    range_val_CT_marker = [250, 900]
    range_val_non_CT_marker = [-700, 0]

    my_first_fake_needle.assign_values(range_val_CT_marker, range_val_non_CT_marker)

    my_first_fake_needle.elastic_transform(10, 6)

    # import dicom2nifti XXX: commenting out to avoid errors
    import nibabel as nib  # nibabel to handle nifti files

    my_CT_dir = "/home/sebquet/EngerLab/Patient_Data/TCIA_data/QIN_Breast/manifest-1542731172463/QIN-BREAST/QIN-BREAST-01-0001/09-01-1991-NA-BREAST PRONE-21963/2.000000-CTAC-02377"
    # dicom2nifti.convert_directory(my_CT_dir, "/home/sebquet/EngerLab/Patient_Data/TCIA_data/QIN_Breast/")

    nifti = nib.load(
        "/home/sebquet/EngerLab/Patient_Data/TCIA_data/QIN_Breast/2_ctac.nii.gz"
    )

    print(nifti)
    print(nifti.shape)
    my_breast_volume = nifti.get_fdata()
    print(nifti.header.get_zooms())
    print(nifti.GetSpacing())

    # my_input, my_mask = my_first_fake_needle.insert_inside_volume(my_breast_volume)
    # my_first_fake_needle.plot(my_mask, vmax=1, title="Final mask")
    # my_first_fake_needle.plot(my_input, vmin = -1000,vmax=1000, title="Final input")

    # line
    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    # line_aa "anti-aliased"
    # array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    #        [  0, 255,  74,   0,   0,   0,   0,   0,   0,   0],
    #        [  0,  74, 255,  74,   0,   0,   0,   0,   0,   0],
    #        [  0,   0,  74, 255,  74,   0,   0,   0,   0,   0],
    #        [  0,   0,   0,  74, 255,  74,   0,   0,   0,   0],
    #        [  0,   0,   0,   0,  74, 255,  74,   0,   0,   0],
    #        [  0,   0,   0,   0,   0,  74, 255,  74,   0,   0],
    #        [  0,   0,   0,   0,   0,   0,  74, 255,  74,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,  74, 255,   0],
    #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)

    # skimage.draw.line_nd()
    # scipy.ndimage.binary_dilation()

    # random_elastic = tio.RandomElasticDeformation(
    #     max_displacement= 2 * np.array(max_displacement),
    #     num_control_points=5,
    # )
    # slice_large_displacement = random_elastic(slice_grid)
    # slice_large_displacement.as_pil()

    # antialiased = True
    # threeD = True
    # dilation_nb_times = 1
    # # if a straight line is not enough, we could try with a spline https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline

    # if not threeD :
    #     img = (np.zeros((100, 100), dtype = np.uint8)+1)*120  # create image
    #     if antialiased :
    #         rows, cols, weights = line_aa(1, 1, 80, 80)    # antialias line
    #         img[rows, cols] = weights * 255
    #     else :
    #         rows, cols = line(1, 1, 80, 80)    # antialias line
    #         img[rows, cols] = 255
    #     for i in range(dilation_nb_times):
    #         img = ndimage.binary_dilation(img).astype(img.dtype)
    #     plt.imshow(img)
    #     plt.show()
    # else :
    #     voxel_spacings = [1, 1, 1]
    #     position_unmarked = 2.5
    #     len_unmarked = 1
    #     img = np.zeros((100, 100,100), dtype = np.uint8) # create image

    #     # trying to code from https://www.geeksforgeeks.org/equation-of-a-line-in-3d/

    #     pt1_marked_line_coord = [20, 15, 1]
    #     pt2_marked_line_coord = [80, 80, 25]
    #     marked_lin = line_nd(pt1_marked_line_coord,pt2_marked_line_coord , endpoint=False)
    #     img[marked_lin] = 255

    #     for i in range(dilation_nb_times):
    #         img = ndimage.binary_dilation(img).astype(img.dtype)
    #     fig, ax = plt.subplots()
    #     # create an IndexTracker and make sure it lives during the whole
    #     # lifetime of the figure by assigning it to a variable
    #     tracker = IndexTracker(fig, ax, img, slider=True)
    #     fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    #     plt.show()
