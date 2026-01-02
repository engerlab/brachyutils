from typing import List, Union

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ai_assisted_brachy.catheter.utils import (
    find_extremal_points_a,
    fit_line,
    distance,
    project_point_to_line,
)


def extrapolate_point(point1, point2, distance:float, reverse=False):
    """
    Extrapolates a third point on the line formed by point1 and point2.

    Args:
    point1 (np.array): The first point.
    point2 (np.array): The second point (starting point for extrapolation).
    distance (float): The distance for extrapolation .

    Returns:
    np.array: The extrapolated third point.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate the direction vector from point1 to point2
    direction = point2 - point1
    if reverse:
        direction = -direction

    # Get distances between the two points
    dist = np.linalg.norm(direction)

    # The 3rd point will be placed at a factor_t * (distance between point 1 and 2) from point2
    factor_t = distance / dist

    # Extrapolate the new point
    extrapolated_point = point2 + factor_t * direction

    return extrapolated_point

class Segment:

    # food for thought: https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python

    def __init__(
        self,
        points: List[float],
        ref_slice_coord: float,
        interslice_ax: int,
        init_line: bool = True,
        init_2D: bool = True,
    ) -> None:
        """_summary_

        Args:
            points (List[float]): 3D points that belong to the same segment.
            ref_slice_coord (float): Coord of the interslice axis, used to recreate the 3D points from 2D points.
            interslice_ax (int): We need to know which axis corresponds to the interslice axis, because
            we work with 2D points only to define new segments but need to keep the 3D info.
        """
        ### 3D info
        self.ref_slice_coord = ref_slice_coord
        self.interslice_ax = interslice_ax
        self.remaining_axes = [0, 1, 2]
        self.remaining_axes.remove(self.interslice_ax)
        points = np.array(points)
        assert points.ndim == 2
        assert points.shape[1] == 3
        if ref_slice_coord is None:
            self.ref_slice_coord = np.mean(np.unique(points[:, interslice_ax]))

        assert isinstance(points, np.ndarray)

        extremum_points, _ = find_extremal_points_a(points)
        extremum_points = extremum_points[0]
        self.extremum_points = self.order_extrem_points(extremum_points)

        if init_line:
            self.mean, self.direction = fit_line(points)
            self.points_on_curve = self.project_on_curve(points)
            self.endpoints = self.project_on_curve(self.extremum_points)
            self.range_distance = distance(self.endpoints[0], self.endpoints[1])
            self.points, self.points_on_curve = self.order_points_from_origin_a(
                self.endpoints[0], points, self.points_on_curve
            )

        if init_2D:
            ### 2D info
            self.points_2D = self.make2D(self.points)
            # Sanity check
            if ref_slice_coord is not None:
                assert np.all(
                    self.points == self.make3D_a(self.points_2D)
                ), "Cannot recreate 3D points from 2D points"
            else:
                print(
                    "WARNING: we had to interpolate the 3D points slice coordinate between different slices"
                )
            extremum_points_2D, _ = find_extremal_points_a(self.points_2D)
            extremum_points_2D = extremum_points_2D[0]
            self.extremum_points_2D = self.order_extrem_points(extremum_points_2D)
            self.mean_2D, self.direction_2D = fit_line(self.points_2D)
            self.endpoints_2D = self.project_on_curve(
                self.extremum_points_2D, two_D=True
            )
            self.range_distance_2D = distance(
                self.endpoints_2D[0], self.endpoints_2D[1]
            )
            self.points_2D_on_curve = self.project_on_curve(self.points_2D, two_D=True)
            self.points_2D, self.points_2D_on_curve = self.order_points_from_origin_a(
                self.endpoints_2D[0], self.points_2D, self.points_2D_on_curve
            )

    def get2Dpoints(self):
        return np.array([self.make2D(pt) for pt in self.points])

    def make2D(self, point3D):
        if point3D.ndim == 1:
            assert point3D.shape[0] == 3
            return point3D[self.remaining_axes]
        else:
            return point3D[:, self.remaining_axes]

    def make3D(self, point2D):
        return np.hstack(
            (
                point2D[: self.interslice_ax],
                self.ref_slice_coord,
                point2D[self.interslice_ax :],
            )
        )

    def make3D_a(self, point2D):
        return np.hstack(
            (
                point2D[:, : self.interslice_ax],
                np.repeat(self.ref_slice_coord, point2D.shape[0]).reshape(-1, 1),
                point2D[:, self.interslice_ax :],
            )
        )

    def order_extrem_points(self, points, two_D=False):
        """
        Order the extremal points of the curve
        Randomly chosing the first axis to order them. (They already share the same slice axis)
        """
        if two_D:
            if points[0][0] > points[1][0]:
                return points[::-1]
            else:
                return points
        else:
            if points[0][self.remaining_axes][0] > points[1][self.remaining_axes][0]:
                return points[::-1]
            else:
                return points

    def project_on_curve_old(self, point, two_D=False):
        """
        DEPRECATED for the array version of the function. see project_on_curve function.
        """

        def dist(pt: List[float]) -> float:
            """
            Compute the distance between a point and a curve at a given point

            Args:
                pt (List[float]): Single point 3D coordinates to feed to the curve equation

            Returns:
                float: distance
            """
            if two_D:
                point_coords = self.mean_2D + pt[:, np.newaxis] * self.direction_2D
            else:
                point_coords = self.mean + pt[:, np.newaxis] * self.direction
            return distance(np.array(point), np.array(point_coords))

        opt = minimize(dist, x0=[0.0], method="BFGS")
        if two_D:
            min_point_coords = self.mean_2D + opt.x * self.direction_2D
        else:
            min_point_coords = self.mean + opt.x * self.direction
        return min_point_coords

    def _get_t_on_curve(self, points, two_D=False):
        """
        Compute the distance between a point and a curve at a given point

        Args:
            point (List[float]): Single point 3D coordinates to feed to the curve equation

        Returns:
            float: distance
        """
        if two_D:
            mean = self.mean_2D
            direction = self.direction_2D
        else:
            mean = self.mean
            direction = self.direction
        t = project_point_to_line(points, mean, direction)
        return t
    
    def project_on_curve(self, points, two_D=False):
        
        if two_D:
            mean = self.mean_2D
            direction = self.direction_2D
        else:
            mean = self.mean
            direction = self.direction
        t = self._get_t_on_curve(points, two_D)
        # Check if t is a scalar or an array
        if t.ndim == 0:
            return mean + t * direction
        else:
            return mean + t[:, np.newaxis] * direction

    def order_points_from_origin(self, origin_point, points, two_D=False):
        """
        Order the points from the origin point
        """
        if two_D:
            return sorted(
                points,
                key=lambda x: distance(
                    self.project_on_curve(x, two_D=True), origin_point
                ),
            )
        else:
            return sorted(
                points, key=lambda x: distance(self.project_on_curve(x), origin_point)
            )

    def order_points_from_origin_a(self, origin_point, points, points_on_curve):
        """
        Order the points from the origin point
        """
        distances = cdist(origin_point[np.newaxis], points_on_curve)
        sorted_indices = np.argsort(distances)
        return points[sorted_indices].reshape(points_on_curve.shape), points_on_curve[
            sorted_indices
        ].reshape(points_on_curve.shape)


class SegmentMerger:
    def __init__(self):
        self.endpts_distance_00 = None
        self.endpts_distance_01 = None
        self.endpts_distance_10 = None
        self.endpts_distance_11 = None
        self.mega_segment = None

    def merge_segments(self, segments: List[Segment]) -> List[Segment]:
        """_summary_

        Args:
            segments (List[Segment]): Segments (one on each slice) to merge for future
            piecewise linear interpolation and placement of the dwell positions.

        Returns:
            List[Segment]: List of merged segments that can be used for the piecewise linear
            interpolation. Each segment will be a line.

        Example:
        This is how we would create segments for piecewise linear interpolation for a needle contour
        that appear on different slices. If a segment is spread over multiple slices, the line z axis
        will be defined as the center position of the slices.
        ---------
                                 |      |    |      |    |                  |      |   |
        Slice 1 :                                        o==================1
                                 |      |    |      |    |                  |      |   |
        Slice 2 :                       o==============================================1
                                 |      |    |      |    |                  |      |   |
        Slice 3 :                o==================1                              o=================1
                                 |      |    |      |    |                  |      |   |
        Slice 4 :  o=========================1
                                 |      |    |      |    |                  |      |   |
                      Segment1     Seg2  Seg3  Seg4  Seg5     Segment6        Seg7  Seg8  Seg9
        ---------
        """
        # Create a mega segment with one 2D line to order segments
        self.mega_segment = Segment(
            np.vstack([segment.points for segment in segments]),
            ref_slice_coord=None,
            interslice_ax=segments[0].interslice_ax,
        )
        # Order endpoints position of segments by their projected position on the 2D line
        # These endpoints will be break points to signal new segments
        all_endpoints_2D = []
        for seg in segments:
            all_endpoints_2D.append(seg.endpoints_2D[0])
            all_endpoints_2D.append(seg.endpoints_2D[1])

        ordered_endpoints_2D = sorted(
            all_endpoints_2D,
            key=lambda pt: distance(
                pt,
                self.mega_segment.endpoints_2D[0],
            ),
        )
        # Filter the ordered endpoints to keep only the ones that are separated
        # by a minimum distance, ie we want actual points between break points
        filtered_ordered_endpoints_2D = []
        previous_dist = -np.inf
        # Min distance:
        # Diagonal between two 1*1*1 mm voxels projected in 2D (1*1mm squares)
        min_dist_between_endpts = np.sqrt(2)
        for pt in ordered_endpoints_2D:
            current_distance = distance(
                pt,
                self.mega_segment.endpoints_2D[0],
            )
            if current_distance > previous_dist + min_dist_between_endpts:
                filtered_ordered_endpoints_2D.append(pt)
            previous_dist = current_distance
        ordered_endpoints_2D = filtered_ordered_endpoints_2D

        # Creating new segment between each breakpoint
        merged_segments = []
        # Index of next endpoint we are trying to reach
        idx_endpt_2D = 1
        merged_segment_points = []
        for pt_idx in range(len(self.mega_segment.points)):
            pt = self.mega_segment.points_2D_on_curve[pt_idx]
            limit = ordered_endpoints_2D[idx_endpt_2D]
            if idx_endpt_2D < len(ordered_endpoints_2D) - 1:
                next_pt = self.mega_segment.points_2D_on_curve[pt_idx + 1]
                distance_pt_to_limit = distance(
                    pt,
                    limit,
                )
                distance_next_pt_to_limit = distance(
                    next_pt,
                    limit,
                )
                if distance_pt_to_limit < distance_next_pt_to_limit:
                    # We are reaching a new segment
                    merged_segments.append(
                        Segment(
                            merged_segment_points,
                            ref_slice_coord=None,
                            interslice_ax=self.mega_segment.interslice_ax,
                        )
                    )
                    merged_segment_points = []
                    idx_endpt_2D += 1
                else:
                    merged_segment_points.append(self.mega_segment.points[pt_idx])
            else:
                merged_segment_points.append(self.mega_segment.points[pt_idx])
        # Add the last segment
        merged_segments.append(
            Segment(
                merged_segment_points,
                ref_slice_coord=None,
                interslice_ax=self.mega_segment.interslice_ax,
            )
        )
        return merged_segments

    @staticmethod
    def condition_inner(distance1, distance2, distance_range_ref):
        return (distance1 < distance_range_ref) and (distance2 < distance_range_ref)

    def check_overlap(self, segment1, segment2):
        """
        DEPRECATED: Please work with the merge_segments method which also works with two segments only.
        You do not need to check for the type of overlap with the merge_segments method.

        Checking if there is an overlap between different segments.
        They are necessarily parallel since they are created on different slices.
        I put the segments on two different slices to make it easier to understand
        the concept of overlap, but they are actually on the same slice/plane.

        - Case 1:
        ********No overlap********
        -----------------------------------------------------------------------------
        Segment 1:   o======================1
        Segment 2:                                o========1
        -----------------------------------------------------------------------------
        Or
        -----------------------------------------------------------------------------
        Segment 1:                                           o======================1
        Segment 2:         o=========================1
        -----------------------------------------------------------------------------

        - Case 2:
        ********Partial overlap********
        -----------------------------------------------------------------------------
        Segment 1:   o================1
        Segment 2:         o=====================1
        -----------------------------------------------------------------------------
        Or
        -----------------------------------------------------------------------------
        Segment 1:                            o================1
        Segment 2:         o==========================1
        -----------------------------------------------------------------------------

        - Case 3:
        ********Complete overlap********
        Complete
        -----------------------------------------------------------------------------
        Segment 1:             o=================1
        Segment 2:         o========================1
        -----------------------------------------------------------------------------
        Or
        Inverse Complete
        -----------------------------------------------------------------------------
        Segment 1:         o========================1
        Segment 2:             o=================1
        -----------------------------------------------------------------------------

        Args:
            endpoints (_type_): _description_
        """
        self.endpts_distance_00 = distance(
            segment1.endpoints_2D[0], segment2.endpoints_2D[0]
        )
        self.endpts_distance_01 = distance(
            segment1.endpoints_2D[0], segment2.endpoints_2D[1]
        )
        self.endpts_distance_10 = distance(
            segment1.endpoints_2D[1], segment2.endpoints_2D[0]
        )
        self.endpts_distance_11 = distance(
            segment1.endpoints_2D[1], segment2.endpoints_2D[1]
        )
        range_distance_ref = segment1.range_distance
        range_distance_to_compare = segment2.range_distance

        overlap = False
        middle_pt_ref = []
        middle_pt_compare = []
        if self.condition_inner(
            self.endpts_distance_00, self.endpts_distance_01, range_distance_to_compare
        ):
            overlap = "partial"
            middle_pt_ref.append(segment1.endpoints_2D[0])
        if self.condition_inner(
            self.endpts_distance_10, self.endpts_distance_11, range_distance_to_compare
        ):
            overlap = "partial"
            middle_pt_ref.append(segment1.endpoints_2D[1])

        if overlap is not None:
            if len(middle_pt_ref) > 1:
                overlap = "complete"
        if self.condition_inner(
            self.endpts_distance_00, self.endpts_distance_10, range_distance_ref
        ):
            middle_pt_compare.append(segment2.endpoints_2D[0])
        if self.condition_inner(
            self.endpts_distance_01, self.endpts_distance_11, range_distance_ref
        ):
            middle_pt_compare.append(segment2.endpoints_2D[1])
        if overlap is not None:
            if len(middle_pt_compare) > 1:
                overlap = "inverse_complete"

        return overlap

    def merge_two_segments(self, segment1: Segment, segment2: Segment):
        """
        DEPRECATED: Please use the merge_segments method which also works with two segments only.
        """
        new_segments = []

        overlap_type = self.check_overlap(segment1, segment2)
        print("we are dealing with a", overlap_type, "overlap")
        if overlap_type == "complete":
            # Segment 1 contained in segment 2
            pts_before_seg1 = []
            pts_seg1_and_2 = []
            pts_after_seg1 = []
            assert (
                segment2.range_distance > segment1.range_distance
            ), "segment2 should be the bigger one"
            assert (
                self.endpts_distance_00 < self.endpts_distance_10
            ), "there is no complete overlap"
            before_seg1 = True
            overlaping_area = True
            for pt_idx in range(len(segment2.points_2D)):
                if before_seg1:
                    limit_pt = segment1.endpoints_2D[0]
                else:
                    limit_pt = segment1.endpoints_2D[1]
                pt = segment2.points_2D[pt_idx]
                dist_of_interest = distance(
                    segment2.project_on_curve(pt, two_D=True), limit_pt
                )
                # print(f"distance of interest: {dist_of_interest}")
                if before_seg1 or overlaping_area:
                    next_pt = segment2.points_2D[pt_idx + 1]
                    dist_next_pt = distance(
                        segment2.project_on_curve(next_pt, two_D=True), limit_pt
                    )
                    # print(f"distance to next pt: {dist_next_pt}")

                if before_seg1:
                    pts_before_seg1.append(segment2.make3D(pt))
                    # print(f"adding pt {pt_idx} to the first segment")
                    if dist_next_pt >= dist_of_interest:
                        before_seg1 = False
                else:
                    if overlaping_area:
                        # print(f"adding pt {pt_idx} to the second segment")
                        pts_seg1_and_2.append(segment2.make3D(pt))
                        if dist_next_pt >= dist_of_interest:
                            overlaping_area = False
                    else:
                        # print(f"adding pt {pt_idx} to the last segment")
                        pts_after_seg1.append(segment2.make3D(pt))
            for pt in segment1.points_2D:
                pts_seg1_and_2.append(segment1.make3D(pt))

            if len(pts_before_seg1) > 1:
                new_segments.append(
                    Segment(
                        pts_before_seg1,
                        ref_slice_coord=segment2.ref_slice_coord,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("ty")
            if len(pts_seg1_and_2) > 1:
                new_segments.append(
                    Segment(
                        pts_seg1_and_2,
                        ref_slice_coord=None,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("y")
            if len(pts_after_seg1) > 1:
                new_segments.append(
                    Segment(
                        pts_after_seg1,
                        ref_slice_coord=segment2.ref_slice_coord,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("yo")
        elif overlap_type == "inverse_complete":
            # Segment 2 contained in segment 1
            pts_before_seg2 = []
            pts_seg1_and_2 = []
            pts_after_seg2 = []
            assert (
                segment1.range_distance > segment2.range_distance
            ), "segment1 should be the bigger one"
            assert (
                self.endpts_distance_00 < segment1.range_distance
                and self.endpts_distance_10 < segment1.range_distance
                and self.endpts_distance_01 < segment1.range_distance
                and self.endpts_distance_11 < segment1.range_distance
            ), "there is no complete overlap"
            before_seg2 = True
            overlaping_area = True
            for pt_idx in range(len(segment1.points_2D)):
                if before_seg2:
                    limit_pt = segment2.endpoints_2D[0]
                else:
                    limit_pt = segment2.endpoints_2D[1]
                pt = segment1.points_2D[pt_idx]
                dist_of_interest = distance(
                    segment1.project_on_curve(pt, two_D=True), limit_pt
                )
                # print(f"distance of interest: {dist_of_interest}")
                if before_seg2 or overlaping_area:
                    next_pt = segment1.points_2D[pt_idx + 1]
                    dist_next_pt = distance(
                        segment1.project_on_curve(next_pt, two_D=True), limit_pt
                    )
                    # print(f"distance to next pt: {dist_next_pt}")

                if before_seg2:
                    pts_before_seg2.append(segment1.make3D(pt))
                    # print(f"adding pt {pt_idx} to the first segment")
                    if dist_next_pt >= dist_of_interest:
                        before_seg2 = False
                else:
                    if overlaping_area:
                        # print(f"adding pt {pt_idx} to the second segment")
                        pts_seg1_and_2.append(segment1.make3D(pt))
                        if dist_next_pt >= dist_of_interest:
                            overlaping_area = False
                    else:
                        # print(f"adding pt {pt_idx} to the last segment")
                        pts_after_seg2.append(segment1.make3D(pt))

            for pt in segment2.points_2D:
                pts_seg1_and_2.append(segment2.make3D(pt))

            if len(pts_before_seg2) > 1:
                new_segments.append(
                    Segment(
                        pts_before_seg2,
                        ref_slice_coord=segment1.ref_slice_coord,
                        interslice_ax=segment1.interslice_ax,
                    )
                )
            else:
                print("t")
            if len(pts_seg1_and_2) > 1:
                new_segments.append(
                    Segment(
                        pts_seg1_and_2,
                        ref_slice_coord=None,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("u")
            if len(pts_after_seg2) > 1:
                new_segments.append(
                    Segment(
                        pts_after_seg2,
                        ref_slice_coord=segment1.ref_slice_coord,
                        interslice_ax=segment1.interslice_ax,
                    )
                )
            else:
                print("v")
        elif overlap_type == "partial":
            # Segment 1 and 2 partially overlap
            pts_before_overlap = []
            pts_overlap = []
            pts_after_overlap = []
            assert (
                self.endpts_distance_00 < segment1.range_distance
                or self.endpts_distance_11 < segment1.range_distance
                or self.endpts_distance_00 < segment2.range_distance
                or self.endpts_distance_11 < segment2.range_distance
            ), "there is no partial overlap"

            # Segment1 on the left
            left_overlap = (
                segment2.range_distance > self.endpts_distance_10
                and segment2.range_distance > self.endpts_distance_00
            )
            if left_overlap:
                print("left overlap")
            # Segment 1 on the right
            right_overlap = (
                segment2.range_distance > self.endpts_distance_01
                and segment2.range_distance > self.endpts_distance_00
            )
            if right_overlap:
                print("right overlap")
            assert left_overlap or right_overlap, "there is no partial overlap"
            assert not (
                left_overlap and right_overlap
            ), "there should only be one partial overlap"
            if left_overlap:
                before_seg2 = True
                for pt_idx in range(len(segment1.points_2D)):
                    pt = segment1.points_2D[pt_idx]
                    if before_seg2:
                        limit_pt = segment2.endpoints_2D[0]
                        dist_of_interest = distance(
                            segment1.project_on_curve(pt, two_D=True), limit_pt
                        )
                    else:
                        limit_pt = None

                    if before_seg2:
                        pts_before_overlap.append(segment1.make3D(pt))
                        if (
                            distance(
                                segment1.project_on_curve(
                                    segment1.points_2D[pt_idx + 1], two_D=True
                                ),
                                limit_pt,
                            )
                            >= dist_of_interest
                        ):
                            before_seg2 = False
                    else:
                        pts_overlap.append(segment1.make3D(pt))

                before_seg1 = True
                for pt_idx in range(len(segment2.points_2D)):
                    pt = segment2.points_2D[pt_idx]
                    if before_seg1:
                        limit_pt = segment1.endpoints_2D[1]
                        dist_of_interest = distance(
                            segment2.project_on_curve(pt, two_D=True), limit_pt
                        )
                    else:
                        limit_pt = None

                    if before_seg1:
                        pts_overlap.append(segment2.make3D(pt))
                        if (
                            distance(
                                segment2.project_on_curve(
                                    segment2.points_2D[pt_idx + 1], two_D=True
                                ),
                                limit_pt,
                            )
                            >= dist_of_interest
                        ):
                            before_seg1 = False
                    else:
                        pts_after_overlap.append(segment2.make3D(pt))

            if right_overlap:
                before_seg1 = True
                for pt_idx in range(len(segment2.points_2D)):
                    pt = segment2.points_2D[pt_idx]
                    if before_seg1:
                        limit_pt = segment1.endpoints_2D[0]
                        dist_of_interest = distance(
                            segment2.project_on_curve(pt, two_D=True), limit_pt
                        )
                    else:
                        limit_pt = None

                    if before_seg1:
                        pts_before_overlap.append(segment2.make3D(pt))
                        if (
                            distance(
                                segment2.project_on_curve(
                                    segment2.points_2D[pt_idx + 1], two_D=True
                                ),
                                limit_pt,
                            )
                            >= dist_of_interest
                        ):
                            before_seg1 = False
                    else:
                        pts_overlap.append(segment2.make3D(pt))

                before_seg2 = True
                for pt_idx in range(len(segment1.points_2D)):
                    pt = segment1.points_2D[pt_idx]
                    if before_seg2:
                        limit_pt = segment2.endpoints_2D[1]
                        dist_of_interest = distance(
                            segment1.project_on_curve(pt, two_D=True), limit_pt
                        )
                    else:
                        limit_pt = None

                    if before_seg2:
                        pts_overlap.append(segment1.make3D(pt))
                        if (
                            distance(
                                segment1.project_on_curve(
                                    segment1.points_2D[pt_idx + 1], two_D=True
                                ),
                                limit_pt,
                            )
                            >= dist_of_interest
                        ):
                            before_seg2 = False
                    else:
                        pts_after_overlap.append(segment1.make3D(pt))

            if len(pts_before_overlap) > 1:
                if left_overlap:
                    ref_slice_coord = segment1.ref_slice_coord
                else:
                    ref_slice_coord = segment2.ref_slice_coord
                new_segments.append(
                    Segment(
                        pts_before_overlap,
                        ref_slice_coord=ref_slice_coord,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("w")
            if len(pts_overlap) > 1:
                new_segments.append(
                    Segment(
                        pts_overlap,
                        ref_slice_coord=None,
                        interslice_ax=segment2.interslice_ax,
                    )
                )
            else:
                print("x")
            if len(pts_after_overlap) > 1:
                if left_overlap:
                    ref_slice_coord = segment2.ref_slice_coord
                else:
                    ref_slice_coord = segment1.ref_slice_coord
                new_segments.append(
                    Segment(
                        pts_after_overlap,
                        ref_slice_coord=ref_slice_coord,
                        interslice_ax=segment1.interslice_ax,
                    )
                )
            else:
                print("abc")
        else:
            new_segments.append(segment1)
            new_segments.append(segment2)

        return new_segments


def create_segments_by_slice(points):
    x, y, z = np.array(points).T
    interslice_coord = None
    interslice_ax = None
    interslice_max_changes = np.inf

    # We want to group the points by slice
    # We need to find when the z (or x or y) coordinate changes
    for ax_idx, axis in enumerate([x, y, z]):
        unique_values = np.unique(axis)
        if len(unique_values) < interslice_max_changes:
            interslice_max_changes = len(unique_values)
            interslice_coord = axis
            interslice_ax = ax_idx
    unique_slice_coords = np.unique(interslice_coord)

    remaining_axes = [0, 1, 2]
    remaining_axes.remove(interslice_ax)

    segments = []
    for slice_coord in unique_slice_coords:
        slice_points = points[np.where(interslice_coord == slice_coord)]
        segment = Segment(
            slice_points, ref_slice_coord=slice_coord, interslice_ax=interslice_ax
        )
        segments.append(segment)
    return segments


class PiecewiseLinear3D:
    def __init__(self, points: Union[List[Segment], List[List[float]]]) -> None:
        """
        Initialize with a list of 3D points defining the piecewise linear path.
        """
        if isinstance(points[0], Segment):
            merger = SegmentMerger()
            catheter_segments = merger.merge_segments(points)
            self.point_pairs = self.get_pairs_from_segment(catheter_segments)
            self.points = None
        else:
            self.points = points
            self.point_pairs = self.get_pairs_from_pts()
        self.segment_lengths, self.total_length = self.calculate_lengths()
        self.segment_ranges = self.calculate_ranges()

    def get_pairs_from_segment(self, segments, extend=True):
        """
        Getting endpoints of each segment in 3D to be able to compute distances.
        Segments are already ordered if created with a SegmentMerger.
        """
        point_pairs = []
        for segment in segments:
            point_pairs.append(
                [segment.make3D(segment.endpoints_2D[i]) for i in range(2)]
            )
        if extend:
            # Add new endpoints at the beginning and end to be able to predict
            # from points before the first endpoint and after the last endpoint.
            point_pairs[0][0] = self.create_point_on_line(
                point_pairs[0][0], point_pairs[0][1]
            )
            point_pairs[-1][1] = self.create_point_on_line(
                point_pairs[-1][0], point_pairs[-1][1], end=True
            )
        return point_pairs

    def get_pairs_from_pts(self):
        """
        Getting endpoints of each segment in 3D to be able to compute distances.
        """
        point_pairs = []
        for i in range(1, len(self.points)):
            point_pairs.append([self.points[i - 1], self.points[i]])
        return point_pairs

    @staticmethod
    def create_point_on_line(point1, point2, distance=10, end=False):
        # Convert points to numpy arrays for easier calculations
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Step 1: Calculate the vector from point1 to point2
        vector = point2 - point1

        # Step 2: Normalize the vector
        normalized_vector = vector / np.linalg.norm(vector)

        # Step 3: Scale the vector by the desired distance
        scaled_vector = normalized_vector * distance

        # Step 4: Calculate the new point
        if end:
            new_point = point2 + scaled_vector
        else:
            new_point = point1 - scaled_vector

        return new_point.tolist()

    def calculate_lengths(self):
        """
        Calculate the length of each segment and the total length.
        """
        lengths = [distance(pair[0], pair[1]) for pair in self.point_pairs]
        total_length = sum(lengths)
        return lengths, total_length

    def calculate_ranges(self, extend=True):
        """
        Calculate the parameter range for each segment.
        """
        ranges = []
        t_start = 0.0
        for length in self.segment_lengths:
            t_end = t_start + (length / self.total_length)
            ranges.append((t_start, t_end))
            t_start = t_end

        # Correcting for floating points errors
        ranges[-1] = (ranges[-1][0], 1.0)

        return ranges

    def evaluate(self, t):
        """
        Evaluate the piecewise linear function at a given parameter t.
        """
        # Find which segment t falls into
        for i, (t_start, t_end) in enumerate(self.segment_ranges):
            if t_start <= t and t <= t_end:
                # Map t to the local parameter of the segment
                local_t = (t - t_start) / (t_end - t_start)
                # Calculate the 3D coordinates
                p1, p2 = self.point_pairs[i]
                x = p1[0] + (p2[0] - p1[0]) * local_t
                y = p1[1] + (p2[1] - p1[1]) * local_t
                z = p1[2] + (p2[2] - p1[2]) * local_t
                return x, y, z
        return None  # t is out of range

    def find_range_on_pwline_form_t(self, t:float=0.0):

        for seg_range_idx, seg_range in enumerate(self.segment_ranges):
            if seg_range[0] <= t and t <= seg_range[1]:
                start_point_t1 = seg_range[0]
                start_point_t2 = seg_range[1]
                start_pt_seg_range_idx = seg_range_idx
                break
        return start_point_t1, start_point_t2, start_pt_seg_range_idx
    
    def project_on_pw_line(self, point: List[float]):
        """
        Project the point on the piecewise linear function.
        """

        def step_func(t: float) -> float:
            point_coords = self.evaluate(t)
            # We are in the same segment so a straight line direction is good
            return abs(distance(point, point_coords))
           
        opt = minimize_scalar(step_func, bounds=(0.0, 1.0), method="bounded")

        return list(self.evaluate(opt.x)), opt.x
    
    def find_last_pt_from_pair(self, pair_idx:int):
        """
        Which point from the pair corresponds to the biggest t.
        """
        max_t = 0.0
        identified_pt = None
        for pt in self.point_pairs[pair_idx]:
            _, pt_t = self.project_on_pw_line(pt)
            if pt_t > max_t:
                max_t = pt_t
                identified_pt = pt
        return identified_pt
    
    def find_first_pt_from_pair(self, pair_idx:int):
        """
        Which point from the pair corresponds to the smallest t.
        """
        min_t = 0.0
        identified_pt = None
        for pt in self.point_pairs[pair_idx]:
            _, pt_t = self.project_on_pw_line(pt)
            if pt_t < min_t:
                min_t = pt_t
                identified_pt = pt
        return identified_pt
    
    def distance_on_pw_line(self, start_point:List[float], end_point:List[float]):
        """
        Compute the distance between two points on the piecewise linear function.
        The path taken is not a straight line but the pw line.
        """
        _, start_point_t = self.project_on_pw_line(start_point)
        _, end_point_t = self.project_on_pw_line(end_point)

        
        _, _, start_pt_seg_range_idx = self.find_range_on_pwline_form_t(start_point_t)
        _, _, end_pt_seg_range_idx = self.find_range_on_pwline_form_t(end_point_t)

        if start_pt_seg_range_idx==end_pt_seg_range_idx:
                # We are in the same segment so a straight line direction is good
                return distance(start_point, end_point)
        else:
            if start_pt_seg_range_idx < end_pt_seg_range_idx:
                # We are in a different segment so the direction goes in multiple
                # segments. We need to add up the distanvce between each segments. 

                # Ranges are ordered so we find the point correspondig to the biggest 
                # t of the range
                total_distance = distance(start_point, self.point_pairs[start_pt_seg_range_idx][1])
                # Find the segment of the t
                for seg_range_idx, seg_range in enumerate(self.segment_ranges):
                    if seg_range_idx <= start_pt_seg_range_idx:
                        continue
                    if seg_range_idx == end_pt_seg_range_idx:
                        total_distance += distance(self.point_pairs[seg_range_idx][0], end_point)   
                        break
                    else:
                        # We skip that segment but add the distance
                        total_distance += self.segment_lengths[seg_range_idx]
            else:
                total_distance =  distance(end_point, self.point_pairs[end_pt_seg_range_idx][1])
                # Find the segment of the t
                for seg_range_idx, seg_range in enumerate(self.segment_ranges):
                    if seg_range_idx <= end_pt_seg_range_idx:
                        continue
                    if seg_range_idx == start_pt_seg_range_idx:
                        total_distance += distance(self.point_pairs[seg_range_idx][0], start_point)
                        break
                    else:
                        # We skip that segment but add the distance
                        total_distance += self.segment_lengths[seg_range_idx]
            return total_distance
        
    def step_in_pw_line(self, start_point:List[float], step:float, bound_min:float=0.0, check:bool=False):
        """
        Given a point and a step, we want to find the point on the piecewise linear function
        that is at a distance step from the point. The point should already be in the pw line.
        """
        projected_start_pt, projected_start_pt_t = self.project_on_pw_line(start_point)
        assert distance(projected_start_pt, start_point) < 1e-2, (
            f"""
            The point should be on the pw line but are distant from {distance(projected_start_pt, start_point)}
            """
        )
        # Some floating point error can happen so we round up to make sure t does not impact
        # the segment chosen.
        
        start_point_t1, start_point_t2, start_pt_seg_range_idx = self.find_range_on_pwline_form_t(np.round(bound_min, 2))
        
        if check:
            _, _, temp_start_pt_seg_range_idx = self.find_range_on_pwline_form_t(np.round(projected_start_pt_t, 2))
            # print("self.point_pairs ", self.point_pairs )
            # print("self.segment_ranges", self.segment_ranges)
            # print("self.segment_lengths", self.segment_lengths)
            # print("self.total_length", self.total_length)
            if temp_start_pt_seg_range_idx != start_pt_seg_range_idx:
                print(
                f"""
                The point is not in the same segment as the bound_min. It can be due to
                the fact that your first segment is super small.
                You have a projected point with value t {projected_start_pt_t} in segment 
                {temp_start_pt_seg_range_idx} and bound_min with value t {bound_min}
                in segment {start_pt_seg_range_idx}
                """
                )

        for seg_range_idx, seg_range in enumerate(self.segment_ranges):
            if seg_range[0] <= bound_min and bound_min < seg_range[1]:
                start_point_t1 = seg_range[0]
                start_point_t2 = seg_range[1]
                start_pt_seg_range_idx = seg_range_idx
                break

        def step_func(t: float) -> float:
            point_coords = self.evaluate(t)
            if start_point_t1 <= t and t < start_point_t2:
                # We are in the same segment so a straight line direction is good
                return abs(distance(start_point, point_coords) - step)
            else:
                # We are in a different segment so the direction goes in multiple
                # segments. We need to add up the distanvce between each segments. 
                total_distance =  distance(start_point, self.point_pairs[start_pt_seg_range_idx][1])
                # Find the segment of the t
                for seg_range_idx, seg_range in enumerate(self.segment_ranges):
                    if seg_range_idx <= start_pt_seg_range_idx:
                        continue
                    if seg_range[0] <= t and t < seg_range[1]:
                        # We are in the corresponding segment of the t
                        total_distance += distance(self.point_pairs[seg_range_idx][0], point_coords)
                        break
                    else:
                        # We skip that segment but add the distance
                        total_distance += self.segment_lengths[seg_range_idx]
                return abs(total_distance - step)

        opt = minimize_scalar(step_func, bounds=(bound_min, 1.0), method="bounded")

        return list(self.evaluate(opt.x)), opt.x, step_func(opt.x) + step
    

def find_segment(points, zoom=True):

    segments = create_segments_by_slice(points)
    merger = SegmentMerger()
    # segments = merger.merge_two_segments(segments[0], segments[5])
    segments = merger.merge_segments(segments)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for color_idx, segment in enumerate(segments):
        points = np.array(segment.points)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=list(mcolors.TABLEAU_COLORS.values())[color_idx],
        )
        print("color used ", list(mcolors.TABLEAU_COLORS.keys())[color_idx])

    if zoom:
        # If you want to set the aspect ratio
        # # Set the aspect ratio and limits
        x_points, y_points, z_points = np.array(merger.mega_segment.points).T
        biggest_range = max(
            max(z_points) - min(z_points),
            max(y_points) - min(y_points),
            max(x_points) - min(x_points),
        )
        # center each axis on the same range size
        ax.set_xlim(
            [
                np.mean(x_points) - biggest_range,
                np.mean(x_points) + biggest_range,
            ]
        )
        ax.set_ylim(
            [
                np.mean(y_points) - biggest_range,
                np.mean(y_points) + biggest_range,
            ]
        )
        ax.set_zlim(
            [
                np.mean(z_points) - biggest_range,
                np.mean(z_points) + biggest_range,
            ]
        )
        ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()
    exit()


if __name__ == "__main__":
    # Testing the function with real needle points.
    pts = np.array(
        [
            (-115.0, -50.0, 73.0000153),
            (-114.0, -50.0, 73.0000153),
            (-117.0, -49.0, 73.0000153),
            (-116.0, -49.0, 73.0000153),
            (-119.0, -48.0, 73.0000153),
            (-118.0, -48.0, 73.0000153),
            (-121.0, -47.0, 73.0000153),
            (-120.0, -47.0, 73.0000153),
            (-123.0, -46.0, 73.0000153),
            (-122.0, -46.0, 73.0000153),
            (-125.0, -45.0, 73.0000153),
            (-124.0, -45.0, 73.0000153),
            (-126.0, -44.0, 73.0000153),
            (-58.0, -80.0, 74.0000153),
            (-60.0, -79.0, 74.0000153),
            (-59.0, -79.0, 74.0000153),
            (-62.0, -78.0, 74.0000153),
            (-61.0, -78.0, 74.0000153),
            (-64.0, -77.0, 74.0000153),
            (-63.0, -77.0, 74.0000153),
            (-66.0, -76.0, 74.0000153),
            (-65.0, -76.0, 74.0000153),
            (-68.0, -75.0, 74.0000153),
            (-67.0, -75.0, 74.0000153),
            (-70.0, -74.0, 74.0000153),
            (-69.0, -74.0, 74.0000153),
            (-72.0, -73.0, 74.0000153),
            (-71.0, -73.0, 74.0000153),
            (-74.0, -72.0, 74.0000153),
            (-73.0, -72.0, 74.0000153),
            (-75.0, -71.0, 74.0000153),
            (-77.0, -70.0, 74.0000153),
            (-76.0, -70.0, 74.0000153),
            (-79.0, -69.0, 74.0000153),
            (-78.0, -69.0, 74.0000153),
            (-80.0, -68.0, 74.0000153),
            (-82.0, -67.0, 74.0000153),
            (-81.0, -67.0, 74.0000153),
            (-84.0, -66.0, 74.0000153),
            (-83.0, -66.0, 74.0000153),
            (-86.0, -65.0, 74.0000153),
            (-85.0, -65.0, 74.0000153),
            (-87.0, -64.0, 74.0000153),
            (-90.0, -63.0, 74.0000153),
            (-89.0, -63.0, 74.0000153),
            (-88.0, -63.0, 74.0000153),
            (-91.0, -62.0, 74.0000153),
            (-94.0, -61.0, 74.0000153),
            (-93.0, -61.0, 74.0000153),
            (-92.0, -61.0, 74.0000153),
            (-95.0, -60.0, 74.0000153),
            (-98.0, -59.0, 74.0000153),
            (-97.0, -59.0, 74.0000153),
            (-96.0, -59.0, 74.0000153),
            (-99.0, -58.0, 74.0000153),
            (-101.0, -57.0, 74.0000153),
            (-100.0, -57.0, 74.0000153),
            (-103.0, -56.0, 74.0000153),
            (-102.0, -56.0, 74.0000153),
            (-105.0, -55.0, 74.0000153),
            (-104.0, -55.0, 74.0000153),
            (-107.0, -54.0, 74.0000153),
            (-106.0, -54.0, 74.0000153),
            (-109.0, -53.0, 74.0000153),
            (-108.0, -53.0, 74.0000153),
            (-111.0, -52.0, 74.0000153),
            (-110.0, -52.0, 74.0000153),
            (-115.0, -51.0, 74.0000153),
            (-114.0, -51.0, 74.0000153),
            (-113.0, -51.0, 74.0000153),
            (-112.0, -51.0, 74.0000153),
            (-117.0, -50.0, 74.0000153),
            (-116.0, -50.0, 74.0000153),
            (-115.0, -50.0, 74.0000153),
            (-114.0, -50.0, 74.0000153),
            (-113.0, -50.0, 74.0000153),
            (-119.0, -49.0, 74.0000153),
            (-118.0, -49.0, 74.0000153),
            (-117.0, -49.0, 74.0000153),
            (-116.0, -49.0, 74.0000153),
            (-115.0, -49.0, 74.0000153),
            (-114.0, -49.0, 74.0000153),
            (-121.0, -48.0, 74.0000153),
            (-120.0, -48.0, 74.0000153),
            (-119.0, -48.0, 74.0000153),
            (-118.0, -48.0, 74.0000153),
            (-117.0, -48.0, 74.0000153),
            (-116.0, -48.0, 74.0000153),
            (-123.0, -47.0, 74.0000153),
            (-122.0, -47.0, 74.0000153),
            (-121.0, -47.0, 74.0000153),
            (-120.0, -47.0, 74.0000153),
            (-119.0, -47.0, 74.0000153),
            (-118.0, -47.0, 74.0000153),
            (-125.0, -46.0, 74.0000153),
            (-124.0, -46.0, 74.0000153),
            (-123.0, -46.0, 74.0000153),
            (-122.0, -46.0, 74.0000153),
            (-121.0, -46.0, 74.0000153),
            (-120.0, -46.0, 74.0000153),
            (-126.0, -45.0, 74.0000153),
            (-125.0, -45.0, 74.0000153),
            (-124.0, -45.0, 74.0000153),
            (-123.0, -45.0, 74.0000153),
            (-122.0, -45.0, 74.0000153),
            (-127.0, -44.0, 74.0000153),
            (-126.0, -44.0, 74.0000153),
            (-125.0, -44.0, 74.0000153),
            (-124.0, -44.0, 74.0000153),
            (-126.0, -43.0, 74.0000153),
            (-36.0, -92.0, 75.0000153),
            (-35.0, -92.0, 75.0000153),
            (-38.0, -91.0, 75.0000153),
            (-37.0, -91.0, 75.0000153),
            (-40.0, -90.0, 75.0000153),
            (-39.0, -90.0, 75.0000153),
            (-41.0, -89.0, 75.0000153),
            (-43.0, -88.0, 75.0000153),
            (-42.0, -88.0, 75.0000153),
            (-45.0, -87.0, 75.0000153),
            (-44.0, -87.0, 75.0000153),
            (-47.0, -86.0, 75.0000153),
            (-46.0, -86.0, 75.0000153),
            (-49.0, -85.0, 75.0000153),
            (-48.0, -85.0, 75.0000153),
            (-51.0, -84.0, 75.0000153),
            (-50.0, -84.0, 75.0000153),
            (-53.0, -83.0, 75.0000153),
            (-52.0, -83.0, 75.0000153),
            (-55.0, -82.0, 75.0000153),
            (-54.0, -82.0, 75.0000153),
            (-58.0, -81.0, 75.0000153),
            (-57.0, -81.0, 75.0000153),
            (-56.0, -81.0, 75.0000153),
            (-60.0, -80.0, 75.0000153),
            (-59.0, -80.0, 75.0000153),
            (-58.0, -80.0, 75.0000153),
            (-57.0, -80.0, 75.0000153),
            (-62.0, -79.0, 75.0000153),
            (-61.0, -79.0, 75.0000153),
            (-60.0, -79.0, 75.0000153),
            (-59.0, -79.0, 75.0000153),
            (-58.0, -79.0, 75.0000153),
            (-64.0, -78.0, 75.0000153),
            (-63.0, -78.0, 75.0000153),
            (-62.0, -78.0, 75.0000153),
            (-61.0, -78.0, 75.0000153),
            (-60.0, -78.0, 75.0000153),
            (-59.0, -78.0, 75.0000153),
            (-66.0, -77.0, 75.0000153),
            (-65.0, -77.0, 75.0000153),
            (-64.0, -77.0, 75.0000153),
            (-63.0, -77.0, 75.0000153),
            (-62.0, -77.0, 75.0000153),
            (-61.0, -77.0, 75.0000153),
            (-68.0, -76.0, 75.0000153),
            (-67.0, -76.0, 75.0000153),
            (-66.0, -76.0, 75.0000153),
            (-65.0, -76.0, 75.0000153),
            (-64.0, -76.0, 75.0000153),
            (-63.0, -76.0, 75.0000153),
            (-70.0, -75.0, 75.0000153),
            (-69.0, -75.0, 75.0000153),
            (-68.0, -75.0, 75.0000153),
            (-67.0, -75.0, 75.0000153),
            (-66.0, -75.0, 75.0000153),
            (-65.0, -75.0, 75.0000153),
            (-72.0, -74.0, 75.0000153),
            (-71.0, -74.0, 75.0000153),
            (-70.0, -74.0, 75.0000153),
            (-69.0, -74.0, 75.0000153),
            (-68.0, -74.0, 75.0000153),
            (-67.0, -74.0, 75.0000153),
            (-74.0, -73.0, 75.0000153),
            (-73.0, -73.0, 75.0000153),
            (-72.0, -73.0, 75.0000153),
            (-71.0, -73.0, 75.0000153),
            (-70.0, -73.0, 75.0000153),
            (-69.0, -73.0, 75.0000153),
            (-75.0, -72.0, 75.0000153),
            (-74.0, -72.0, 75.0000153),
            (-73.0, -72.0, 75.0000153),
            (-72.0, -72.0, 75.0000153),
            (-71.0, -72.0, 75.0000153),
            (-77.0, -71.0, 75.0000153),
            (-76.0, -71.0, 75.0000153),
            (-75.0, -71.0, 75.0000153),
            (-74.0, -71.0, 75.0000153),
            (-73.0, -71.0, 75.0000153),
            (-79.0, -70.0, 75.0000153),
            (-78.0, -70.0, 75.0000153),
            (-77.0, -70.0, 75.0000153),
            (-76.0, -70.0, 75.0000153),
            (-75.0, -70.0, 75.0000153),
            (-80.0, -69.0, 75.0000153),
            (-79.0, -69.0, 75.0000153),
            (-78.0, -69.0, 75.0000153),
            (-77.0, -69.0, 75.0000153),
            (-76.0, -69.0, 75.0000153),
            (-82.0, -68.0, 75.0000153),
            (-81.0, -68.0, 75.0000153),
            (-80.0, -68.0, 75.0000153),
            (-79.0, -68.0, 75.0000153),
            (-78.0, -68.0, 75.0000153),
            (-84.0, -67.0, 75.0000153),
            (-83.0, -67.0, 75.0000153),
            (-82.0, -67.0, 75.0000153),
            (-81.0, -67.0, 75.0000153),
            (-80.0, -67.0, 75.0000153),
            (-86.0, -66.0, 75.0000153),
            (-85.0, -66.0, 75.0000153),
            (-84.0, -66.0, 75.0000153),
            (-83.0, -66.0, 75.0000153),
            (-82.0, -66.0, 75.0000153),
            (-81.0, -66.0, 75.0000153),
            (-87.0, -65.0, 75.0000153),
            (-86.0, -65.0, 75.0000153),
            (-85.0, -65.0, 75.0000153),
            (-84.0, -65.0, 75.0000153),
            (-83.0, -65.0, 75.0000153),
            (-90.0, -64.0, 75.0000153),
            (-89.0, -64.0, 75.0000153),
            (-88.0, -64.0, 75.0000153),
            (-87.0, -64.0, 75.0000153),
            (-86.0, -64.0, 75.0000153),
            (-85.0, -64.0, 75.0000153),
            (-91.0, -63.0, 75.0000153),
            (-90.0, -63.0, 75.0000153),
            (-89.0, -63.0, 75.0000153),
            (-88.0, -63.0, 75.0000153),
            (-87.0, -63.0, 75.0000153),
            (-94.0, -62.0, 75.0000153),
            (-93.0, -62.0, 75.0000153),
            (-92.0, -62.0, 75.0000153),
            (-91.0, -62.0, 75.0000153),
            (-90.0, -62.0, 75.0000153),
            (-89.0, -62.0, 75.0000153),
            (-88.0, -62.0, 75.0000153),
            (-95.0, -61.0, 75.0000153),
            (-94.0, -61.0, 75.0000153),
            (-93.0, -61.0, 75.0000153),
            (-92.0, -61.0, 75.0000153),
            (-91.0, -61.0, 75.0000153),
            (-98.0, -60.0, 75.0000153),
            (-97.0, -60.0, 75.0000153),
            (-96.0, -60.0, 75.0000153),
            (-95.0, -60.0, 75.0000153),
            (-94.0, -60.0, 75.0000153),
            (-93.0, -60.0, 75.0000153),
            (-92.0, -60.0, 75.0000153),
            (-99.0, -59.0, 75.0000153),
            (-98.0, -59.0, 75.0000153),
            (-97.0, -59.0, 75.0000153),
            (-96.0, -59.0, 75.0000153),
            (-95.0, -59.0, 75.0000153),
            (-101.0, -58.0, 75.0000153),
            (-100.0, -58.0, 75.0000153),
            (-99.0, -58.0, 75.0000153),
            (-98.0, -58.0, 75.0000153),
            (-97.0, -58.0, 75.0000153),
            (-96.0, -58.0, 75.0000153),
            (-103.0, -57.0, 75.0000153),
            (-102.0, -57.0, 75.0000153),
            (-101.0, -57.0, 75.0000153),
            (-100.0, -57.0, 75.0000153),
            (-99.0, -57.0, 75.0000153),
            (-105.0, -56.0, 75.0000153),
            (-104.0, -56.0, 75.0000153),
            (-103.0, -56.0, 75.0000153),
            (-102.0, -56.0, 75.0000153),
            (-101.0, -56.0, 75.0000153),
            (-100.0, -56.0, 75.0000153),
            (-107.0, -55.0, 75.0000153),
            (-106.0, -55.0, 75.0000153),
            (-105.0, -55.0, 75.0000153),
            (-104.0, -55.0, 75.0000153),
            (-103.0, -55.0, 75.0000153),
            (-102.0, -55.0, 75.0000153),
            (-109.0, -54.0, 75.0000153),
            (-108.0, -54.0, 75.0000153),
            (-107.0, -54.0, 75.0000153),
            (-106.0, -54.0, 75.0000153),
            (-105.0, -54.0, 75.0000153),
            (-104.0, -54.0, 75.0000153),
            (-111.0, -53.0, 75.0000153),
            (-110.0, -53.0, 75.0000153),
            (-109.0, -53.0, 75.0000153),
            (-108.0, -53.0, 75.0000153),
            (-107.0, -53.0, 75.0000153),
            (-106.0, -53.0, 75.0000153),
            (-113.0, -52.0, 75.0000153),
            (-112.0, -52.0, 75.0000153),
            (-111.0, -52.0, 75.0000153),
            (-110.0, -52.0, 75.0000153),
            (-109.0, -52.0, 75.0000153),
            (-108.0, -52.0, 75.0000153),
            (-114.0, -51.0, 75.0000153),
            (-113.0, -51.0, 75.0000153),
            (-112.0, -51.0, 75.0000153),
            (-111.0, -51.0, 75.0000153),
            (-110.0, -51.0, 75.0000153),
            (-115.0, -50.0, 75.0000153),
            (-114.0, -50.0, 75.0000153),
            (-113.0, -50.0, 75.0000153),
            (-112.0, -50.0, 75.0000153),
            (-117.0, -49.0, 75.0000153),
            (-116.0, -49.0, 75.0000153),
            (-119.0, -48.0, 75.0000153),
            (-118.0, -48.0, 75.0000153),
            (-121.0, -47.0, 75.0000153),
            (-120.0, -47.0, 75.0000153),
            (-123.0, -46.0, 75.0000153),
            (-122.0, -46.0, 75.0000153),
            (-125.0, -45.0, 75.0000153),
            (-124.0, -45.0, 75.0000153),
            (-126.0, -44.0, 75.0000153),
            (-30.0, -95.0, 76.0000153),
            (-29.0, -95.0, 76.0000153),
            (-32.0, -94.0, 76.0000153),
            (-31.0, -94.0, 76.0000153),
            (-36.0, -93.0, 76.0000153),
            (-35.0, -93.0, 76.0000153),
            (-34.0, -93.0, 76.0000153),
            (-33.0, -93.0, 76.0000153),
            (-38.0, -92.0, 76.0000153),
            (-37.0, -92.0, 76.0000153),
            (-36.0, -92.0, 76.0000153),
            (-35.0, -92.0, 76.0000153),
            (-34.0, -92.0, 76.0000153),
            (-40.0, -91.0, 76.0000153),
            (-39.0, -91.0, 76.0000153),
            (-38.0, -91.0, 76.0000153),
            (-37.0, -91.0, 76.0000153),
            (-36.0, -91.0, 76.0000153),
            (-35.0, -91.0, 76.0000153),
            (-41.0, -90.0, 76.0000153),
            (-40.0, -90.0, 76.0000153),
            (-39.0, -90.0, 76.0000153),
            (-38.0, -90.0, 76.0000153),
            (-37.0, -90.0, 76.0000153),
            (-43.0, -89.0, 76.0000153),
            (-42.0, -89.0, 76.0000153),
            (-41.0, -89.0, 76.0000153),
            (-40.0, -89.0, 76.0000153),
            (-39.0, -89.0, 76.0000153),
            (-45.0, -88.0, 76.0000153),
            (-44.0, -88.0, 76.0000153),
            (-43.0, -88.0, 76.0000153),
            (-42.0, -88.0, 76.0000153),
            (-41.0, -88.0, 76.0000153),
            (-47.0, -87.0, 76.0000153),
            (-46.0, -87.0, 76.0000153),
            (-45.0, -87.0, 76.0000153),
            (-44.0, -87.0, 76.0000153),
            (-43.0, -87.0, 76.0000153),
            (-42.0, -87.0, 76.0000153),
            (-49.0, -86.0, 76.0000153),
            (-48.0, -86.0, 76.0000153),
            (-47.0, -86.0, 76.0000153),
            (-46.0, -86.0, 76.0000153),
            (-45.0, -86.0, 76.0000153),
            (-44.0, -86.0, 76.0000153),
            (-51.0, -85.0, 76.0000153),
            (-50.0, -85.0, 76.0000153),
            (-49.0, -85.0, 76.0000153),
            (-48.0, -85.0, 76.0000153),
            (-47.0, -85.0, 76.0000153),
            (-46.0, -85.0, 76.0000153),
            (-53.0, -84.0, 76.0000153),
            (-52.0, -84.0, 76.0000153),
            (-51.0, -84.0, 76.0000153),
            (-50.0, -84.0, 76.0000153),
            (-49.0, -84.0, 76.0000153),
            (-48.0, -84.0, 76.0000153),
            (-55.0, -83.0, 76.0000153),
            (-54.0, -83.0, 76.0000153),
            (-53.0, -83.0, 76.0000153),
            (-52.0, -83.0, 76.0000153),
            (-51.0, -83.0, 76.0000153),
            (-50.0, -83.0, 76.0000153),
            (-57.0, -82.0, 76.0000153),
            (-56.0, -82.0, 76.0000153),
            (-55.0, -82.0, 76.0000153),
            (-54.0, -82.0, 76.0000153),
            (-53.0, -82.0, 76.0000153),
            (-52.0, -82.0, 76.0000153),
            (-58.0, -81.0, 76.0000153),
            (-57.0, -81.0, 76.0000153),
            (-56.0, -81.0, 76.0000153),
            (-55.0, -81.0, 76.0000153),
            (-54.0, -81.0, 76.0000153),
            (-58.0, -80.0, 76.0000153),
            (-57.0, -80.0, 76.0000153),
            (-56.0, -80.0, 76.0000153),
            (-60.0, -79.0, 76.0000153),
            (-59.0, -79.0, 76.0000153),
            (-62.0, -78.0, 76.0000153),
            (-61.0, -78.0, 76.0000153),
            (-64.0, -77.0, 76.0000153),
            (-63.0, -77.0, 76.0000153),
            (-66.0, -76.0, 76.0000153),
            (-65.0, -76.0, 76.0000153),
            (-68.0, -75.0, 76.0000153),
            (-67.0, -75.0, 76.0000153),
            (-70.0, -74.0, 76.0000153),
            (-69.0, -74.0, 76.0000153),
            (-72.0, -73.0, 76.0000153),
            (-71.0, -73.0, 76.0000153),
            (-74.0, -72.0, 76.0000153),
            (-73.0, -72.0, 76.0000153),
            (-75.0, -71.0, 76.0000153),
            (-77.0, -70.0, 76.0000153),
            (-76.0, -70.0, 76.0000153),
            (-79.0, -69.0, 76.0000153),
            (-78.0, -69.0, 76.0000153),
            (-80.0, -68.0, 76.0000153),
            (-82.0, -67.0, 76.0000153),
            (-81.0, -67.0, 76.0000153),
            (-84.0, -66.0, 76.0000153),
            (-83.0, -66.0, 76.0000153),
            (-86.0, -65.0, 76.0000153),
            (-85.0, -65.0, 76.0000153),
            (-87.0, -64.0, 76.0000153),
            (-90.0, -63.0, 76.0000153),
            (-89.0, -63.0, 76.0000153),
            (-88.0, -63.0, 76.0000153),
            (-91.0, -62.0, 76.0000153),
            (-94.0, -61.0, 76.0000153),
            (-93.0, -61.0, 76.0000153),
            (-92.0, -61.0, 76.0000153),
            (-95.0, -60.0, 76.0000153),
            (-98.0, -59.0, 76.0000153),
            (-97.0, -59.0, 76.0000153),
            (-96.0, -59.0, 76.0000153),
            (-99.0, -58.0, 76.0000153),
            (-101.0, -57.0, 76.0000153),
            (-100.0, -57.0, 76.0000153),
            (-103.0, -56.0, 76.0000153),
            (-102.0, -56.0, 76.0000153),
            (-105.0, -55.0, 76.0000153),
            (-104.0, -55.0, 76.0000153),
            (-107.0, -54.0, 76.0000153),
            (-106.0, -54.0, 76.0000153),
            (-109.0, -53.0, 76.0000153),
            (-108.0, -53.0, 76.0000153),
            (-111.0, -52.0, 76.0000153),
            (-110.0, -52.0, 76.0000153),
            (-113.0, -51.0, 76.0000153),
            (-112.0, -51.0, 76.0000153),
            (-30.0, -96.0, 77.0000153),
            (-29.0, -96.0, 77.0000153),
            (-32.0, -95.0, 77.0000153),
            (-31.0, -95.0, 77.0000153),
            (-30.0, -95.0, 77.0000153),
            (-29.0, -95.0, 77.0000153),
            (-28.0, -95.0, 77.0000153),
            (-34.0, -94.0, 77.0000153),
            (-33.0, -94.0, 77.0000153),
            (-32.0, -94.0, 77.0000153),
            (-31.0, -94.0, 77.0000153),
            (-30.0, -94.0, 77.0000153),
            (-29.0, -94.0, 77.0000153),
            (-35.0, -93.0, 77.0000153),
            (-34.0, -93.0, 77.0000153),
            (-33.0, -93.0, 77.0000153),
            (-32.0, -93.0, 77.0000153),
            (-31.0, -93.0, 77.0000153),
            (-36.0, -92.0, 77.0000153),
            (-35.0, -92.0, 77.0000153),
            (-34.0, -92.0, 77.0000153),
            (-33.0, -92.0, 77.0000153),
            (-38.0, -91.0, 77.0000153),
            (-37.0, -91.0, 77.0000153),
            (-40.0, -90.0, 77.0000153),
            (-39.0, -90.0, 77.0000153),
            (-41.0, -89.0, 77.0000153),
            (-43.0, -88.0, 77.0000153),
            (-42.0, -88.0, 77.0000153),
            (-45.0, -87.0, 77.0000153),
            (-44.0, -87.0, 77.0000153),
            (-47.0, -86.0, 77.0000153),
            (-46.0, -86.0, 77.0000153),
            (-49.0, -85.0, 77.0000153),
            (-48.0, -85.0, 77.0000153),
            (-51.0, -84.0, 77.0000153),
            (-50.0, -84.0, 77.0000153),
            (-53.0, -83.0, 77.0000153),
            (-52.0, -83.0, 77.0000153),
            (-55.0, -82.0, 77.0000153),
            (-54.0, -82.0, 77.0000153),
            (-57.0, -81.0, 77.0000153),
            (-56.0, -81.0, 77.0000153),
            (-30.0, -95.0, 78.0000153),
            (-29.0, -95.0, 78.0000153),
            (-32.0, -94.0, 78.0000153),
            (-31.0, -94.0, 78.0000153),
            (-34.0, -93.0, 78.0000153),
            (-33.0, -93.0, 78.0000153),
        ]
    )

    find_segment(pts)
