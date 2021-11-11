"""
Structure module.

Copyright Marc-Andre Renaud, 2017
"""

import pydicom as dicom

from matplotlib.path import Path

import numpy

import pyclipper
from pyclipper import scale_from_clipper, scale_to_clipper

from scipy.spatial import ConvexHull

from pyRad.utils.math_utils import find_nearest


class Structure(object):
    """Wrapper for a structure inside a DICOM file."""

    def __init__(self, attrs):
        """
        Constructor.

        :param roi_num: Sequence index for the structure inside the DICOM file.
        :param roi_name: Name of the structure.
        :param struct_path: Full path to DICOM RT Structure file.
        :param frame_of_ref_uid: Frame of reference UID
        (Optional) :param contours: Pre-generated list of contour points.

        """
        self.struct_path = attrs["struct_path"]
        self.name, self.id = self.get_patient_info()
        # Can provide either the ROI num or the ROI name
        self.roi_num = attrs["roi_num"] if "roi_num" in attrs else self.get_num(attrs["roi_name"])
        self.roi_name = attrs["roi_name"] if "roi_name" in attrs else self.get_name(attrs["roi_num"])
        self.roi_index = self.get_index()
        self.frame_of_ref_uid = attrs.get("frame_of_ref_uid", None)

        if "contours" in attrs:
            self.contours = attrs["contours"]

        self.sanity_test()

    def get_name(self, roi_num):
        struct_path = self.struct_path
        struct_file = dicom.read_file(struct_path, force=True)

        for roi in struct_file.StructureSetROISequence:
            if roi.ROINumber == roi_num:
                return roi.ROIName

        return None

    def get_num(self, roi_name):
        struct_path = self.struct_path
        struct_file = dicom.read_file(struct_path, force=True)

        for roi in struct_file.StructureSetROISequence:
            if roi.ROIName == roi_name:
                return roi.ROINumber

    def sanity_test(self):
        """
        If both roi_num and roi_name have been pre-propulated, check to make sure they
        point to the right structure.
        """
        struct_file = dicom.read_file(self.struct_path, force=True)
        index_name = struct_file.StructureSetROISequence[self.roi_index].ROIName.decode("utf-8")
        if index_name != self.roi_name:
            raise Exception("roi_num: {}, roi_name: {}, index_name: {}".format(self.roi_num, self.roi_name, index_name))
        if "ReferencedROINumber" in struct_file.ROIContourSequence:
            assert(struct_file.ROIContourSequence[self.roi_index].ReferencedROINumber == self.roi_num)

    def get_patient_info(self):
        struct_path = self.struct_path
        struct_file = dicom.read_file(struct_path, force=True)
        return struct_file.PatientName, struct_file.PatientID

    def get_roi(self):
        struct_path = self.struct_path
        struct_file = dicom.read_file(struct_path, force=True)
        return struct_file.ROIContourSequence[self.roi_index]

    def get_index(self):
        struct_path = self.struct_path
        struct_file = dicom.read_file(struct_path, force=True)

        for index, roi in enumerate(struct_file.StructureSetROISequence):
            if roi.ROIName.decode("utf-8") == self.roi_name:
                return index

    def get_points(self):
        """
        Return a list of all points making up the structure contour.

        The points are given in the [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn] format.
        """
        try:
            return self._points
        except AttributeError:
            roi_points = []

            roi = self.get_roi()

            if "ContourSequence" in roi:
                for contour in roi.ContourSequence:
                    roi_points += zip(*([iter(contour.ContourData)] * 3))

            self._points = numpy.array(roi_points)
            return self._points

    def get_centroid(self):
        """Return the centroid of the ROI."""
        try:
            return self._centroid
        except AttributeError:
            points = self.get_points()

            self._centroid = numpy.mean(numpy.array(points), axis=0).tolist()
            return self._centroid

    def get_convex_hull(self, grow=False):
        """
        Return the convex hull of the ROI.

        :param grow: Naive grow of the convex hull centered around the centroid.
        """
        points = self.get_points()

        if grow:
            centroid = self.get_centroid()
            direction = points - centroid
            # Divide each direction by its respective norm
            direction /= numpy.linalg.norm(direction, axis=1)[:, None]

            return ConvexHull(points + direction * grow)
        else:
            return ConvexHull(points)

    def get_bounding_box(self, z_first=None, z_last=None):
        """Return the bounding box of the ROI."""
        points = self.get_points()

        if z_first is not None and z_last is not None:
            lower_mask = points[:, 2] >= z_first
            upper_mask = points[:, 2] <= z_last

            combined_mask = numpy.logical_and(lower_mask, upper_mask)
            points = points[combined_mask]

        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]

        bounding_box = {}
        bounding_box["min"] = numpy.array([min(x_points), min(y_points), min(z_points)])
        bounding_box["max"] = numpy.array([max(x_points), max(y_points), max(z_points)])

        return bounding_box

    # ASK
    def get_contour_slices(self, image_set):
        """
        Return contour points segmented by image_set slices.

        :param image_set: Coordinate system of images on top
        of which the contours are overlayed.
        """
        if hasattr(self, "contours"):
            return self.contours
        else:
            roi_dict = {}
            roi = self.get_roi()
            if "ContourSequence" in roi:
                for contour in roi.ContourSequence:
                    slice_number = image_set.slice_from_z(contour.ContourData[2])
                    if slice_number is not None and slice_number not in roi_dict:
                        roi_dict[slice_number] = []
                    roi_dict[slice_number].append(numpy.array(contour.ContourData).tolist())

            self.contours = roi_dict
            return roi_dict

    def get_boolean_slices(self, as_path=False):
        """
        Return contour points optimised for performing boolean operations.

        :param as_path: If True, returns points as a Path object.
        """
        if hasattr(self, "boolean_slices"):
            return self.boolean_slices
        else:
            assert(hasattr(self, "contours"))

            vertices = {}
            for ct_slice in self.contours:
                intslice = int(ct_slice)
                contours = self.contours[ct_slice]
                vertices[intslice] = []
                for contour in contours:
                    cur_contour = [pos[0:2] for pos in zip(*([iter(contour)] * 3))]
                    if as_path:
                        vertices[intslice].append(Path(cur_contour))
                    else:
                        vertices[intslice].append(cur_contour)

            self.boolean_slices = vertices
            return vertices

    def get_mask_slices(self, coordinates):
        """
        Return contour points optimised for calculating structure masks.
        Note: This uses nearest neighbor z_slice 

        :param coordinates: Coordinate system of images on top
        of which the contours are overlayed.
        - coordinates should be modified to be the same coordinate system as
        the structure
        """
        roi_dict = {}
        roi = self.get_roi()
        if "ContourSequence" in roi:
            for contour in roi.ContourSequence:
                if "ContourData" in contour:
                    z_value = contour.ContourData[2]
                    contour_data = Path(numpy.array([pos[0:2] for pos in zip(*([iter(contour.ContourData)] * 3))]))
                    # Using float as dict key is okay here because they'll just be iterated over.
                    if z_value not in roi_dict:
                        roi_dict[z_value] = []
                    roi_dict[z_value].append(contour_data)

        slice_dict = {}
        z_positions = sorted(roi_dict.keys())
        ar_z_positions = numpy.array(z_positions)
        structure_spacing = ar_z_positions[1] - ar_z_positions[0]

        for vox in range(coordinates.num_voxels[2]):
            slice_dict[vox] = []
            z_pos = coordinates.slice_coordinates[vox]
            idx = find_nearest(ar_z_positions, z_pos)
            if idx is not None:
                ar_z_pos = ar_z_positions[idx]
                if coordinates.spacing[2] >= structure_spacing:
                    if numpy.abs(ar_z_pos - z_pos) <= 0.5 * coordinates.spacing[2]:
                        slice_dict[vox].append(roi_dict[z_positions[idx]])
                else:
                    if numpy.abs(ar_z_pos - z_pos) <= 0.5 * structure_spacing:
                        slice_dict[vox].append(roi_dict[z_positions[idx]])
        return slice_dict

    def get_slices_from_mask(self, coords, mask, new_coords):
        """
            Return contour points optimized for calculating structure masks from
            a boolean mask that represents a contour.

            :param mask: matrix of booleans that represent a filled contour
            :param coords: Coordinate system of the masked contour
            :param new_coords: Coordinate system of images on top of which the
            contours are overlayed
        """
        roi_dict = {}
        for slice, i in enumerate(mask):
            if numpy.any(i):
                xy = find_contours(i,1) # double check for contours with holes
                x_pos = xy[:,0] * coords.spacing[0] * coords.orient[0] + coords.img_pos[0]
                y_pos = xy[:,1] * coords.spacing[1] * coords.orient[1] + coords.img_pos[1]
                z_pos = np.repeat(coords.slice_coordinates[i], x_pos.shape[0])
                contour_data = Path(numpy.array([pos[0:2] for i in zip(x_pos, y_pos, z_pos)]))

                if z_value not in roi_dict:
                    roi_dict[z_value] = []
                roi_dict[z_value].append(contour_data)

        slice_dict = {}
        z_positions = sorted(roi_dict.keys())
        ar_z_positions = numpy.array(z_positions)
        structure_spacing = ar_z_positions[1] - ar_z_positions[0]

        for vox in range(new_coords.num_voxels[2]):
            slice_dict[vox] = []
            z_pos = new_coords.slice_coordinates[vox]
            idx = find_nearest(ar_z_positions, z_pos)
            if idx is not None:
                ar_z_pos = ar_z_positions[idx]
                if new_coords.spacing[2] >= structure_spacing:
                    if numpy.abs(ar_z_pos - z_pos) <= 0.5 * new_coords.spacing[2]:
                        slice_dict[vox].append(roi_dict[z_positions[idx]])
                else:
                    if numpy.abs(ar_z_pos - z_pos) <= 0.5 * structure_spacing:
                        slice_dict[vox].append(roi_dict[z_positions[idx]])
        return slice_dict

    def get_ct_mask_slices(self, ct):
        """
        Return contour points optimised for calculating structure masks.

        :param coordinates: Coordinate system of images on top
        of which the contours are overlayed.
        """
        roi_dict = {}
        roi = self.get_roi()
        if "ContourSequence" in roi:
            for contour in roi.ContourSequence:
                ref_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
                if "ContourData" in contour:
                    contour_data = Path(numpy.array([pos[0:2] for pos in zip(*([iter(contour.ContourData)] * 3))]))

                    if ref_uid not in roi_dict:
                        roi_dict[ref_uid] = []
                    roi_dict[ref_uid].append(contour_data)

        slice_dict = {}
        for slice_uid, contours in roi_dict.iteritems():
            slice_number = ct.uids.index(slice_uid)

            if slice_number is not None:
                if slice_number not in slice_dict:
                    slice_dict[slice_number] = []
                slice_dict[slice_number].append(contours)

        return slice_dict

    def get_mask(self, coordinates, slice_data=None, phantom=False):
        """
        Return a boolean mask for voxels included inside the contour.

        :param coordinates: Coordinate system of images on top
        of which the contours are overlayed
        :param  slice_data: If slice data has already been found.
        - Adds a utility option for if the previous method was used. eg. get_slices_from_mask()
        could be explicity called
        :param phantom: If True, specifies that the mask if being generated
        for a phantom file and not for the original CT
        """
        slice_data = self.get_mask_slices(coordinates) if not slice_data else slice_data
        positions = coordinates.get_phantom_voxel_positions() if phantom else coordinates.get_voxel_positions()

        voxels = coordinates.num_voxels
        grid_mask = numpy.zeros((voxels[2], voxels[1], voxels[0]), dtype=numpy.bool)

        # We only want to XOR contours that came from the same z value,
        # even if they were assigned to the same slice.
        for slice_num, path_list in slice_data.iteritems():
            if path_list:
                mask = numpy.zeros(voxels[1] * voxels[0], dtype=numpy.bool)
                for path in path_list:
                    local_mask = numpy.zeros(voxels[1] * voxels[0], dtype=numpy.bool)
                    for contour in path:
                        contour_mask = contour.contains_points(positions)
                        local_mask = numpy.logical_xor(local_mask, contour_mask)
                    mask = numpy.logical_or(mask, local_mask)

                grid_mask[slice_num] = mask.reshape((voxels[1], voxels[0]))

        if coordinates.slice_flipped:
            grid_mask = grid_mask[::-1]

        return grid_mask

    def get_ct_mask(self, ct, phantom=False):
        """
        Return a structure mask on a CT dataset. This situation is treated differently
        than the general get_mask method because each structure contour has an associated
        CT slice so we can obtain a mask without doing any coordinate-to-slice-number math.
        """
        slice_data = self.get_ct_mask_slices(ct)

        positions = ct.coords.get_voxel_positions()

        ct_voxels = ct.coords.num_voxels
        grid_mask = numpy.zeros((ct_voxels[2], ct_voxels[1], ct_voxels[0]), dtype=numpy.bool)

        for slice_num, paths in slice_data.iteritems():
            for path in paths:
                for contour in path: #path class has contains_points
                    contour_mask = contour.contains_points(positions).reshape((ct_voxels[1], ct_voxels[0]))
                    grid_mask[slice_num] = numpy.logical_xor(grid_mask[slice_num], contour_mask)

        return grid_mask

    def boolean_and(self, path, clip_path):
        """
        Boolean AND operation.

        :param path: Single contour slice of first ROI.
        :param clip_path: Single contour slice of clipping ROI.
        """
        pc = pyclipper.Pyclipper()
        if len(path) == 1:
            pc.AddPath(scale_to_clipper(path[0]), pyclipper.PT_SUBJECT, True)
        else:
            pc.AddPaths(scale_to_clipper(path), pyclipper.PT_SUBJECT, True)

        if len(clip_path) == 1:
            pc.AddPath(scale_to_clipper(clip_path[0]), pyclipper.PT_CLIP, True)
        else:
            pc.AddPaths(scale_to_clipper(clip_path), pyclipper.PT_CLIP, True)

        solution = scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
        return solution

    def boolean_or(self, path, clip_path):
        """
        Boolean OR operation.

        :param path: Single contour slice of first ROI.
        :param clip_path: Single contour slice of clipping ROI.
        """
        pc = pyclipper.Pyclipper()
        if len(path) == 1:
            pc.AddPath(scale_to_clipper(path[0]), pyclipper.PT_SUBJECT, True)
        else:
            pc.AddPaths(scale_to_clipper(path), pyclipper.PT_SUBJECT, True)

        if len(clip_path) == 1:
            pc.AddPath(scale_to_clipper(clip_path[0]), pyclipper.PT_CLIP, True)
        else:
            pc.AddPaths(scale_to_clipper(clip_path), pyclipper.PT_CLIP, True)

        solution = scale_from_clipper(pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
        return solution

    def boolean_not(self, path, clip_path):
        """
        Boolean NOT operation.

        :param path: Single contour slice of first ROI.
        :param clip_path: Single contour slice of clipping ROI.
        """
        pc = pyclipper.Pyclipper()
        if len(path) == 1:
            pc.AddPath(scale_to_clipper(path[0]), pyclipper.PT_SUBJECT, True)
        else:
            pc.AddPaths(scale_to_clipper(path), pyclipper.PT_SUBJECT, True)

        if len(clip_path) == 1:
            pc.AddPath(scale_to_clipper(clip_path[0]), pyclipper.PT_CLIP, True)
        else:
            pc.AddPaths(scale_to_clipper(clip_path), pyclipper.PT_CLIP, True)

        solution = scale_from_clipper(pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
        return solution

    def boolean_xor(self, path, clip_path):
        """
        Boolean XOR operation.

        :param path: Single contour slice of first ROI.
        :param clip_path: Single contour slice of clipping ROI.
        """
        pc = pyclipper.Pyclipper()
        if len(path) == 1:
            pc.AddPath(scale_to_clipper(path[0]), pyclipper.PT_SUBJECT, True)
        else:
            pc.AddPaths(scale_to_clipper(path), pyclipper.PT_SUBJECT, True)

        if len(clip_path) == 1:
            pc.AddPath(scale_to_clipper(clip_path[0]), pyclipper.PT_CLIP, True)
        else:
            pc.AddPaths(scale_to_clipper(clip_path), pyclipper.PT_CLIP, True)

        solution = scale_from_clipper(pc.Execute(pyclipper.CT_XOR, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD))
        return solution

    def boolean_with(self, other_structure, operation):
        """
        Boolean operation between two structures.

        :param other_structure: Clipping structure for boolean operation.
        :param operation: Type of boolean operation requested (AND, OR, NOT, XOR).
        """
        if not hasattr(self, "boolean_slices"):
            self.get_boolean_slices()

        new_contours = {}
        if operation == "AND":
            for contour_slice, paths in self.boolean_slices.iteritems():
                if contour_slice in other_structure.boolean_slices:
                    clipped_contours = []
                    other_paths = other_structure.boolean_slices[contour_slice]
                    clipped_contours = self.boolean_and(paths, other_paths)
                    if len(clipped_contours) > 0:
                        new_contours[contour_slice] = clipped_contours

        elif operation == "OR":
            subject_slices = set(self.boolean_slices.keys())
            clip_slices = set(other_structure.boolean_slices.keys())
            all_slices = list(subject_slices.union(clip_slices))

            for cslice in all_slices:
                if cslice in subject_slices and cslice in clip_slices:
                    paths = self.boolean_slices[cslice]
                    other_paths = other_structure.boolean_slices[cslice]
                    clipped_contours = self.boolean_or(paths, other_paths)

                    if len(clipped_contours) > 0:
                        new_contours[cslice] = clipped_contours

                elif cslice in subject_slices and cslice not in clip_slices:
                    new_contours[cslice] = self.boolean_slices[cslice]
                else:
                    new_contours[cslice] = other_structure.boolean_slices[cslice]

        elif operation == "NOT":
            subject_slices = set(self.boolean_slices.keys())
            clip_slices = set(other_structure.boolean_slices.keys())
            all_slices = list(subject_slices.union(clip_slices))

            for cslice in all_slices:
                if cslice in subject_slices and cslice in clip_slices:
                    paths = self.boolean_slices[cslice]
                    other_paths = other_structure.boolean_slices[cslice]
                    clipped_contours = self.boolean_not(paths, other_paths)
                    if len(clipped_contours) > 0:
                        new_contours[cslice] = clipped_contours

                elif cslice in subject_slices and cslice not in clip_slices:
                    new_contours[cslice] = self.boolean_slices[cslice]

        elif operation == "XOR":
            subject_slices = set(self.boolean_slices.keys())
            clip_slices = set(other_structure.boolean_slices.keys())
            all_slices = list(subject_slices.union(clip_slices))

            for cslice in all_slices:
                if cslice in subject_slices and cslice in clip_slices:
                    paths = self.boolean_slices[cslice]
                    clipped_contours = []
                    other_paths = other_structure.boolean_slices[cslice]
                    clipped_contours = self.boolean_xor(paths, other_paths)
                    if len(clipped_contours) > 0:
                        new_contours[cslice] = clipped_contours

                elif cslice in subject_slices and cslice not in clip_slices:
                    new_contours[cslice] = self.boolean_slices[cslice]
                else:
                    new_contours[cslice] = other_structure.boolean_slices[cslice]

        return new_contours

    def offset(self, offset_value):
        """
        Offset (grow, shrink) a contour.

        :param offset_value: size of offset.
        """
        new_contours = {}
        boolean_slices = self.get_boolean_slices()
        pco = pyclipper.PyclipperOffset()

        for contour_slice, paths in boolean_slices.iteritems():
            if len(paths) == 1:
                pco.AddPath(scale_to_clipper(paths[0], 10), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            else:
                pco.AddPaths(scale_to_clipper(paths, 10), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            solution = scale_from_clipper(pco.Execute(offset_value * 10), 10)
            new_contours[contour_slice] = solution
            pco.Clear()

        return new_contours
