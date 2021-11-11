"""
StructureSet module.

Copyright Marc-Andre Renaud, 2017
"""
import json

import pydicom as dicom
import numpy
from pyRad.Structure import Structure


class BooleanOperation(object):
    """
    Boolean equation wrapper.

    A boolean equation has two members and an operation. A member can itself
    be a boolean equation.

    Example: ["PTV", "BODY", "AND"]
    Example 2: ["PTV", ["PROSTATE", "URETHRA", "XOR"], "AND"]

    """

    def __init__(self, subject, clip, operation):
        """
        Constructor.

        :param subject: Either ROI object or list representing a nested boolean operation.
        :param clip: Clipping ROI, either ROI obj or list representing a nested boolean operation.
        :param str operation: One of AND, OR, NOT, XOR.
        """
        if isinstance(subject, list):
            self.subject = BooleanOperation(subject[0], subject[1], subject[2]).consume()
        elif isinstance(subject, Structure):
            self.subject = subject

        if isinstance(clip, list):
            self.clip = BooleanOperation(clip[0], clip[1], clip[2]).consume()
        elif isinstance(clip, Structure):
            self.clip = clip

        self.operation = operation

    def consume(self):
        """Perform boolean operation."""
        return self.subject.boolean_with(self.clip, self.operation)


class StructureSet(object):
    """
    Wrapper for a whole DICOM RT Structure set.

    Has methods for retrieving and modifying contours.
    """

    def __init__(self, attrs):
        """
        Constructor.

        :param struct_path: Path to RT Structure DICOM file.
        :param ref_image_set: Referenced CT image.
        """
        self.struct_path = attrs["struct_path"]
        self.ref_image_set = attrs["ref_image_set"]

        if not hasattr(self, "structures"):
            self.structures = self._init_structures()

    def _init_structures(self):
        structures = []

        struct_file = dicom.read_file(self.struct_path, force=True)
        for index, roi in enumerate(struct_file.StructureSetROISequence):
            roi_dict = {
                "roi_name": roi.ROIName.decode("utf-8"),
                "roi_num": roi.ROINumber,
                "struct_path": self.struct_path,
                "frame_of_ref_uid": roi.ReferencedFrameOfReferenceUID
            }
            structures.append(Structure(roi_dict))

        return structures

    def get_struct_list(self):
        """Return list of structures with metadata."""
        struct_file = dicom.read_file(self.struct_path, force=True)
        num_roi = len(struct_file.StructureSetROISequence)
        struct_list = []
        for roi_i in range(num_roi):
            roi_name = struct_file.StructureSetROISequence[roi_i].ROIName
            # For some reason, not all ROIs are given colours.
            if "ROIDisplayColor" in struct_file.ROIContourSequence[roi_i]:
                roi_colour = [int(i) for i in struct_file.ROIContourSequence[roi_i].ROIDisplayColor]
            else:
                roi_colour = [0, 0, 255]

            my_struct = {
                "ROI_number": struct_file.StructureSetROISequence[roi_i].ROINumber,
                "name": roi_name,
                "colour": roi_colour
            }
            struct_list.append(my_struct)

        struct_list.sort(key=lambda roi: roi["name"])

        return struct_list

    def get_contour_dump(self):
        """Return list of structures with contour data included."""
        struct_list = self.get_struct_list()
        for roi in struct_list:
            roi_obj = self.get_roi_object(roi["ROI_number"])
            contours = roi_obj.get_contour_slices(self.ref_image_set)
            roi["contours"] = contours

        return struct_list

    def get_roi_object(self, roi):
        """
        Return Structure object from roi name or roi number.

        :param roi: ROI name if string, ROI number if int.
        """
        if isinstance(roi, basestring):
            for structure in self.structures:
                if structure.roi_name == roi:
                    return structure
        elif isinstance(roi, int):
            for structure in self.structures:
                if structure.roi_num == roi:
                    return structure
        else:
            raise Exception(type(roi))

        return None

    def _parse_boolean_equation(self, equation):
        """
        Preprocess boolean equation to contain the right information.

        A boolean equation has two members and an operation. A member can itself
        be a boolean equation.

        Example: ["PTV", "BODY", "AND"]
        Example 2: ["PTV", ["PROSTATE", "URETHRA", "XOR"], "AND"]

        :param equation: Boolean equation.
        """
        parse_depth = equation
        while isinstance(parse_depth, list):
            subject = parse_depth[0]
            if isinstance(subject, list):
                parse_depth = subject
            elif isinstance(subject, (basestring, int)):
                obj = self.get_roi_object(subject)
                obj.get_contour_slices(self.ref_image_set)
                obj.get_boolean_slices()
                parse_depth[0] = obj
                parse_depth = subject

        parse_depth = equation
        while isinstance(parse_depth, list):
            clip = parse_depth[1]
            if isinstance(clip, list):
                parse_depth = clip
            elif isinstance(clip, (basestring, int)):
                obj = self.get_roi_object(clip)
                obj.get_contour_slices(self.ref_image_set)
                obj.get_boolean_slices()
                parse_depth[1] = obj
                parse_depth = subject

        return equation

    def boolean_structures(self, name, equation, colour):
        """
        Perform nested boolean operation and create a new structure from it.

        The [R, G, B] colour values are between 0 and 255.

        :param name: Name of new structure.
        :param equation: Boolean equation.
        :param colour: [R, G, B] colour of new structure.
        """
        parsed_equation = self._parse_boolean_equation(equation)
        new_contours = BooleanOperation(parsed_equation[0],
                                        parsed_equation[1],
                                        parsed_equation[2]).consume()

        contours = self.to_dicom_contours(new_contours)

        new_roi = self.make_new_structure(name, colour, contours)

        return new_roi

    def offset_structure(self, ref_name, new_name, offset, colour):
        """
        Perform structure offset operation and create a new structure.

        A positive offset value will grow the structure, and a
        negative value will shrink it.

        :param ref_name: Name of structure to offset.
        :param name: Name of new structure.
        :param offset: Offset value in mm.
        :param colour: [R, G, B] colour of new structure.
        """
        roi_obj = self.get_roi_object(ref_name)
        roi_obj.get_contour_slices(self.ref_image_set)

        new_contours = roi_obj.offset(offset)
        contours = self.to_dicom_contours(new_contours)

        new_roi = self.make_new_structure(new_name, colour, contours)

        return new_roi

    def to_dicom_contours(self, data_slices):
        """
        Convert contours back to DICOM format after boolean operation.

        :param image_set: Image set on top of which contour is overlayed.
        """
        contours = {}
        for ct_slice, paths in data_slices.iteritems():
            ct_z = self.ref_image_set.slice_coordinates[ct_slice]
            contour = []
            for path in paths:
                unravelled = []
                for point in path:
                    unravelled += [point[0], point[1], ct_z]
                contour.append(unravelled)
            contours[ct_slice] = contour

        return contours


    def remove_structure(self, roi_name):
        """
        Remove a structure from an RT DICOM file.

        This step is irreversible.
        :param roi_name: Name of ROI to remove.
        """
        roi = self.get_roi_object(roi_name)
        if not roi:
            return self.get_struct_list()

        roi_num = roi.roi_num
        rs_file = dicom.read_file(self.struct_path, force=True)
        roi_index = roi.get_index()

        if rs_file.StructureSetROISequence[roi_index].ROIName != roi_name or rs_file.StructureSetROISequence[roi_index].ROINumber != roi_num:
            error = "%s does not match DICOM records: %s" % (roi_name, rs_file.StructureSetROISequence[roi_index].ROIName)
            raise Exception(error)

        del rs_file.ROIContourSequence[roi_index]
        del rs_file.RTROIObservationsSequence[roi_index]
        del rs_file.StructureSetROISequence[roi_index]

        new_order = []

        # Redo ROI numbers
        for index, roi_obs in enumerate(rs_file.RTROIObservationsSequence):
            roi_obs.ObservationNumber = index + 1
            roi_obs.ReferencedROINumber = index + 1

        for index, roi_info in enumerate(rs_file.StructureSetROISequence):
            roi_info.ROINumber = index + 1
            new_order.append({"name": roi_info.ROIName, "ROI_number": roi_info.ROINumber})

        for index, roi in enumerate(rs_file.ROIContourSequence):
            roi.ReferencedROINumber = rs_file.StructureSetROISequence[index].ROINumber

        rs_file.save_as(self.struct_path)

        return new_order

    def modify_structure(self, current_name, new_name, new_colour):
        """
        Change the name and colour of an ROI inside a DICOM file.

        :param current_name: Current ROI name.
        :param new_name: New ROI name.
        :param new_colour: New ROI colour.
        """
        rs_file = dicom.read_file(self.struct_path, force=True)
        num_roi = len(rs_file.StructureSetROISequence)
        for i in range(num_roi):
            roi_name = rs_file.StructureSetROISequence[i].ROIName
            if roi_name == current_name:
                rs_file.ROIContourSequence[i].ROIDisplayColor = new_colour
                rs_file.StructureSetROISequence[i].ROIName = new_name
                break

        rs_file.save_as(self.struct_path)
        return "success"

    def add_couch(self, couch_pos, couch_colour, couch_path):
        """
        Add couch contour to structure set.

        :param couch_pos: Couch position.
        :param couch_colour: Colour of couch contour.
        :param couch_path: Path of file containing contour points.
        """
        jsonfile = open(couch_path)
        couch_points = json.load(jsonfile)
        jsonfile.close()

        new_structures = []

        for contour in couch_points["contours"]:
            points_3d = numpy.array(zip(*([iter(contour["points"])]*3)))

            contours = {}
            for i, z_val in enumerate(self.ref_image_set.slice_coordinates):
                if i not in contours:
                    contours[i] = []

                translated_points = points_3d + numpy.array([float(couch_pos["x"]),
                                                             float(couch_pos["y"]),
                                                             float(z_val)])

                contours[i].append(translated_points.flatten().tolist())

            struct_dict = {
                "name": contour["name"],
                "colour": couch_colour,
                "contours": contours
            }

            new_struct = self.make_new_structure(contour["name"], couch_colour, contours)
            struct_dict["ROI_number"] = new_struct.roi_num

            new_structures.append(struct_dict)

        return new_structures

    def make_new_structure(self, name, colour, contours):
        """
        Create a new ROI inside a DICOM RT Structure file based on some processing.

        :param name: ROI name
        :param colour: RGB colour, e.g. [255, 255, 0]
        :param contours: Dict of contour data in DICOM format with slice number for keys.

        One slice can have multiple contours to accomodate disjointed ROIs.

        Example contours:
        {
            0: [
                   [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn],
                   [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn],
               ],
            1: [
                   [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn],
                   [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn],
               ],
            ...
        }
        """

        roi_info = dicom.dataset.Dataset()
        roi_info.ROIName = name
        roi_info.ROIGenerationAlgorithm = "MANUAL"
        roi_info.ROINumber = len(self.structures) + 1
        roi_info.ReferencedFrameOfReferenceUID = self.structures[0].frame_of_ref_uid

        roi_obs = dicom.dataset.Dataset()
        roi_obs.ObservationNumber = roi_info.ROINumber
        roi_obs.ReferencedROINumber = roi_info.ROINumber
        roi_obs.ROIObservationLabel = roi_info.ROIName
        roi_obs.RTROIInterpretedType = ''
        roi_obs.ROIInterpreter = ''

        roi_contours = dicom.dataset.Dataset()
        roi_contours.ROIDisplayColor = colour
        roi_contours.ContourSequence = dicom.sequence.Sequence()

        for _, contour_list in contours.iteritems():
            for contour in contour_list:
                roi_contour = dicom.dataset.Dataset()
                roi_contour.NumberOfContourPoints = len(contour)
                roi_contour.ContourData = contour
                roi_contour.ContourGeometricType = "CLOSED_PLANAR"

                roi_contours.ContourSequence.append(roi_contour)

        rs_file = dicom.read_file(self.struct_path, force=True)

        rs_file.RTROIObservationsSequence.append(roi_obs)
        rs_file.StructureSetROISequence.append(roi_info)
        rs_file.ROIContourSequence.append(roi_contours)

        rs_file.save_as(self.struct_path)

        new_structure = Structure({"roi_num": roi_info.ROINumber,
                                   "roi_name": name,
                                   "struct_path": self.struct_path,
                                   "contours": contours})

        self.structures.append(new_structure)

        return new_structure
