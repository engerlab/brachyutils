"""
ControlPoint module.

Copyright Marc-Andre Renaud, 2017
"""
import math

import numpy

import scipy.ndimage

from utils import dicom_to_spherical


class ControlPoint(object):
    """
    Specify the state of the linac at a given moment during treatment.

    All coordinates are in millimeters.

    Attributes:
    energy (float): Nominal energy of the beam for this control point.
    dsource (float): Distance from (virtual) radiation source to isocenter.
    sad (float): Distance from (real) radiation source to isocenter.
    d_coll (float): Distance between (real) radiation source and MLC.
    gantry_angle (float): Gantry angle (deg).
    couch_angle (float): Couch angle (deg).
    col_angle (float): Collimator angle (deg).
    iso (list): [x, y, z] coordinates of isocenter in the CT coordinate system.
    ptv (Structure instance): Structure object of the PTV.
    """

    def __init__(self, attrs=None):
        """Constructor."""
        if attrs is not None:
            for k, v in attrs.items():
                setattr(self, k, v)

        if isinstance(self.iso, list):
            self.iso = numpy.array(self.iso)

        if hasattr(self, "beamlet_mask"):
            self.beamlet_mask = numpy.array(self.beamlet_mask)

        if hasattr(self, "ptv"):
            self._preprocess()

        if hasattr(self, "iso_col_size") and hasattr(self, "iso_row_size") and hasattr(self, "d_coll"):
            self._preprocess_mlc()

        if not hasattr(self, "sad"):
            self.sad = 1000.0

    def as_dict(self):
        """Serialize everything to fit into a dictionary."""
        cpt_dict = {}
        for key, value in self.__dict__.iteritems():
            if not key.startswith("_"):
                if type(value) == numpy.ndarray:
                    cpt_dict[key] = value.tolist()
                else:
                    cpt_dict[key] = value

        if "ptv" in cpt_dict:
            cpt_dict["ptv"] = self.ptv.roi_num

        return cpt_dict

    def _preprocess(self):
        self._init_rot_matrix()
        self._determine_beam_position()

    def _preprocess_mlc(self):
        self._determine_mlc_beamlet_sizes()
        if not hasattr(self, "beamlet_rows") or not hasattr(self, "beamlet_columns"):
            self._determine_field_size()

    def _init_rot_matrix(self):
        theta, phi, phicol = dicom_to_spherical(self.gantry_angle,
                                                self.couch_angle,
                                                self.col_angle)

        ct = math.cos(theta)
        st = math.sin(theta)

        cp = math.cos(phi)
        sp = math.sin(phi)

        cc = math.cos(phicol)
        sc = math.sin(phicol)

        self.r1 = numpy.array([ct * cp * cc + sp * sc,
                                 -ct * cp * sc + sp * cc,
                                 -st * cp])

        self.r2 = numpy.array([ct * sp * cc - cp * sc,
                                 -ct * sp * sc - cp * cc,
                                 -st * sp])

        self.r3 = numpy.array([-st * cc, st * sc, -ct])

    def _determine_beam_position(self):
        unrotated = numpy.array([0, 0, -self.sad])
        self._beam_position = numpy.array([self.r1.dot(unrotated),
                                           self.r2.dot(unrotated),
                                           self.r3.dot(unrotated)])
        self._beam_position += self.iso

    def _determine_mlc_beamlet_sizes(self):
        # Beamlet size at the MLC is calculated by projecting the desired
        # beamlet size from isocenter to MLC plane.
        mlc_projection = float(self.d_coll) / self.sad
        self.mlc_row_size = self.iso_row_size * mlc_projection
        self.mlc_col_size = self.iso_col_size * mlc_projection

    def _determine_field_size(self):
        """
        Set the default field size.

        A raytracing step will be performed to identify which beamlets
        are actually activated.
        """
        self.beamlet_rows = int(400.0 / self.iso_row_size)
        self.beamlet_columns = int(400.0 / self.iso_col_size)

    def _determine_beamlet_direction(self, beamlet):
        dir_vector = beamlet - self._beam_position
        return dir_vector / numpy.sqrt(dir_vector.dot(dir_vector))

    def _rotate_beamlet(self, beamlet):
        return [self.r1.dot(beamlet),
                self.r2.dot(beamlet),
                self.r3.dot(beamlet)]

    def _alt_check_intersection(self, beamlet, ptv_convex):
        b_dir = self._determine_beamlet_direction(beamlet)
        # Intersection of ray R(t) = A + t * dir, with plane
        # defined by n dot X = d, where n is the plane's normal.
        # t = \frac{(Pt - A) dot n}{n_dot_dir} where Pt is a point
        # on the plane.

        points = (ptv_convex.equations[:, 0:3].T * -ptv_convex.equations[:, 3]).T
        interm = points - beamlet
        numerators = numpy.sum(interm * ptv_convex.equations[:, 0:3], axis=1)
        denominators = ptv_convex.equations[:, 0:3].dot(b_dir)

        positive = denominators > 0
        negative = denominators <= 0

        positive_ts = numerators[positive] / denominators[positive]
        negative_ts = numerators[negative] / denominators[negative]

        min_pos = positive_ts.min()
        max_neg = negative_ts.max()

        return not (min_pos < max_neg)

    def _check_intersection(self, beamlet, ptv_convex):
        max_te = 0
        min_tl = 999999
        b_dir = self._determine_beamlet_direction(beamlet)
        # Intersection of ray R(t) = A + t * dir, with plane
        # defined by n dot X = d, where n is the plane's normal.
        # t = \frac{(Pt - A) dot n}{n_dot_dir} where Pt is a point
        # on the plane.

        for plane in ptv_convex.equations:
            d = -plane[3]
            point = d * plane[:3]

            numerator = (point - beamlet).dot(plane[:3])
            denominator = plane[0] * b_dir[0] + plane[1] * b_dir[1] + plane[2] * b_dir[2]

            if denominator > 0:
                t = numerator / denominator
                # Leaving the polyhedron
                if t < min_tl:
                    min_tl = t
                    if min_tl < max_te:
                        return False
            else:
                t = numerator / denominator
                # Entering the polyhedron
                if t > max_te:
                    max_te = t
                    if max_te > min_tl:
                        return False

        return True

    def get_base_beamlets(self):
        try:
            return self.base_beamlets
        except AttributeError:
            x_min = -self.beamlet_columns / 2.0 * self.mlc_col_size
            x_max = self.beamlet_columns / 2.0 * self.mlc_col_size
            y_min = -self.beamlet_rows / 2.0 * self.mlc_row_size
            y_max = self.beamlet_rows / 2.0 * self.mlc_row_size

            x_positions = numpy.arange(x_min, x_max, self.mlc_col_size) + 0.5 * self.mlc_col_size
            y_positions = numpy.arange(y_min, y_max, self.mlc_row_size) + 0.5 * self.mlc_row_size

            position_grid = numpy.meshgrid(x_positions, y_positions)
            z_pos = self.sad - self.d_coll

            self.base_beamlets = numpy.array(zip(position_grid[0].flatten(),
                                                 position_grid[1].flatten(),
                                                 [-z_pos] * position_grid[0].size))

            return self.base_beamlets

    def get_iso_beamlets(self):
        try:
            return self.iso_beamlets
        except AttributeError:
            x_min = -self.beamlet_columns / 2.0 * self.iso_col_size
            x_max = self.beamlet_columns / 2.0 * self.iso_col_size
            y_min = -self.beamlet_rows / 2.0 * self.iso_row_size
            y_max = self.beamlet_rows / 2.0 * self.iso_row_size

            x_positions = numpy.arange(x_min, x_max, self.iso_col_size) + 0.5 * self.iso_col_size
            y_positions = numpy.arange(y_min, y_max, self.iso_row_size) + 0.5 * self.iso_row_size

            position_grid = numpy.meshgrid(x_positions, y_positions)
            z_pos = self.sad

            self.iso_beamlets = numpy.array(zip(position_grid[0].flatten(),
                                                 position_grid[1].flatten(),
                                                 [-z_pos] * position_grid[0].size))
            return self.iso_beamlets

    def get_rotated_beamlets(self):
        try:
            return self.rotated_beamlets
        except AttributeError:
            base_beamlets = self.get_base_beamlets()
            self.rotated_beamlets = numpy.array([self._rotate_beamlet(b) for b in base_beamlets])
            self.rotated_beamlets += self.iso
            return self.rotated_beamlets

    def prettyprint_mask(self):
        reshaped = self.beamlet_mask.reshape((self.beamlet_rows, self.beamlet_columns))
        for row in reshaped:
            row_string = "|"
            for column in row:
                if column:
                    row_string += " "
                else:
                    row_string += "="

            row_string += "|"
            print row_string

        print ""

    def output_mask(self, index="0"):
        filename = "cpt_{gantry}_{couch}_{col}_{index}.mask".format(gantry=self.gantry_angle, couch=self.couch_angle, col=self.col_angle, index=index)
        with open(filename, "w") as myfile:
            reshaped = self.beamlet_mask.reshape((self.beamlet_rows, self.beamlet_columns))
            for row in reshaped:
                row_string = "|"
                for column in row:
                    if column:
                        row_string += " "
                    else:
                        row_string += "="

                row_string += "|"
                myfile.write(row_string + "\n")

            myfile.write("\n")
            num_nonzero = numpy.count_nonzero(self.beamlet_mask)
            myfile.write("num beamlets: {}\n".format(num_nonzero))


    def bbox2(self, img):
        """Find the bounding box of a mask to have rectangular masks"""
        rows = numpy.any(img, axis=1)
        cols = numpy.any(img, axis=0)
        rmin, rmax = numpy.where(rows)[0][[0, -1]]
        cmin, cmax = numpy.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def get_beamlet_mask(self, bound=True):
        try:
            return self.beamlet_mask
        except AttributeError:
            ptv_convex = self.ptv.get_convex_hull()
            rotated_beamlets = self.get_rotated_beamlets()
            beamlet_mask = numpy.array([self._check_intersection(beamlet, ptv_convex) for beamlet in rotated_beamlets])

            # Add 1 beamlet's worth of padding to the mask to cover PTV properly.
            reshaped = beamlet_mask.reshape((self.beamlet_rows, self.beamlet_columns))
            beamlet_mask = scipy.ndimage.morphology.binary_dilation(reshaped).astype(beamlet_mask.dtype)

            if bound:
                (rmin, rmax, cmin, cmax) = self.bbox2(beamlet_mask)
                beamlet_mask[rmin:rmax+1,cmin:cmax+1] = True

            self.beamlet_mask = beamlet_mask.flatten()
            self.output_mask()

            return self.beamlet_mask
