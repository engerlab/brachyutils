import math
import numpy
from ..utils import dicom_to_spherical


class BaseballStitchTrajectory(object):
    """
        Attributes:
        sad (float, mm): SAD
        ptv (Structure instance): Structure instance of PTV.
        d_source (float, mm): Distance from radiation source to isocenter.
        d_coll (float, mm): Distance between radiation source and MLC.
        energy (int): Nominal energy of beam (MV).
        couch_angle (float, deg): Angle of treatment couch.
        start_angle (float, deg): Start angle of arc.
        end_angle (float, deg): End angle of arc.
        segment_size (float, deg): Angle spanned by one control point.
    """

    defaults = {
        "iso_col_size": 10,
        "iso_row_size": 10,
        "d_source": 1000.0,
        "sad": 1000.0,
        "d_coll": 510.0,  # temporary, from TrueBeam STX
        "col_angle": 0,
        "energy": 6,
        "segment_size": 2.0  # degrees
    }

    def __init__(self, attrs=None):
        for k, v in self.defaults.iteritems():
            setattr(self, k, v)

        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

        self.col_angle = float(self.col_angle)
        self.sad = float(self.sad)
        self.iso_col_size = float(self.iso_col_size)
        self.iso_row_size = float(self.iso_row_size)

    def create_control_points(self, cpts=None):
        fac = 1

        gantry_angles = [-180 + 2 * fac * x for x in range(0, 180 / fac + 1)]
        couch_angles = [(90 - fac * x) % 360 for x in range(0, 180 / fac + 1)]

        idx = 0
        cpt_list = []

        for gantry_angle, couch_angle in zip(gantry_angles, couch_angles):
            cpt_dict = {}
            cpt_dict["index"] = idx
            cpt_dict["ptv"] = self.ptv.roi_name
            cpt_dict["energy"] = int(self.energy)

            cpt_dict["sad"] = float(self.sad)
            cpt_dict["d_coll"] = float(self.d_coll)
            cpt_dict["d_source"] = float(self.d_source)

            cpt_dict["iso_row_size"] = self.iso_row_size
            cpt_dict["iso_col_size"] = self.iso_col_size

            cpt_dict["arclength_scaling"] = float(self.segment_size)

            cpt_dict["gantry_angle"] = gantry_angle
            cpt_dict["couch_angle"] = couch_angle
            cpt_dict["col_angle"] = self.col_angle

            theta, phi, phicol = dicom_to_spherical(cpt_dict["gantry_angle"],
                                                    cpt_dict["couch_angle"],
                                                    cpt_dict["col_angle"])
            cpt_dict["theta"] = math.degrees(theta)
            cpt_dict["phi"] = math.degrees(phi)
            cpt_dict["phicol"] = math.degrees(phicol)

            x_dir = math.cos(phi) * math.sin(theta)
            y_dir = math.sin(phi) * math.sin(theta)
            z_dir = math.cos(theta)

            iso = self.ptv.get_centroid()

            direction = -numpy.array([x_dir, y_dir, z_dir])

            real_iso = numpy.array(iso) + direction * (1000.0 - self.sad)

            cpt_dict["iso"] = real_iso.tolist()

            cpt_list.append(cpt_dict)
            idx += 1

        self.control_points = cpt_list

        return cpt_list
