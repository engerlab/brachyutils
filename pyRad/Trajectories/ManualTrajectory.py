import math
import numpy

from ..utils import dicom_to_spherical


class ManualTrajectory(object):
    """
        Attributes:
        d_source (float): Distance from radiation source to isocenter.
        d_coll (float): Distance between radiation source and MLC.
    """

    defaults = {
        "iso_col_size": 10,
        "iso_row_size": 10,
        "d_source": 1000.0,
        "sad": 1000.0,
        "d_coll": 504.5,  # temporary, from CL21EX beam model
        "override_iso": False
    }

    def __init__(self, attrs=None):
        for k, v in self.defaults.iteritems():
            setattr(self, k, v)

        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def create_control_points(self, cpt_dicts):
        cpt_list = []
        for index, cpt in enumerate(cpt_dicts["cpts"]):
            cpt_dict = {}
            cpt_dict["index"] = index
            cpt_dict["ptv"] = cpt["ptv"].roi_name
            cpt_dict["sad"] = float(self.sad)
            cpt_dict["d_coll"] = float(self.d_coll)
            cpt_dict["d_source"] = float(self.d_source)

            cpt_dict["iso_row_size"] = float(self.iso_row_size)
            cpt_dict["iso_col_size"] = float(self.iso_col_size)

            cpt_dict["energy"] = int(cpt["energy"])
            cpt_dict["col_angle"] = float(cpt["col_angle"])
            cpt_dict["gantry_angle"] = float(cpt["gantry_angle"])
            cpt_dict["couch_angle"] = float(cpt["couch_angle"])

            theta, phi, phicol = dicom_to_spherical(cpt_dict["gantry_angle"],
                                                    cpt_dict["couch_angle"],
                                                    cpt_dict["col_angle"])
            cpt_dict["theta"] = math.degrees(theta)
            cpt_dict["phi"] = math.degrees(phi)
            cpt_dict["phicol"] = math.degrees(phicol)

            x_dir = math.cos(phi) * math.sin(theta)
            y_dir = math.sin(phi) * math.sin(theta)
            z_dir = math.cos(theta)

            direction = -numpy.array([x_dir, y_dir, z_dir])

            if self.override_iso is True:
                cpt_dict["iso"] = [float(x) for x in cpt["iso"]]
            else:
                iso = cpt["ptv"].get_centroid()
                real_iso = numpy.array(iso) + direction * (1000.0 - float(cpt["sad"]))
                cpt_dict["iso"] = real_iso.tolist()

            cpt_list.append(cpt_dict)

        self.control_points = cpt_list

        return cpt_list
