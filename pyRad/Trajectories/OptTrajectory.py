import math
from ..utils import dicom_to_spherical

class OptTrajectory(object):
    """
        Attributes:
        d_source (float): Distance from radiation source to isocenter.
        d_coll (float): Distance between radiation source and MLC.
        plan (Plan obj): Plan object
    """

    defaults = {
        "iso_col_size": 2,
        "iso_row_size": 2,
        "d_source": 1000.0,
        "sad": 1000.0,
        "d_coll": 515.785,  # temporary, from CL21EX beam model
    }

    def __init__(self, attrs):
        for key in self.defaults:
            setattr(self, key, self.defaults[key])

        for k, v in attrs.items():
            setattr(self, k, v)

    def create_control_points(self, cpt_dicts):
        cpt_list = []
        cpt_index = 0
        photon_cpts = self.plan.get("photon_cpts", None)
        if photon_cpts is not None:
            for cpt in photon_cpts:
                cpt = cpt.copy()
                cpt["ptv"] = self.ptv.roi_name
                cpt["d_coll"] = self.d_coll
                theta, phi, phicol = dicom_to_spherical(cpt["gantry_angle"],
                                                        cpt["couch_angle"],
                                                        cpt["col_angle"])
                cpt["theta"] = math.degrees(theta)
                cpt["phi"] = math.degrees(phi)
                cpt["phicol"] = math.degrees(phicol)

                cpt_list.append(cpt)
                cpt_index += 1

        electron_cpts = self.plan.get("electron_cpts", None)
        if electron_cpts is not None:
            for cpt in electron_cpts:
                cpt = cpt.copy()
                cpt["ptv"] = self.ptv.roi_name
                cpt["d_coll"] = self.d_coll
                theta, phi, phicol = dicom_to_spherical(cpt["gantry_angle"],
                                                        cpt["couch_angle"],
                                                        cpt["col_angle"])
                cpt["theta"] = math.degrees(theta)
                cpt["phi"] = math.degrees(phi)
                cpt["phicol"] = math.degrees(phicol)

                cpt_list.append(cpt)
                cpt_index += 1

        self.control_points = cpt_list

        return cpt_list
