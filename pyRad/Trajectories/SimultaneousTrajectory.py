import math
import numpy
from ..utils import dicom_to_spherical


class SimultaneousTrajectory(object):
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
        "gantry_spacing": 10.0,
        "couch_spacing": 5.0,
        "iso_col_size": 10,
        "iso_row_size": 10,
        "d_source": 1000.0,
        "sad": 1000.0,
        "d_coll": 510.0,  # temporary, from CL21EX beam model
        "couch_angle": 0,
        "col_angle": 0,
        "energy": 6,
        "gantry_start": 0,
        "end_gantry": 360,
        "couch_start": -90,
        "end_couch": 90,
        "override_iso": False
    }

    def __init__(self, attrs=None):
        for k, v in self.defaults.iteritems():
            setattr(self, k, v)

        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def create_control_points(self, cpts=None):
        cpt_list = []

        self.gantry_start = float(self.gantry_start)
        self.gantry_end = float(self.gantry_end)
        self.gantry_spacing = float(self.gantry_spacing)
        self.couch_spacing = float(self.couch_spacing)
        self.couch_start = float(self.couch_start)
        self.couch_end = float(self.couch_end)

        num_gantries = int((self.gantry_end - self.gantry_start) / self.gantry_spacing) + 1
        num_couches = int((self.couch_end - self.couch_start) / self.couch_spacing) + 1

        idx = 0
        for gant_i in range(num_gantries):
            for couch_i in range(num_couches):
                cpt_dict = {}
                cpt_dict["arc_index"] = 0
                cpt_dict["index"] = idx
                cpt_dict["ptv"] = self.ptv.roi_name
                cpt_dict["energy"] = self.energy

                cpt_dict["sad"] = self.sad
                cpt_dict["d_coll"] = float(self.d_coll)
                cpt_dict["d_source"] = float(self.d_source)
                cpt_dict["joel"] = True

                cpt_dict["iso_row_size"] = float(self.iso_row_size)
                cpt_dict["iso_col_size"] = float(self.iso_col_size)

                cpt_dict["gantry_angle"] = self.gantry_start + gant_i * self.gantry_spacing
                cpt_dict["couch_angle"] = self.couch_start + couch_i * self.couch_spacing
                cpt_dict["col_angle"] = float(self.col_angle)

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
                    cpt_dict["iso"] = [float(x) for x in self.iso]
                else:
                    iso = self.ptv.get_centroid()
                    real_iso = numpy.array(iso) + direction * (1000.0 - float(self.sad))
                    cpt_dict["iso"] = real_iso.tolist()

                cpt_list.append(cpt_dict)
                idx += 1

            self.control_points = cpt_list

        return cpt_list
