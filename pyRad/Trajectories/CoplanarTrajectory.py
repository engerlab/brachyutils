import math
import numpy
from ..utils import dicom_to_spherical


class CoplanarTrajectory(object):
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
        "d_coll": 510.0,  # temporary, from CL21EX beam model
        "couch_angle": 0,
        "col_angle": 0,
        "energy": 6,
        "start_angle": 0,
        "end_angle": 360,
        "segment_size": 2.0,  # degrees
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

        for arc_index, arc in enumerate(cpts["arcs"]):
            num_cpt = int((float(arc["gantry_end"]) - float(arc["gantry_start"])) / float(arc["segment_size"]))

            for i in range(num_cpt):
                cpt_dict = {}
                cpt_dict["arc_index"] = arc_index
                cpt_dict["index"] = len(cpt_list)
                cpt_dict["ptv"] = arc["ptv"].roi_name
                cpt_dict["energy"] = int(arc["energy"])

                cpt_dict["sad"] = float(arc["sad"])
                cpt_dict["d_coll"] = float(self.d_coll)
                cpt_dict["d_source"] = float(self.d_source)

                cpt_dict["iso_row_size"] = float(self.iso_row_size)
                cpt_dict["iso_col_size"] = float(self.iso_col_size)

                cpt_dict["arclength_scaling"] = float(arc["segment_size"])

                cpt_dict["gantry_angle"] = float(arc["gantry_start"]) + (i + 0.5) * float(arc["segment_size"])
                cpt_dict["couch_angle"] = float(arc["couch_angle"])
                cpt_dict["col_angle"] = float(arc["col_angle"])

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
                    cpt_dict["iso"] = [float(x) for x in arc["iso"]]
                else:
                    iso = arc["ptv"].get_centroid()
                    real_iso = numpy.array(iso) + direction * (1000.0 - float(arc["sad"]))
                    cpt_dict["iso"] = real_iso.tolist()

                cpt_list.append(cpt_dict)

        self.control_points = cpt_list

        return cpt_list
