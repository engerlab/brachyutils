import math
from ..utils import dicom_to_spherical


class KVATTrajectory(object):
    """
        Attributes:
        sad (float, mm): SAD
        ptv (Structure instance): Structure instance of PTV.
        energy (int): Nominal energy of beam (MV).
        couch_angle (float, deg): Angle of treatment couch.

        start_angle (float, deg): Start angle of arc.
        end_angle (float, deg): End angle of arc.
        segment_size (float, deg): Distance between control points.
    """

    sad = 1000.0
    couch_angle = 0
    col_angle = 0
    energy = 200.0
    start_angle = 0
    end_angle = 360
    target_size = 4
    model = "breast"
    override_iso = False
    segment_size = 2.0  # degrees

    model_data = {
        "breast": {
            "num_beamlets": {1: 27, 2: 27, 3: 31, 4: 27},
            "dsource": 300.0,
            "sad": 608.952
        },
        "prostate": {
            "num_beamlets": {1: 27, 2: 27, 3: 27, 4: 27},
            "dsource": 300.0,
            "sad": 608.952
        },
        "lung": {
            "num_beamlets": {1: 31, 2: 31, 3: 31, 4: 29},
            "dsource": 300.0,
            "sad": 608.952
        },
        "phantom": {
            "num_beamlets": 9,
            "dsource": 300.0,
            "sad": 608.952
        },
        "cylinder": {
            "num_beamlets": 1,
            "dsource": 300.0,
            "sad": 608.952
        },
        "PrecRT": {
            "num_beamlets": {4: 70},
            "dsource": 400.0,
            "sad": 493.90
        }
    }


    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

        self.couch_angle = float(self.couch_angle)
        self.col_angle = float(self.col_angle)

        self.start_angle = float(self.start_angle)
        self.end_angle = float(self.end_angle)
        self.segment_size = float(self.segment_size)

    def create_control_points(self, cpts=None):
        num_cpt = int((self.end_angle - self.start_angle) / float(self.segment_size))

        cpt_list = []
        for i in range(num_cpt):
            cpt_dict = {}
            cpt_dict["index"] = i
            cpt_dict["ptv"] = self.ptv.roi_num

            if self.override_iso is not False:
                cpt_dict["iso"] = [float(x) for x in self.iso]
            else:
                cpt_dict["iso"] = self.ptv.get_centroid()

            cpt_dict["energy"] = int(self.energy)

            cpt_dict["gantry_angle"] = self.start_angle + (i + 0.5) * self.segment_size
            cpt_dict["couch_angle"] = self.couch_angle
            cpt_dict["col_angle"] = self.col_angle

            theta, phi, phicol = dicom_to_spherical(cpt_dict["gantry_angle"],
                                                    cpt_dict["couch_angle"],
                                                    cpt_dict["col_angle"])
            cpt_dict["theta"] = math.degrees(theta)
            cpt_dict["phi"] = math.degrees(phi)
            cpt_dict["phicol"] = math.degrees(phicol)

            cpt_dict["target_size"] = int(self.target_size)
            cpt_dict["dsource"] = self.model_data[self.model]["dsource"]
            cpt_dict["sad"] = self.model_data[self.model]["sad"]

            if self.model == "breast" or self.model == "lung" or self.model == "prostate" or self.model == "PrecRT":
                cpt_dict["beamlet_columns"] = self.model_data[self.model]["num_beamlets"][int(self.target_size)]
            elif self.model == "cylinder":
                cpt_dict["beamlet_columns"] = self.model_data[self.model]["num_beamlets"]
            else:
                cpt_dict["beamlet_columns"] = self.model_data[self.model]["num_beamlets"]

            cpt_dict["model"] = self.model

            cpt_list.append(cpt_dict)

        self.control_points = cpt_list

        return cpt_list
