import os
import json

class KVATOptimisation(object):
    """
        Attributes:
        name (string)
        beamlet_generation (BeamletCreation instance)
        roi_rules (list of ROIRule instances)
        control_points (list of ControlPoint instances)
        total_voxels (Integer, TEMPORARY)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "kvat"
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels
        opt_dict["cpts"] = [cpt.as_dict() for cpt in self.control_points]
        opt_dict["structures"] = self.roi_rules

        if not hasattr(self.control_points[0], "beamlet_columns"):
            opt_dict["beamlet_columns"] = 9
        else:
            opt_dict["beamlet_columns"] = self.control_points[0].beamlet_columns

        if hasattr(self, "max_apertures"):
            opt_dict["max_apertures"] = self.max_apertures

        if hasattr(self, "beamlet_generation"):
            photon_beamlet_name = self.beamlet_generation.name
            photon_beamlet_folder = self.beamlet_generation.beamlet_path

        for cpt_index, cpt in enumerate(self.control_points):
            opt_dict["cpts"][cpt_index]["beamlets"] = []
            for b_index in range(opt_dict["beamlet_columns"]):
                b_filename = "%s_%ikeV_%.1f_%i.minidos" % (photon_beamlet_name, int(cpt.energy), cpt.gantry_angle, b_index)
                beamlet_path = os.path.join(photon_beamlet_folder, b_filename)
                opt_dict["cpts"][cpt_index]["beamlets"].append(beamlet_path)

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
