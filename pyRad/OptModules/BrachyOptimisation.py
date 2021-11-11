import os
import json

class BrachyOptimisation(object):
    """
        Attributes:
        name (string)
        beamlet_generation (BeamletCreation instance)
        roi_rules (list of ROIRule instances)
        dwells (list of DwellPosition instances)
        total_voxels (Integer, TEMPORARY)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "brachy"
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels
        opt_dict["structures"] = self.roi_rules
        opt_dict["dwells"] = []

        if hasattr(self, "max_apertures"):
            opt_dict["max_apertures"] = self.max_apertures

        if hasattr(self, "max_iterations"):
            opt_dict["max_iterations"] = self.max_iterations

        beamlet_name = self.beamlet_generation.name
        beamlet_folder = self.beamlet_generation.beamlet_path

        angle_increment = 0
        if self.n_angles > 0:
            angle_increment = 360.0 / self.n_angles
            shield_angles = [angle_increment * i for i in range(self.n_angles)]
        else:
            shield_angles = [0]

        for catheter_index, catheter in enumerate(self.dwells):
            for pos_index, dwell in enumerate(catheter["positions"]):
                for angle_index, angle in enumerate(shield_angles):
                    if self.n_angles > 0:
                        filename = "{}_{}_{}_{}.minidos".format(beamlet_name,
                                                                catheter_index,
                                                                pos_index,
                                                                angle_index)
                    else:
                        filename = "{}_{}_{}.minidos".format(beamlet_name,
                                                             catheter_index,
                                                             pos_index)

                    file_path = os.path.join(beamlet_folder, filename)
                    dwell_dict = {
                        "catheter_index": catheter_index,
                        "angle_index": angle_index,
                        "shield_angle": angle,
                        "position": [dwell["x"], dwell["y"], dwell["z"]],
                        "dose_filename": file_path
                    }

                    opt_dict["dwells"].append(dwell_dict)

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
