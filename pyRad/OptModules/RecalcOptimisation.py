import json


class RecalcOptimisation(object):
    """
        Attributes:
        name (string)
        beamlet_generation (BeamletCreation instance)
        roi_rules (list of ROIRule instances)
        beam_paths (list of full paths to aperture doses)
        total_voxels (Integer, TEMPORARY)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "recalc"
        opt_dict["name"] = self.name
        opt_dict["original_filename"] = self.original_dose
        opt_dict["aperture_sets"] = self.beamlet_sets
        opt_dict["structures"] = self.roi_rules
        opt_dict["control_points"] = getattr(self, "control_points", None)
        opt_dict["output_modalities"] = getattr(self, "output_modalities", None)

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
