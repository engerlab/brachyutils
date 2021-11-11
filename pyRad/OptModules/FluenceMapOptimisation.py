import os
import json


class FluenceMapOptimisation(object):
    """
        Attributes:
        name (string)
        roi_rules (list of ROIRule instances)
        beam_paths (list of full paths to beamlet doses)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "sparse_fluence_map"
        opt_dict["name"] = self.name
        opt_dict["structures"] = self.roi_rules
        opt_dict["filenames"] = self.beam_paths

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
