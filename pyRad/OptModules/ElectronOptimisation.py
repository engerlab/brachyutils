import os
import json

class ElectronOptimisation(object):
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

    def segment_by_cpt(self, filenames):
        cpt_filenames = {}
        for filename in filenames:
            # Schema: UID_CPT#_BEAMLET#_ENERGY.minidos
            # Example filename: UID_0_100_6MeV.minidos
            splitted = filename.split("_")
            cpt_num = int(splitted[1])
            if cpt_num not in cpt_filenames:
                cpt_filenames[cpt_num] = []
            cpt_filenames[cpt_num].append(int(splitted[2]))

        for cpt_num in cpt_filenames:
            # Sort by beamlet number within a control point
            cpt_filenames[cpt_num] = set(cpt_filenames[cpt_num])

        return cpt_filenames

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "electron"
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels
        opt_dict["control_points"] = [cpt.as_dict() for cpt in self.control_points]
        opt_dict["structures"] = self.roi_rules

        if hasattr(self, "max_apertures"):
            opt_dict["max_apertures"] = self.max_apertures

        beamlet_name = self.beamlet_generation.name
        beamlet_folder = self.beamlet_generation.beamlet_path
        segmented_filenames = self.segment_by_cpt(self.beam_paths)

        for cpt_index, cpt in enumerate(self.control_points):
            opt_dict["control_points"][cpt_index]["beamlets"] = []
            active_beamlets = segmented_filenames[cpt_index]
            for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                if beam_i in active_beamlets:
                    b_filename = "%s_%i_%i_%iMeV.minidos" % (beamlet_name, cpt_index, beam_i, cpt.energy)
                    beamlet_path = os.path.join(beamlet_folder, b_filename)
                else:
                    beamlet_path = "Inactive"

                opt_dict["control_points"][cpt_index]["beamlets"].append(beamlet_path)

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
