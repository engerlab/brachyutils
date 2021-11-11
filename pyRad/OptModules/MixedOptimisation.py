import os
import json
import numpy

class MixedOptimisation(object):
    """
        Attributes:
        name (string)
        beamlet_generation (BeamletCreation instance)
        roi_rules (list of ROIRule instances)
        electron_cpts (list of ControlPoint instances)
        photon_cpts (list of ControlPoint instances)
        total_voxels (Integer, TEMPORARY)
    """

    def __init__(self, attrs):
        self.opt_type = "mixed"
        for k, v in attrs.items():
            setattr(self, k, v)

    def segment_by_cpt(self, filenames):
        cpt_filenames = {}
        for filename in filenames:
            # Schema: UID_CPT#_BEAMLET#_ENERGY.bindos
            # Example filename: UID_0_100_6MeV.bindos
            splitted = filename.split("_")
            cpt_num = int(splitted[1])
            if cpt_num not in cpt_filenames:
                cpt_filenames[cpt_num] = {}
            cpt_filenames[cpt_num][int(splitted[2])] = filename

        return cpt_filenames

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = self.opt_type
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels
        opt_dict["electron_cpts"] = [cpt.as_dict() for cpt in self.electron_cpts]
        opt_dict["photon_cpts"] = [cpt.as_dict() for cpt in self.photon_cpts]
        opt_dict["structures"] = self.roi_rules

        if hasattr(self, "SA_after"):
            opt_dict["SA_after"] = self.SA_after

        if hasattr(self, "max_apertures"):
            opt_dict["max_apertures"] = self.max_apertures

        if hasattr(self, "mixing_scheme"):
            opt_dict["mixing_scheme"] = self.mixing_scheme

        if hasattr(self, "output_modalities"):
            opt_dict["output_modalities"] = self.output_modalities

        # Check if it has photon component
        if hasattr(self, "beamlet_generation"):
            p_name = self.beamlet_generation.name
            p_beamlet_folder = self.beamlet_generation.beamlet_path
            segmented_p_filenames = self.segment_by_cpt(self.beam_paths)

        # Check if it has electron component
        if hasattr(self, "electron_beamlets_name"):
            e_name = self.electron_beamlets_name
            e_beamlet_folder = self.electron_beamlets_folder
            segmented_e_filenames = self.segment_by_cpt(self.electron_beam_paths)

        for cpt_index, cpt in enumerate(self.electron_cpts):
            opt_dict["electron_cpts"][cpt_index]["beamlets"] = []
            active_beamlets = segmented_e_filenames[cpt_index]
            for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                if beam_i in active_beamlets:
                    b_filename = active_beamlets[beam_i]
                    beamlet_path = os.path.join(e_beamlet_folder, b_filename)
                else:
                    beamlet_path = "Inactive"

                opt_dict["electron_cpts"][cpt_index]["beamlets"].append(beamlet_path)


        for cpt_index, cpt in enumerate(self.photon_cpts):
            opt_dict["photon_cpts"][cpt_index]["beamlets"] = []
            active_beamlets = segmented_p_filenames[cpt_index]
            for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                if beam_i in active_beamlets:
                    b_filename = active_beamlets[beam_i]
                    beamlet_path = os.path.join(p_beamlet_folder, b_filename)
                else:
                    beamlet_path = "Inactive"

                opt_dict["photon_cpts"][cpt_index]["beamlets"].append(beamlet_path)

        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
