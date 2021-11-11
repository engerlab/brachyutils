import os
import json
import numpy

class RobustOptimisation(object):
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
            # Example filename: {UID}_{cpt_num}_{b_num}_6MeV_{sc_num}.bindos
            splitted = filename.split("_")
            cpt_num = int(splitted[1])
            beamlet_num = int(splitted[2])
            scenario_num = int(splitted[-1].strip(".bindos"))
            if cpt_num not in cpt_filenames:
                cpt_filenames[cpt_num] = {}

            if scenario_num not in cpt_filenames[cpt_num]:
                cpt_filenames[cpt_num][scenario_num] = {}

            cpt_filenames[cpt_num][scenario_num][beamlet_num] = filename

        return cpt_filenames

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = self.opt_type
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels
        opt_dict["electron_cpts"] = [cpt.as_dict() for cpt in self.electron_cpts]
        opt_dict["photon_cpts"] = [cpt.as_dict() for cpt in self.photon_cpts]
        opt_dict["structures"] = self.roi_rules

        # HARDCODED FOR NOW
        opt_dict["num_scenarios"] = 7

        if hasattr(self, "scenario_weights"):
            opt_dict["scenario_weights"] = self.scenario_weights

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
            active_beamlets = segmented_e_filenames[cpt_index][0]
            for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                if beam_i in active_beamlets:
                    b_filename = active_beamlets[beam_i]
                    beamlet_path = os.path.join(e_beamlet_folder, b_filename)
                else:
                    beamlet_path = "Inactive"

                opt_dict["electron_cpts"][cpt_index]["beamlets"].append(beamlet_path)

            opt_dict["electron_cpts"][cpt_index]["robust_beamlets"] = []
            for sc_id, beamlets in segmented_e_filenames[cpt_index].iteritems():
                if sc_id != 0:
                    robust_beamlets = []
                    for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                        if beam_i in beamlets:
                            b_filename = beamlets[beam_i]
                            beamlet_path = os.path.join(e_beamlet_folder, b_filename)
                        else:
                            beamlet_path = "Inactive"

                        robust_beamlets.append(beamlet_path)

                    opt_dict["electron_cpts"][cpt_index]["robust_beamlets"].append(robust_beamlets)


        for cpt_index, cpt in enumerate(self.photon_cpts):
            opt_dict["photon_cpts"][cpt_index]["beamlets"] = []
            active_beamlets = segmented_p_filenames[cpt_index][0]
            for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                if beam_i in active_beamlets:
                    b_filename = active_beamlets[beam_i]
                    beamlet_path = os.path.join(p_beamlet_folder, b_filename)
                else:
                    beamlet_path = "Inactive"

                opt_dict["photon_cpts"][cpt_index]["beamlets"].append(beamlet_path)

            opt_dict["photon_cpts"][cpt_index]["robust_beamlets"] = []
            for sc_id, beamlets in segmented_p_filenames[cpt_index].iteritems():
                if sc_id != 0:
                    robust_beamlets = []
                    for beam_i in range(cpt.beamlet_rows * cpt.beamlet_columns):
                        if beam_i in beamlets:
                            b_filename = beamlets[beam_i]
                            beamlet_path = os.path.join(p_beamlet_folder, b_filename)
                        else:
                            beamlet_path = "Inactive"

                        robust_beamlets.append(beamlet_path)

                    opt_dict["photon_cpts"][cpt_index]["robust_beamlets"].append(robust_beamlets)



        return opt_dict

    # Public methods
    def create_opt_file(self):
        opt_dict = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return filename
