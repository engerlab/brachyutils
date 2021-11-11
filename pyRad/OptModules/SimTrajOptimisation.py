import os
import json
import numpy

class SimTrajOptimisation(object):
    """
        Attributes:
        name (string)
        beamlet_generation (BeamletCreation instance)
        roi_rules (list of ROIRule instances)
        control_points (list of ControlPoint instances)
        total_voxels (Integer, TEMPORARY)
    """

    bindos = True

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def segment_by_cpt(self, filenames):
        cpt_masks = {}
        for filename in filenames:
            base_file = os.path.basename(filename)
            # Schema: {UID}{GANTRY*10}_{COUCH*10}_{X}_{Y}.bindos
            # Example filename: uHLef2xmLprD9N3vM6JVjn900_900_-99_-99.bindos
            if self.bindos:
                splitted = base_file.replace(self.beamlet_generation.name, "").replace(".bindos", "").split("_")
            else:
                splitted = base_file.replace(self.beamlet_generation.name, "").replace(".minidos", "").split("_")

            gantry_angle = int(splitted[0])
            couch_angle = int(splitted[1])
            cpt_id = (gantry_angle, couch_angle)
            x = int(splitted[2])
            y = int(splitted[3])

            if cpt_id not in cpt_masks:
                cpt_masks[cpt_id] = {}

            if y not in cpt_masks[cpt_id]:
                cpt_masks[cpt_id][y] = set()

            if x not in cpt_masks[cpt_id][y]:
                cpt_masks[cpt_id][y].add(x)

        return cpt_masks

    def make_cpt_filenames(self, cpt, cpt_db):
        beamlet_folder = self.beamlet_generation.beamlet_path

        gantry_id = cpt.gantry_angle
        couch_id = cpt.couch_angle if cpt.couch_angle >= 0 else cpt.couch_angle + 360.0
        cpt_id = (int(gantry_id * 10), int(couch_id * 10))

        beamlets_found = 0
        beamlets = []

        for row in range(cpt.beamlet_rows):
            row_y = int((-cpt.beamlet_rows / 2 + row) * cpt.iso_row_size * 10)
            # If there are beamlets in this row, then iterate through all columns to see
            # which ones are active
            if row_y in cpt_db[cpt_id]:
                for col in range(cpt.beamlet_columns):
                    col_x = int((-cpt.beamlet_columns / 2 + col) * cpt.iso_col_size * 10)
                    if col_x in cpt_db[cpt_id][row_y]:
                        filename = "{name}{gantry_id}_{couch_id}_{col_x}_{col_y}.{ext}".format(
                            name=self.beamlet_generation.name,
                            gantry_id=int(gantry_id * 10),
                            couch_id=int(couch_id * 10),
                            col_x=col_x,
                            col_y=row_y,
                            ext="bindos" if self.bindos else "minidos"
                        )

                        beamlets.append(os.path.join(beamlet_folder, filename))
                        beamlets_found += 1
                    else:
                        beamlets.append("Inactive")
            else:
                for _ in range(cpt.beamlet_columns):
                    beamlets.append("Inactive")

        return (beamlets, beamlets_found)

    def get_json_cpt_filename(self, cpt_dict):
        gantry_id = cpt_dict["gantry_angle"]
        couch_id = cpt_dict["couch_angle"] if cpt_dict["couch_angle"] >= 0 else cpt_dict["couch_angle"] + 360.0

        filename = "f{gantry}_{couch}.json".format(
            gantry=int(gantry_id*10),
            couch=int(couch_id*10)
        )

        return filename

    def make_json_cpt_file(self, cpt_dict):
        cpt_filename = self.get_json_cpt_filename(cpt_dict)

        with open(cpt_filename, "w") as myfile:
            json.dump(cpt_dict, myfile)

        return cpt_filename

    def as_dict(self):
        opt_dict = {}
        opt_dict["opt_type"] = "sim_traj"
        opt_dict["name"] = self.name
        opt_dict["total_voxels"] = self.total_voxels

        opt_dict["structures"] = self.roi_rules
        opt_dict["path_to_beamlets"] = os.path.join(self.beamlet_generation.server.get_path("optimisation"), self.name)

        beamlet_folder = self.beamlet_generation.beamlet_path
        opt_dict["cpt_mask"] = os.path.join(beamlet_folder, "cpt_mask.json")

        if "minidos" in self.beam_paths[0]:
            self.bindos = False

        cpt_db = self.segment_by_cpt(self.beam_paths)

        control_points = [cpt.as_dict() for cpt in self.photon_cpts]

        beamlets_found = 0
        for cpt_index, cpt in enumerate(self.photon_cpts):
            beamlet_list, num_beamlets = self.make_cpt_filenames(cpt, cpt_db)
            control_points[cpt_index]["beamlets"] = beamlet_list
            beamlets_found += num_beamlets

        cpt_json_filenames = [self.make_json_cpt_file(cpt_dict) for cpt_dict in control_points]

        beamlet_path = opt_dict["path_to_beamlets"]
        if hasattr(self, "trajectory_start") and hasattr(self, "trajectory_end"):
            opt_dict["photon_cpts"] = [os.path.join(beamlet_path, self.get_json_cpt_filename(self.trajectory_start)),
                                       os.path.join(beamlet_path, self.get_json_cpt_filename(self.trajectory_end))]
        else:
            opt_dict["photon_cpts"] = [os.path.join(beamlet_path, cpt_json_filenames[0]),
                                    os.path.join(beamlet_path, cpt_json_filenames[-1])]

        return (opt_dict, cpt_json_filenames)

    # Public methods
    def create_opt_file(self):
        opt_dict, cpt_json_filenames = self.as_dict()

        filename = self.name + ".json"
        with open(filename, "w") as myfile:
            myfile.write(json.dumps(opt_dict))

        return (filename, cpt_json_filenames)
