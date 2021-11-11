import os
import numpy
import importlib
from pyRad.utils import dicom_to_spherical


class PHSPBeamNRC(object):
    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

        if hasattr(self, "sim_params"):
            self._load_beam_model(self.sim_params["beam_model"])

    def _load_beam_model(self, name):
        beam_model = importlib.import_module("pyRad.SimCodes.BeamModels.{}".format(name))
        beamClass = getattr(beam_model, name)
        self.beam_model = beamClass()
        return self.beam_model

    def _make_beamnrc_file(self, params):
        return self.beam_model.write_beamnrc_phsp_input(params)

    def _make_jaws_string(self, cpt):
        jaw_dict = {}
        jaw_dict["x_jaw"] = "{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_pos_front:.5}, {x_jaw_pos_back:.5}, {x_jaw_neg_front:.5}, {x_jaw_neg_back:.5}".format(**cpt)
        jaw_dict["y_jaw"] = "{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_pos_front:.5}, {y_jaw_pos_back:.5}, {y_jaw_neg_front:.5}, {y_jaw_neg_back:.5}".format(**cpt)

        return jaw_dict

    def _make_mlc_string(self, aperture):
        mlc_string = ""
        for position in aperture:
            mlc_string += "{neg:.5}, {pos:.5}, 1\n".format(neg=position[0], pos=position[1])
        mlc_string = mlc_string.rstrip()

        return mlc_string

    def _generate_phsp_beamlets(self, beamlet_creation):
        server = beamlet_creation.server
        files_to_send = [beamlet_creation.phantom_filename]
        dosxyz_inputs = []

        for cpt in beamlet_creation.control_points:
            cpt_dosxyz_inputs = self._generate_cpt_beamlets(cpt, beamlet_creation)
            dosxyz_inputs += cpt_dosxyz_inputs
            files_to_send += cpt_dosxyz_inputs

        beamnrc_path = server.get_path("BeamNRC")
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        server.put_files(files_to_send, dosxyznrc_path)

        for dosxyz_filename in dosxyz_inputs:
            egsinp_stripped = ".".join(dosxyz_filename.split(".")[:-1])
            command = '''
            exb dosxyznrc %s %s short &
            exit
            ''' % (egsinp_stripped, self.beam_model.pegs_file, int(self.sim_params["nthreads"]))

            server.exec_shell_command(command)

    def _get_phsp_beamlets(self, beamlet_columns, beamlet_rows, beamlet_size):
        """
            Generate a list of beamlet positions for the entire treatment field.
        """
        x_min = -beamlet_columns / 2.0 * beamlet_size
        x_max = beamlet_columns / 2.0 * beamlet_size
        y_min = -beamlet_rows / 2.0 * beamlet_size
        y_max = beamlet_rows / 2.0 * beamlet_size

        x_positions = numpy.arange(x_min, x_max, beamlet_size) + 0.5 * beamlet_size
        y_positions = numpy.arange(y_min, y_max, beamlet_size) + 0.5 * beamlet_size

        position_grid = numpy.meshgrid(x_positions, y_positions)

        beamlets = list(zip(position_grid[0].flatten(),
                            position_grid[1].flatten()))

        return beamlets

    def _get_phsp_cpts(self, energies):
        x_jaw = [-20, 20]
        y_jaw = [-20, 20]

        beamlet_columns = 40
        beamlet_rows = 40
        beamlet_size = 10.0  # in mm because get_leaf_numbers expects leaf positions in mm
        beamlet_positions = self._get_phsp_beamlets(beamlet_columns, beamlet_rows, beamlet_size)

        mlc_projection = self.beam_model.dcoll / 100.0

        cpts = []

        for energy in energies:
            cpt_dict = {}
            cpt_dict["energy"] = energy

            cpt_dict["y_z_jaw_front"] = self.beam_model.y_z_jaw_front
            cpt_dict["y_z_jaw_back"] = self.beam_model.y_z_jaw_back

            cpt_dict["x_z_jaw_front"] = self.beam_model.x_z_jaw_front
            cpt_dict["x_z_jaw_back"] = self.beam_model.x_z_jaw_back

            cpt_dict["y_jaw_neg_front"] = y_jaw[0] * (cpt_dict["y_z_jaw_front"] / 100.0)
            cpt_dict["y_jaw_pos_front"] = y_jaw[1] * (cpt_dict["y_z_jaw_front"] / 100.0)
            cpt_dict["y_jaw_neg_back"] = y_jaw[0] * (cpt_dict["y_z_jaw_back"] / 100.0)
            cpt_dict["y_jaw_pos_back"] = y_jaw[1] * (cpt_dict["y_z_jaw_back"] / 100.0)

            cpt_dict["x_jaw_neg_front"] = x_jaw[0] * (cpt_dict["x_z_jaw_front"] / 100.0)
            cpt_dict["x_jaw_pos_front"] = x_jaw[1] * (cpt_dict["x_z_jaw_front"] / 100.0)
            cpt_dict["x_jaw_neg_back"] = x_jaw[0] * (cpt_dict["x_z_jaw_back"] / 100.0)
            cpt_dict["x_jaw_pos_back"] = x_jaw[1] * (cpt_dict["x_z_jaw_back"] / 100.0)

            cpt_dict["beamlets"] = []
            for position in beamlet_positions:
                aperture = [[-10.325, -10.325] for i in range(self.beam_model.num_leaves)]
                leaf_numbers = self.beam_model.get_leaf_numbers(position[1], beamlet_size)

                neg_pos = (position[0] - 0.5 * beamlet_size) / 10.0  # cm
                pos_pos = (position[0] + 0.5 * beamlet_size) / 10.0

                for leaf_number in leaf_numbers:
                    aperture[leaf_number] = [neg_pos * mlc_projection, pos_pos * mlc_projection]

                cpt_dict["beamlets"].append(aperture)

            cpts.append(cpt_dict)

        return cpts

    def _create_phsp_inputs(self, server):
        files_created = {}
        files_created["beamnrc_files"] = []

        self._load_beam_model("CL21EX_E")

        energies = [6, 9, 12, 16, 20]
        hist_dict = {
            6: 100000000,
            9: 83920000,
            12: 67857000,
            16: 46428000,
            20: 25000000
        }

        cpts = self._get_phsp_cpts(energies)

        for cpt in cpts:
            cpt["beamnrc_files"] = []
            for index, beamlet in enumerate(cpt["beamlets"]):
                phsp_name = self.beam_model.name + "_%iMeV_%i" % (cpt["energy"], index)
                params = {"cpt": cpt}
                params["name"] = phsp_name

                params["energy"] = cpt["energy"]
                params["pegs_file"] = self.beam_model.pegs_file
                params["beam_inputfile"] = phsp_name + ".egsinp"
                params["beam_model"] = self.beam_model.folder
                params["title"] = "Beamlet apertures"
                params["nhist"] = hist_dict[int(cpt["energy"])]

                params["mlc_string"] = self._make_mlc_string(beamlet)
                jaw_dict = self._make_jaws_string(cpt)
                params["x_jaw"] = jaw_dict["x_jaw"]
                params["y_jaw"] = jaw_dict["y_jaw"]

                cpt["beamnrc_files"].append(self._make_beamnrc_file(params))

        return cpts

    def generate_phsp(self):
        from pyRad.Server import Server
        server = Server({
            "name": "webtps",
            "username": "webtps",
            "ip": "172.17.248.84",
            "port": 22,
            "sim_paths": {
                "BeamNRC": "/home/webtps/egsnrc"
            }
        })

        cpts = self._create_phsp_inputs(server)
        beamnrc_path = server.get_path("BeamNRC")
        model_path = os.path.join(beamnrc_path, self.beam_model.folder)

        for cpt in cpts:
            server.put_files(cpt["beamnrc_files"], model_path)
