import os
import errno
import numpy
import math
import copy
import json
from pyRad.utils import egsdose_to_dicom
from pyRad.utils import dicom_to_spherical


class Supify(object):
    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

        self.sim_params = attrs["sim_params"]


    def submit_sim(self, simulation):
        """
        Create input files and submit Supify simulation.

        :param simulation: Simulation instance with all parameters required
            to submit a simulation.
        """

        cpts = self._process_beams(simulation.beams)

        files_to_send = self._make_sim_input(cpts, simulation, simulation.settings.get("commissioning", False))

        server = simulation.server
        supify_path = server.get_path("Supify")
        sim_path = os.path.join(supify_path, "simulations")

        queue = self.sim_params.get("queue", "gpu")
        mem = self.sim_params.get("mem", 5000)

        server.put_files([files_to_send["input_file"]], sim_path)

        if not simulation.settings.get("commissioning", False):
            server.put_files([simulation.phantom_filename], sim_path)

        input_file = files_to_send["input_file"]

        command = '''
        cd {sim_path}
        python ../submit_supify.py {inputfile} -q {queue} -m {mem}
        exit
        '''.format(sim_path=sim_path,
                   inputfile=input_file,
                   queue=queue,
                   mem=mem)
        server.exec_shell_command(command)

        return [input_file]

    def _process_beams(self, beams, positioning_error=None):
        """
        Transform beam data into the format required by BeamNRC.

        DICOM defines most attributes at isocenter, BeamNRC needs them at
        the actual physical height of the components.
        """
        if positioning_error is None:
            positioning_error = [0.0, 0.0, 0.0]

        processed_cpts = []

        cum_weight = 0.0
        for beam in beams:
            for cpt in beam["cpts"]:
                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)

                cpt_dict = {
                    "y_jaws": [-cpt.y_jaw[1] / 10.0, -cpt.y_jaw[0] / 10.0],
                    "x_jaws": [cpt.x_jaw[0] / 10.0, cpt.x_jaw[1] / 10.0],
                    "weight": cpt.weight,
                    "iso": [(x + x_err) / 10.0 for (x, x_err) in zip(cpt.iso, positioning_error)],
                    "energy": cpt.energy,
                    "theta": math.degrees(theta),
                    "phi": math.degrees(phi),
                    "phicol": math.degrees(phicol),
                    "sad": cpt.sad / 10.0
                }

                cum_weight += cpt.weight

                processed_cpts.append(cpt_dict)

        return processed_cpts

    def _generate_cpt_beamlets(self, beamlet_creation, scenarios):
        cpts_created = []
        for cpt in beamlet_creation.control_points:
            beamlets = cpt.get_iso_beamlets()
            model_leakage = beamlet_creation.settings.get("model_leakage", True)
            beamlet_mask = cpt.get_beamlet_mask(bound=model_leakage)

            theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
            cpt_dict = {
                "weight": 1.0,
                "energy": cpt.energy,
                "theta": math.degrees(theta),
                "phi": math.degrees(phi),
                "phicol": math.degrees(phicol),
                "sad": cpt.sad / 10.0,
            }

            for sc_i, shift in enumerate(scenarios):
                cpt_dict["iso"] = [(x + x_shift) / 10.0 for x, x_shift in zip(cpt.iso, shift)]

                beamlet_index = 0
                for beamlet, included in zip(beamlets, beamlet_mask):
                    if included:
                        cpt_dict["x_jaws"] = [(beamlet[0] - 0.5 * cpt.iso_col_size) / 10.0,
                                            (beamlet[0] + 0.5 * cpt.iso_col_size) / 10.0]
                        cpt_dict["y_jaws"] = [(beamlet[1] - 0.5 * cpt.iso_row_size) / 10.0,
                                            (beamlet[1] + 0.5 * cpt.iso_row_size) / 10.0]

                        cpt_name = "{name}_{cpt_index}_{beamlet_index}_{energy}MV_{scenario_index}".format(
                            name=beamlet_creation.name,
                            cpt_index=cpt.index,
                            scenario_index=sc_i,
                            beamlet_index=beamlet_index,
                            energy=str(cpt.energy)
                        )

                        if getattr(cpt, "joel", False):
                            couch_id = cpt.couch_angle if cpt.couch_angle >= 0 else cpt.couch_angle + 360.0
                            y_index = beamlet_index / cpt.beamlet_columns
                            x_index = beamlet_index % cpt.beamlet_columns
                            x_id = (-cpt.beamlet_columns / 2 + x_index) * cpt.iso_col_size
                            y_id = (-cpt.beamlet_rows / 2 + y_index) * cpt.iso_row_size

                            cpt_name = "{name}{gantry}_{couch}_{x}_{y}".format(
                                name=beamlet_creation.name,
                                gantry=int(cpt.gantry_angle * 10),
                                couch=int(couch_id * 10),
                                x=int(x_id * 10),
                                y=int(y_id * 10)
                            )

                        cpt_dict["name"] = cpt_name

                        cpts_created.append(copy.deepcopy(cpt_dict))

                    beamlet_index += 1

        return cpts_created

    def _make_sim_input(self, cpts, simulation, commissioning=False):
        supify_path = simulation.server.get_path("Supify")
        sim_path = os.path.join(supify_path, "simulations")
        phantom_filename = simulation.phantom_filename
        beam_model = simulation.settings.get("beam_model", "TrueBeam_6X")

        input_dict = {
            "att_filename": os.path.join(supify_path, "beam_models", beam_model + ".att"),
            "kernel_filename": os.path.join(supify_path, "beam_models", beam_model + ".skernel"),
            "phantom_filename": os.path.join(sim_path, phantom_filename),
            "output_every_cpt": False,
            "dose_output": "3ddose",
            "dose_output_threshold": 0.001,
            "cpts": cpts
        }

        input_filename = simulation.name + ".json"
        with open(input_filename, "w") as myfile:
            if commissioning:
                json.dump(input_dict, myfile, sort_keys=True, indent=4)
            else:
                json.dump(input_dict, myfile)

        return {"input_file": input_filename}

    def _make_beamlet_input(self, cpts, beamlet_creation):
        supify_path = beamlet_creation.server.get_path("Supify")
        phantom_filename = beamlet_creation.phantom_filename
        beam_model = beamlet_creation.settings.get("beam_model", "TrueBeam_6X")

        input_dict = {
            "att_filename": os.path.join(supify_path, "beam_models", beam_model + ".att"),
            "kernel_filename": os.path.join(supify_path, "beam_models", beam_model + ".skernel"),
            "phantom_filename": os.path.join(beamlet_creation.beamlet_path, phantom_filename),
            "output_every_cpt": True,
            "dose_output": "bindos",
            "dose_output_threshold": 0.001,
            "cpts": cpts
        }

        input_filename = beamlet_creation.name + ".json"
        with open(input_filename, "w") as myfile:
            json.dump(input_dict, myfile)

        return input_filename

    def _process_scenarios(self, uncert_coords):
        scenarios = [[0.0, 0.0, 0.0]]
        positioning_uncertainty = [abs(float(x)) for x in uncert_coords]
        for sc_i, uncert in enumerate(positioning_uncertainty):
            if uncert > 0.0:
                if (sc_i > 2):
                    uncert = -uncert
                iso_shift = [0.0, 0.0, 0.0]
                iso_shift[sc_i % 3] = uncert
                scenarios.append(iso_shift)

        return scenarios

    def generate_beamlets(self, beamlet_creation):
        server = beamlet_creation.server
        beamlet_directory = beamlet_creation.beamlet_path
        supify_path = server.get_path("Supify")
        submit_path = os.path.join(supify_path, "submit_supify.py")

        files_to_send = [beamlet_creation.phantom_filename]

        if beamlet_creation.settings.get("robust", False) and "uncert_coords" in beamlet_creation.settings:
            scenarios = self._process_scenarios(beamlet_creation.settings["uncert_coords"])
        else:
            scenarios = [[0.0, 0.0, 0.0]]

        cpts = self._generate_cpt_beamlets(beamlet_creation, scenarios)
        inputfile = self._make_beamlet_input(cpts, beamlet_creation)

        files_to_send.append(inputfile)

        server.put_files(files_to_send, beamlet_directory, make_dir=True, delete_original=True)

        command = '''
        cd {beamlet_path}
        python ../supify_beamlets.py -n {nsplit} {inputfile} {submit_path}
        exit
        '''.format(beamlet_path=beamlet_directory,
                   nsplit=beamlet_creation.settings["n_gpus"],
                   inputfile=inputfile,
                   submit_path=submit_path)
        server.exec_shell_command(command)

    def check_beamlet_progress(self, beam_gen):
        server = beam_gen.server
        beamlet_path = server.get_path("beamlets")
        sim_path = os.path.join(beamlet_path, beam_gen.name)

        stdin, stdout, stderr = server.exec_command("cd {};python ../count_supify_beamlets.py {}.json".format(sim_path, beam_gen.name))
        output = stdout.read()
        num_doses = int(output.split("/")[0])
        num_inputs = int(output.split("/")[1])

        if num_doses == num_inputs:
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def check_aperture_progress(self, aperture_gen):
        return self.check_beamlet_progress(aperture_gen)
