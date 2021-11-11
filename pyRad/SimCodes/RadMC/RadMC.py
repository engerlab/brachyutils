"""
RadMC simulation code module.

Copyright Marc-Andre Renaud, 2017
"""


import os
import importlib
import math
import json
import errno

import numpy
from pyRad.utils import dicom_to_spherical
from pyRad.utils import egsdose_to_dicom

class RadMC(object):
    """
    Module for Radify to interact with RadMC installations on remote servers.
    """

    def __init__(self, attrs=None):
        if "sim_params" in attrs:
            self.sim_params = attrs["sim_params"]
            if isinstance(self.sim_params["beam_model"], basestring):
                self._load_beam_model(self.sim_params["beam_model"])
            else:
                self._load_beam_model(self.sim_params["beam_model"]["name"])

    def submit_sim(self, simulation):
        """
        Create input files and submit BeamNRC simulation.

        :param simulation: Simulation instance with all parameters required
            to submit a simulation.
        """
        files_to_send = self.beam_model.make_sim_inputs(simulation)

        server = simulation.server
        beamnrc_path = server.get_path("BeamNRC")
        beam_folder = self.beam_model.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        radmc_path = server.get_path("RadMC")
        sim_path = os.path.join(radmc_path, "simulations")

        beam_files = [files_to_send["beamnrc_file"]]
        if "mlc_file" in files_to_send:
            beam_files.append(files_to_send["mlc_file"])
        if "jaws_file" in files_to_send:
            beam_files.append(files_to_send["jaws_file"])

        server.put_files(beam_files, beam_model_path)

        radmc_input = self._parse_dosxyz_input(files_to_send["dosxyznrc_file"])
        radmc_input["phantom_filename"] = simulation.phantom_filename
        radmc_filename = simulation.name + ".json"
        with open(radmc_filename, "w") as myfile:
            myfile.write(json.dumps(radmc_input, indent=4, separators=(',', ': ')))

        os.remove(files_to_send["dosxyznrc_file"])

        radmc_files = [simulation.phantom_filename,
                       radmc_filename]

        server.put_files(radmc_files, sim_path)

        nthreads = int(self.sim_params["nthreads"])
        if nthreads > 1:
            command = '''
            cd {}
            python ../submit_radmc.py -mm -n {} {}
            exit
            '''.format(sim_path, int(self.sim_params["nthreads"]), radmc_filename)
        else:
            command = '''
            cd {}
            python ../submit_radmc.py {}
            exit
            '''.format(sim_path, radmc_filename)

        server.exec_shell_command(command)

    def generate_beamlets(self, beamlet_creation):
        """
        Generate beamlets for optimisation purposes.

        Attributes:
        beamlet_creation (BeamletCreation):
            BeamletCreation instance with simulation parameters.
        """

        server = beamlet_creation.server
        beamlet_directory = beamlet_creation.beamlet_path
        radmc_directory = server.get_path("RadMC")
        n_threads = int(beamlet_creation.settings["nthreads"])
        beamlet_maker = self._beamlet_maker_template(radmc_directory, n_threads)

        files_to_send = [beamlet_creation.phantom_filename, beamlet_maker]

        if beamlet_creation.settings.get("robust", False) and "uncert_coords" in beamlet_creation.settings:
            positioning_uncertainty = [abs(float(x)) for x in beamlet_creation.settings["uncert_coords"]]
        else:
            positioning_uncertainty = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        scenarios = [[0.0, 0.0, 0.0]]
        for sc_i, uncert in enumerate(positioning_uncertainty):
            if uncert > 0.0:
                if (sc_i > 2):
                    uncert = -uncert
                iso_shift = [0.0, 0.0, 0.0]
                iso_shift[sc_i % 3] = uncert
                scenarios.append(iso_shift)

        for cpt in beamlet_creation.control_points:
            files_to_send += self._generate_cpt_beamlets(cpt, beamlet_creation, scenarios)

        server.put_files(files_to_send, beamlet_directory, make_dir=True)

        command = '''
        cd %s
        python radmc_beamlet_maker.py &
        exit
        ''' % (beamlet_directory)
        server.exec_shell_command(command)

    def generate_aperture_beamlets(self, aperture_recalc, submit_phantom=True):
        server = aperture_recalc.server

        files_to_send = self.beam_model.make_aperture_inputs(aperture_recalc)

        beamnrc_path = server.get_path("BeamNRC")
        beam_folder = self.beam_model.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        radmc_path = server.get_path("RadMC")
        beamlet_path = aperture_recalc.aperture_path

        beam_files = files_to_send["beamnrc_files"]
        if "mlc_files" in files_to_send:
            beam_files += files_to_send["mlc_files"]
        if "jaws_files" in files_to_send:
            beam_files += files_to_send["jaws_files"]

        server.put_files(beam_files, beam_model_path)
        if submit_phantom:
            server.put_files([aperture_recalc.phantom_filename], beamlet_path, make_dir=True)

        radmc_files = []
        for b_index, dosxyznrc_file in enumerate(files_to_send["dosxyznrc_files"]):
            radmc_input = self._parse_dosxyz_input(dosxyznrc_file)
            radmc_input["source_parameters"]["type"] = "PhspExt"
            radmc_input["dose_output"] = 1
            radmc_input["phantom_filename"] = aperture_recalc.phantom_filename
            radmc_filename = aperture_recalc.name + "_ap{}".format(b_index) + ".json"
            with open(radmc_filename, "w") as myfile:
                myfile.write(json.dumps(radmc_input, indent=4, separators=(',', ': ')))

            os.remove(dosxyznrc_file)

            radmc_files.append(radmc_filename)

        server.put_files(radmc_files, beamlet_path)

        nthreads = int(self.sim_params["nthreads"])
        for radmc_filename in radmc_files:
            if nthreads > 1:
                command = '''
                cd {}
                python {}/submit_radmc.py -mm -n {} {}
                exit
                '''.format(beamlet_path, radmc_path, int(self.sim_params["nthreads"]), radmc_filename)
            else:
                command = '''
                cd {}
                python {}/submit_radmc.py {}
                exit
                '''.format(beamlet_path, radmc_path, radmc_filename)

            server.exec_shell_command(command)

        return aperture_recalc.control_points


    def _load_beam_model(self, name):
        beam_model = importlib.import_module("pyRad.SimCodes.BeamNRC.BeamModels.{}.{}".format(name, name))
        beam_class = getattr(beam_model, name)
        self.beam_model = beam_class()
        return self.beam_model

    def _parse_dosxyz_input(self, dosxyz_filename):
        with open(dosxyz_filename) as myfile:
            myfile.readline()  # title
            myfile.readline()  # dummy
            myfile.readline()  # phantom filename
            myfile.readline()  # global electron/photon cuts
            myfile.readline()  # zeroairdose, random stuff
            line = myfile.readline()  # source number, number of control points, nsplit
            line = [x.strip() for x in line.split(",")]
            num_fields = int(line[2])
            n_split = int(line[4])
            fields = []
            for _ in range(num_fields):
                field_line = myfile.readline()
                field_line = [x.strip() for x in field_line.split(",")]
                field = {
                    "iso_x": float(field_line[0]),
                    "iso_y": float(field_line[1]),
                    "iso_z": float(field_line[2]),
                    "theta": float(field_line[3]),
                    "phi": float(field_line[4]),
                    "phicol": float(field_line[5]),
                    "sad": float(field_line[6]),
                    "index": float(field_line[7])
                }
                fields.append(field)

            myfile.readline()  # useless
            beamline = myfile.readline()  # beam parameters
            beamline = [x.strip() for x in beamline.split(",")]
            nhist = int(myfile.readline().split(",")[0])  # number of histories

            if len(beamline) > 3:
                input_dict = {
                    "n_histories": nhist,
                    "e_cut": 0.2,
                    "p_cut": 0.01,
                    "n_split": n_split,
                    "source_parameters": {
                        "type": "BEAMnrc",
                        "beam_accelerator": beamline[0],
                        "input_file": beamline[1],
                        "pegs_file": beamline[2],
                        "fields": fields
                    }
                }
            elif len(beamline) == 3:
                input_dict = {
                    "n_histories": nhist,
                    "e_cut": 0.2,
                    "p_cut": 0.01,
                    "n_split": n_split,
                    "source_parameters": {
                        "type": "PhspExt",
                        "beam_accelerator": beamline[0],
                        "phsp_path": beamline[1],
                        "pegs_file": "radify",
                        "input_file": beamline[2],
                        "fields": fields
                    }
                }

        return input_dict

    def _generate_cpt_beamlets(self, cpt, beamlet_creation, scenarios=None):
        if scenarios is None:
            scenarios = [[0.0, 0.0, 0.0]]

        beamlet_mask = cpt.get_beamlet_mask(bound=False)
        theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle,
                                                cpt.couch_angle,
                                                cpt.col_angle)

        cpt_dict = {
            "weight": 1.0,
            "energy": cpt.energy,
            "theta": math.degrees(theta),
            "phi": math.degrees(phi),
            "phicol": math.degrees(phicol),
            "d_source": self.beam_model.phsp_dsource,
        }

        files_created = []

        for sc_i, shift in enumerate(scenarios):
            cpt_dict["iso"] = [(x + x_shift) / 10.0 for x, x_shift in zip(cpt.iso, shift)]


            for b_index, included in enumerate(beamlet_mask):
                if included:
                    input_dict = self._make_beamlet_phsp_input(b_index,
                                                            cpt_dict,
                                                            beamlet_creation)
                    filename = "{name}_{cpt_index}_{scenario_index}_{beamlet_index}_{energy}MV.json".format(
                        name=beamlet_creation.name,
                        cpt_index=cpt.index,
                        scenario_index=sc_i,
                        beamlet_index=b_index,
                        energy=str(cpt.energy)
                    )

                    with open(filename, "w") as myfile:
                        myfile.write(json.dumps(input_dict,
                                                sort_keys=True,
                                                indent=4,
                                                separators=(',', ': ')))

                    files_created.append(filename)

        return files_created

    def _make_beamlet_phsp_input(self, b_index, cpt, beamlet_creation):
        server = beamlet_creation.server
        beamnrc_path = server.get_path("BeamNRC")

        phantom_filename = beamlet_creation.phantom_filename
        phantom_path = os.path.join(beamlet_creation.beamlet_path,
                                    phantom_filename)

        phsp_path = os.path.join(beamnrc_path, "phsp", "{}E".format(int(cpt.energy)))
        phsp_file = "TrueBeam_{energy}E_beamlets_{index}.egsphsp1".format(energy=int(cpt["energy"]), index=b_index)
        phsp_path = os.path.join(phsp_path, phsp_file)

        input_dict = {
            "n_histories": int(beamlet_creation.settings["nhist"]),
            "p_cut": 0.01,
            "e_cut": 0.2,
            "n_split": 1,
            "phantom_filename": phantom_path,
            "zero_threshold": 0.2,
            "dose_output": 1,
            "source_parameters": {
                "phsp_path": phsp_path,
                "fields": [
                    {
                        "iso_x": cpt["iso"][0],
                        "iso_y": cpt["iso"][1],
                        "iso_z": cpt["iso"][2],
                        "index": 0.0,
                        "phicol": cpt["phicol"],
                        "theta": cpt["theta"],
                        "phi": cpt["phi"],
                        "sad": cpt["d_source"]
                    }
                ],
                "type": "EGSPhsp"
            }
        }

        return input_dict

    def _beamlet_maker_template(self, path_to_radmc, n_threads):
        path_to_submit = os.path.join(path_to_radmc, "submit_radmc.py")

        maker_str = ""
        maker_str += "from subprocess import call\n"
        maker_str += "import glob\n"
        maker_str += "import time\n"
        maker_str += "import os\n\n"

        maker_str += "beamlets = glob.glob('*.json')\n"
        maker_str += "doses = glob.glob('*.bindos')\n"
        maker_str += "beamlets = set(['.'.join(b.split('.')[:-1]) for b in beamlets])\n"
        maker_str += "doses = set(['.'.join(d.split('.')[:-1]) for d in doses])\n"
        maker_str += "beamlets_left = list(beamlets.symmetric_difference(doses))\n"
        maker_str += "beamlets_left = [b + '.json' for b in beamlets_left]\n"
        maker_str += "print('%i beamlets left' % len(beamlets_left))\n"

        maker_str += "for filename in beamlets_left:\n"
        maker_str += "    call('python {} -n {} %s' % filename, shell=True)\n".format(path_to_submit, n_threads)

        beamlet_maker = "radmc_beamlet_maker.py"
        with open(beamlet_maker, "w") as myfile:
            myfile.write(maker_str)

        return beamlet_maker

    def check_sim_progress(self, simulation):
        """
        Return a number between 0 and 100 representing the progress of the simulation.

        Radify passes ssh and sftp channels, along with path to mc_code
        and simulation uid.

        If an error is found, return a string with the error message.
        """
        server = simulation.server
        radmc_path = server.get_path("RadMC")
        sim_path = os.path.join(radmc_path, "simulations")

        path_to_3ddose = os.path.join(sim_path, simulation.name + ".3ddose")
        path_to_logfile = os.path.join(sim_path, simulation.name + ".log")
        # If simulation is done, grab the dose. Otherwise, check for a ".lock"
        # to get the status of the simulation. If ".lock" file doesn't exist,
        # then either the simulation crashed or it hasn't started yet.
        try:
            # Check if 3ddose exists
            server.stat(path_to_3ddose)
            return "Done"
        except IOError, e:
            if e.errno == errno.ENOENT:
                try:
                    # Check if lock file exists
                    server.stat(path_to_logfile)
                    _, stdout, _ = server.exec_command("/usr/bin/tail %s" % path_to_logfile)
                    tail_result = stdout.readlines()

                    for line in tail_result[::-1]:
                        if "Histories done" in line:
                            splitted = line.strip().split(":")[-1].strip().split("-")[0].split("/")
                            hists_done = float(splitted[0])
                            total_hists = float(splitted[1])

                            response = round(hists_done / (total_hists) * 100, 0)
                            break

                    return response
                except IOError, error:
                    # If error is "file not found" for lock file
                    if error.errno == errno.ENOENT:
                        return {"error": "Could not find a 3ddose or lock file. If the simulation was just submitted, it may not be started yet. Otherwise, the simulation crashed."}
                    else:
                        return {"error": "Expected file not found error, got something else."}
            else:
                return {"error": "Expected file not found error, got something else."}

    def check_beamlet_progress(self, beam_gen):
        """
        Compare number of minidos files to number of input files.

        Returns the progress as %.
        """
        server = beam_gen.server
        path = server.get_path("beamlets")
        path = os.path.join(path, beam_gen.name)
        _, stdout, _ = server.exec_command("cd %s;ls *.minidos | wc" % (path))
        output = stdout.read()
        num_doses = int(output.split()[0])
        _, stdout, _ = server.exec_command("cd %s;ls *.json | wc" % path)
        output = stdout.read()
        num_inputs = int(output.split()[0])

        if num_doses == num_inputs:
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def check_aperture_progress(self, aperture_recalc):
        return self.check_beamlet_progress(aperture_recalc)

    def get_finished(self, simulation):
        """
        Retrieve the output of a finished BeamNRC simulation.

        :param simulation: Simulation instance with unique identifier.
        """
        server = simulation.server
        beamnrc_path = server.get_path("RadMC")
        sim_path = os.path.join(beamnrc_path, "simulations")

        path_to_3ddose = os.path.join(sim_path, simulation.name + ".3ddose")

        local_dose_name = simulation.name + ".3ddose"
        server.get_file(path_to_3ddose, local_dose_name)

        plan_uid = self.sim_params["plan_uid"]
        patient_uid = self.sim_params["patient_uid"]
        orient = self.sim_params["orient"]
        if "manual_norm" in self.sim_params and self.sim_params["manual_norm"]:
            norm_dose = self.sim_params["norm_dose"]
            norm_point = numpy.array([float(x) for x in self.sim_params["norm_point"]])
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm_dose, norm_point=norm_point)
        else:
            norm = self.beam_model.get_calibration(simulation.ref_plan.get_plan_type_object(), simulation)
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm)
        return dose_template, error_template