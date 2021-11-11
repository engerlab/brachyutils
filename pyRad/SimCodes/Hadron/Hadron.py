import os
import json
import errno
from numpy import cumsum
from pyRad.utils import dicom_to_spherical
from pyRad.utils import egsdose_to_dicom

class Hadron(object):
    input_extension = ".inp"
    beamlet_extension = ".minidos"

    @staticmethod
    def check_beamlet_progress(server, sim_path, dose_extension="minidos"):
        stdin, stdout, stderr = server.exec_command("cd %s;ls *.%s | wc" % (sim_path, dose_extension))
        output = stdout.read()
        num_doses = int(output.split()[0])
        stdin, stdout, stderr = server.exec_command("cd %s;ls *.inp | wc" % sim_path)
        output = stdout.read()
        num_inputs = int(output.split()[0])

        if num_doses == num_inputs:
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def _make_proton_cpt_input(self, simulation):
        filename = simulation.name + ".beams"
        beams = simulation.beams

        input_dict = {}
        input_dict["beams"] = []
        for beam in beams:
            beam_dict = {}
            if "norm_point" not in beam:
                beam["norm_point"] = list(beam["cpts"][0].iso)

            if "norm_value" not in beam:
                beam["norm_value"] = 30.0

            beam_dict["norm_point"] = beam["norm_point"]
            beam_dict["norm_value"] = beam["norm_value"]
            beam_dict["control_points"] = []
            total_weight = sum([cpt.weight for cpt in beam["cpts"]])
            for cpt in beam["cpts"]:
                # Swap X and Y directions
                cpt.spot_weights = cpt.spot_weights[::-1]
                cpt.spot_positions = cpt.spot_positions[::-1]
                cpt.virtual_sad = cpt.virtual_sad[::-1]
                cpt.spot_size = cpt.spot_size[::-1]

                spot_weights = cumsum(cpt.spot_weights)
                spot_weights /= spot_weights[-1]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle,
                                                        cpt.couch_angle,
                                                        cpt.col_angle)

                cpt_dict = {}
                cpt_dict["theta"] = theta
                cpt_dict["phi"] = phi
                cpt_dict["phicol"] = phicol
                cpt_dict["iso"] = list(cpt.iso)
                cpt_dict["snout_position"] = cpt.snout_position
                cpt_dict["spot_size"] = cpt.spot_size
                cpt_dict["spot_positions"] = cpt.spot_positions
                cpt_dict["spot_weights"] = spot_weights.tolist()
                cpt_dict["energy"] = cpt.energy
                cpt_dict["virtual_sad"] = cpt.virtual_sad
                cpt_dict["weight"] = cpt.weight / total_weight

                beam_dict["control_points"].append(cpt_dict)

            input_dict["beams"].append(beam_dict)

        with open(filename, "w") as cpt_file:
            cpt_file.write(json.dumps(input_dict))

        return filename

    def _make_proton_input(self, simulation):
        """

            Makes a Hadron proton input file from a Simulation object.
            A lot of the Hadron parameters are hardcoded, to be refactored.

            Args:
                simulation (Simulation instance)

        """
        phantom_filename = simulation.phantom_filename
        cpt_filename = self._make_proton_cpt_input(simulation)
        hadron_path = simulation.server.get_path("Hadron")

        phantom_path = os.path.join(hadron_path, phantom_filename)
        cpt_file_path = os.path.join(hadron_path, cpt_filename)

        input_string = ""

        input_string += "/world/phantom {}\n".format(phantom_path)
        input_string += "/hadron/beams {}\n".format(cpt_file_path)
        input_string += "/run/numberOfThreads {}\n".format(self.sim_params["nthreads"])

        input_string += "/run/initialize\n"
        input_string += "/control/verbose 0\n"
        input_string += "/run/verbose 0\n"
        input_string += "/tracking/verbose 0\n"

        input_string += "/hadron/run {}\n".format(self.sim_params["nhists"])

        input_filename = simulation.name + ".mac"
        with open(input_filename, "w") as myfile:
            myfile.write(input_string)

        files_to_send = {"input_file": input_filename, "cpt_file": cpt_filename}

        return files_to_send

    def submit_sim(self, simulation):
        """
            Queues a simulation to the simulation server.

            Args:
            simulation (Simulation instance)

        """
        server = simulation.server
        hadron_path = server.get_path("Hadron")

        files_created = self._make_proton_input(simulation)
        server.put_files([files_created["input_file"],
                          files_created["cpt_file"],
                          simulation.phantom_filename], hadron_path)

        command = '''
        cd %s
        python submit_hadron.py -n %i %s
        ''' % (hadron_path, int(self.sim_params["nthreads"]), files_created["input_file"])

        server.exec_shell_command(command)

    def check_sim_progress(self, simulation):
        server = simulation.server
        progress_path = os.path.join(self.sim_params["sim_path"],
                                     simulation.name + ".progress")
        path_to_3ddose = os.path.join(self.sim_params["sim_path"],
                                      simulation.name + ".3ddose")

        # If simulation is done, grab the dose. Otherwise, check for
        # a ".progress" file to get the status of the simulation.
        # If ".progress" file doesn't exist, then either the simulation crashed
        # or it hasn't started yet.
        try:
            # Check if 3ddose exists
            server.stat(path_to_3ddose)
            return self._get_finished(simulation)
        except IOError, e:
            # If the error is "file not found" for 3ddose
            if e.errno == errno.ENOENT:
                try:
                    # Check if lock file exists
                    server.stat(progress_path)
                    stdin, stdout, stderr = server.exec_command("/bin/cat %s"
                                                             % progress_path)
                    cat_result = stdout.read()

                    # Rough approximation of the progress. The current batch
                    # being processed won't show up in the lock file so there
                    # is a 10%/Nthreads error associated with the progress bar.
                    hists = cat_result.strip().split("/")
                    hists_done = float(hists[0])
                    hists_total = float(hists[1])
                    response = round(hists_done / hists_total * 100, 0)
                except IOError, error:
                    # If error is "file not found" for lock file
                    if error.errno == errno.ENOENT:
                        return {"error": "Could not find a 3ddose or progress file. If the simulation was just submitted, it may not be started yet. Otherwise, the simulation crashed."}
                    else:
                        return {"error": "Expected file not found error, got something else."}
            else:
                return {"error": "Expected file not found error, got something else."}
        return response

    def _get_finished(self, simulation):
        server = simulation.server

        dose_filename = simulation.name + ".3ddose"
        dose_path = os.path.join(self.sim_params["sim_path"],
                                 dose_filename)

        local_dose_name = simulation.name + ".3ddose"
        server.get_file(dose_path, dose_filename)

        plan_uid = self.sim_params["plan_uid"]
        patient_uid = self.sim_params["patient_uid"]
        orient = self.sim_params["orient"]
        if self.sim_params.get("manual_norm", False):
            norm_dose = self.sim_params["norm_dose"]
            norm_point = numpy.array([float(x) for x in self.sim_params["norm_point"]])
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm_dose, norm_point=norm_point)
        else:
            norm = 1.0
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm)
        return dose_template, error_template