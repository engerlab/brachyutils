import os
import math
import json
import errno
import numpy
from numpy import cumsum
from pyRad.utils import dicom_to_spherical_2
from pyRad.utils import egsdose_to_dicom


class PMC(object):
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

    def _generate_cpt_beamlets(self, cpt, beamlet_creation):
        beamlets = cpt.get_base_beamlets()
        beamlet_mask = cpt.get_beamlet_mask()

        files_created = []

        beamlet_index = 0
        for beamlet, included in zip(beamlets, beamlet_mask):
            if included:
                #input_string = self._make_beamlet_input(beamlet, cpt, beamlet_creation)
                input_string = self._make_beamlet_phsp_input(beamlet_index, cpt, beamlet_creation)
                filename = "beamlet_%i_%i_%iMeV" % (cpt.index, beamlet_index, cpt.energy)
                filename += self.input_extension

                with open(filename, "w") as myfile:
                    myfile.write(input_string)

                files_created.append(filename)

            beamlet_index += 1

        return files_created

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

            beam_dict["norm_point"] = [x / 10.0 for x in beam["norm_point"]]
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

                theta, phi, phicol = dicom_to_spherical_2(cpt.gantry_angle,
                                                          cpt.couch_angle,
                                                          cpt.col_angle)

                cpt_dict = {}
                cpt_dict["theta"] = theta
                cpt_dict["phi"] = phi
                cpt_dict["phicol"] = phicol
                cpt_dict["iso"] = [x / 10.0 for x in cpt.iso]
                cpt_dict["snout_position"] = cpt.snout_position / 10.0
                cpt_dict["spot_size"] = [x / 10.0 for x in cpt.spot_size]
                cpt_dict["spot_positions"] = [x / 10.0 for x in cpt.spot_positions]
                cpt_dict["spot_weights"] = spot_weights.tolist()
                cpt_dict["energy"] = cpt.energy
                cpt_dict["virtual_sad"] = [x / 10.0 for x in cpt.virtual_sad]
                cpt_dict["weight"] = cpt.weight / total_weight

                beam_dict["control_points"].append(cpt_dict)

            input_dict["beams"].append(beam_dict)

        with open(filename, "w") as cpt_file:
            cpt_file.write(json.dumps(input_dict))

        return filename

    def _make_proton_input(self, simulation):
        """

            Makes a PMC proton input file from a Simulation object.
            A lot of the PMC parameters are hardcoded, to be refactored.

            Args:
                simulation (Simulation instance)

        """
        phantom_filename = simulation.phantom_filename
        cpt_filename = self._make_proton_cpt_input(simulation)
        pmc_path = simulation.server.get_path("PMC")

        phantom_path = os.path.join(pmc_path, phantom_filename)
        cpt_file_path = os.path.join(pmc_path, cpt_filename)

        input_string = ""

        input_string += "source = 3\n"
        input_string += "n_hists = 1\n"
        input_string += "cutoff = 0.2\n"

        input_string += "phantom = {}\n".format(phantom_path)
        input_string += "beam_file = {}\n".format(cpt_file_path)

        input_string += "dose_output = 0\n"
        input_string += "threshold = 0.005\n"
        input_string += "num_energies = 13\n"
        input_string += "num_batches = 10\n"
        input_string += "num_tracks = 40000\n"
        input_string += "zero_air_dose = 1\n"
        input_string += "tracks_per_batch = 4000\n"

        input_filename = simulation.name + ".inp"
        with open(input_filename, "w") as myfile:
            myfile.write(input_string)

        files_to_send = {"input_file": input_filename, "cpt_file": cpt_filename}

        return files_to_send

    def _make_beamlet_input(self, beamlet, cpt, beamlet_creation):
        server = beamlet_creation.server
        phantom_filename = self.sim_params["phantom_filename"]

        spectrum_filename = "beam_{}mev.spectrum".format(int(cpt.energy))
        spectrum_path = os.path.join(server.get_path("PMC"), spectrum_filename)

        phantom_path = os.path.join(beamlet_creation.beamlet_path, phantom_filename)

        pmc_dcoll = cpt.d_coll - (cpt.sad - cpt.d_source)

        theta, phi, phicol = dicom_to_spherical_2(cpt.gantry_angle,
                                                  cpt.couch_angle,
                                                  cpt.col_angle)

        input_string = ""

        input_string += "source = 4\n"
        input_string += "energy = {}\n".format(cpt.energy)
        input_string += "n_hists = 10\n"
        input_string += "cutoff = 0.2\n"

        input_string += "iso.x = {}\n".format(cpt.iso[0] / 10.0)
        input_string += "iso.y = {}\n".format(cpt.iso[1] / 10.0)
        input_string += "iso.z = {}\n".format(cpt.iso[2] / 10.0)

        input_string += "theta = {}\n".format(math.degrees(theta))
        input_string += "phi = {}\n".format(math.degrees(phi))
        input_string += "phicol = {}\n".format(math.degrees(phicol))

        input_string += "phantom = {}\n".format(phantom_path)
        input_string += "dsource = {}\n".format(cpt.d_source / 10.0)
        input_string += "dcoll = {}\n".format(pmc_dcoll / 10.0)

        input_string += "x_neg = {}\n".format((beamlet[0] - 0.5 * cpt.mlc_col_size) / 10.0)
        input_string += "x_pos = {}\n".format((beamlet[0] + 0.5 * cpt.mlc_col_size) / 10.0)
        input_string += "y_neg = {}\n".format((beamlet[1] - 0.5 * cpt.mlc_row_size) / 10.0)
        input_string += "y_pos = {}\n".format((beamlet[1] + 0.5 * cpt.mlc_row_size) / 10.0)

        input_string += "energy_mode = 1\n"
        input_string += "spectrum = {}\n".format(spectrum_path)

        input_string += "dose_output = 1\n"
        input_string += "threshold = 0.005\n"
        input_string += "num_energies = 13\n"
        input_string += "num_batches = 10\n"
        input_string += "num_tracks = 40000\n"
        input_string += "zero_air_dose = 1\n"
        input_string += "tracks_per_batch = 4000\n"

        return input_string

    def _make_beamlet_phsp_input(self, beamlet_index, cpt, beamlet_creation):
        phantom_filename = self.sim_params["phantom_filename"]

        phantom_path = os.path.join(beamlet_creation.beamlet_path, phantom_filename)

        theta, phi, phicol = dicom_to_spherical_2(cpt.gantry_angle,
                                                  cpt.couch_angle,
                                                  cpt.col_angle)

        temp_path = "/media/data/home/marc/beamlets"
        phsp_file = "CL21EX_E_%iMeV_%i.egsphsp1" % (cpt.energy, beamlet_index)
        phsp_path = os.path.join(temp_path, "CL21EX_E_beamlets", phsp_file)

        input_string = ""

        input_string += "source = 5\n"
        input_string += "energy = {}\n".format(cpt.energy)
        input_string += "n_hists = 10\n"
        input_string += "cutoff = 0.2\n"

        input_string += "iso.x = {}\n".format(cpt.iso[0] / 10.0)
        input_string += "iso.y = {}\n".format(cpt.iso[1] / 10.0)
        input_string += "iso.z = {}\n".format(cpt.iso[2] / 10.0)

        input_string += "theta = {}\n".format(math.degrees(theta))
        input_string += "phi = {}\n".format(math.degrees(phi))
        input_string += "phicol = {}\n".format(math.degrees(phicol))

        input_string += "phantom = {}\n".format(phantom_path)
        input_string += "dsource = {}\n".format(cpt.d_source / 10.0 - 51.51092)

        input_string += "phsp_file = {}\n".format(phsp_path)

        input_string += "dose_output = 1\n"
        input_string += "threshold = 0.005\n"
        input_string += "num_energies = 13\n"
        input_string += "num_batches = 10\n"
        input_string += "num_tracks = 40000\n"
        input_string += "zero_air_dose = 1\n"
        input_string += "tracks_per_batch = 4000\n"

        return input_string

    def beamlet_maker_template(self, path_to_pmc):
        path_to_submit = os.path.join(path_to_pmc, "submit_pmc.py")

        maker_str = ""
        maker_str += "from subprocess import call\n"
        maker_str += "import glob\n"
        maker_str += "import time\n"
        maker_str += "import os\n\n"

        maker_str += "beamlets = glob.glob('*.inp')\n"
        maker_str += "doses = glob.glob('*.minidos')\n"
        maker_str += "beamlets = set(['.'.join(b.split('.')[:-1]) for b in beamlets])\n"
        maker_str += "doses = set(['.'.join(d.split('.')[:-1]) for d in doses])\n"
        maker_str += "beamlets_left = list(beamlets.symmetric_difference(doses))\n"
        maker_str += "beamlets_left = [b + '.inp' for b in beamlets_left]\n"
        maker_str += "print('%i beamlets left' % len(beamlets_left))\n"

        maker_str += "for filename in beamlets_left:\n"
        maker_str += "    call('python {} %s' % script_file, shell=True)\n".format(path_to_submit)

        beamlet_maker = "pmc_beamlet_maker.py"
        with open(beamlet_maker, "w") as myfile:
            myfile.write(maker_str)

        return beamlet_maker

    def generate_beamlets(self, beamlet_creation):
        server = beamlet_creation.server
        beamlet_directory = beamlet_creation.beamlet_path
        pmc_directory = server.get_path("PMC")

        files_to_send = [self.sim_params["phantom_filename"]]
        files_to_send.append(self.beamlet_maker_template(pmc_directory))

        for cpt in beamlet_creation.control_points:
            files_to_send += self._generate_cpt_beamlets(cpt, beamlet_creation)

        server.put_files(files_to_send, beamlet_directory, make_dir=True)

        command = '''
        cd %s
        python pmc_beamlet_maker.py &
        exit
        ''' % (beamlet_directory)
        server.exec_shell_command(command)

    def submit_sim(self, simulation):
        """
            Queues a simulation to the simulation server.

            Args:
            simulation (Simulation instance)

        """
        server = simulation.server
        pmc_directory = server.get_path("PMC")

        files_created = self._make_proton_input(simulation)
        server.put_files([files_created["input_file"],
                          files_created["cpt_file"],
                          simulation.phantom_filename], pmc_directory)

        command = '''
        cd %s
        python submit_pmc.py %s
        exit
        ''' % (pmc_directory, files_created["input_file"])

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