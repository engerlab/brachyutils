import os
import errno
import numpy
import pydicom as dicom
import copy
from pyRad.utils import egsdose_to_dicom


class BrachySource(object):
    """BrachySource simulation module."""


    def __init__(self, attrs):
        """
        Constructor.

        :params attrs: Dictionary with required parameters for simulation.
        """
        self.sim_params = copy.deepcopy(attrs["sim_params"])

    def submit_sim(self, simulation):
        """
        Create input files and submit BrachySource simulation

        :param simulation: Simulation instance with all parameters required
            to submit a simulation.
        """

        files_to_send = self._generate_input(simulation)
        files_to_send.append(simulation.phantom_filename)
        nthreads = int(self.sim_params["nthreads"])

        server = simulation.server

        brachysource_path = server.get_path("BrachySource")
        server.put_files(files_to_send, brachysource_path)

        inputfile_path = os.path.join(brachysource_path, simulation.name + ".mac")

        command = '''
        cd %s
        python submit_brachysource.py -n %i %s
        exit
        ''' % (brachysource_path, nthreads, inputfile_path)

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
            return "Done"
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

    def check_beamlet_progress(self, beam_gen):
        server = beam_gen.server
        sim_path = server.get_path("beamlets")
        sim_path = os.path.join(sim_path, beam_gen.name)

        stdin, stdout, stderr = server.exec_command("cd {sim_path};ls *.minidos | wc".format(sim_path=sim_path))
        output = stdout.read()
        num_doses = int(output.split()[0])
        stdin, stdout, stderr = server.exec_command("cd {sim_path};ls *.mac | wc".format(sim_path=sim_path))
        output = stdout.read()
        num_inputs = int(output.split()[0])

        if num_doses == num_inputs:
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def _generate_input(self, simulation):
        if "dose_output" not in self.sim_params:
            self.sim_params["dose_output"] = "3ddose"

        self.sim_params["phantom_filename"] = simulation.phantom_filename

        catheters = simulation.beams["catheters"]  # beams == catheters for Brachy

        self.sim_params["file_id"] = self.sim_params["name"]
        geant_core_name = "G4_%s" % (self.sim_params["core"]["isotope"].split("-")[0])
        self.sim_params["g4_core"] = geant_core_name

        catheter_mode = self.sim_params.get("catheter_mode", "sequential")
        if catheter_mode == "multi-catheter":
            plan_file = self._create_multicath_plan(catheters)
        else:
            plan_file = self._create_sequential_plan(catheters)

        self.sim_params["ref_ak"] = simulation.beams["ref_akr"]

        mac_file = self.gen_mac_file()

        return [mac_file, plan_file]

    def _generate_shielded_beamlets(self):
        n_angles = 16
        angle_increment = 360.0 / n_angles
        shield_angles = [angle_increment * i for i in range(n_angles)]

        created_files = []
        bigcounter = 0
        name = self.sim_params["name"]

        for catheter_index, dwell in enumerate(self.sim_params["dwells"]["data"]):
            for pos_index, position in enumerate(dwell["positions"]):
                for angle_index, angle in enumerate(shield_angles):
                    applicator_positions = [
                        {"x": position["x"],
                         "y": position["y"],
                         "z": position["z"],
                         "angle": angle,
                         "time": 3600.0}
                    ]

                    self.sim_params["applicator_positions"] = applicator_positions
                    self.sim_params["file_id"] = name + "_%i_%i_%i" % (catheter_index, pos_index,
                                                                       angle_index)
                    plan_file = self._create_sequential_plan([applicator_positions])
                    mac_file = self.gen_mac_file()
                    created_files += [mac_file, plan_file]
                    bigcounter += 1

        return created_files

    def _generate_unshielded(self):
        created_files = []
        bigcounter = 0
        for catheter_index, dwell in enumerate(self.sim_params["dwells"]["data"]):
            for pos_index, position in enumerate(dwell["positions"]):
                applicator_positions = [
                    {"x": position["x"],
                     "y": position["y"],
                     "z": position["z"],
                     "time": 3600.0
                    }
                ]
                self.sim_params["applicator_positions"] = applicator_positions
                name = self.sim_params["name"]
                self.sim_params["file_id"] = "{}_{}_{}".format(name, catheter_index, pos_index)
                plan_file = self._create_sequential_plan([applicator_positions])
                mac_file = self.gen_mac_file()
                created_files += [mac_file, plan_file]
                bigcounter += 1

        return created_files

    def _create_multicath_plan(self, catheters):
        total_time = 0.0
        max_cumul_time = 0.0
        total_positions = 0
        current_pos = {}
        for index, catheter in enumerate(catheters):
            total_time += sum(pos["time"] for pos in catheter)
            total_positions += len(catheter)
            current_pos[index] = 0

        for catheter in catheters:
            catheter.sort(key=lambda x: (x["z"], x["angle"]))
            cumul_time = 0
            for pos in catheter:
                cumul_time += pos["time"]
                pos["cumul_time"] = cumul_time
            if catheter[-1]["cumul_time"] > max_cumul_time:
                max_cumul_time = catheter[-1]["cumul_time"]

        self.sim_params["total_time"] = total_time / 3600.0  # in hours

        current_time = 0
        cpt_list = []

        while current_time < max_cumul_time:
            cpt = {}

            # Get the cumulative dwell times of the current position of all catheters
            dwell_times = numpy.array([cath[current_pos[i]]["cumul_time"]
                                       for i, cath in enumerate(catheters)])

            relative_times = (dwell_times - current_time)
            changing_catheter = -1
            smallest_time = 1e20
            for cath_index, time in enumerate(relative_times):
                if 0 <= time < smallest_time and current_pos[cath_index] >= 0:
                    smallest_time = time
                    changing_catheter = cath_index

            cpt_weight = smallest_time

            cpt["weight"] = cpt_weight
            cpt["positions"] = []
            for cath_index, time in enumerate(relative_times):
                if time > 0:
                    pos = catheters[cath_index][current_pos[cath_index]]
                    cpt["positions"].append(pos)

            current_time = catheters[changing_catheter][current_pos[changing_catheter]]["cumul_time"]

            if (current_pos[changing_catheter] < len(catheters[changing_catheter]) - 1):
                current_pos[changing_catheter] += 1
            else:
                current_pos[changing_catheter] = -1

            if cpt["weight"] > 0:
                cpt_list.append(cpt)

        plan_filename = self.sim_params["file_id"] + ".plan"
        with open(plan_filename, "w") as plan_file:
            plan_file.write("Treatment Plan\n")
            plan_file.write("%i Control Points\n" % len(cpt_list))
            for cpt in cpt_list:
                plan_file.write("Control Point\n")
                plan_file.write("weight = %e\n" % (cpt["weight"] * len(cpt["positions"]) / total_time))
                plan_file.write("%i Dwell Position\n" % len(cpt["positions"]))
                for pos in cpt["positions"]:
                    plan_file.write("{},{},{},{}\n".format(pos["x"], pos["y"], pos["z"], pos["angle"]))

        return plan_filename

    def _create_sequential_plan(self, catheters):
        plan_filename = self.sim_params["file_id"] + ".plan"
        total_time = 0.0
        total_positions = 0
        for catheter in catheters:
            total_time += sum(pos["time"] for pos in catheter)
            total_positions += len(catheter)

        self.sim_params["total_time"] = total_time / 3600.0  # in hours

        with open(plan_filename, "w") as plan_file:
            plan_file.write("Treatment Plan\n")
            plan_file.write("%i Control Points\n" % total_positions)
            for catheter in catheters:
                for pos in catheter:
                    plan_file.write("Control Point\n")
                    plan_file.write("weight = %e\n" % (pos["time"] / total_time))
                    plan_file.write("1 Dwell Position\n")
                    angle = 0
                    if "angle" in pos:
                        angle = pos["angle"]
                    plan_file.write("{x},{y},{z},{angle}\n".format(x=pos["x"], y=pos["y"], z=pos["z"], angle=angle))

        return plan_filename

    def _get_catheters_from_plan(self, plan_catheters):
        catheters = []
        for cath_num, catheter in plan_catheters.items():
            positions = []
            for dwell in catheter:
                positions.append({"x": dwell["pos"][0],
                                  "y": dwell["pos"][1],
                                  "z": dwell["pos"][2],
                                  "time": dwell["weight"],
                                  "angle": dwell["angle"]})
            catheters.append(positions)

        return catheters

    def gen_mac_file(self):
        input_filename = self.sim_params["file_id"] + ".mac"
        plan_filename = self.sim_params["file_id"] + ".plan"
        self.sim_params["plan_filename"] = plan_filename

        template_dir = os.path.dirname(__file__)
        if "shield_radius" in self.sim_params:
            template_filename = "brachysource_shielded.template"
        else:
            template_filename = "brachysource_unshielded.template"
        template_path = os.path.join(template_dir, template_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        with open(input_filename, "w") as myfile:
            myfile.write(template_string.format(**self.sim_params))

        return input_filename

    def beamlet_maker_template(self, bsource_path):
        maker_str = ""

        maker_str += "from subprocess import call\n"
        maker_str += "import glob\n"
        maker_str += "import time\n"
        maker_str += "import os\n\n"

        maker_str += "beamlets = glob.glob('*.mac')\n"
        maker_str += "doses = glob.glob('*.minidos')\n"
        maker_str += "beamlets = set(['.'.join(b.split('.')[:-1]) for b in beamlets])\n"
        maker_str += "doses = set(['.'.join(d.split('.')[:-1]) for d in doses])\n"
        maker_str += "beamlets_left = list(beamlets.symmetric_difference(doses))\n"
        maker_str += "beamlets_left = [b + '.mac' for b in beamlets_left]\n"
        maker_str += "print('%i beamlets left' % len(beamlets_left))\n"

        maker_str += "for filename in beamlets_left:\n"
        maker_str += "    call('python {}/submit_brachysource.py -n 1 %s' % filename, shell=True)\n".format(bsource_path)

        beamlet_maker = "brachysource_beamlet_maker.py"
        with open(beamlet_maker, "w") as myfile:
            myfile.write(maker_str)

        return beamlet_maker

    def generate_beamlets(self, beamlet_creation):
        self.sim_params["nthreads"] = 1
        self.sim_params["dose_output"] = "minidos"
        geant_core_name = "G4_%s" % (self.sim_params["core"]["isotope"].split("-")[0])
        self.sim_params["g4_core"] = geant_core_name

        self.sim_params["ref_ak"] = 3.7e10 * self.sim_params["core"]["ak_per_history"]

        if "Shielded" in self.sim_params["source"]["name"]:
            created_files = self._generate_shielded_beamlets()
        else:
            created_files = self._generate_unshielded()

        server = beamlet_creation.server
        bsource_path = server.get_path("BrachySource")
        files_to_send = [beamlet_creation.phantom_filename,
                         self.beamlet_maker_template(bsource_path)]
        files_to_send += created_files

        beamlet_directory = beamlet_creation.beamlet_path
        server.put_files(files_to_send, beamlet_directory, make_dir=True, delete_original=True)

        command = '''
        cd %s
        python brachysource_beamlet_maker.py
        exit
        ''' % (beamlet_directory)
        server.exec_shell_command(command)

        return created_files

    def get_finished(self, simulation):
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
