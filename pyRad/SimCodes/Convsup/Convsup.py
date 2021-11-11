import os
import errno
import numpy
import math
import copy
import json
from pyRad.utils import egsdose_to_dicom
from pyRad.utils import dicom_to_spherical


class Convsup(object):
    input_extension = ".inp"
    beamlet_extension = ".bindos"
    default_dsource = 100.0
    default_dcoll = 50.45
    abut_gap = 0.03
    leaf_radius = 8.0

    num_leaves = 80

    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def _make_beam_dicts(self, cpts):
        beams = []
        for cpt in cpts:
            mlc_projection = cpt["d_coll"] / cpt["d_source"]

            beam_dict = copy.deepcopy(cpt)
            beam_dict["x_size"] = beam_dict["x_size"] * mlc_projection
            beam_dict["y_size"] = beam_dict["y_size"] * mlc_projection
            beams.append(beam_dict)

        return beams

    def _process_beams(self, beams):
        dcoll = self.default_dcoll

        total_mus = sum([beam["beam_meterset"] for beam in beams])
        processed_cpts = []

        for beam in beams:
            for cpt in beam["cpts"]:
                cpt_dict = {}

                if hasattr(cpt, "apertures") and len(cpt.apertures) > 0:
                    cpt_dict["apertures"] = []
                    # Leaves are specified in opposite order compared to DICOM
                    reverse_leaves = cpt.apertures[::-1]

                    # First ten leaves are 1 cm
                    for ap_index, x in enumerate(reverse_leaves):
                        # Convert between aperture positions defined at SSD 100 cm to MLC plane.
                        projected = [x[0] * (dcoll / 100.0) / 10.0, x[1] * (dcoll / 100.0) / 10.0]
                        real_a = projected[0] + self.leaf_radius * (math.sqrt(projected[0] * projected[0] + dcoll * dcoll) / dcoll - 1)
                        real_b = projected[1] + self.leaf_radius * (math.sqrt(projected[1] * projected[1] + dcoll * dcoll) / dcoll - 1)

                        real_b = -1.0 * real_b  # Flip the sign of the B leaf to match coordinate system.

                        # If leaves are touching, must add abutting leaf gap
                        if abs(real_a) - abs(real_b) < self.abut_gap:
                            real_b -= 0.5 * self.abut_gap
                            real_a += 0.5 * self.abut_gap

                        if ap_index < 10 or ap_index >= 50:
                            cpt_dict["apertures"].append([real_b, real_a])
                            cpt_dict["apertures"].append([real_b, real_a])
                        else:
                            cpt_dict["apertures"].append([real_b, real_a])

                else:
                    # Fully open MLC-defined field
                    cpt_dict["apertures"] = [[20.0, -20.0] for i in range(self.num_leaves)]

                cpt_dict["weight"] = cpt.weight / total_mus
                cpt_dict["iso"] = [x / 10.0 for x in cpt.iso]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol)
                cpt_dict["energy"] = cpt.energy
                cpt_dict["d_coll"] = self.default_dcoll

                if hasattr(cpt, "d_source"):
                    cpt_dict["d_source"] = cpt.d_source
                else:
                    cpt_dict["d_source"] = self.default_dsource

                # Maximum field size
                cpt_dict["x_size"] = 40.0
                cpt_dict["y_size"] = 40.0
                cpt_dict["offset_y"] = 0.0

                processed_cpts.append(cpt_dict)

        return processed_cpts

    def _process_aperture_cpts(self, beamlet_cpts):
        """
            Creates BeamNRC input files for individual apertures.
            Used in full MC recalculation after optimising from beamlets.
        """
        dcoll = self.default_dcoll
        default_sad = 100.0
        individual_cpts = []

        for cpt in beamlet_cpts:
            if not hasattr(cpt, "apertures"):
                continue

            if not isinstance(cpt.apertures, list):
                continue

            for aperture in cpt.apertures:
                cpt_dict = {}
                cpt_dict["energy"] = cpt.energy
                cpt_dict["apertures"] = []

                for x in aperture["rows"]:
                    cpt_dict["apertures"].append([(x[0] / 10.0) * (dcoll / default_sad),
                                                  (x[1] / 10.0) * (dcoll / default_sad)])

                cpt_dict["iso"] = [x / 10.0 for x in cpt.iso]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol)

                cpt_dict["weight"] = 1.0
                cpt_dict["d_source"] = self.default_dsource
                cpt_dict["d_coll"] = dcoll
                cpt_dict["x_size"] = cpt.beamlet_columns * cpt.iso_col_size / 10.0
                cpt_dict["y_size"] = cpt.beamlet_rows * cpt.iso_row_size / 10.0
                cpt_dict["offset_y"] = 0.0

                individual_cpts.append(cpt_dict)

        return individual_cpts

    def _make_sim_inputs(self, simulation):
        cpts = self._process_beams(simulation.beams)
        convsup_path = simulation.server.get_path("Convsup")
        phantom_filename = simulation.phantom_filename
        input_filename = simulation.name + self.input_extension
        beams = self._make_beam_dicts(cpts)

        input_dict = {
            "mu_path": os.path.join(convsup_path, "beam_models", "water_mu.dat"),
            "phantom_path": os.path.join(convsup_path, "simulations", phantom_filename),
            "kernel_path": os.path.join(convsup_path, "beam_models", "truebeam.kernel"),
            "spectrum_path": os.path.join(convsup_path, "beam_models", "truebeam.spectrum"),
            "dose_output_type": 0,
            "dose_output_threshold": 0.001,
            "unit_conversion": 1.0,
            "beams": beams,
        }

        with open(input_filename, "w") as myfile:
            myfile.write(json.dumps(input_dict, sort_keys=True, indent=4, separators=(',', ': ')))

        return [input_filename, phantom_filename]

    def _generate_cpt_beamlets(self, cpt, beamlet_creation, scenarios=None):
        if scenarios is None:
            scenarios = [[0.0, 0.0, 0.0]]

        beamlets = cpt.get_base_beamlets()
        beamlet_mask = cpt.get_beamlet_mask(bound=beamlet_creation.settings.get("model_leakage", True))

        theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
        cpt_dict = {
            "weight": 1.0,
            "x_size": cpt.iso_col_size / 10.0,
            "y_size": cpt.iso_row_size / 10.0,
            "energy": cpt.energy,
            "theta": math.degrees(theta),
            "phi": math.degrees(phi),
            "phicol": math.degrees(phicol),
            "d_source": cpt.d_source / 10.0,
            "d_coll": cpt.d_coll / 10.0
        }

        files_created = []

        for sc_i, shift in enumerate(scenarios):
            cpt_dict["iso"] = [(x + x_shift) / 10.0 for x, x_shift in zip(cpt.iso, shift)]

            beamlet_index = 0
            for beamlet, included in zip(beamlets, beamlet_mask):
                if included:
                    cpt_dict["apertures"] = [[(beamlet[0] - 0.5 * cpt.mlc_col_size) / 10.0,
                                        (beamlet[0] + 0.5 * cpt.mlc_col_size) / 10.0]]
                    cpt_dict["offset_y"] = beamlet[1] / 10.0

                    input_dict = self._make_beamlet_input(beamlet, cpt_dict, beamlet_creation)
                    filename = "{name}_{cpt_index}_{beamlet_index}_{energy}MV_{scenario_index}.inp".format(
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

                        filename = "{name}{gantry}_{couch}_{x}_{y}.inp".format(
                            name=beamlet_creation.name,
                            gantry=int(cpt.gantry_angle * 10),
                            couch=int(couch_id * 10),
                            x=int(x_id * 10),
                            y=int(y_id * 10)
                        )

                    with open(filename, "w") as myfile:
                        myfile.write(json.dumps(input_dict, sort_keys=True, indent=4, separators=(',', ': ')))

                    files_created.append(filename)

                beamlet_index += 1

        return files_created

    def _make_beamlet_input(self, beamlet, cpt, beamlet_creation):
        convsup_path = beamlet_creation.server.get_path("Convsup")
        phantom_filename = beamlet_creation.phantom_filename

        beams = self._make_beam_dicts([cpt])

        input_dict = {
            "mu_path": os.path.join(convsup_path, "beam_models", "water_mu.dat"),
            "spectrum_path": os.path.join(convsup_path, "beam_models", "spectral_6X.spectrum"),
            "kernel_path": os.path.join(convsup_path, "beam_models", "TrueBeam_6X_feb6.kernel"),
            "phantom_path": os.path.join(beamlet_creation.beamlet_path, phantom_filename),
            "dose_output_type": 2,  # BINDOSE
            "dose_output_threshold": 0.001,
            "unit_conversion": 1.0,
            "beams": beams
        }

        return input_dict

    def _beamlet_maker_template(self, convsup_path):
        submit_path = os.path.join(convsup_path, "submit_convsup.py")

        maker_str = ""

        maker_str += "from subprocess import call\n"
        maker_str += "import glob\n"
        maker_str += "import time\n"
        maker_str += "import os\n\n"

        maker_str += "beamlets = glob.glob('*.inp')\n"
        maker_str += "doses = glob.glob('*.bindos')\n"
        maker_str += "beamlets = set(['.'.join(b.split('.')[:-1]) for b in beamlets])\n"
        maker_str += "doses = set(['.'.join(d.split('.')[:-1]) for d in doses])\n"
        maker_str += "beamlets_left = list(beamlets.symmetric_difference(doses))\n"
        maker_str += "beamlets_left = [b + '.inp' for b in beamlets_left]\n"
        maker_str += "print('%i beamlets left' % len(beamlets_left))\n"

        maker_str += "for filename in beamlets_left:\n"
        maker_str += "    call('python {} %s' % filename, shell=True)\n".format(submit_path)

        beamlet_maker = "convsup_beamlet_maker.py"
        with open(beamlet_maker, "w") as myfile:
            myfile.write(maker_str)

        return beamlet_maker

    def generate_beamlets(self, beamlet_creation):
        server = beamlet_creation.server
        beamlet_directory = beamlet_creation.beamlet_path
        convsup_path = server.get_path("Convsup")

        files_to_send = [beamlet_creation.phantom_filename]
        files_to_send.append(self._beamlet_maker_template(convsup_path))

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

        server.put_files(files_to_send, beamlet_directory, make_dir=True, delete_original=True)

        command = '''
        cd %s
        python convsup_beamlet_maker.py
        exit
        ''' % (beamlet_directory)
        server.exec_shell_command(command)

    def make_aperture_inputs(self, aperture_recalc):
        cpts = self._process_aperture_cpts(aperture_recalc.control_points)
        server = aperture_recalc.server
        convsup_path = server.get_path("Convsup")
        beamlet_path = server.get_path("beamlets")

        beamlet_name = aperture_recalc.settings["name"]
        phantom_filename = aperture_recalc.phantom_filename

        mu_path = os.path.join(convsup_path, "beam_models", "water_mu.dat")
        spectrum_path = os.path.join(convsup_path, "beam_models", "spectral_6X.spectrum")
        kernel_path = os.path.join(convsup_path, "beam_models", "TrueBeam_6X_feb6.kernel")
        phantom_path = os.path.join(beamlet_path, beamlet_name, phantom_filename)

        files_created = []
        for index, cpt in enumerate(cpts):
            ap_name = aperture_recalc.settings["name"] + "_ap%i" % index
            input_filename = ap_name + self.input_extension
            beams = self._make_beam_dicts([cpt])

            input_dict = {
                "mu_path": mu_path,
                "spectrum_path": spectrum_path,
                "kernel_path": kernel_path,
                "phantom_path": phantom_path,
                "dose_output_type": 1,
                "dose_output_threshold": 0.001,
                "unit_conversion": 1.0,
                "beams": beams
            }

            with open(input_filename, "w") as myfile:
                myfile.write(json.dumps(input_dict, sort_keys=True, indent=4, separators=(',', ': ')))

            files_created.append(input_filename)

        return files_created

    def generate_aperture_beamlets(self, aperture_recalc, submit_phantom=True):
        files_created = self.make_aperture_inputs(aperture_recalc)
        path_to_beamlets = aperture_recalc.server.get_path("beamlets")
        path_to_beamlets = os.path.join(path_to_beamlets, aperture_recalc.settings["name"])

        convsup_path = aperture_recalc.server.get_path("Convsup")

        files_created.append(self._beamlet_maker_template(convsup_path))
        if submit_phantom:
            files_created.append(aperture_recalc.phantom_filename)

        aperture_recalc.server.put_files(files_created, path_to_beamlets, make_dir=True, delete_original=True)

        command = '''
        cd %s
        python convsup_beamlet_maker.py
        exit
        ''' % (path_to_beamlets)
        aperture_recalc.server.exec_shell_command(command)

    def submit_sim(self, simulation):
        server = simulation.server
        files_to_send = self._make_sim_inputs(simulation)

        # Will make this cleaner later.
        input_filename = files_to_send[0]

        convsup_path = server.get_path("Convsup")
        script_path = os.path.join(convsup_path, "submit_convsup.py")
        sim_path = os.path.join(convsup_path, "simulations")
        server.put_files(files_to_send, sim_path)

        input_path = os.path.join(sim_path, input_filename)

        command = '''
        python {} {}
        exit
        '''.format(script_path, input_path)

        server.exec_shell_command(command)

    def check_sim_progress(self, simulation):
        server = simulation.server
        progress_path = os.path.join(self.sim_params["sim_path"], "simulations",
                                     simulation.name + ".progress")
        path_to_3ddose = os.path.join(self.sim_params["sim_path"], "simulations",
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

    def get_finished(self, simulation):
        server = simulation.server
        convsup_path = server.get_path("Convsup")
        dose_path = os.path.join(convsup_path, "simulations", simulation.name + ".3ddose")

        local_dose_name = simulation.name + ".3ddose"
        server.get_file(dose_path, local_dose_name)

        plan_uid = self.sim_params["plan_uid"]
        patient_uid = self.sim_params["patient_uid"]
        orient = self.sim_params["orient"]
        if "manual_norm" in self.sim_params and self.sim_params["manual_norm"]:
            norm_dose = self.sim_params["norm_dose"]
            norm_point = numpy.array([float(x) for x in self.sim_params["norm_point"]])
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm_dose, norm_point=norm_point)
        else:
            norm = self.norm
            dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm)
        return dose_template, error_template

    def check_beamlet_progress(self, beam_gen):
        server = beam_gen.server
        beamlet_path = server.get_path("beamlets")
        checker_path = os.path.join(beamlet_path, "count_beamlets.py")
        sim_path = os.path.join(beamlet_path, beam_gen.name)

        if server.stat(checker_path):
            stdin, stdout, stderr = server.exec_command("python {} {}".format(checker_path, sim_path))
            output = stdout.read()
            num_doses = int(output.split("/")[0])
            num_inputs = int(output.split("/")[1])
        else:
            stdin, stdout, stderr = server.exec_command("cd %s;ls *.bindos | wc" % (sim_path))
            output = stdout.read()
            num_doses = int(output.split()[0])
            stdin, stdout, stderr = server.exec_command("cd %s;ls *.inp | wc" % sim_path)
            output = stdout.read()
            num_inputs = int(output.split()[0])

        if num_doses == num_inputs:
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def check_aperture_progress(self, aperture_gen):
        return self.check_beamlet_progress(aperture_gen)
