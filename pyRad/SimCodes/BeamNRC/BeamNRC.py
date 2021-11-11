"""
BeamNRC simulation code module.

Copyright Marc-Andre Renaud, 2017
"""
import errno
import importlib
import os

import numpy

from pyRad.utils import egsdose_to_dicom, simdose_to_dicom


class BeamNRC(object):
    """BeamNRC simulation module."""

    def __init__(self, attrs=None):
        """
        Constructor.

        Acts as a factory class to instantiate the appropriate BeamModel
        class as well. A BeamModel module with the same name as
        sim_params["beam_model"] must exist.
        """
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

        if hasattr(self, "sim_params"):
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
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        queue = self.sim_params.get("queue", "user1")
        mem = self.sim_params.get("mem", 400)

        beam_files = files_to_send["beamnrc_file"]
        if "mlc_file" in files_to_send:
            beam_files += files_to_send["mlc_file"]
        if "jaws_file" in files_to_send:
            beam_files += files_to_send["jaws_file"]
        if "xjaws_file" in files_to_send:
            beam_files += files_to_send["xjaws_file"]
        if "yjaws_file" in files_to_send:
            beam_files += files_to_send["yjaws_file"]

        server.put_files(beam_files, beam_model_path)

        dosxyz_files = files_to_send["dosxyznrc_file"]

        server.put_files(dosxyz_files, dosxyznrc_path)

        if not simulation.settings.get("commissioning", False):
            server.put_files([simulation.phantom_filename], dosxyznrc_path)

        simulations_submitted = []

        for filename in files_to_send["dosxyznrc_file"]:
            # Strip extension from dosxyznrc filename
            dosxyz_filename = os.path.splitext(filename)[0]

            nthreads = int(self.sim_params["nthreads"])
            if nthreads > 1:
                command = '''
                cd {dosxyz_path}
                exb dosxyznrc {dosxyz_input} {pegs} {queue} p={nthreads} mem={mem} &
                exit
                '''.format(dosxyz_path=dosxyznrc_path,
                        dosxyz_input=dosxyz_filename,
                        pegs=self.beam_model.pegs_file,
                        queue=queue,
                        mem=mem,
                        nthreads=int(self.sim_params["nthreads"]))
            else:
                command = '''
                cd {dosxyz_path}
                exb dosxyznrc {dosxyz_input} {pegs} {queue} mem={mem} &
                exit
                '''.format(dosxyz_path=dosxyznrc_path,
                        dosxyz_input=dosxyz_filename,
                        pegs=self.beam_model.pegs_file,
                        queue=queue,
                        mem=mem)

            server.exec_shell_command(command)

            simulations_submitted.append(filename)

        return simulations_submitted

    def check_beamlet_progress(self, aperture_recalc):
        server = aperture_recalc.server
        beam_path = server.get_path("BeamNRC")
        path_to_3ddose = os.path.join(beam_path, "dosxyznrc")
        stdin, stdout, stderr = server.exec_command("cd %s;ls %s*.bindos | wc" % (path_to_3ddose, aperture_recalc.name))
        output = stdout.read()
        num_doses = int(output.split()[0])
        stdin, stdout, stderr = server.exec_command("cd %s;ls %s*.egsinp | wc" % (path_to_3ddose, aperture_recalc.name))
        output = stdout.read()
        num_inputs = int(output.split()[0])
        if num_doses == num_inputs:
            self._move_finished_beamlets(aperture_recalc)
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def check_aperture_progress(self, aperture_recalc):
        server = aperture_recalc.server
        beam_path = server.get_path("BeamNRC")
        path_to_3ddose = os.path.join(beam_path, "dosxyznrc")
        stdin, stdout, stderr = server.exec_command("cd %s;ls %s*.3ddose | wc" % (path_to_3ddose, aperture_recalc.name))
        output = stdout.read()
        num_doses = int(output.split()[0])
        stdin, stdout, stderr = server.exec_command("cd %s;ls %s*.egsinp | wc" % (path_to_3ddose, aperture_recalc.name))
        output = stdout.read()
        num_inputs = int(output.split()[0])
        if num_doses == num_inputs:
            self._process_finished_apertures(aperture_recalc)
            return "Done"
        else:
            return round(num_doses / float(num_inputs) * 100, 1)

    def check_sim_progress(self, simulation):
        """
        Return a number between 0 and 100 representing the progress of the simulation.

        Radify passes ssh and sftp channels, along with path to mc_code
        and simulation uid.

        If an error is found, return a string with the error message.
        """
        server = simulation.server
        beamnrc_path = server.get_path("BeamNRC")
        dosxyz_path = os.path.join(beamnrc_path, "dosxyznrc")

        sims_submitted = simulation.settings["sims_submitted"]
        nhist = int(simulation.settings["nhist"])

        progress = []
        sims_done = []
        for sim in sims_submitted:
            lock_path = os.path.join(dosxyz_path, sim.replace(".egsinp", ".lock"))
            dose_path = os.path.join(dosxyz_path, sim.replace(".egsinp", ".bindos"))
            try:
                server.stat(dose_path)
                progress.append(100.0)
                sims_done.append(sim)
            except IOError, e:

                server.stat(lock_path)
                stdin, stdout, stderr = server.exec_command("/bin/cat %s" % lock_path)
                cat_result = stdout.read()
                lock_content = cat_result.split()

                hists_left = float(lock_content[0])
                hists_done = float(lock_content[1])
                progress.append(hists_done / nhist * 100.0)

        if len(sims_done) == len(sims_submitted):
            return "Done"
        else:
            return round(sum(progress) / len(sims_submitted), 1)

    def get_finished(self, simulation):
        """
        Retrieve the output of a finished BeamNRC simulation.

        :param simulation: Simulation instance with unique identifier.
        """
        server = simulation.server
        beamnrc_path = server.get_path("BeamNRC")
        dosxyz_path = os.path.join(beamnrc_path, "dosxyznrc")
        command = '''
        cd {dosxyz_path}
        python clean.py {name}
        exit
        '''.format(dosxyz_path=dosxyz_path, name=simulation.name)
        server.exec_shell_command(command)

        doses = []
        for sim in simulation.settings["sims_submitted"]:
            dose_filename = sim.replace(".egsinp", ".bindos")
            dose_path = os.path.join(dosxyz_path, dose_filename)
            server.get_file(dose_path, dose_filename)
            doses.append(dose_filename)

        final_dose = self.beam_model.process_finished_doses(simulation, doses)

        plan_uid = self.sim_params["plan_uid"]
        patient_uid = self.sim_params["patient_uid"]
        orient = self.sim_params["orient"]
        if "manual_norm" in self.sim_params and self.sim_params["manual_norm"]:
            norm_dose = self.sim_params["norm_dose"]
            norm_point = numpy.array([float(x) for x in self.sim_params["norm_point"]])
            dose_template, error_template = simdose_to_dicom(final_dose, patient_uid, plan_uid, orient, norm=norm_dose, norm_point=norm_point)
        else:
            norm = self.beam_model.get_calibration(simulation.ref_plan.get_plan_type_object(), simulation)
            dose_template, error_template = simdose_to_dicom(final_dose, patient_uid, plan_uid, orient, norm=norm)

        for dose in doses:
            os.remove(dose)

        return dose_template, error_template

    def _load_beam_model(self, name):
        beam_model = importlib.import_module("pyRad.SimCodes.BeamNRC.BeamModels.{}.{}".format(name, name))
        beam_class = getattr(beam_model, name)
        self.beam_model = beam_class()
        return self.beam_model

    def generate_aperture_beamlets(self, aperture_recalculation, submit_phantom=True, run_jobs=True):
        server = aperture_recalculation.server

        files_to_send = self.beam_model.make_aperture_inputs(aperture_recalculation)

        beamnrc_path = server.get_path("BeamNRC")
        beam_folder = self.beam_model.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        beam_files = (files_to_send["beamnrc_files"] +
                      files_to_send["mlc_files"] +
                      files_to_send["jaws_files"])
        server.put_files(beam_files, beam_model_path)

        dosxyz_files = files_to_send["dosxyznrc_files"]
        server.put_files(dosxyz_files, dosxyznrc_path)

        if submit_phantom:
            server.put_files([aperture_recalculation.phantom_filename], dosxyznrc_path)

        nthreads = int(self.sim_params["nthreads"])

        if run_jobs:
            for dosxyz_filename in files_to_send["dosxyznrc_files"]:
                egsinp_stripped = os.path.splitext(dosxyz_filename)[0]

                if nthreads > 1:
                    command = '''
                    cd {dosxyz_path}
                    exb dosxyznrc {input} {pegs} p={nthreads} &
                    exit
                    '''.format(dosxyz_path=dosxyznrc_path,
                            input=egsinp_stripped,
                            pegs=self.beam_model.pegs_file,
                            nthreads=int(self.sim_params["nthreads"]))
                else:
                    command = '''
                    cd {dosxyz_path}
                    exb dosxyznrc {input} {pegs} &
                    exit
                    '''.format(dosxyz_path=dosxyznrc_path, input=egsinp_stripped, pegs=self.beam_model.pegs_file)

                server.exec_shell_command(command)

        return files_to_send.get("processed_cpts", [])

    def generate_beamlets(self, beamlet_creation):
        server = beamlet_creation.server
        files_to_send = [beamlet_creation.phantom_filename]

        files_to_send += self.beam_model.generate_cpt_beamlets(beamlet_creation)

        beamnrc_path = server.get_path("BeamNRC")
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        server.put_files(files_to_send, dosxyznrc_path)

        this_dir = os.path.dirname(__file__)
        checkmissing = os.path.join(this_dir, "checkmissing.py")
        server.put_files([checkmissing], dosxyznrc_path, delete_original=False)

        command = '''
        cd %s
        python %s %s %s
        exit
        ''' % (dosxyznrc_path, "checkmissing.py", self.sim_params["name"], self.beam_model.pegs_file)

        server.exec_shell_command(command)

    def _move_finished_beamlets(self, aperture_recalc):
        server = aperture_recalc.server
        beam_path = server.get_path("BeamNRC")
        path_to_3ddose = os.path.join(beam_path, "dosxyznrc")
        beamlet_path = server.get_path("beamlets")
        beamlet_folder = os.path.join(beamlet_path, aperture_recalc.name)
        dose_files_mv = os.path.join(path_to_3ddose, "%s*.bindos" % aperture_recalc.name)
        stdin, stdout, stderr = server.exec_command("mkdir %s;mv %s %s" % (beamlet_folder, dose_files_mv, beamlet_folder))

    def _process_finished_apertures(self, aperture_recalc):
        server = aperture_recalc.server
        beam_path = server.get_path("BeamNRC")
        path_to_3ddose = os.path.join(beam_path, "dosxyznrc")
        beamlet_path = server.get_path("beamlets")
        beamlet_folder = os.path.join(beamlet_path, aperture_recalc.name)
        dose_files_mv = os.path.join(path_to_3ddose, "%s*.3ddose" % aperture_recalc.name)
        stdin, stdout, stderr = server.exec_command("mkdir %s;mv %s %s" % (beamlet_folder, dose_files_mv, beamlet_folder))
        stdout.read()

        idx = 0
        for index, cpt in enumerate(aperture_recalc.control_points):
            mu_calibration = 1.0 / self.beam_model.get_calibration_factor(aperture_recalc, aperture_recalc.settings["dicom_cpts"][index])
            if cpt.apertures:
                for _ in cpt.apertures:
                    # TEMPORARY assume sc0 for now
                    beamlet_filename = "{name}_ap{index}_sc0.3ddose".format(name=aperture_recalc.name, index=idx)
                    beamlet_filepath = os.path.join(beamlet_folder, beamlet_filename)
                    stdin, stdout, stderr = server.exec_command("/usr/bin/rad_process -r {norm} smooth -i -b {file_path}".format(norm=mu_calibration, file_path=beamlet_filepath))
                    # Try a nonblocking call by commenting out stdout.read()
                    stdout.read()
                    idx += 1
