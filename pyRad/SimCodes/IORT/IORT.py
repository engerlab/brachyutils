import os
import math
import numpy
import errno
from pyRad.utils import dicom_to_spherical
from pyRad.utils import egsdose_to_dicom

class IORT(object):
    input_extension = ".inp"
    beamlet_extension = ".minidos"

    @staticmethod
    def check_beamlet_progress(server, sim_path, dose_extension="minidos"):
        stdin, stdout, stderr = server.exec_command("cd %s;ls *.%s | wc" % (sim_path, dose_extension))
        output = stdout.read()
        num_doses = int(output.split()[0])
        stdin, stdout, stderr = server.exec_command("cd %s;ls *.mac | wc" % sim_path)
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


    def _generate_input(self, simulation):
        if "dose_output" not in self.sim_params:
            self.sim_params["dose_output"] = "3ddose"

        self.sim_params["phantom_filename"] = simulation.phantom_filename

        self.sim_params["file_id"] = self.sim_params["name"]
        mac_file = self._gen_mac_file()

        return [mac_file, simulation.phantom_filename]

    def get_phsp_filename(self, radius):
        if str(radius) == "0.0":
            filename = "bareprobe"
        elif str(radius) == "3.0":
            filename = "app30"
        elif str(radius) == "3.5":
            filename = "app35"
        elif str(radius) == "4.0":
            filename = "app40"
        elif str(radius) == "4.5":
            filename = "app45"
        else:
            print "Radius does not match known radii: ", radius
            print "Returning bareprobe"
            filename = "bareprobe"

        return filename

    def _gen_slurm_file(self):
        slurm_filename = self.sim_params["name"] + ".sh"
        with open(slurm_filename, "w") as slurmfile:
            slurmfile.write("""#!/bin/bash
source setup_geant_env.sh
$BSOURCE_DIR/IORT {}
""".format(self.sim_params["name"] + ".mac"))

        return slurm_filename

    def _gen_mac_file(self):
        input_filename = self.sim_params["file_id"] + ".mac"
        self.sim_params["phsp_filename"] = self.get_phsp_filename(self.sim_params["intraprobe"]["radius"])

        template_dir = os.path.dirname(__file__)
        template_filename = "IORT.template"
        template_path = os.path.join(template_dir, template_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        with open(input_filename, "w") as myfile:
            myfile.write(template_string.format(**self.sim_params))

        return input_filename

    def submit_sim(self, simulation):
        server = simulation.server
        files_to_send = self._generate_input(simulation)
        slurm_file = self._gen_slurm_file()
        files_to_send.append(slurm_file)

        iort_path = server.get_path("IORT")
        server.put_files(files_to_send, iort_path, delete_original=True)

        command = '''
        cd %s
        sbatch -n1 %s
        exit
        ''' % (iort_path, slurm_file)
        server.exec_shell_command(command)

    def check_sim_progress(self, simulation):
        """
            WebTPS passes ssh and sftp channels, along with path to mc_code
            and simulation uid.

            If the simulation is still in progress, return a number between
            0 and 100 representing the progress of the simulation.

            If the simulation is finished, retrieve the dose/uncertainties,
            convert to dicom and return the dicom objects.

            If an error is found, return a string with the error message.
        """
        server = simulation.server
        sim_path = server.get_path("IORT")
        progress_path = os.path.join(sim_path,
                                     simulation.name + ".progress")
        path_to_3ddose = os.path.join(sim_path,
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
                    stdin, stdout, stderr = ssh.exec_command("/bin/cat %s"
                                                             % progress_path)
                    cat_result = stdout.read()
                    progress = cat_result.split("/")

                    # Rough approximation of the progress. The current batch
                    # being processed won't show up in the lock file so there
                    # is a 10%/Nthreads error associated with the progress bar.
                    hists_done = float(progress[0])
                    hists_total = float(progress[1])
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
        norm_dose = self.sim_params["norm_dose"]
        norm_point = numpy.array([float(x) for x in self.sim_params["norm_point"]])
        dose_template, error_template = egsdose_to_dicom(local_dose_name, patient_uid, plan_uid, orient, remove_original=True, norm=norm_dose, norm_point=norm_point)

        return dose_template, error_template