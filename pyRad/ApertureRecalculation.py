import importlib

class ApertureRecalculation(object):
    """
        Attributes:
        name (string): Name of beamlet creation instance.
        server (Server instance): Server where beamlets are to be created.
        aperture_path (str): Path where apertures are stored on server.
        components (list): List of plan components
        settings (dict): Simulation settings for sim program

        There must be a module and class with the same name as sim_program in
        the SimCodes folder.

    """

    sim_program = None
    aperture_path = None
    settings = {}
    server = None
    control_points = []

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

        simModule = importlib.import_module("pyRad.SimCodes.{}".format(self.sim_program))
        simClass = getattr(simModule, self.sim_program)
        self.sim_program = simClass({"sim_params": self.settings})

    def get_beamlet_filenames(self):
        beamlet_files = self.server.as_object().get_file_list(self.aperture_path)

        beam_list = [dose_file for dose_file
                     in beamlet_files if self.sim_program.dose_extension in dose_file]

        return beam_list

    def generate_aperture_beamlets(self, submit_phantom=True, run_jobs=True):
        return self.sim_program.generate_aperture_beamlets(self, submit_phantom, run_jobs)

    def check_progress(self):
        return self.sim_program.check_aperture_progress(self)

    def get_dicom_cpts(self):
        if hasattr(self, "sim_program") and hasattr(self.sim_program, "beam_model"):
            # MLC type is specified either in the beam model itself, or as a parameter in the case
            # of generic models.
            try:
                mlc_type = self.settings["linac"]["mlc"]
            except KeyError:
                mlc_type = self.sim_program.beam_model.mlc_type

            return self.sim_program.beam_model.opt_to_dicom_cpts(self.control_points, mlc_type)
        else:
            return []