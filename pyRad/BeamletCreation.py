import importlib


class BeamletCreation(object):
    """
        Attributes:
        name (string): Name of beamlet creation instance.
        server (Server instance): Server where beamlets are to be created.
        beamlet_path (str): Path where beamlets are stored on server.
        sim_program (str): Simulation program used to generate beamlets.
        settings (dict): Simulation settings for sim program.
        total_filesize (int): Total size of beamlet folder in megabytes.
        (Optional) control_points (list): Control points for beamlet generation.


        There must be a module and class with the same name as sim_program in
        the SimCodes folder.

    """

    def __init__(self, attrs=None):
        if attrs is None:
            raise Exception("BeamletCreation must be instanciated with attributes. See documentation.")\

        self.name = attrs["name"]
        self.server = attrs["server"]
        self.beamlet_path = attrs["beamlet_path"]
        self.sim_program = attrs["sim_program"]
        self.settings = attrs["settings"]
        self.phantom_filename = attrs["phantom_filename"]
        self.folder_size = attrs["folder_size"]

        if "control_points" in attrs:
            self.control_points = attrs["control_points"]

        sim_module = importlib.import_module("pyRad.SimCodes.{}".format(self.sim_program))
        sim_class = getattr(sim_module, self.sim_program)
        self.sim_program = sim_class({"sim_params": self.settings})

    def get_beamlet_filenames(self):
        beamlet_files = self.server.as_object().get_file_list(self.beamlet_path)

        beam_list = [dose_file for dose_file
                     in beamlet_files if self.sim_program.dose_extension in dose_file]

        return beam_list

    def check_progress(self):
        return self.sim_program.check_beamlet_progress(self)

    def get_beamlet_progress(self):
        """Included for backwards compatibility."""
        return self.check_progress()

    def generate_beamlets(self):
        return self.sim_program.generate_beamlets(self)
