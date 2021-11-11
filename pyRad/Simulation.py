"""
Simulation module.

Copyright Marc-Andre Renaud, 2017
"""
import importlib


class Simulation(object):
    """
    Wrapper for all simulation programs.

    A SimCode module must exist with the same name as the sim_program attribute.
    """

    def __init__(self, attrs):
        """
        Constructor.

        :param server: Server where beamlets are to be created.
        :param ref_plan: Plan instance pointing to DICOM RTPlan file.
        :param sim_program: Simulation program used to generate beamlets.
        :param settings: Simulation settings for sim program
        :param plan_data: Plan data for the simulation. Usually holds data like num. monitor units.
        :param phantom_filename: Filename of phantom file for simulations.
        :param phantom_parameters: Phantom parameters.
        """
        for k, v in attrs.items():
            setattr(self, k, v)

        sim_module = importlib.import_module("pyRad.SimCodes.{}".format(self.sim_program))
        sim_class = getattr(sim_module, self.sim_program)
        self.sim_program = sim_class({"sim_params": self.settings})

    def get_sim_progress(self):
        """Get progress of remote simulation run."""
        return self.sim_program.check_sim_progress(self)

    def get_finished(self):
        """Retrieve the output of a finished simulation."""
        return self.sim_program.get_finished(self)

    def submit_sim(self):
        """Submit a simulation to a remote server."""
        return self.sim_program.submit_sim(self)
