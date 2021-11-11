"""
Generic Varian beam model module.

Copyright Marc-Andre Renaud, 2017
"""
import os

import numpy

from pyRad.SimCodes.BeamNRC.BeamModels.GenericElektaAgility import GenericElektaAgility

class AgilityStat(GenericElektaAgility.GenericElektaAgility):
    """
    Generic Varian statistical beam model. Inherit everything from GenericElektaAgility
    and only redefine the input file writing routines.
    """

    model_name = "AgilityStat"
    folder = "BEAM_AgilityStat"
    pegs_file = "radify"
    particle = "photon"

    output_corr = {
        "fs": numpy.array([30, 40, 50, 100, 200, 300]),
    }

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this.
        """
        self.template_folder = os.path.dirname(__file__)

    def make_sim_inputs(self, simulation):
        """
        Create input files for DOSXYZnrc simulatation using a BEAMnrc beam model.
        """

        sim_pos_error = simulation.settings.get("positioning_error", False)
        error_coords = [float(x) for x in simulation.settings["error_coords"]] if sim_pos_error else None

        cpts = self._process_beams(simulation.beams,
                                   simulation.settings["linac"]["mlc"])

        beam_energy = cpts[0].get("energy", 6)
        energy_identifier = str(int(beam_energy))
        if simulation.beams[0].get("FFF", False):
            energy_identifier += "FFF"

        # Specific data related to Generic models
        generic_params = simulation.settings["linac"]["beam_parameters"][energy_identifier]

        beamnrc_path = simulation.server.get_path("BeamNRC")
        beam_model_path = os.path.join(beamnrc_path, self.folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        beam_inputfile = simulation.name + "_beam.egsinp"

        # DBS radius should be larger than the largest simulation field. Actual size
        # depends on whether user is concerned about out of field dose. Default setting is
        # quite large compared to field size.

        params = {
            "name": simulation.name,
            "mlc_type": simulation.settings["linac"]["mlc"],
            "e_energy": generic_params["e_energy"],
            "e_fwhm": generic_params["e_fwhm"],
            "FFF": simulation.beams[0].get("FFF", False),
            "cpts": cpts,
            "energy": beam_energy,
            "nhist": simulation.settings["nhist"],
            "title": "Plan recalculation",
            "zero_air_dose": int(bool(simulation.settings.get("zero_air_dose", 1))),
            "delivery_mode": 1,  # dynamic
            "pegs_file": self.pegs_file,
            "beam_model": self.folder,
            "beam_inputfile": beam_inputfile,
            "phantom_path": os.path.join(dosxyznrc_path, simulation.phantom_filename),
            "dbs_radius": round(self._find_max_jaw(simulation.beams) / 10.0 + 10.0, 3),
            "z_phsp" : simulation.settings["z_phsp"]
        }

        params["filler_depth"] = simulation.settings.get("filler_depth", 0.0) + self.min_filler
        params["dbs_rejection"] = 60.0 + params["filler_depth"] - 1.0  # dbs rejection plane 1 cm above scoring plane.

        params["commissioning"] = simulation.settings.get("commissioning", False)
        params["commissioning_type"] = simulation.settings.get("commissioning_type", None)

        if params["commissioning_type"] == "profile":
            measurement_depth = simulation.settings.get("measurement_depth", 10.0)
            comm_dict = self._process_commissioning(simulation.beams, measurement_depth)
            params.update(comm_dict)
        elif params["commissioning_type"] == "calibration":
            measurement_depth = simulation.settings.get("measurement_depth", 1.5)
            params["calib_vox_size"] = 0.2
            params["phantom_depth"] = measurement_depth - 1.5 * params["calib_vox_size"]

        params["beam_model_path"] = beam_model_path
        mlc_filename = self._make_mlc_file(params)
        jaws_filename = self._make_jaws_file(params)
        params["mlc_file"] = os.path.join(beam_model_path,
                                          mlc_filename)
        params["jaws_file"] = os.path.join(beam_model_path,
                                           jaws_filename)

        files_created = {}
        files_created["mlc_file"] = [mlc_filename]
        files_created["jaws_file"] = [jaws_filename]
        files_created["beamnrc_file"] = [self._write_beamnrc_input(params)]
        files_created["dosxyznrc_file"] = [self._write_dosxyznrc_input(params)]

        return files_created

    def get_beamnrc_template(self, params):
        """
        Return filename of BEAMnrc template. Same template file, only stats
        model filename changes.
        """

        template = self.model_name + ".egsinp"

        return template

    def _get_stats_model(self, params):
        """
        Return the filename of the statistical model for the given
        beam quality, energy and spot size.
        """
        ident = params["energy"]
        if params["FFF"] is True:
            ident = str(ident) + "FFF"

        e_fwhm = params["e_fwhm"]
        e_energy = params["e_energy"]

        return "AgilityStat_{ident}_{energy}_{fwhm}.mcmodel".format(ident=ident, energy=e_energy, fwhm=e_fwhm)

    def _write_beamnrc_input(self, params):
        """
        Write BEAMnrc input file from params dict.

        The template string is formatted twice. Once to include the MLC
        template, then once more to fill in all the parameters defined by
        the params dict.
        """
        beamnrc_filename = self.get_beamnrc_template(params)
        params["stats_model"] = os.path.join(params["beam_model_path"], self._get_stats_model(params))
        #params["z_phsp"] = self.z_phsp

        template_path = os.path.join(self.template_folder, beamnrc_filename)
        mlc_path = os.path.join(self.template_folder, params["mlc_type"] + ".mlc")
        with open(template_path) as template_file:
            template_string = template_file.read()

        with open(mlc_path) as mlc_template_file:
            mlc_string = mlc_template_file.read()

        template_string = template_string.format(mlc_template=mlc_string)

        beam_filename = params["name"] + "_beam.egsinp"
        with open(beam_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beam_filename
