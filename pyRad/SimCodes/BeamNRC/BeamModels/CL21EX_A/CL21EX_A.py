"""
CL21EX photon module.

Copyright Marc-Andre Renaud, 2017
"""
import os

import numpy

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel


class CL21EX_A(BeamModel):
    """CL21EX photon model with Millenium MLC."""

    model_name = "CL21EX_A"
    folder = "BEAM_CL21EX_A"
    pegs_file = "radify"
    particle = "photon"

    calibration_factors = {
        "6": 1.12721e-14  # MU / primary history
    }

    default_dsource = 40.0

    y_z_jaw_front = 28.0
    y_z_jaw_back = 35.6485
    x_z_jaw_front = 36.7
    x_z_jaw_back = 44.3485
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16

    mlc_type = "VMLC"

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this
        """
        self.template_folder = os.path.dirname(__file__)

    def get_beamnrc_template(self, params):
        """Override default BeamModel implementation of this method."""
        return self.model_name + "_{}.egsinp".format(params["energy"])

    def make_sim_inputs(self, simulation):
        cpts = self._process_beams(simulation.beams, self.mlc_type)

        files_created = {}

        beamnrc_path = simulation.server.get_path("BeamNRC")
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params = {"cpts": cpts}
        params["name"] = simulation.name
        try:
            params["energy"] = cpts[0]["energy"]
        except KeyError:
            print "Energy not found, assuming 6 MV"
            params["energy"] = 6

        params["pegs_file"] = self.pegs_file
        params["beam_inputfile"] = params["name"] + "_beam.egsinp"
        params["beam_model"] = self.folder
        params["title"] = "Plan recalculation"
        params["nhist"] = simulation.settings["nhist"]
        params["phantom_path"] = os.path.join(dosxyznrc_path,
                                              simulation.phantom_filename)

        params["dbs_radius"] = round(self._find_max_jaw(simulation.beams) / 10.0 + 10.0, 3)

        zero_air_dose = simulation.settings.get("zero_air_dose", 1)
        if not zero_air_dose:
            zero_air_dose = 0
        else:
            zero_air_dose = 1
        params["zero_air_dose"] = zero_air_dose

        # Automatically put dynamic delivery mode since static modes
        # are taken into account by duplicating control points.
        params["delivery_mode"] = 1

        mlc_filename = self._make_mlc_file(params)
        jaws_filename = self._make_jaws_file(params)
        params["mlc_file"] = os.path.join(beam_model_path,
                                          mlc_filename)
        params["jaws_file"] = os.path.join(beam_model_path,
                                           jaws_filename)

        files_created["mlc_file"] = mlc_filename
        files_created["jaws_file"] = jaws_filename
        files_created["beamnrc_file"] = self._write_beamnrc_input(params)
        files_created["dosxyznrc_file"] = self._write_dosxyznrc_input(params)

        return files_created

    def make_aperture_inputs(self, beamlet_creation):
        files_created = {}
        files_created["mlc_files"] = []
        files_created["jaws_files"] = []
        files_created["beamnrc_files"] = []
        files_created["dosxyznrc_files"] = []

        cpts = self._process_aperture_cpts(beamlet_creation.control_points, self.mlc_type)
        server = beamlet_creation.server

        beamnrc_path = server.get_path("BeamNRC")
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")
        for index, cpt in enumerate(cpts):
            cpt_copy = cpt.copy()
            cpt_copy["weight"] = 0.0
            params = {"cpts": [cpt_copy, cpt]}
            params["name"] = beamlet_creation.settings["name"] + "_ap%i" % index

            params["energy"] = cpt["energy"]
            params["pegs_file"] = self.pegs_file
            params["beam_inputfile"] = params["name"] + ".egsinp"
            params["beam_model"] = self.folder
            params["title"] = "Beamlet apertures"
            params["nhist"] = beamlet_creation.settings["nhist"]

            params["phantom_path"] = os.path.join(dosxyznrc_path,
                                                  beamlet_creation.phantom_filename)
            params["delivery_mode"] = 2  # Step and shoot

            zero_air_dose = beamlet_creation.settings.get("zero_air_dose", 1)
            if not zero_air_dose:
                zero_air_dose = 0
            else:
                zero_air_dose = 1
            params["zero_air_dose"] = zero_air_dose

            mlc_filename = self._make_mlc_file(params)
            jaws_filename = self._make_jaws_file(params)

            params["mlc_file"] = os.path.join(beam_model_path,
                                              mlc_filename)
            params["jaws_file"] = os.path.join(beam_model_path,
                                               jaws_filename)

            beamnrc_filename = self._write_beamnrc_input(params)
            dosxyznrc_filename = self._write_dosxyznrc_input(params)

            files_created["mlc_files"].append(mlc_filename)
            files_created["jaws_files"].append(jaws_filename)
            files_created["beamnrc_files"].append(beamnrc_filename)
            files_created["dosxyznrc_files"].append(dosxyznrc_filename)

        return files_created
