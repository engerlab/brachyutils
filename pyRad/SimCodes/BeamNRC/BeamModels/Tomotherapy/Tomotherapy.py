"""
Tomotherapy module.

Copyright Marc-Andre Renaud, 2017
"""
import math
import os

import numpy

from pyRad.utils import dicom_to_spherical
from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel

class Tomotherapy(BeamModel):
    """Tomotherapy beam model."""

    model_name = "tomo"
    folder = "BEAM_tomo"
    pegs_file = "radify"
    num_leaves = 64  # There's actually 66 leaves but the first and last are always closed
    last_cm_z = 401.1  # in mm
    calibration_factor = 4.5366e16  # Obtained from 8.88 Gy/min at Dmax for a 40x5 field.
    default_spacing = 360.0 / 51  # Control points are sampled every 360 / 51 degrees.

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this
        """
        self.template_folder = os.path.dirname(__file__)

    def make_sim_inputs(self, simulation):
        sim_pos_error = simulation.settings.get("positioning_error", False)
        error_coords = [float(x) for x in simulation.settings["error_coords"]] if sim_pos_error else None

        cpts = self._process_beams(simulation.beams, positioning_error=error_coords)

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

        zero_air_dose = int(bool(simulation.settings.get("zero_air_dose", 1)))
        params["zero_air_dose"] = zero_air_dose


        mlc_filename = self._make_mlc_file(params)
        params["mlc_file"] = os.path.join(beam_model_path,
                                          mlc_filename)

        # Find appropriate tomo model based on the field size of the first
        # control point.
        self.tomo_model = self._find_tomo_model(simulation.beams[0]["cpts"][0])

        files_created["mlc_file"] = [mlc_filename]

        files_created["beamnrc_file"] = [self._write_beamnrc_input(params)]
        files_created["dosxyznrc_file"] = [self._write_dosxyznrc_input(params)]

        return files_created

    def get_beamnrc_template(self, params):
        """
        Override default BeamModel implementation of this method.

        Assumes that _find_tomo_model() has already been called.
        """
        return self.tomo_model

    def get_calibration(self, params):
        """Override the default implementation since tomo doesn't use MUs."""
        total_time = params["total_meterset"]
        return total_time * self.calibration_factor

    def _make_mlc_file(self, params):
        """Override default implementation due to tomo special MLC logic."""
        mlc_filename = params["name"] + ".mlc"
        num_cpts = len(params["cpts"])

        with open(mlc_filename, "w") as mlc_file:
            mlc_file.write("SYNCVMLC file\n")
            mlc_file.write("{}\n".format(num_cpts))
            for cpt in params["cpts"]:
                mlc_file.write("{weight:.10}\n".format(weight=cpt["weight"]))
                mlc_file.write("-5.0, -4.9, 1\n")

                for ap in cpt["apertures"]:
                    mlc_file.write("{neg:.5}, {pos:.5}, 1\n".format(neg=ap[0], pos=ap[1]))

                mlc_file.write("4.9, 5.0, 1\n")

        return mlc_filename

    def _find_tomo_model(self, first_cpt):
        allowable_sizes = numpy.array([10, 25, 50])
        field_size = -first_cpt.y_jaw[0] + first_cpt.y_jaw[1]

        field_index = numpy.argmin(numpy.abs(allowable_sizes - field_size))
        model_name = self.model_name + "_%imm.egsinp" % allowable_sizes[field_index]

        return model_name

    def _process_beams(self, beams, positioning_error=None):
        """Override the default implementation due to special tomo parsing."""
        if positioning_error is None:
            positioning_error = [0.0, 0.0, 0.0]

        processed_cpts = []

        total_weight = sum([beam["cpts"][-1].cum_weight for beam in beams])

        accumulated_weight = 0.0
        for beam in beams:
            for cpt in beam["cpts"]:
                cpt_dict = {}
                cpt_dict["apertures"] = cpt.apertures
                cpt_dict["weight"] = (accumulated_weight + cpt.cum_weight) / total_weight
                # Perturb iso in the opposite of position error direction if an error is specified.
                cpt_dict["iso"] = [(x - x_err) / 10.0 for (x, x_err) in zip(cpt.iso, positioning_error)]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle + self.default_spacing / 2.0,
                                                        cpt.couch_angle,
                                                        cpt.col_angle)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol) - 90.0

                cpt_dict["dsource"] = (cpt.sad - self.last_cm_z) / 10.0

                processed_cpts.append(cpt_dict)

            accumulated_weight += beam["cpts"][-1].cum_weight

        return processed_cpts
