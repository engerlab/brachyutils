"""
CL21EX electron beam model module.

Copyright Marc-Andre Renaud, 2017
"""
import math
import os

import numpy

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel
from pyRad.utils import dicom_to_spherical


class CL21EX_E(BeamModel):
    """CL21EX electron model with Millenium MLC."""

    model_name = "CL21EX_E"
    folder = "BEAM_CL21EX_E"
    pegs_file = "radify"
    particle = "electron"

    dcoll = 51.5785  # Distance between source and middle of MLC plane
    default_dsource = 40.0
    phsp_dsource = 48.4215

    y_z_jaw_front = 28.0
    y_z_jaw_back = 35.6485
    x_z_jaw_front = 36.7
    x_z_jaw_back = 44.3485

    num_leaves = 60
    leaf_radius = 8.0
    abut_gap = 0.03

    leaf_boundaries = numpy.array([-200.0, -190.0, -180.0, -170.0, -160.0, -150.0, -140.0, -130.0, -120.0, -110.0, -100.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0])

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
        cpts = self._process_beams(simulation.beams)

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
        params["delivery_mode"] = simulation.settings["delivery_mode"]

        zero_air_dose = simulation.settings.get("zero_air_dose", 1)
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

        cpts = self._process_aperture_cpts(beamlet_creation.control_points)
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

    def generate_cpt_beamlets(self, beamlet_creation):
        global_files_created = []
        for cpt in beamlet_creation.control_points:
            beamlets = cpt.get_base_beamlets()
            beamlet_mask = cpt.get_beamlet_mask()

            files_created = []

            beamlet_index = 0
            for beamlet, included in zip(beamlets, beamlet_mask):
                if included:
                    filename = self._make_beamlet_input(beamlet_index, cpt, beamlet_creation)
                    files_created.append(filename)

                beamlet_index += 1

            global_files_created += files_created

        return global_files_created

    def _write_dosxyznrc_phsp_input(self, params, filename):
        template_filename = "dosxyznrc_phsp_template.egsinp"
        template_path = os.path.join(self.template_folder, template_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        input_string = template_string.format(**params)
        with open(filename, "w") as myfile:
            myfile.write(input_string)

        return filename

    def _make_beamlet_input(self, beamlet_index, cpt, beamlet_creation):
        server = beamlet_creation.server
        beamnrc_path = server.get_path("BeamNRC")
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        phantom_filename = beamlet_creation.phantom_filename

        phantom_path = os.path.join(dosxyznrc_path, phantom_filename)

        theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle,
                                                cpt.couch_angle,
                                                cpt.col_angle)

        temp_path = "/home/webtps/CL21EX_E_raytraced_phsp"
        phsp_file = "CL21EX_E_%iMeV_%i.egsphsp1" % (cpt.energy, beamlet_index)
        phsp_path = os.path.join(temp_path, phsp_file)
        params = {
            "phsp_path": phsp_path,
            "phantom_path": phantom_path,
            "theta": math.degrees(theta),
            "phi": math.degrees(phi),
            "phicol": math.degrees(phicol),
            "dsource": self.phsp_dsource,
            "iso_x": cpt.iso[0] / 10.0,
            "iso_y": cpt.iso[1] / 10.0,
            "iso_z": cpt.iso[2] / 10.0,
            "nhist": 1000000,
            "title": "beamletGeneration"
        }

        filename = "%s_%i_%i_%iMeV.egsinp" % (beamlet_creation.settings["name"], cpt.index, beamlet_index, cpt.energy)
        self._write_dosxyznrc_phsp_input(params, filename)

        return filename
