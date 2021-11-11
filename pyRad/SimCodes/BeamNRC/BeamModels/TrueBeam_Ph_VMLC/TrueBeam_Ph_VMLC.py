"""
TrueBeam photon VMLC module.

Copyright Marc-Andre Renaud, 2017
"""
import os
import math

import numpy

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel
from pyRad.utils import dicom_to_spherical

class TrueBeam_Ph_VMLC(BeamModel):
    """TrueBeam photon model with Millenium MLC."""

    model_name = "TrueBeam_Ph_VMLC"
    folder = "BEAM_TrueBeam_M"
    pegs_file = "radify"
    particle = "photon"

    calibration_factors = {
        "6": 1.4374e-14,  # MU / primary history
        "6FFF": 3.09763e-14,
        "10": 3.22125e-14,
        "10FFF": 1.05566472e-13
    }

    default_dsource = 100.0  # dosxyz source 20 uses SAD as dsource if IAEA phsp file
    phsp_dsource = 45.093

    y_z_jaw_front = 27.89
    y_z_jaw_back = 35.66
    x_z_jaw_front = 36.61
    x_z_jaw_back = 44.41
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16

    mlc_type = "VMLC"
    min_filler = 0.01

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this
        """
        self.template_folder = os.path.dirname(__file__)

    def make_sim_inputs(self, simulation):
        orient = simulation.settings.get("orient", "HFS")
        sim_pos_error = simulation.settings.get("positioning_error", False)
        error_coords = [float(x) for x in simulation.settings["error_coords"]] if sim_pos_error else None

        cpts = self._process_beams(simulation.beams, self.mlc_type,
                                   positioning_error=error_coords, orient=orient)

        files_created = {}

        beamnrc_path = simulation.server.get_path("BeamNRC")
        beam_model_path = os.path.join(beamnrc_path, self.folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        beam_inputfile = simulation.name + "_beam.egsinp"

        energy = str(cpts[0].get("energy", 6))
        if "FFF" in simulation.beams[0] and simulation.beams[0]["FFF"] is True:
            energy += "FFF"

        phsp_name = "{model}_{energy}.IAEAphsp".format(model=self.model_name, energy=int(energy))

        params = {
            "name": simulation.name,
            "cpts": cpts,
            "nhist": simulation.settings["nhist"],
            "title": "Plan recalculation",
            "zero_air_dose": int(bool(simulation.settings.get("zero_air_dose", 1))),
            "delivery_mode": 1,
            "pegs_file": self.pegs_file,
            "beam_model": self.folder,
            "beam_inputfile": beam_inputfile,
            "phantom_path": os.path.join(dosxyznrc_path, simulation.phantom_filename),
            "energy": energy,
            "phsp_path": os.path.join(beam_model_path, phsp_name),
            "commissioning": simulation.settings.get("commissioning", False),
            "commissioning_type": simulation.settings.get("commissioning_type", None)
        }

        params["filler_depth"] = simulation.settings.get("filler_depth", 0.0) + self.min_filler

        if params["commissioning_type"] == "calibration" or params["commissioning_type"] == "output":
            measurement_depth = simulation.settings.get("measurement_depth", 1.5)
            params["calib_vox_size"] = 0.2
            params["phantom_depth"] = measurement_depth - 1.5 * params["calib_vox_size"]
        elif params["commissioning_type"] == "profile":
            measurement_depth = simulation.settings.get("measurement_depth", 10.0)
            comm_dict = self._process_profile_commissioning(simulation.beams, measurement_depth)
            params.update(comm_dict)

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

    def make_aperture_inputs(self, beamlet_creation):
        files_created = {
            "mlc_files": [],
            "jaws_files": [],
            "beamnrc_files": [],
            "dosxyznrc_files": [],
            "processed_cpts": []
        }

        dicom_cpts = self.opt_to_dicom_cpts(beamlet_creation.control_points, self.mlc_type)
        scenario_cpts = self._process_aperture_cpts(dicom_cpts, self.mlc_type, beamlet_creation.settings)

        server = beamlet_creation.server
        beamnrc_path = server.get_path("BeamNRC")
        beam_model_path = os.path.join(beamnrc_path, self.folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")
        phantom_path = os.path.join(dosxyznrc_path, beamlet_creation.phantom_filename)

        zero_air_dose = int(bool(beamlet_creation.settings.get("zero_air_dose", 1)))

        for cpt_index, cpts in enumerate(scenario_cpts):
            dicom_cpt = dicom_cpts[cpt_index]
            # Keep track of aperture "beamlet" file names for this control point
            dicom_cpt["beamlets"] = []
            for sc_index, cpt in enumerate(cpts):
                cpt_copy = cpt.copy()
                cpt_copy["weight"] = 0.0
                ap_name = beamlet_creation.settings["name"] + "_ap%i_sc%i" % (cpt_index, sc_index)
                # To be used later
                dicom_cpt["beamlets"].append(ap_name)

                energy = str(int(cpt.get("energy", 6)))
                if cpt.get("FFF", False):
                    energy += "FFF"

                phsp_name = "{model}_{energy}.IAEAphsp".format(model=self.model_name, energy=energy)

                params = {
                    "beam_model": self.folder,
                    "title": "Beamlet apertures",
                    "nhist": beamlet_creation.settings["nhist"],
                    "phantom_path": phantom_path,
                    "delivery_mode": 2,
                    "zero_air_dose": zero_air_dose,
                    "cpts": [cpt_copy, cpt],
                    "name": ap_name,
                    "energy": energy,
                    "beam_inputfile": "{name}.egsinp".format(name=ap_name),
                    "phsp_path": os.path.join(beam_model_path, phsp_name),
                    "filler_depth": self.min_filler
                }

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

            files_created["processed_cpts"].append(dicom_cpt)

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

    def _process_profile_commissioning(self, beams, measurement_depth=10.0):
        x_jaws_iso = [x / 10.0 for x in beams[0]["cpts"][0].x_jaw]
        y_jaws_iso = [y / 10.0 for y in beams[0]["cpts"][0].y_jaw]
        phantom_neg_x = x_jaws_iso[0] - 5.0
        phantom_neg_y = y_jaws_iso[0] - 5.0

        depth_vox_size = 0.5
        phantom_depth = measurement_depth - 0.5 * (3 * depth_vox_size)

        vox_size = 0.25
        num_vox_x = int(2 * abs(phantom_neg_x) / vox_size)
        num_vox_y = int(2 * abs(phantom_neg_y) / vox_size)

        return {
            "phantom_neg_x": phantom_neg_x,
            "phantom_neg_y": phantom_neg_y,
            "num_vox_x": num_vox_x,
            "num_vox_y": num_vox_y,
            "phantom_depth": phantom_depth,
            "vox_size": vox_size,
            "depth_vox_size": depth_vox_size
        }

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

        temp_path = "/home/webtps/phsp_files/Photons/beamlets"
        phsp_file = "TrueBeam_%iX_beamlets_%i.egsphsp1" % (cpt.energy, beamlet_index)
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

        filename = "%s_%i_%i_%iMV.egsinp" % (beamlet_creation.settings["name"], cpt.index, beamlet_index, cpt.energy)
        self._write_dosxyznrc_phsp_input(params, filename)

        return filename

    def get_dosxyznrc_template(self, params=None):
        """
        Return filename of DOSXYZnrc template. All energies should share
        the same template.
        """
        if isinstance(params, dict):
            commissioning_type = params.get("commissioning_type", None)
        else:
            commissioning_type = None

        if commissioning_type is not None:
            template = "dosxyznrc_{ctype}_commissioning.egsinp".format(ctype=commissioning_type)
        else:
            template = "dosxyznrc_template.egsinp"

        return template
