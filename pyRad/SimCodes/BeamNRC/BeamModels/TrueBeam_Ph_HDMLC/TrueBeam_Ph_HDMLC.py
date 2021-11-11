"""
TrueBeam photon HDMLC module.

Copyright Marc-Andre Renaud, 2017
"""
import os

import numpy

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel


class TrueBeam_Ph_HDMLC(BeamModel):
    """TrueBeam photon model with HDMLC."""

    model_name = "TrueBeam_Ph_HDMLC"
    folder = "BEAM_TrueBeam_HD"
    pegs_file = "radify"
    particle = "photon"

    calibration_factors = {
        "6": 1.43245e-14,  # MU / primary history
        "6FFF": 3.09763e-14,
        "10": 3.22125e-14,
        "10FFF": 1.05566472e-13
    }

    default_dsource = 100.0  # dosxyz source 20 uses SAD as dsource if IAEA phsp file

    y_z_jaw_front = 27.89
    y_z_jaw_back = 35.66
    x_z_jaw_front = 36.61
    x_z_jaw_back = 44.41
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16

    mlc_type = "HDMLC"

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
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params = {"cpts": cpts}
        params["name"] = simulation.name
        try:
            params["energy"] = str(cpts[0]["energy"])
            if "FFF" in simulation.beams[0] and simulation.beams[0]["FFF"] is True:
                params["energy"] += "FFF"
        except KeyError:
            print "Energy not found, assuming 6 MV"
            params["energy"] = "6"

        params["pegs_file"] = self.pegs_file
        params["beam_inputfile"] = params["name"] + "_beam.egsinp"
        params["beam_model"] = self.folder
        params["title"] = "Plan recalculation"
        params["nhist"] = simulation.settings["nhist"]
        params["phantom_path"] = os.path.join(dosxyznrc_path,
                                              simulation.phantom_filename)

        phsp_name = self.model_name + "_%s.IAEAphsp" % params["energy"]
        params["phsp_path"] = os.path.join(beam_model_path, phsp_name)

        zero_air_dose = int(bool(simulation.settings.get("zero_air_dose", 1)))
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