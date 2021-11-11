"""
Generic Varian beam model module.

Copyright Marc-Andre Renaud, 2017
"""
import os

import numpy

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel

class GenericVarian(BeamModel):
    """Generic Varian beam model."""

    model_name = "GenericVarian"
    folder = "BEAM_GenericVarian"
    pegs_file = "radify"
    particle = "photon"

    default_dsource = 40.0
    min_filler = 0.01

    y_z_jaw_front = 28.0
    y_z_jaw_back = 35.6485
    x_z_jaw_front = 36.7
    x_z_jaw_back = 44.3485
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16

    output_corr = {
        "fs": numpy.array([30, 40, 50, 100, 200, 300]),
        #"6": numpy.array([0.880 / 0.881, 0.906 / 0.911, 0.928 / 0.932, 1.0, 1.067 / 1.057, 1.104 / 1.093])
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
                                   simulation.settings["linac"]["mlc"],
                                   positioning_error=error_coords,
                                   orient=simulation.settings.get("orient", "HFS"))

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
            "dbs_radius": round(self._find_max_jaw(simulation.beams) / 10.0 + 10.0, 3)
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

    def make_aperture_inputs(self, beamlet_creation):
        files_created = {}
        files_created["mlc_files"] = []
        files_created["jaws_files"] = []
        files_created["beamnrc_files"] = []
        files_created["dosxyznrc_files"] = []

        mlc_type = beamlet_creation.settings["linac"]["mlc"]

        # dicom_cpts is only needed to get dbs radius... kind of a hacky way to get to it.
        dicom_cpts = self.opt_to_dicom_cpts(beamlet_creation.control_points, mlc_type)

        scenario_cpts = self._process_aperture_cpts(beamlet_creation.control_points, mlc_type, beamlet_creation.settings)
        server = beamlet_creation.server
        beamnrc_path = server.get_path("BeamNRC")
        beam_model_path = os.path.join(beamnrc_path, self.folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params = {
            "beam_model": self.folder,
            "title": "Beamlet apertures",
            "nhist": beamlet_creation.settings["nhist"],
            "pegs_file": self.pegs_file,
            "mlc_type": mlc_type,
            "delivery_mode": 2,  # Step and shoot
            "phantom_path": os.path.join(dosxyznrc_path, beamlet_creation.phantom_filename),
            "zero_air_dose": int(bool(beamlet_creation.settings.get("zero_air_dose", True)))
        }

        for cpt_index, cpts in enumerate(scenario_cpts):
            for sc_index, cpt in enumerate(cpts):
                cpt_copy = cpt.copy()
                cpt_copy["weight"] = 0.0
                params["cpts"] = [cpt_copy, cpt]

                params["name"] = beamlet_creation.settings["name"] + "_ap%i_sc%i" % (cpt_index, sc_index)
                params["beam_inputfile"] = params["name"] + ".egsinp"

                params["dbs_radius"] = round(self._find_max_jaw([{"cpts": [dicom_cpts[cpt_index]]}]) / 10.0 + 10.0, 3)

                params["energy"] = str(int(cpt.get("energy", 6)))
                params["FFF"] = cpt.get("FFF", False)

                energy_identifier = str(int(params["energy"]))
                if params["FFF"]:
                    energy_identifier += "FFF"

                # Specific data related to Generic models
                generic_params = beamlet_creation.settings["linac"]["beam_parameters"][energy_identifier]
                params["e_energy"] = generic_params["e_energy"]
                params["e_fwhm"] = generic_params["e_fwhm"]

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

    def get_beamnrc_template(self, params):
        """
        Return filename of BEAMnrc template. Different energies sometimes
        have different template files.
        """
        ident = params["energy"]
        if params["FFF"] is True:
            ident = str(ident) + "FFF"

        template = self.model_name + "_{}.egsinp".format(ident)

        return template

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

    def get_calibration(self, plan_obj, simulation):
        """
        Return the calibration factor to go from DOSXYZnrc dose to absolute dose.

        :param plan_obj: Plan object with the number of monitor units.
        """
        parsed_plan = plan_obj.parse_plan()

        try:
            energy = str(int(parsed_plan["beams"][0]["cpts"][0]["energy"]))
        except KeyError:
            energy = "6"

        if "FFF" in parsed_plan["beams"][0] and parsed_plan["beams"][0]["FFF"]:
            energy += "FFF"

        if energy not in simulation.settings["linac"]["beam_parameters"]:
            raise Exception("Calibration factor not found for energy: %s" % energy)

        calibration = simulation.settings["linac"]["beam_parameters"][energy]["calibration_factor"]

        return parsed_plan["total_mus"] * (1.0 / calibration)

    def get_calibration_factor(self, sim, cpt):
        """
        Return the calibration factor to go from DOSXYZnrc dose to dose / MU.
        """

        try:
            energy = str(int(cpt["energy"]))
        except KeyError:
            energy = "6"

        if cpt.get("FFF", False):
            energy += "FFF"

        if energy not in sim.settings["linac"]["beam_parameters"]:
            raise Exception("Calibration factor not found for energy: %s" % energy)

        calibration = sim.settings["linac"]["beam_parameters"][energy]["calibration_factor"]

        return calibration

    def _process_commissioning(self, beams, measurement_depth=10.0):
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

    def _write_beamnrc_input(self, params):
        """
        Write BEAMnrc input file from params dict.

        The template string is formatted twice. Once to include the MLC
        template, then once more to fill in all the parameters defined by
        the params dict.
        """
        beamnrc_filename = self.get_beamnrc_template(params)
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
