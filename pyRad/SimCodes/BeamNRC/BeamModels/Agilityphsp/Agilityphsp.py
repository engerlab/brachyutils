"""
Generic Varian beam model module.

Copyright Marc-Andre Renaud, 2017
"""
import math
import os

import numpy
from scipy.interpolate import interp1d

from pyRad.utils import dicom_to_spherical

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel

class Agilityphsp(BeamModel):
    """Generic Elekta Agility beam model."""

    model_name = "Agilityphsp"
    folder = "BEAM_Agilityphsp"
    pegs_file = "radify"
    particle = "photon"

    default_dsource = 40.0
    airslab_end = 59.99
    min_filler = 0.01
    ssd = default_dsource + airslab_end + min_filler

    jaws_cm = "SYNCEJAWS"
    y_z_jaw_front = 43.2
    y_z_jaw_back = y_z_jaw_front + 7.7

    jaw_directions = 'y'

    # 50% transmission
    mlc_XR = [-154.72, -149.72, -139.73, -129.73, -119.73, -109.73, -99.79, -89.73, -79.73, -69.74, -59.74, -49.74, -39.74, -29.74, -19.74, -9.75, 0.25, 10.25, 20.25, 30.25, 40.25, 50.25, 60.25, 70.25, 80.25, 90.25, 100.24, 110.24, 120.24, 130.24, 140.24, 150.24, 160.24, 170.24, 180.24, 190.24, 200.24]
    mlc_tip = [-149.19, -144.56, -135.25, -125.90, -116.51, -107.06, -97.57, -88.03, -78.45, -68.81, -59.12, -49.39, -39.61, -29.78, -19.9, -9.98, 0.00, 10.02, 20.10, 30.22, 40.39, 50.61, 60.88, 71.19, 81.55, 91.97, 102.43, 112.94, 123.49, 134.10, 144.75, 155.44, 166.19, 176.98, 187.82, 198.71, 209.64]
    mlc_light = [-155.0] + [-150.0 + 10.0 * i for i in range(36)]
    mlc_interp = interp1d(mlc_XR, mlc_tip)
    mlc_light_interp = interp1d(mlc_light, mlc_tip)

    diaphragm_XR = [-129.75, -119.75, -109.75, -99.75, -89.75, -79.75, -69.75, -59.76, -49.76, -39.76, -29.76, -19.76, -9.76, 0.24, 10.24, 20.24, 30.24, 40.24, 50.24, 60.24, 70.24, 80.24, 90.23, 100.23, 110.23, 120.23, 130.23, 140.23, 150.23, 160.23, 170.23, 180.23, 190.23, 200.23]
    diaphragm_tip = [-127.57, -117.93, -108.26, -98.56, -88.83, -79.08, -69.29, -59.48, -49.64, -39.77, -29.87, -19.94, -9.99, 0.0, 10.01, 20.06, 30.13, 40.23, 50.36, 60.52, 70.71, 80.92, 91.17, 101.44, 111.74, 122.07, 132.43, 142.82, 153.23, 163.68, 174.15, 184.64, 195.17, 205.72]
    diaphragm_light = [-130.0 + i * 10.0 for i in range(34)]
    diaphragm_interp = interp1d(diaphragm_XR, diaphragm_tip)
    diaphragm_light_interp = interp1d(diaphragm_light, diaphragm_tip)

    #mlc backproject
    mlc_front = 31.18
    mlc_back = mlc_front + 9.0
    mlc_zrcurve = 34.93 #zmin + 3.75 as per belec doc
    mlc_curvature_radius = 17.0

    diaphragm_curvature_radius = 13.5 #nominal 13.5
    diaphragm_zrcurve = y_z_jaw_front + 3.5

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this.
        """
        self.template_folder = os.path.dirname(__file__)

    def _process_beams(self, beams, mlc_type):
        """
        Transform beam data into the format required by BeamNRC.

        DICOM defines most attributes at isocenter, BeamNRC needs them at
        the actual physical height of the components.
        """
        #dcoll = self.mlcs[mlc_type]["dcoll"]

        total_mus = sum([beam["beam_meterset"] for beam in beams])

        processed_cpts = []

        for beam in beams:
            is_static = beam.get("static", False)

            for cpt in beam["cpts"]:
                cpt_dict = {}

                # JAW POSITIONS
                if hasattr(cpt, "y_jaw"): cpt_dict.update(self._process_y_jaws(cpt.y_jaw))
                #if hasattr(cpt, "x_jaw"): cpt_dict.update(self._process_x_jaws(cpt.x_jaw))

                # MLC POSITIONS
                cpt_dict["apertures"] = self._process_mlc(cpt, is_static, mlc_type)

                cpt_dict["weight"] = cpt.cum_weight / total_mus
                cpt_dict["iso"] = [x / 10.0 for x in cpt.iso]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol)

                cpt_dict["dsource"] = self.default_dsource
                cpt_dict["energy"] = cpt.energy

                processed_cpts.append(cpt_dict)

        return processed_cpts

    def _process_mlc(self, cpt, is_static=False, mlc_type="80mlce"):
        minimum_leaf_gap = 0.125 #1mm minimum isocentric leaf gap as specified in agiltiy and integrity document

        aperture = []

        dcoll = self.mlc_zrcurve
        leaf_radius = self.mlc_curvature_radius

        for leaf_pos in cpt.apertures[::-1]:

            tip_a = self.mlc_interp(leaf_pos[0]) * (dcoll / 100.0)
            tip_b = self.mlc_interp(leaf_pos[1]) * (dcoll / 100.0)

            # Flip the sign of B leaf since it's the one in the negative direction
            tip_b = -1.0 * tip_b
            leaf_gap = abs(tip_a - tip_b)
            if leaf_gap < minimum_leaf_gap:
                remainder = minimum_leaf_gap - leaf_gap
                tip_a += 0.5 * remainder
                tip_b -= 0.5 * remainder

            # The leaf positions are specified at the center of curvature of the leaves.
            # Convert to cm
            aperture.append([tip_a / 10.0 + leaf_radius, tip_b / 10.0 - leaf_radius])

        return aperture


    def _process_y_jaws(self, jaw_pos):

        """
        Convert from DICOM-specified y jaw positions to BEAMnrc. DICOM jaw
        positions are defined as the field projection at isocenter.

        On Varian linacs, the Y jaws travel in an arc based on the projected
        field such that the jaw face is parallel to the beam divergence.
        BEAMnrc does not allow left/right jaws to have a different height,
        so the average height of both jaws is taken. These are very small
        adjustments and should not introduce any inaccuracies.
        """

        jaw_dict = {
            "y_z_jaw_front": self.y_z_jaw_front,
            "y_z_jaw_back": self.y_z_jaw_back
        }


        # Make both jaw position values positive in the quadrant they represent
        # ie left jaw is negative if it goes into the right part of the field,
        # and vice versa.
        left_jaw_rad = jaw_pos[1]
        right_jaw_rad = -jaw_pos[0]

        left_jaw_tip = self.diaphragm_interp(left_jaw_rad)
        left_jaw = left_jaw_tip * (self.diaphragm_zrcurve / 100.0) / 10.0

        right_jaw_tip = self.diaphragm_interp(right_jaw_rad)
        right_jaw = right_jaw_tip * (self.diaphragm_zrcurve / 100.0) / 10.0

        # Jaw position is specified at the center of curvature, not at the tip
        jaw_dict["y_jaw_neg"] = -left_jaw - self.diaphragm_curvature_radius
        jaw_dict["y_jaw_pos"] = right_jaw + self.diaphragm_curvature_radius

        return jaw_dict

    def _find_max_jaw(self, beams):
        beam_maxes = []
        for beam in beams:
            cpt_jaws = [cpt.y_jaw for cpt in beam["cpts"]]
            max_jaw_neg = max([abs(cpt_y_jaw[0]) for cpt_y_jaw in cpt_jaws])
            max_jaw_pos = max([abs(cpt_y_jaw[1]) for cpt_y_jaw in cpt_jaws])
            max_jaw = max([max_jaw_neg, max_jaw_pos])

            mlc_maxes = []
            for cpt in beam["cpts"]:
                max_a = max([leaf_pos[0] for leaf_pos in cpt.apertures])
                max_b = max([leaf_pos[1] for leaf_pos in cpt.apertures])
                mlc_maxes.append(max([max_a, max_b]))

            max_mlc = max(mlc_maxes)

            beam_maxes.append(max([max_jaw, max_mlc]))

        return max(beam_maxes)

    def make_sim_inputs(self, simulation):
        """
        Create input files for DOSXYZnrc simulation using a BEAMnrc beam model.
        """
        params = {}
        params["name"] = simulation.name
        params["mlc_type"] = simulation.settings["linac"]["mlc"]
        cpts = self._process_beams(simulation.beams, params["mlc_type"])

        params["cpts"] = cpts
        params["name"] = simulation.name
        try:
            params["energy"] = str(cpts[0]["energy"])
        except KeyError:
            print "Energy not found, assuming 6 MV"
            params["energy"] = "6"

        params["FFF"] = simulation.beams[0].get("FFF", False)

        energy_identifier = str(int(params["energy"]))
        if params["FFF"]:
            energy_identifier += "FFF"

        # Specific data related to Generic models
        generic_params = simulation.settings["linac"]["beam_parameters"][energy_identifier]
        params["e_energy"] = generic_params["e_energy"]
        params["e_fwhm"] = generic_params["e_fwhm"]
        params["mlc_type"] = simulation.settings["linac"]["mlc"]

        beamnrc_path = simulation.server.get_path("BeamNRC")
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params["pegs_file"] = self.pegs_file
        params["beam_inputfile"] = params["name"] + "_beam.egsinp"
        params["beam_model"] = self.folder
        params["title"] = "Plan recalculation"
        params["nhist"] = simulation.settings["nhist"]
        params["phantom_path"] = os.path.join(dosxyznrc_path,
                                              simulation.phantom_filename)

        # DBS radius should be larger than the largest simulation field. Actual size
        # depends on whether user is concerned about out of field dose. Default setting is
        # quite large compared to field size.
        params["dbs_radius"] = round(self._find_max_jaw(simulation.beams) / 10.0 + 10.0, 3)

        zero_air_dose = int(bool(simulation.settings.get("zero_air_dose", 1)))
        params["zero_air_dose"] = zero_air_dose

        # Automatically put dynamic delivery mode since static modes
        # are taken into account by duplicating control points.
        params["delivery_mode"] = 1

        params["commissioning"] = simulation.settings.get("commissioning", False)
        params["commissioning_type"] = simulation.settings.get("commissioning_type", None)
        params["filler_depth"] = simulation.settings.get("filler_depth", 0.0) + self.min_filler
        params["dbs_rejection"] = 60.0 + params["filler_depth"] - 1.0  # dbs rejection plane 1 cm above scoring plane.
        if params["commissioning_type"] == "profile":
            measurement_depth = simulation.settings.get("measurement_depth", 10.0)
            comm_dict = self._process_commissioning(simulation.beams, measurement_depth)
            params.update(comm_dict)
        elif params["commissioning_type"] == "calibration":
            measurement_depth = simulation.settings.get("measurement_depth", 1.5)
            params["calib_vox_size"] = 0.2
            params["phantom_depth"] = measurement_depth - 1.5 * params["calib_vox_size"]
        elif params["commissioning_type"] == "output":
            measurement_depth = simulation.settings.get("measurement_depth", 5.0)
            depth_vox_size = 0.5
            params["phantom_depth"] = measurement_depth - 1.5 * depth_vox_size

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

        cpts = self._process_aperture_cpts(beamlet_creation.control_points, mlc_type)
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
            params["dbs_radius"] = round(self._find_max_jaw([{"cpts": [dicom_cpts[index]]}]) / 10.0 + 10.0, 3)

            params["energy"] = str(int(cpt.get("energy", 6)))
            params["FFF"] = cpt.get("FFF", False)

            energy_identifier = str(int(params["energy"]))
            if params["FFF"]:
                energy_identifier += "FFF"

            # Specific data related to Generic models
            generic_params = beamlet_creation.settings["linac"]["beam_parameters"][energy_identifier]
            params["e_energy"] = generic_params["e_energy"]
            params["e_fwhm"] = generic_params["e_fwhm"]
            params["mlc_type"] = mlc_type

            params["pegs_file"] = self.pegs_file
            params["beam_inputfile"] = params["name"] + ".egsinp"
            params["beam_model"] = self.folder

            params["title"] = "Beamlet apertures"
            params["nhist"] = beamlet_creation.settings["nhist"]
            params["phantom_path"] = os.path.join(dosxyznrc_path,
                                                  beamlet_creation.phantom_filename)
            params["delivery_mode"] = 2  # Step and shoot

            zero_air_dose = int(bool(beamlet_creation.settings.get("zero_air_dose", True)))
            params["zero_air_dose"] = zero_air_dose

            mlc_filename = self._make_mlc_file(params)
            jaws_filename = self._make_jaws_file(params, jaw_direction=self.jaw_directions)

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

        leaf_sizes = []
        for beam in beams:
            for cpt in beam["cpts"]:
                try: apertures = cpt.apertures
                except AttributeError:
                    apertures = cpt["apertures"]
                max_leaf_neg = max([abs(aperture[0]) for aperture in apertures])
                max_leaf_pos = max([abs(aperture[1]) for aperture in apertures])

                leaf_sizes.append(max([max_leaf_neg, max_leaf_pos]))

        #x_jaws_iso = [x / 10.0 for x in beams[0]["cpts"][0].x_jaw]
        y_jaws_iso = [y / 10.0 for y in beams[0]["cpts"][0].y_jaw]
        phantom_neg_x = -max(leaf_sizes) / 10.0 - 5.0
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
            mlc_string = mlc_template_file.read().rstrip()

        template_string = template_string.format(mlc_template=mlc_string)

        beam_filename = params["name"] + "_beam.egsinp"
        with open(beam_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beam_filename
