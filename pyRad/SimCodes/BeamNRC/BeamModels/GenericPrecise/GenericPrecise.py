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

class GenericPrecise(BeamModel):
    """Generic Elekta Precise beam model."""

    model_name = "GenericPrecise"
    folder = "BEAM_GenericPrecise"
    pegs_file = "radify"
    particle = "photon"

    default_dsource = 40.0
    airslab_end = 59.9
    min_filler = 0.01
    ssd = default_dsource + airslab_end + min_filler


    jaws_cm = { 'x' : "SYNCJAWS", 'y' : "SYNCEJAWS" }
    '''
    y_z_jaw_front = 37.8
    y_z_jaw_back = 40.8
    x_z_jaw_front = 41.8
    x_z_jaw_back = 49.6
    '''
    y_z_jaw_front = 39.6
    y_z_jaw_back = 42.6
    x_z_jaw_front = 43.1
    x_z_jaw_back = 50.9

    yjaw_zrcurve = (y_z_jaw_back - y_z_jaw_front) * 0.4 + y_z_jaw_front #typical setup
    yjaw_curvature_radius = 6.0 #200% of thickness

    '''
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16
    '''

    jaw_directions = 'xy'

    mlc_front = 29.8
    mlc_back = 37.3
    mlc_curvature_radius = 15.0
    mlc_zrcurve = (29.8 + 37.3)/2. - 0.7 #0.7 cm above leaf midplane

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
                if hasattr(cpt, "x_jaw"): cpt_dict.update(self._process_x_jaws(cpt.x_jaw))

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


    def _process_mlc(self, cpt, is_static=False, mlc_type="40mlce"):

        #adjustment = -0.24 # approximate adjustment to the light field edge as per agility & integrity doc

        adjustment = 0.
        minimum_leaf_gap = 0.1 #5mm minimum isocentric leaf gap as specified in agiltiy and integrity document

        aperture = []
        for leaf_pos in cpt.apertures[::-1]:

            leaf_pos_a = leaf_pos[0]
            leaf_pos_b = leaf_pos[1]

            isocentric_gap = leaf_pos_a + leaf_pos_b
            if isocentric_gap < minimum_leaf_gap:
                missing_gap = minimum_leaf_gap - isocentric_gap
                leaf_pos_a += missing_gap/2.
                leaf_pos_b -= missing_gap/2.


            xl_neg = (-abs(leaf_pos_b) + adjustment) / 10. #light field edge at isocenter
            xl_pos = (abs(leaf_pos_a) - adjustment) / 10.

            try:
                adj_leaf_tip_b = (numpy.sign(xl_neg) * -self.mlc_curvature_radius * math.sqrt((self.ssd / xl_neg) ** 2 + 1) + self.mlc_zrcurve) * xl_neg / self.ssd #this is the coordinate of the center of the circle along which the jaw/mlc curvature lies
            except ZeroDivisionError:
                adj_leaf_tip_b = -self.mlc_curvature_radius

            try:
                adj_leaf_tip_a = (numpy.sign(xl_pos) * self.mlc_curvature_radius * math.sqrt((self.ssd / xl_pos) ** 2 + 1) + self.mlc_zrcurve) * xl_pos / self.ssd #this is the coordinate of the center of the circle along which the jaw/mlc curvature lies
            except ZeroDivisionError:
                adj_leaf_tip_a = self.mlc_curvature_radius

            aperture.append([adj_leaf_tip_a, adj_leaf_tip_b])
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
        jaw_dict = {}

        adjustment = -0.24 # approximate adjustment to the light field edge as per agility & integrity doc

        jaw_dict["y_z_jaw_front"] = self.y_z_jaw_front
        jaw_dict["y_z_jaw_back"] = self.y_z_jaw_back


        xl_neg = (-abs(jaw_pos[0]) + adjustment) / 10.
        xl_pos = (abs(jaw_pos[1]) - adjustment) / 10.

        try:
            jaw_dict["y_jaw_neg"] = (numpy.sign(xl_neg) * -self.yjaw_curvature_radius * math.sqrt((self.ssd / xl_neg) ** 2 + 1) + self.yjaw_zrcurve) * xl_neg / self.ssd #this is the coordinate of the center of the circle along which the jaw/mlc curvature lies
        except ZeroDivisionError:
            jaw_dict["y_jaw_neg"] = -self.yjaw_curvature_radius

        try:
            jaw_dict["y_jaw_pos"] = (numpy.sign(xl_pos) * self.yjaw_curvature_radius * math.sqrt((self.ssd / xl_pos) ** 2 + 1) + self.yjaw_zrcurve) * xl_pos / self.ssd #this is the coordinate of the center of the circle along which the jaw/mlc curvature lies
        except ZeroDivisionError:
            jaw_dict["y_jaw_pos"] = self.yjaw_curvature_radius

        return jaw_dict

    def _process_x_jaws(self, jaw_pos):

        """
        Convert from DICOM-specified y jaw positions to BEAMnrc. DICOM jaw
        positions are defined as the field projection at isocenter.

        On Varian linacs, the Y jaws travel in an arc based on the projected
        field such that the jaw face is parallel to the beam divergence.
        BEAMnrc does not allow left/right jaws to have a different height,
        so the average height of both jaws is taken. These are very small
        adjustments and should not introduce any inaccuracies.
        """
        # Calculate left y jaw height
        jaw_dict = {}

        # JAW POSITIONS
        # DICOM->BEAMnrc: Swap signs for x
        x_jaws = [-jaw_pos[1] / 10.0, -jaw_pos[0] / 10.0]

        jaw_dict["x_z_jaw_front"] = self.x_z_jaw_front
        jaw_dict["x_z_jaw_back"] = self.x_z_jaw_back
        #adjusted_x_jaw_width = x_z_jaw_back - x_z_jaw_front

        # Jaws focus to a 3 mm^2 square at the target plane, not to a point.
        projection_size = 0.0
        x_neg_front = (x_jaws[0] + projection_size/2.) * (jaw_dict["x_z_jaw_front"] / 100.0) - projection_size/2.
        x_neg_back = (x_jaws[0] + projection_size/2.) * (jaw_dict["x_z_jaw_back"] / 100.0) - projection_size/2.
        x_pos_front = (x_jaws[1] - projection_size/2.) * (jaw_dict["x_z_jaw_front"] / 100.0) + projection_size/2.
        x_pos_back = (x_jaws[1] - projection_size/2.) * (jaw_dict["x_z_jaw_back"] / 100.0) + projection_size/2.


        # Translate the jaws back to their position if thex were projected to a point,
        # but keep the angle defined bx the 3 mm projection.
        #diff_neg = abs(projection_size/2. * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 - projection_size/2.)
        #diff_pos = abs(-projection_size/2. * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 + projection_size/2.)

        jaw_dict["x_jaw_neg_front"] = x_neg_front
        jaw_dict["x_jaw_neg_back"] = x_neg_back
        jaw_dict["x_jaw_pos_front"] = x_pos_front
        jaw_dict["x_jaw_pos_back"] = x_pos_back

        return jaw_dict

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
        xjaws_filename, yjaws_filename = self._make_jaws_file(params)
        params["mlc_file"] = os.path.join(beam_model_path,
                                          mlc_filename)
        params["xjaws_file"] = os.path.join(beam_model_path,
                                           xjaws_filename)
        params["yjaws_file"] = os.path.join(beam_model_path,
                                           yjaws_filename)

        files_created = {}
        files_created["mlc_file"] = [mlc_filename]
        files_created["xjaws_file"] = [xjaws_filename]
        files_created["yjaws_file"] = [yjaws_filename]
        files_created["beamnrc_file"] = [self._write_beamnrc_input(params)]
        files_created["dosxyznrc_file"] = [self._write_dosxyznrc_input(params)]

        return files_created

    def _make_jaws_file(self, params):
        """
        Create a BeamNRC jaw file for control points in a plan.

        :param params: Dict with processed control points including jaw positions

        Assumes 2 sets of jaws by default.

        Maybe want to make this so it works with any jaw ordering (yx, xy)
        """
        try: jaw_directions = self.jaw_directions
        except AttributeError: jaw_directions = 'xy'
        #try: jaws_cm = self.jaws_cm
        #except AttributeError: jaws_cm = "SYNCJAWS"

        num_cpts = len(params["cpts"])
        jaws_filenames = []

        for direction in jaw_directions:
            jaws_cm = self.jaws_cm[direction]
            jaws_filename = params["name"] + "." + direction + "jaws"
            jaws_filenames.append(jaws_filename)

            with open(jaws_filename, "w") as jaws_file:
                jaws_file.write(direction + " JAW file\n")
                jaws_file.write("{}\n".format(num_cpts))
                for cpt in params["cpts"]:
                    jaws_file.write("{weight:.6}\n".format(weight=cpt["weight"]))
                    if jaws_cm == "SYNCJAWS":
                        if 'y' in direction: jaws_file.write("{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_pos_front:.5}, {y_jaw_pos_back:.5}, {y_jaw_neg_front:.5}, {y_jaw_neg_back:.5}\n".format(**cpt))
                        if 'x' in direction: jaws_file.write("{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_pos_front:.5}, {x_jaw_pos_back:.5}, {x_jaw_neg_front:.5}, {x_jaw_neg_back:.5}\n".format(**cpt))
                    elif jaws_cm == "SYNCEJAWS":
                        if 'y' in direction: jaws_file.write("{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_neg:.5}, {y_jaw_pos:.5}\n".format(**cpt))
                        if 'x' in direction: jaws_file.write("{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_neg:.5}, {x_jaw_pos:.5}\n".format(**cpt))
                    else:
                        raise Exception("Wrong jaws_cm name")

        return jaws_filenames


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
            mlc_string = mlc_template_file.read()

        template_string = template_string.format(mlc_template=mlc_string)

        beam_filename = params["name"] + "_beam.egsinp"
        with open(beam_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beam_filename
