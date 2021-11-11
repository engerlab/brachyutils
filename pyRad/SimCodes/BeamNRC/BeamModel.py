"""
BeamNRC Beam model base class.

Copyright Marc-Andre Renaud, 2017
"""

import math
import os

import numpy

from pyRad.utils import dicom_to_spherical
from pyRad.utils import SimDose
from pyRad import ControlPoint

class BeamModel(object):
    """
    BeamModel class.

    Defines a basic implementation of all methods needed to create linac-based BeamNRC simulations.
    Unless the basic implementation is overridden, all Beam models are expected to define:
        model_name
        folder
        pegs_file
        particle
        dosxyznrc_template_filename

        y_z_jaw_front
        y_z_jaw_back
        x_z_jaw_front
        x_z_jaw_back
        dcoll
        default_dsource
        leaf_radius
        abut_gap
        num_leaves
        leaf_boundaries
        calibration_factors
    """

    mlcs = {
        "HDMLC": {
            "num_leaves": 60,
            "leaf_boundaries": numpy.array([-110.0, -105.0, -100.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -37.5, -35.0, -32.5, -30.0, -27.5, -25.0, -22.5, -20.0, -17.5, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0]),
            "leaf_radius": 16.0,
            "abut_gap": 0.03,
            "dcoll": 51.01
        },
        "VMLC": {
            "num_leaves": 60,
            "leaf_boundaries": numpy.array([-200.0, -190.0, -180.0, -170.0, -160.0, -150.0, -140.0, -130.0, -120.0, -110.0, -100.0, -95.0, -90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]),
            "leaf_radius": 8.0,
            "abut_gap": 0.015,
            "dcoll": 51.5785
        }
    }

    def get_calibration(self, plan_obj, simulation=None):
        """
        Return the calibration factor to go from DOSXYZnrc dose to absolute dose.

        :param plan_obj: Plan object with the number of monitor units.
        """
        parsed_plan = plan_obj.parse_plan()

        try:
            energy = str(parsed_plan["beams"][0]["cpts"][0]["energy"])
        except KeyError:
            energy = "6"

        if "FFF" in parsed_plan["beams"][0] and parsed_plan["beams"][0]["FFF"]:
            energy += "FFF"

        if energy not in self.calibration_factors:
            raise Exception("Calibration factor not found for energy: %s" % energy)

        return parsed_plan["total_mus"] * (1.0 / self.calibration_factors[energy])

    def get_calibration_factor(self, sim, cpt):
        energy = str(int(cpt["energy"]))
        if cpt.get("FFF", False):
            energy += "FFF"

        return self.calibration_factors[energy]

    def get_output_correction(self, cpt, fff=False):
        """
        Correction factor between measurement OFs and MC OFs.
        """
        energy = str(int(cpt.energy))
        if fff:
            energy += "FFF"

        y_field = abs(cpt.y_jaw[1] - cpt.y_jaw[0])
        x_field = abs(cpt.x_jaw[1] - cpt.x_jaw[0])
        eq_field = 4 * y_field * x_field / (2 * (x_field + y_field))
        try:
            field_sizes = self.output_corr["fs"]
            factors = self.output_corr[energy]
            correction = numpy.interp(eq_field, field_sizes, factors)
        except AttributeError:
            correction = 1.0
        except KeyError:
            correction = 1.0

        return correction

    def process_finished_doses(self, simulation, doses):
        """
        If multiple simulations were needed to recalculate a plan,
        this method combines them into a single dose file.
        """
        return SimDose.from_file(doses[0])

    def get_leaf_numbers(self, pos, thickness, mlc_type):
        """
        Return open leaves to produce a field centered at pos.

        :param pos: center of the field
        :param thickness: y width of the field
        """
        leaf_boundaries = self.mlcs[mlc_type]["leaf_boundaries"]
        num_leaves = self.mlcs[mlc_type]["num_leaves"]

        neg_limit = pos - 0.5 * thickness
        pos_limit = pos + 0.5 * thickness
        leaves_included = (leaf_boundaries > neg_limit) & (leaf_boundaries <= pos_limit)

        return -(leaves_included.nonzero()[0] - 1) + num_leaves - 1

    def _find_max_jaw(self, beams):
        """
        Return the maximum jaw opening.

        Usually used to determine maximum DBS radius.
        """
        jaw_sizes = []
        for beam in beams:
            # Control points are sometimes passed as classes and
            # sometimes as dicts. Kind of gross.

            has_y_jaws = False
            has_x_jaws = False
            try: some_cpt = beam["cpts"][0] #assume sets of jaws are the same for all cpts in beam
            except IndexError: continue #go to next beam if no cpts

            if isinstance(some_cpt, dict):
                if "y_jaw" in some_cpt:
                    has_y_jaws = True
                    cpt_y_jaws = [cpt["y_jaw"] for cpt in beam["cpts"]]

                if "x_jaw" in some_cpt:
                    has_x_jaws = True
                    cpt_x_jaws = [cpt["x_jaw"] for cpt in beam["cpts"]]
            else: #cpts passed as class
                if hasattr(some_cpt, "y_jaw"):
                    has_y_jaws = True
                    cpt_y_jaws = [cpt.y_jaw for cpt in beam["cpts"]]
                if hasattr(some_cpt, "x_jaw"):
                    has_x_jaws = True
                    cpt_x_jaws = [cpt.x_jaw for cpt in beam["cpts"]]

            if has_y_jaws:
                max_jaw_neg = max([abs(cpt_y_jaw[0]) for cpt_y_jaw in cpt_y_jaws])
                max_jaw_pos = max([abs(cpt_y_jaw[1]) for cpt_y_jaw in cpt_y_jaws])

                jaw_sizes.append(max([max_jaw_neg, max_jaw_pos]))

            if has_x_jaws:
                max_jaw_neg = max([abs(cpt_x_jaw[0]) for cpt_x_jaw in cpt_x_jaws])
                max_jaw_pos = max([abs(cpt_x_jaw[1]) for cpt_x_jaw in cpt_x_jaws])

                jaw_sizes.append(max([max_jaw_neg, max_jaw_pos]))

        return max(jaw_sizes)

    def get_beamnrc_template(self, params):
        """Return beamnrc template filename."""
        return "{}.egsinp".format(self.model_name)

    def get_dosxyznrc_template(self, params=None):
        """Return dosxyznrc template filename."""
        return "dosxyznrc_template.egsinp"

    def _process_beams(self, beams, mlc_type, positioning_error=None, orient="HFS"):
        """
        Transform beam data into the format required by BeamNRC.

        DICOM defines most attributes at isocenter, BeamNRC needs them at
        the actual physical height of the components.
        """
        if positioning_error is None:
            positioning_error = [0.0, 0.0, 0.0]

        processed_cpts = []

        cum_weight = 0.0
        for beam in beams:
            is_static = beam.get("static", False)

            for cpt in beam["cpts"]:
                cpt_dict = {}

                # JAW POSITIONS
                if hasattr(cpt, "y_jaw"):
                    cpt_dict.update(self._process_y_jaws(cpt.y_jaw))
                if hasattr(cpt, "x_jaw"):
                    cpt_dict.update(self._process_x_jaws(cpt.x_jaw))

                # MLC POSITIONS
                cpt_dict["apertures"] = self._process_mlc(cpt, is_static, mlc_type)

                correction = self.get_output_correction(cpt)
                corrected_weight = cpt.weight * correction
                cpt_dict["weight"] = corrected_weight + cum_weight
                cum_weight += corrected_weight

                # Perturb iso in the position error direction if an error is specified.
                cpt_dict["iso"] = [(x + x_err) / 10.0 for (x, x_err) in zip(cpt.iso, positioning_error)]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle, orient=orient)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol)

                cpt_dict["dsource"] = self.default_dsource
                cpt_dict["energy"] = cpt.energy

                processed_cpts.append(cpt_dict)

        for cpt in processed_cpts:
            cpt["weight"] /= cum_weight

        return processed_cpts

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
        # Calculate left y jaw height
        jaw_dict = {}

        theta = math.atan2(abs(jaw_pos[0]) / 10.0, 100.0)
        costheta = math.cos(theta)
        left_jaw_front = self.y_jaw_arc_radius * costheta

        # Right y jaw height
        theta = math.atan2(abs(jaw_pos[1]) / 10.0, 100.0)
        costheta = math.cos(theta)
        right_jaw_front = self.y_jaw_arc_radius * costheta

        # Take the mean of the two in case of asymmetric jaws.
        # Difference in height between the "zero" position and the arc position
        # Should be a negative value.
        difference = 0.5 * (left_jaw_front + right_jaw_front) - self.y_jaw_arc_radius

        y_z_jaw_front = self.y_z_jaw_front + difference
        y_z_jaw_back = (y_z_jaw_front + self.y_jaw_width) * costheta

        jaw_dict["y_z_jaw_front"] = y_z_jaw_front
        jaw_dict["y_z_jaw_back"] = y_z_jaw_back
        adjusted_y_jaw_width = y_z_jaw_back - y_z_jaw_front

        # JAW POSITIONS
        # DICOM->BEAMnrc: Swap signs for y
        y_jaws = [-jaw_pos[1] / 10.0, -jaw_pos[0] / 10.0]

        # Jaws focus to a 3 mm^2 square at the target plane, not to a point.
        y_neg_front = (y_jaws[0] + 0.15) * (jaw_dict["y_z_jaw_front"] / 100.0) - 0.15
        y_neg_back = (y_jaws[0] + 0.15) * (jaw_dict["y_z_jaw_back"] / 100.0) - 0.15
        y_pos_front = (y_jaws[1] - 0.15) * (jaw_dict["y_z_jaw_front"] / 100.0) + 0.15
        y_pos_back = (y_jaws[1] - 0.15) * (jaw_dict["y_z_jaw_back"] / 100.0) + 0.15

        # Translate the jaws back to their position if they were projected to a point,
        # but keep the angle defined by the 3 mm projection.
        diff_neg = abs(0.15 * (jaw_dict["y_z_jaw_front"] + 0.5 * adjusted_y_jaw_width) / 100.0 - 0.15)
        diff_pos = abs(-0.15 * (jaw_dict["y_z_jaw_front"] + 0.5 * adjusted_y_jaw_width) / 100.0 + 0.15)

        jaw_dict["y_jaw_neg_front"] = y_neg_front + diff_neg
        jaw_dict["y_jaw_neg_back"] = y_neg_back + diff_neg
        jaw_dict["y_jaw_pos_front"] = y_pos_front - diff_pos
        jaw_dict["y_jaw_pos_back"] = y_pos_back - diff_pos


        # Old way of doing things
        jaw_dict["y_z_jaw_front"] = self.y_z_jaw_front
        jaw_dict["y_z_jaw_back"] = self.y_z_jaw_back
        jaw_dict["y_jaw_neg_front"] = y_jaws[0] * (self.y_z_jaw_front / 100.0)
        jaw_dict["y_jaw_neg_back"] = y_jaws[0] * (self.y_z_jaw_back / 100.0)
        jaw_dict["y_jaw_pos_front"] = y_jaws[1] * (self.y_z_jaw_front / 100.0)
        jaw_dict["y_jaw_pos_back"] = y_jaws[1] * (self.y_z_jaw_back / 100.0)


        return jaw_dict

    def _process_x_jaws(self, jaw_pos):
        """
        Convert from DICOM-specified x jaw positions to BEAMnrc. DICOM jaw
        positions are defined as the field projection at isocenter.

        On Varian linacs, the X jaws travel linearly with field size. However,
        they rotate such that the jaw face stays parallel to the beam
        divergence. Front of jaws stays at constant height.
        """
        # Calculate left x jaw back height
        jaw_dict = {}

        cm_jaws = [jaw_pos[0] / 10.0, jaw_pos[1] / 10.0]
        costheta = 100.0 / math.sqrt(100.0 * 100.0 + cm_jaws[0] * cm_jaws[0])
        left_jaw_back = self.x_z_jaw_front + self.x_jaw_width * costheta

        costheta = 100.0 / math.sqrt(100.0 * 100.0 + cm_jaws[1] * cm_jaws[1])
        right_jaw_back = self.x_z_jaw_front + self.x_jaw_width * costheta

        jaw_dict["x_z_jaw_front"] = self.x_z_jaw_front
        # Take the mean of the two back jaw positions
        jaw_dict["x_z_jaw_back"] = 0.5 * (left_jaw_back + right_jaw_back)
        adjusted_x_jaw_width = jaw_dict["x_z_jaw_back"] - self.x_z_jaw_front

        # Jaws focus to a 3 mm^2 square at the target plane, not to a point.
        x_neg_front = (cm_jaws[0] + 0.15) * (jaw_dict["x_z_jaw_front"] / 100.0) - 0.15
        x_neg_back = (cm_jaws[0] + 0.15) * (jaw_dict["x_z_jaw_back"] / 100.0) - 0.15
        x_pos_front = (cm_jaws[1] - 0.15) * (jaw_dict["x_z_jaw_front"] / 100.0) + 0.15
        x_pos_back = (cm_jaws[1] - 0.15) * (jaw_dict["x_z_jaw_back"] / 100.0) + 0.15

        # Translate the jaws back to their position if they were projected to a point,
        # but keep the angle defined by the 3 mm projection.
        diff_neg = abs(0.15 * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 - 0.15)
        diff_pos = abs(-0.15 * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 + 0.15)

        #jaw_dict["x_jaw_neg_front"] = x_neg_front + diff_neg
        #jaw_dict["x_jaw_neg_back"] = x_neg_back + diff_neg
        #jaw_dict["x_jaw_pos_front"] = x_pos_front - diff_pos
        #jaw_dict["x_jaw_pos_back"] = x_pos_back - diff_pos

        # Old way of doing things
        jaw_dict["x_z_jaw_front"] = self.x_z_jaw_front
        jaw_dict["x_z_jaw_back"] = self.x_z_jaw_back
        jaw_dict["x_jaw_neg_front"] = cm_jaws[0] * (self.x_z_jaw_front / 100.0)
        jaw_dict["x_jaw_neg_back"] = cm_jaws[0] * (self.x_z_jaw_back / 100.0)
        jaw_dict["x_jaw_pos_front"] = cm_jaws[1] * (self.x_z_jaw_front / 100.0)
        jaw_dict["x_jaw_pos_back"] = cm_jaws[1] * (self.x_z_jaw_back / 100.0)

        return jaw_dict

    def _process_mlc(self, cpt, is_static=False, mlc_type="VMLC"):
        """
        Convert DICOM-specified MLC positions to BEAMnrc. In DICOM, MLC
        leaf positions are defined as the projected field at isocenter.
        Positions must be backprojected to the MLC mid-plane. In addition,
        an offset is included to account for MLC rounded leaf ends.
        """
        dcoll = self.mlcs[mlc_type]["dcoll"]
        leaf_radius = self.mlcs[mlc_type]["leaf_radius"]
        num_leaves = self.mlcs[mlc_type]["num_leaves"]
        abut_gap = self.mlcs[mlc_type]["abut_gap"]

        if not (hasattr(cpt, "apertures") and len(cpt.apertures) > 0):
            return [[20.0, -20.0] for _ in range(num_leaves)]

        aperture = []
        # Leaves are specified in opposite order compared to DICOM
        for leaf_pos in cpt.apertures[::-1]:
            # Convert between aperture positions defined at SSD 100 cm to MLC plane.
            projected = [leaf_pos[0] * (dcoll / 100.0) / 10.0, leaf_pos[1] * (dcoll / 100.0) / 10.0]
            real_a = projected[0] + leaf_radius * (math.sqrt(projected[0] * projected[0] + dcoll * dcoll) / dcoll - 1)
            real_b = projected[1] + leaf_radius * (math.sqrt(projected[1] * projected[1] + dcoll * dcoll) / dcoll - 1)

            real_b = -1.0 * real_b  # Flip the sign of the B leaf to match coordinate system.
            if not is_static:
                # If leaves are touching in DICOM, must add abutting leaf gap
                if abs(real_a - real_b) < abut_gap:
                    real_b -= 0.5 * abut_gap
                    real_a += 0.5 * abut_gap

            aperture.append([real_a, real_b])

        return aperture

    def _find_largest_mlc_field(self, cpts, mlc_type):
        """
            Find largest MLC field from optimisation. Used to set the jaw positions.
        """

        epsilon = 1e-3
        left_x = -1e10
        right_x = -1e10

        top_y = 1e10
        bottom_y = -1e10

        for cpt in cpts:
            first_open = None
            last_open = 0

            for l_no, leaf_pair in enumerate(cpt["apertures"]):
                # Check if leaf pair is opened. At this point the leaf pair coordinates are given
                # as distance from 0 (closed). For the left pair, the distance is positive in the "left"
                # direction, and for the right pair, the distance is positive in the "right" direction.
                # Silliness.

                if abs(leaf_pair[1] + leaf_pair[0]) > epsilon:
                    if first_open is None:
                        first_open = l_no

                    last_open = l_no

                    if leaf_pair[0] >= left_x:
                        left_x = leaf_pair[0]

                    if leaf_pair[1] >= right_x:
                        right_x = leaf_pair[1]

            if first_open <= top_y:
                top_y = first_open

            if last_open >= bottom_y:
                bottom_y = last_open

        leaf_boundaries = self.mlcs[mlc_type]["leaf_boundaries"]
        top_y_coord = leaf_boundaries[top_y]
        bottom_y_coord = leaf_boundaries[bottom_y+1]

        return {
            "x_mlc": [-right_x, left_x],
            "y_mlc": [top_y_coord, bottom_y_coord]
        }

    def opt_to_dicom_cpts(self, beamlet_cpts, mlc_type):
        """
        Converts control points and apertures from optimisation to DICOM format.

        This is a bit redundant as we are only really interested in BEAMnrc format, but
        I want to have only one method converting from DICOM to BEAMnrc.
        """
        individual_cpts = []

        for cpt in beamlet_cpts:
            if not hasattr(cpt, "apertures"):
                continue

            if not isinstance(cpt.apertures, list):
                continue

            row_positions = numpy.arange(cpt.beamlet_rows) * cpt.iso_row_size - (cpt.beamlet_rows / 2.0 - 0.5) * cpt.iso_row_size
            leaf_numbers = [self.get_leaf_numbers(pos, cpt.iso_row_size, mlc_type) for pos in row_positions]

            for aperture in cpt.apertures:
                cpt_dict = {
                    "FFF": getattr(cpt, "FFF", False),
                    "energy": cpt.energy,
                    "gantry_angle": cpt.gantry_angle,
                    "couch_angle": cpt.couch_angle,
                    "col_angle": cpt.col_angle,
                    "iso_row_size": cpt.iso_row_size,
                    "iso_col_size": cpt.iso_col_size,
                    "beamlet_rows": cpt.beamlet_rows,
                    "beamlet_columns": cpt.beamlet_columns,
                    "iso": list(cpt.iso),
                    "apertures": [[0.0, 0.0] for i in range(self.mlcs[mlc_type]["num_leaves"])],
                    "static": True,
                    "particle": self.particle,
                    "weight": 1.0,
                    "cum_weight": 1.0
                }

                for leaf_nos, x_pos in zip(leaf_numbers, aperture["rows"]):
                    for leaf_no in leaf_nos:
                        #cpt_dict["apertures"][leaf_no] = [x_pos[0], x_pos[1]]
                        cpt_dict["apertures"][leaf_no] = [x_pos[1], -x_pos[0]]

                individual_cpts.append(cpt_dict)

        if self.particle is "electron":
            # Not worried about leakage for electrons, make jaws as big as possible.
            x_jaw = [-175.0, 175.0]
            y_jaw = [-175.0, 175.0]
        else:
            # Set jaw positions based on largest MLC field + 1 cm margin
            largest_mlc = self._find_largest_mlc_field(individual_cpts, mlc_type)
            x_jaw = [largest_mlc["x_mlc"][0] - 5, largest_mlc["x_mlc"][1] + 5]
            y_jaw = [largest_mlc["y_mlc"][0] - 5, largest_mlc["y_mlc"][1] + 5]

        for cpt in individual_cpts:
            cpt["x_jaw"] = x_jaw[:]  # Making sure to give a copy instead of a reference
            cpt["y_jaw"] = y_jaw[:]

        return individual_cpts

    def _process_aperture_cpts(self, dicom_cpts, mlc_type, settings=None):
        """
        Create BeamNRC input files for individual apertures.

        Used in full MC recalculation after optimising from beamlets.
        """
        if settings is None:
            settings = {}

        simulations_to_run = []

        if settings.get("robust", False) and "uncert_coords" in settings:
            positioning_uncertainty = [abs(float(x)) for x in settings["uncert_coords"]]
        else:
            positioning_uncertainty = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        scenarios = [[0.0, 0.0, 0.0]]
        for sc_i, uncert in enumerate(positioning_uncertainty):
            if uncert > 0.0:
                if (sc_i > 2):
                    uncert = -uncert
                iso_shift = [0.0, 0.0, 0.0]
                iso_shift[sc_i % 3] = uncert
                scenarios.append(iso_shift)

        for cpt in dicom_cpts:
            scenario_cpts = []
            for scenario in scenarios:
                beam_dict = {
                    "beam_meterset": 1.0,
                    "cpts": [ControlPoint(cpt)],
                    "static": True,
                    "FFF": cpt.get("FFF", False)
                }

                processed_cpts = self._process_beams([beam_dict], mlc_type, positioning_error=scenario)

                scenario_cpts += processed_cpts

            simulations_to_run.append(scenario_cpts)

        return simulations_to_run

    def _make_mlc_file(self, params):
        """
        Create a BeamNRC mlc file for control points in a plan.

        :param params: Dict with processed control points including MLC positions
        """
        mlc_filename = params["name"] + ".mlc"
        num_cpts = len(params["cpts"])

        with open(mlc_filename, "w") as mlc_file:
            mlc_file.write("MLC file\n")
            mlc_file.write("{}\n".format(num_cpts))
            for cpt in params["cpts"]:
                mlc_file.write("{weight:.6}\n".format(weight=cpt["weight"]))
                for ap in cpt["apertures"]:
                    mlc_file.write("{neg:.5}, {pos:.5}, 1\n".format(neg=ap[1], pos=ap[0]))

        return mlc_filename

    def _make_jaws_file(self, params):
        """
        Create a BeamNRC jaw file for control points in a plan.

        :param params: Dict with processed control points including jaw positions

        Assumes 2 sets of jaws by default.

        Maybe want to make this so it works with any jaw ordering (yx, xy)
        """
        try:
            jaw_directions = self.jaw_directions
        except AttributeError:
            jaw_directions = 'xy'

        try:
            jaws_cm = self.jaws_cm
        except AttributeError:
            jaws_cm = "SYNCJAWS"

        jaws_filename = params["name"] + ".jaws"
        num_cpts = len(params["cpts"])

        with open(jaws_filename, "w") as jaws_file:
            jaws_file.write("JAW file\n")
            jaws_file.write("{}\n".format(num_cpts))
            for cpt in params["cpts"]:
                jaws_file.write("{weight:.6}\n".format(weight=cpt["weight"]))
                if jaws_cm == "SYNCJAWS":
                    if 'y' in jaw_directions: jaws_file.write("{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_pos_front:.5}, {y_jaw_pos_back:.5}, {y_jaw_neg_front:.5}, {y_jaw_neg_back:.5}\n".format(**cpt))
                    if 'x' in jaw_directions: jaws_file.write("{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_pos_front:.5}, {x_jaw_pos_back:.5}, {x_jaw_neg_front:.5}, {x_jaw_neg_back:.5}\n".format(**cpt))
                elif jaws_cm == "SYNCEJAWS":
                    if 'y' in jaw_directions: jaws_file.write("{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_neg:.5}, {y_jaw_pos:.5}\n".format(**cpt))
                    if 'x' in jaw_directions: jaws_file.write("{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_neg:.5}, {x_jaw_pos:.5}\n".format(**cpt))
                else:
                    raise Exception("Wrong jaws_cm name")

        return jaws_filename

    def _write_beamnrc_input(self, params):
        beamnrc_filename = self.get_beamnrc_template(params)
        template_path = os.path.join(self.template_folder, beamnrc_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        beam_filename = params["name"] + "_beam.egsinp"
        with open(beam_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beam_filename

    def _write_dosxyznrc_input(self, params):
        cpt_strings = []
        cpt_string = "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.6f"
        for cpt in params["cpts"]:
            formatted_string = cpt_string % (cpt["iso"][0], cpt["iso"][1], cpt["iso"][2],
                                             cpt["theta"], cpt["phi"], cpt["phicol"],
                                             cpt["dsource"], cpt["weight"])

            cpt_strings.append(formatted_string)

        full_string = "\n".join(cpt_strings)
        params["cpt_string"] = full_string
        params["num_cpts"] = len(params["cpts"])

        dosxyznrc_filename = self.get_dosxyznrc_template(params)
        template_path = os.path.join(self.template_folder, dosxyznrc_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        dosxyznrc_filename = params["name"] + ".egsinp"
        with open(dosxyznrc_filename, "w") as myfile:
            #print params
            myfile.write(template_string.format(**params))

        return dosxyznrc_filename
