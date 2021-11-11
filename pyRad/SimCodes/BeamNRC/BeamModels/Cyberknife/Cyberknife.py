# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:05:13 2018

@author: vengj
"""
import os

import math
import numpy as np

from pyRad.SimCodes.BeamNRC.BeamModel import BeamModel
from pyRad.utils import SimDose

class Cyberknife(BeamModel):
    model_name = "cyberknife"
    folder = "BEAM_cyberknife"
    pegs_file = "radify"
    particle = "photon"

    col_dict = {
        5: '5',
        7.5: '7-5',
        10: '10',
        12.5: '12-5',
        15: '15',
        20: '20',
        25: '25',
        30: '30',
        35: '35',
        40: '40',
        50: '50',
        60: '60'
    }

    calibration_factors = {
        "6": 8.7915996e-14
    }

    # Reads in the dictionary of beam parameters
    def __init__(self):
        self.template_folder = os.path.dirname(__file__)

    def make_sim_inputs(self, simulation):
        col_sizes = self._process_beams(simulation.beams[0])

        beamnrc_path = simulation.server.get_path("BeamNRC")
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params = {
            "name": simulation.name,
            "title": "Plan recalculation",
            "nhist": simulation.settings["nhist"],
            "zero_air_dose": int(bool(simulation.settings.get("zero_air_dose", 1))),
            "delivery_mode": 1,
            "pegs_file": self.pegs_file,
            "beam_model": self.folder,
            "phantom_path": os.path.join(dosxyznrc_path, simulation.phantom_filename),
        }

        files_created = {}
        files_created["mlc_file"] = []
        files_created["jaws_file"] = []
        files_created["beamnrc_file"] = []
        files_created["dosxyznrc_file"] = []

        nhist_total = float(params["nhist"])
        col_sizes_split = {}

        for cpt in simulation.beams[0]["cpts"]:
            if cpt.col_size not in col_sizes_split:
                col_sizes_split[cpt.col_size] = []

            col_sizes_split[cpt.col_size].append(cpt)

        total_mus = 0
        for col_size, cpts in col_sizes_split.iteritems():
            total_mus += sum([cpt.weight for cpt in cpts])

        for col_size in col_sizes:
            cpts = col_sizes_split[col_size]
            col_weight = sum([cpt.weight for cpt in cpts])
            relative_weight = col_weight / total_mus
            col_size_coeff = np.square(col_size/60.0)
            params['nhist'] = int(round(relative_weight*nhist_total*col_size_coeff))

            params["col_size"] = col_size
            params["cpts"] = col_sizes[col_size]
            params["name"] = simulation.name + "_" + str(col_size)

            files_created["beamnrc_file"].append(self._write_beamnrc_input(params))
            files_created["dosxyznrc_file"].append(self._write_dosxyznrc_input(params))

        return files_created

    def process_finished_doses(self, simulation, doses):
        """
        If multiple simulations were needed to recalculate a plan,
        this method combines them into a single dose file.
        """
        parsed_plan = simulation.ref_plan.parse_plan()

        total_mus = parsed_plan["total_mus"]
        col_sizes = self._split_col_size(parsed_plan["beams"][0])
        relative_weights = {}

        for col_size, cpts in col_sizes.iteritems():
            col_weight = sum([cpt["weight"] for cpt in cpts])
            relative_weight = col_weight / total_mus
            relative_weights[col_size] = relative_weight

        sim_doses = []
        for dose in doses:
            sim_dose = SimDose.from_file(dose)
            name = os.path.splitext(dose)[0]
            splitted_name = name.split("_")
            # eg. ASWz6GcUAFDkXvPfVNyViJ_5.0.3ddose
            col_size = float(splitted_name[1])
            sim_dose.doses *= relative_weights[col_size]
            sim_doses.append(sim_dose)

        final_dose = sim_doses[0]
        for dose in sim_doses[1:]:
            final_dose.doses += dose.doses

        return final_dose

    def _split_col_size(self, beam):
        col_sizes = {}

        for cpt in beam["cpts"]:
            if cpt["col_size"] not in col_sizes:
                col_sizes[cpt["col_size"]] = []

            col_sizes[cpt["col_size"]].append(cpt)

        return col_sizes

    def _process_beams(self, beam, positioning_error=None, orient="HFS"):
        """
        Transform beam data into the format required by BeamNRC.

        DICOM defines most attributes at isocenter, BeamNRC needs them at
        the actual physical height of the components.
        """
        if positioning_error is None:
            positioning_error = [0.0, 0.0, 0.0]

        processed_cpts = []

        col_sizes = {}

        for cpt in beam["cpts"]:
            cpt_dict = {}

            cpt_dict["weight"] = cpt.weight

            x = cpt.node_position[0] - cpt.iso[0]
            y = cpt.node_position[1] - cpt.iso[1]
            z = cpt.node_position[2] - cpt.iso[2]
            xpysq = x * x + y * y
            SAD = np.sqrt(x*x + y*y + z*z)
            theta = 90.0 - math.degrees(math.atan2(z, math.sqrt(xpysq)))
            if math.atan2(y,x) < 0:
                phi = 360 + math.degrees(math.atan2(y, x))
            else:
                phi = math.degrees(math.atan2(y, x))
            phicol = 0.0

            cpt_dict["theta"] = theta
            cpt_dict["phi"] = phi
            cpt_dict["phicol"] = phicol

            cpt_dict["iso"] = [(x + x_err) / 10.0 for (x, x_err) in zip(cpt.iso, positioning_error)]

            cpt_dict["dsource"] = round(SAD / 10.0 - 50.0, 2) # Must change this if the air gap is changed in the model

            if cpt.col_size not in col_sizes:
                col_sizes[cpt.col_size] = []

            col_sizes[cpt.col_size].append(cpt_dict)

        processed_cpts = {}
        for col_size in col_sizes:
            processed_cpts[col_size] = []

        for col_size, cpts in col_sizes.iteritems():
            cum_weight = 0.0
            for cpt in cpts:
                cpt_weight = cpt["weight"]

                cpt_copy = cpt.copy()
                cpt_copy["weight"] = cum_weight
                cpt["weight"] = cum_weight + cpt_weight

                processed_cpts[col_size].append(cpt_copy)
                processed_cpts[col_size].append(cpt)

                cum_weight += cpt_weight

            for cpt in processed_cpts[col_size]:
                cpt["weight"] /= cum_weight


        return processed_cpts

    def _get_beamnrc_template(self, params):
        col_size = params["col_size"]
        col_str = self.col_dict[col_size]

        return "ckfixv6_{col_str}mm.egsinp".format(col_str=col_str)

    def _write_beamnrc_input(self, params):
        template_filename = self._get_beamnrc_template(params)
        template_path = os.path.join(self.template_folder, template_filename)

        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        beamnrc_filename = params["name"] + "_beam.egsinp"
        with open(beamnrc_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beamnrc_filename

    def _write_dosxyznrc_input(self,  params):
		cpt_strings = []
		cpt_string = "%.4f, %.4f, %.4f, %.3f, %.3f, %.3f, %.3f, %.6f"
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
			myfile.write(template_string.format(**params))

		return dosxyznrc_filename
