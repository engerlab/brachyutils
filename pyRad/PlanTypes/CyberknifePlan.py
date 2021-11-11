# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:26:23 2018

@author: vengj
"""

import xml.etree.ElementTree as xml
import glob, os

from pyRad.utils import create_uid

class CyberknifePlan(object):
    ### attribute is directory = path to folder enclosing beam and plan xml files
    def __init__(self, attrs):
        self.rtplan_path = attrs["rtplan_path"]
        if os.path.isdir(self.rtplan_path):
            self.directory = self.rtplan_path
        else:
            self.directory = os.path.dirname(self.rtplan_path)

    def parse_plan(self):
        ### Returns dictionary of beam parameters (beam_param)
        """
        Really rough xml parsing of the cyberknife xml output. No idea if it's robust at all.
        """
        beam_path_names = []
        xml_plan_name = None
        plan_uid = create_uid()

        glob_string = os.path.join(self.directory, "*.xml")

        for xmlfile in glob.glob(glob_string):
            if xmlfile.endswith("plan.xml"):
                xml_plan_name = xmlfile

        tree = xml.parse(xml_plan_name)
        root = tree.getroot()

        beam_dict = {
            "cpts": []
        }

        total_MUs = 0.0

        plan_setup = root.find("{http://www.accuray.com/cyris}PLAN_SETUP")
        prescription_dose = float(plan_setup.find("{http://www.accuray.com/cyris}PRESCRIBED_DOSE").text) / 100.0

        beamset = root.find("{http://www.accuray.com/cyris}BEAMSET")
        for beam in beamset.findall("{http://www.accuray.com/cyris}BEAM"):
            cpt = {}
            cpt["beamid"] = int(beam.attrib["id"])

            weight = float(beam.find("{http://www.accuray.com/cyris}WEIGHT").text)
            if weight > 0:
                cpt["weight"] = weight
                total_MUs += weight
                cpt["col_size"] = float(beam.find("{http://www.accuray.com/cyris}COLLIMATOR_SIZE").text)
                cpt["node_id"] = int(beam.find("{http://www.accuray.com/cyris}NODE_ID").text)

                node_coord = beam.find("{http://www.accuray.com/cyris}NODE_COORD")
                node_position = [
                    float(node_coord.find("{http://www.accuray.com/cyris}X").text),
                    float(node_coord.find("{http://www.accuray.com/cyris}Y").text),
                    float(node_coord.find("{http://www.accuray.com/cyris}Z").text),
                ]
                cpt["node_position"] = node_position

                target_coord = beam.find("{http://www.accuray.com/cyris}TARGET_COORD")
                target_position = [
                    float(target_coord.find("{http://www.accuray.com/cyris}X").text),
                    float(target_coord.find("{http://www.accuray.com/cyris}Y").text),
                    float(target_coord.find("{http://www.accuray.com/cyris}Z").text),
                ]
                cpt["iso"] = target_position

                beam_dict["cpts"].append(cpt)

            beam_dict["cpts"].sort(key=lambda cpt: cpt["node_id"])

        beam_param = {
            "uid": plan_uid,
            "beams": [beam_dict],
            "prescription_dose": prescription_dose,
            "total_mus": total_MUs
        }

        return beam_param
