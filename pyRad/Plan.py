"""
Wrapper for plan files.

Copyright Marc-Andre Renaud, 2017
"""
import os

import pydicom as dicom

from pyRad.PlanTypes import BrachyPlan, LinacPlan, ProtonPlan, TomoPlan, CyberknifePlan


class Plan(object):
    """Wrapper for DICOM RTPlan file."""

    plan_associations = {
        "Proton": ProtonPlan,
        "Tomo": TomoPlan,
        "Brachy": BrachyPlan,
        "Linac": LinacPlan,
        "Cyberknife": CyberknifePlan,
        "unknown": None
    }

    def __init__(self, attrs):
        """
        Determine plan type from reading the file.

        :param str rtplan_path: Path to DICOM RTPlan file.
        """
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, "plan_type"):
            self._preprocess_plan()
        else:
            self._plan_type = self.plan_type

        self._create_plan_object()

    def _preprocess_plan(self):
        """Identify plan type from common fingerprints in file."""
        filename, extension = os.path.splitext(self.rtplan_path)

        if "xml" in extension or os.path.isdir(self.rtplan_path):
            plan_type = "Cyberknife"
            self.name = "TPS Plan"
        else:
            rp = dicom.read_file(self.rtplan_path, force=True)
            if "IonBeamSequence" in rp:
                plan_type = "Proton"
            elif "SourceSequence" in rp:
                plan_type = "Brachy"
            elif "BeamSequence" in rp:
                if "SeriesDescription" in rp and "TomoTherapy" in rp.SeriesDescription:
                    plan_type = "Tomo"
                else:
                    plan_type = "Linac"
            else:
                plan_type = "unknown"

            self.uid = rp.SOPInstanceUID

            if "RTPlanName" in rp:
                self.name = rp.RTPlanName
            elif "RTPlanLabel" in rp:
                self.name = rp.RTPlanLabel
            else:
                self.name = "TPS Plan"

        self._plan_type = plan_type


    def _create_plan_object(self):
        """Create more specific plan object from fingerprinting routine."""
        plan_dict = {"rtplan_path": self.rtplan_path}
        if hasattr(self, "parsed_plan"):
            plan_dict["parsed_plan"] = self.parsed_plan

        if self._plan_type is not "unknown":
            self._plan = self.plan_associations[self._plan_type](plan_dict)
        else:
            self._plan = None

    def get_plan_type_object(self):
        """Getter for specific plan object."""
        return self._plan

    def parse_plan(self):
        """Parse plan using specific plan object routines."""
        if not hasattr(self, 'parsed_plan'):
            if self._plan:
                plan_dict = self._plan.parse_plan()
                plan_dict["plan_type"] = self._plan_type

                self.parsed_plan = plan_dict
            else:
                self.parsed_plan = {"plan_type": "unknown"}

        if not hasattr(self, 'uid'):
            self.uid = self.parsed_plan["uid"]

        return self.parsed_plan

    def transpose_plan(self, new_iso, collapse_gantry=False, new_gantry=0):
        """
        Transpose all control points from a plan onto a new iso.
        """
        return self._plan.transpose_plan(new_iso, collapse_gantry, new_gantry)