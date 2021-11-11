import pydicom as dicom
import math
import numpy
from ..utils import dicom_to_spherical
from xml.etree import ElementTree


class LinacPlan(object):
    """
        Attributes:
        rtplan_path (str): path to DICOM RTPlan file.
    """
    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def _find_meterset(self, beam, meterset_sequence):
        for ref_beam in meterset_sequence:
            if beam.BeamNumber == ref_beam.ReferencedBeamNumber:
                if "BeamMeterset" in ref_beam:
                    return float(ref_beam.BeamMeterset)
                else:
                    return -1

        return -1

    def _parse_beam(self, beam, beam_meterset, accumulated_mus):
        cpts = []
        previous_energy = 6
        previous_couch_angle = 0
        previous_gantry_angle = 0
        previous_col_angle = 0
        prev_weight = 0.0
        prev_x_jaw = []
        prev_y_jaw = []

        previous_mlc = []

        for cpt in beam.ControlPointSequence:
            cpt_dict = {}
            cpt_dict["arc_index"] = int(beam.BeamNumber)
            cpt_dict["sad"] = float(beam.SourceAxisDistance)
            cpt_dict["couch_angle"] = 0

            if "NominalBeamEnergy" in cpt:
                cpt_dict["energy"] = str(int(cpt.NominalBeamEnergy))
                previous_energy = cpt_dict["energy"]
            else:
                cpt_dict["energy"] = previous_energy

            if "DoseRateSet" in cpt:
                cpt_dict["dose_rate"] = float(cpt.DoseRateSet)

            if "BeamLimitingDeviceAngle" in cpt:
                cpt_dict["col_angle"] = float(cpt.BeamLimitingDeviceAngle)
                previous_col_angle = float(cpt.BeamLimitingDeviceAngle)
            else:
                cpt_dict["col_angle"] = previous_col_angle

            if "GantryAngle" in cpt:
                cpt_dict["gantry_angle"] = float(cpt.GantryAngle)
                previous_gantry_angle = float(cpt.GantryAngle)
            else:
                cpt_dict["gantry_angle"] = previous_gantry_angle

            if "PatientSupportAngle" in cpt:
                cpt_dict["couch_angle"] = float(cpt.PatientSupportAngle)
                previous_couch_angle = cpt_dict["couch_angle"]
            else:
                cpt_dict["couch_angle"] = previous_couch_angle

            if "IsocenterPosition" in cpt:
                cpt_dict["iso"] = [float(x) for x in cpt.IsocenterPosition]
                previous_iso = [float(x) for x in cpt.IsocenterPosition]
            else:
                cpt_dict["iso"] = previous_iso

            cum_weight = float(cpt.CumulativeMetersetWeight)
            cpt_dict["weight"] = (cum_weight - prev_weight) * beam_meterset
            prev_weight = cum_weight

            cpt_dict["beam_cum_weight"] = cum_weight * beam_meterset
            cpt_dict["cum_weight"] = cum_weight * beam_meterset + accumulated_mus

            if "BeamLimitingDevicePositionSequence" in cpt:
                for device in cpt.BeamLimitingDevicePositionSequence:
                    if "MLC" in device.RTBeamLimitingDeviceType:
                        positions = numpy.array([float(x) for x in device.LeafJawPositions])
                        A_leaves = positions[len(positions)/2:]

                        # B leaves have sign flipped to make both leaf banks have positive values
                        # in the plane that they cover.
                        B_leaves = -numpy.array(positions[:len(positions)/2])
                        cpt_dict["apertures"] = list(zip(A_leaves, B_leaves))
                        previous_mlc = cpt_dict["apertures"][:]
                    elif "ASYMX" in device.RTBeamLimitingDeviceType:
                        cpt_dict["x_jaw"] = [float(x) for x in device.LeafJawPositions]
                        prev_x_jaw = [float(x) for x in device.LeafJawPositions]
                    elif "ASYMY" in device.RTBeamLimitingDeviceType:
                        cpt_dict["y_jaw"] = [float(x) for x in device.LeafJawPositions]
                        prev_y_jaw = [float(x) for x in device.LeafJawPositions]
                    elif device.RTBeamLimitingDeviceType == "X":
                        cpt_dict["x_jaw"] = [float(x) for x in device.LeafJawPositions]
                        prev_x_jaw = [float(x) for x in device.LeafJawPositions]
                    elif device.RTBeamLimitingDeviceType == "Y":
                        cpt_dict["y_jaw"] = [float(x) for x in device.LeafJawPositions]
                        prev_y_jaw = [float(x) for x in device.LeafJawPositions]

            if "BlockSequence" in beam:
                cpt_dict["cutout"] = [float(x) for x in beam.BlockSequence[0].BlockData]
                cpt_dict["cutout_thickness"] = float(beam.BlockSequence[0].BlockThickness)

            if "ApplicatorSequence" in beam:
                cpt_dict["applicator"] = beam.ApplicatorSequence[0].ApplicatorID

            if "apertures" not in cpt_dict:
                if len(previous_mlc) > 0:
                    cpt_dict["apertures"] = previous_mlc[:]
                else:
                    cpt_dict["apertures"] = []

            if "x_jaw" not in cpt_dict:
                # Make sure we get a copy and not a reference by using [:]
                cpt_dict["x_jaw"] = prev_x_jaw[:]

            if "y_jaw" not in cpt_dict:
                cpt_dict["y_jaw"] = prev_y_jaw[:]

            theta, phi, phicol = dicom_to_spherical(cpt_dict["gantry_angle"], cpt_dict["couch_angle"], cpt_dict["col_angle"])
            cpt_dict["theta"] = theta * 180.0 / math.pi
            cpt_dict["phi"] = phi * 180.0 / math.pi
            cpt_dict["phicol"] = phicol * 180.0 / math.pi

            cpts.append(cpt_dict)

        return cpts

    def parse_plan(self):
        plan_dict = {}
        rp = dicom.read_file(self.rtplan_path, force=True)
        self.rp = rp
        accumulated_meterset = 0.0
        accumulated_mus = 0.0
        if "BeamSequence" not in rp:
            return

        if "RTPlanName" in rp:
            plan_dict["name"] = rp.RTPlanName
        elif "RTPlanLabel" in rp:
            plan_dict["name"] = rp.RTPlanLabel
        else:
            plan_dict["name"] = "TPS_Plan"

        beams = []
        mlc_dict = {}
        for arc_index, beam in enumerate(rp.BeamSequence):
            beam_dict = {}
            beam_dict["index"] = arc_index

            cpts = []

            # Don't want to simulate setup beams
            if beam.TreatmentDeliveryType == "SETUP":
                continue

            if "BeamType" in beam:
                beam_dict["static"] = beam.BeamType == "STATIC"
            else:
                beam_dict["static"] = False

            for device in beam.BeamLimitingDeviceSequence:
                # Assume only one type of MLC is used during treatment... makes sense
                if "LeafPositionBoundaries" in device and "boundaries" not in mlc_dict:
                    mlc_dict["boundaries"] = [float(x) for x in device.LeafPositionBoundaries]

            beam_meterset = self._find_meterset(beam, rp.FractionGroupSequence[0].ReferencedBeamSequence)
            beam_dict["beam_meterset"] = beam_meterset
            if beam_meterset < 0:
                raise Exception("Beam Meterset not provided")

            if "PrimaryFluenceModeSequence" in beam:
                temp = beam.PrimaryFluenceModeSequence[0]
                if "FluenceModeID" in temp and temp.FluenceModeID == "FFF":
                    beam_dict["FFF"] = True

            cpts = self._parse_beam(beam, beam_meterset, accumulated_mus)

            beam_dict["cpts"] = cpts
            if "RadiationType" in beam:
                if beam.RadiationType == "ELECTRON":
                    beam_dict["radiation_type"] = "electron"

                    if "BlockSequence" in beam:
                        assert len(beam.BlockSequence) == 1
                        block = beam.BlockSequence[0]
                        assert block.MaterialID.lower() == 'cutout insert'
                        blockdata = numpy.array(block.BlockData).tolist()
                        assert len(blockdata) == 2 * int(block.BlockNumberOfPoints)

                        #cutout = { 'x_values' : blockdata[::2], 'y_values' : blockdata[1::2] }
                        cutout = zip(blockdata[::2], blockdata[1::2])
                        beam_dict["cutout"] = cutout

                else:
                    beam_dict["radiation_type"] = "photon"

            beams.append(beam_dict)

            accumulated_meterset += beam.FinalCumulativeMetersetWeight
            accumulated_mus += beam.FinalCumulativeMetersetWeight * beam_meterset

        plan_dict["total_meterset"] = accumulated_meterset
        plan_dict["total_mus"] = accumulated_mus * rp.FractionGroupSequence[0].NumberOfFractionsPlanned
        plan_dict["beams"] = beams
        plan_dict["fraction_mus"] = accumulated_mus

        if "boundaries" in mlc_dict:
            plan_dict["mlc_boundaries"] = mlc_dict["boundaries"]

        self.parsed_plan = plan_dict
        return plan_dict

    def _make_xml_file(self, mlc, couch_pos):
        if not hasattr(self, "parsed_plan"):
            self.parse_plan()

        top = ElementTree.Element("VarianResearchBeam")
        top.set("SchemaVersion", "1.0")

        beam_dom = ElementTree.SubElement(top, "SetBeam")
        beam_id = ElementTree.SubElement(beam_dom, "Id")
        beam_id.text = "0"

        mlc_model = ElementTree.SubElement(beam_dom, "MLCModel")
        mlc_model.text = mlc

        # No accessories
        ElementTree.SubElement(beam_dom, "Accs")

        control_points = ElementTree.SubElement(beam_dom, "ControlPoints")

        first_iso = numpy.array(self.parsed_plan["beams"][0]["cpts"][0]["iso"])
        if couch_pos is not None:
            first_couch = numpy.array([couch_pos["lat"], couch_pos["vert"], couch_pos["long"]])
        else:
            first_couch = None

        cpt_doms = []

        for b_index, beam in enumerate(self.parsed_plan["beams"]):
            cpt_doms += self._make_xml_beam(beam, b_index, first_couch, first_iso)

        control_points.extend(cpt_doms)

        xml_filename = self.parsed_plan["name"].replace(" ", "_") + ".xml"

        tree = ElementTree.ElementTree(top)
        tree.write(xml_filename)

        return xml_filename

    def _make_xml_beam(self, beam, b_index, first_couch, first_iso):
        cpt_doms = []
        rad_type = beam.get("radiation_type", "photon")
        is_FFF = beam.get("FFF", False)

        if b_index == 0:
            # Build first control point separately as it has more info
            if first_couch is not None:
                current_couch = numpy.array(first_couch)
                if current_couch[0] > 900:
                    current_couch[0] -= 900
                elif current_couch[0] < 100:
                    current_couch[0] += 100

                if current_couch[1] > 900:
                    current_couch[1] -= 900
                elif current_couch[1] < 100:
                    current_couch[1] += 100
            else:
                current_couch = None

            cpt_doms.append(self._make_xml_cpt(beam["cpts"][0], current_couch, rad_type, FFF=is_FFF, first=True))
            start_index = 1
        else:
            start_index = 0

        for cpt in beam["cpts"][start_index:]:
            if current_couch is not None:
                iso_shift = (first_iso - numpy.array(cpt["iso"])) / 10.0
                current_couch = (iso_shift + first_couch) % 1000.0
                if current_couch[0] > 900:
                    current_couch[0] -= 900
                elif current_couch[0] < 100:
                    current_couch[0] += 100

                if current_couch[1] >= 900:
                    current_couch[1] -= 900
                elif current_couch[1] < 100:
                    current_couch[1] += 100

            cpt_doms.append(self._make_xml_cpt(cpt, current_couch, rad_type, FFF=is_FFF))

        return cpt_doms

    def _make_xml_cpt(self, cpt, current_couch, rad_type, FFF=False, first=False):
        cpt_dom = ElementTree.Element("Cp")

        if first:
            sub = ElementTree.SubElement(cpt_dom, "SubBeam")
            seq = ElementTree.SubElement(sub, "Seq")
            seq.text = str(0)
            name = ElementTree.SubElement(sub, "Name")
            name.text = "Something"

            energy = ElementTree.SubElement(cpt_dom, "Energy")
            energy.text = cpt["energy"]

            if rad_type == "electron":
                energy.text = str(cpt["energy"]) + "e"
            else:
                if FFF:
                    energy.text = str(cpt["energy"]) + "FFF"
                else:
                    energy.text = str(cpt["energy"]) + "x"

        mu = ElementTree.SubElement(cpt_dom, "Mu")
        mu.text = "%.1f" % cpt["cum_weight"]

        if "dose_rate" in cpt:
            drate = ElementTree.SubElement(cpt_dom, "DRate")
            drate.text = str(cpt["dose_rate"])

        if "gantry_angle" in cpt:
            gantry_angle = ElementTree.SubElement(cpt_dom, "GantryRtn")
            xml_gantry_angle = (-cpt["gantry_angle"] + 180.0) % 360
            gantry_angle.text = str(xml_gantry_angle)

        if "col_angle" in cpt:
            col_angle = ElementTree.SubElement(cpt_dom, "CollRtn")
            xml_col_angle = (-cpt["col_angle"] + 180.0) % 360
            col_angle.text = str(xml_col_angle)

        if current_couch is not None:
            couchvrt = ElementTree.SubElement(cpt_dom, "CouchVrt")
            couchvrt.text = str(current_couch[1])

            couchlat = ElementTree.SubElement(cpt_dom, "CouchLat")
            couchlat.text = str(current_couch[0])

            couchlng = ElementTree.SubElement(cpt_dom, "CouchLng")
            couchlng.text = str(current_couch[2])

        if "couch_angle" in cpt:
            couch_angle = ElementTree.SubElement(cpt_dom, "CouchRtn")
            xml_couch_angle = (-cpt["couch_angle"] + 180.0) % 360
            couch_angle.text = str(xml_couch_angle)

        if "y_jaw" in cpt:
            y1 = ElementTree.SubElement(cpt_dom, "Y1")
            y1.text = str(cpt["y_jaw"][1] / 10.0)

            y2 = ElementTree.SubElement(cpt_dom, "Y2")
            y2.text = str(-cpt["y_jaw"][0] / 10.0)

        if "x_jaw" in cpt:
            x1 = ElementTree.SubElement(cpt_dom, "X1")
            x1.text = str(-cpt["x_jaw"][0] / 10.0)

            x2 = ElementTree.SubElement(cpt_dom, "X2")
            x2.text = str(cpt["x_jaw"][1] / 10.0)

        if "apertures" in cpt:
            mlc = ElementTree.SubElement(cpt_dom, "Mlc")
            mlc_id = ElementTree.SubElement(mlc, "ID")
            mlc_id.text = str(1)

            b_leaf_positions = [leaf[1] / 10.0 for leaf in cpt["apertures"]]
            a_leaf_positions = [leaf[0] / 10.0 for leaf in cpt["apertures"]]

            b_leaves = ElementTree.SubElement(mlc, "B")
            b_leaves.text = " ".join([str(x) for x in b_leaf_positions])

            a_leaves = ElementTree.SubElement(mlc, "A")
            a_leaves.text = " ".join([str(x) for x in a_leaf_positions])

        return cpt_dom

    def as_xml(self, couch_pos=None, mlc="NDS120"):
        valid_MLCs = ("NDS120", "NDS120HD")
        if mlc not in valid_MLCs:
            raise Exception("Invalid MLC, choices are: " + valid_MLCs)

        xml_filename = self._make_xml_file(mlc, couch_pos)
        return xml_filename

    def transpose_plan(self, new_iso, collapse_gantry, new_gantry):
        """
        Take a patient plan and translate the isocenter of each control point
        onto a new iso. Typically used to recalculate DQA plans.

        The first control point of the plan is set to new_iso, and all other
        control points are translated with respect to that.
        """
        if not hasattr(self, "parsed_plan"):
            self.parse_plan()

        first_iso = numpy.array(self.parsed_plan["beams"][0]["cpts"][0]["iso"])

        new_iso = numpy.array(new_iso)

        new_plan = self.parsed_plan.copy()

        for beam in new_plan["beams"]:
            for cpt in beam["cpts"]:
                iso_offset = numpy.array(cpt["iso"]) - first_iso
                cpt["iso"] = list(new_iso + iso_offset)
                if collapse_gantry:
                    cpt["gantry_angle"] = new_gantry

        return new_plan