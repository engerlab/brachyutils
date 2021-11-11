import pydicom as dicom
import numpy


class TomoPlan(object):
    """
        Attributes:
        rtplan_path (str): path to DICOM RTPlan file.
    """
    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def _find_meterset(self, beam, fraction_group):
        num_fractions = int(fraction_group.NumberOfFractionsPlanned)
        for ref_beam in fraction_group.ReferencedBeamSequence:
            if beam.BeamNumber == ref_beam.ReferencedBeamNumber:
                if "BeamMeterset" in ref_beam:
                    return ref_beam.BeamMeterset * num_fractions
                else:
                    return -1

        return -1

    def _alternate_process_local_cpt(self, cpt, linac_state):
        # Private field for sinogram
        local_cpts = []
        sinogram = cpt[(0x300d, 0x10a7)].value.split("\\")
        try:
            leaves = numpy.array([float(x) for x in sinogram])
        except ValueError:
            # Empty control point
            leaves = numpy.zeros(linac_state["num_leaves"])

        open_leaves = numpy.nonzero(leaves)[0]
        if len(open_leaves) == 0:
            linac_state["cum_weight"] += linac_state["weight_per_cpt"]
            aperture = numpy.zeros((linac_state["num_leaves"], 2))
            aperture[::2] = [4.9, 5.0]
            aperture[1::2] = [-5.0, -4.9]

            cpt_dict = {}
            cpt_dict["arc_index"] = linac_state["beam_index"]
            cpt_dict["apertures"] = aperture.tolist()
            cpt_dict["x_jaw"] = linac_state["x_jaw"]
            cpt_dict["y_jaw"] = linac_state["y_jaw"]
            cpt_dict["sad"] = linac_state["sad"]
            cpt_dict["iso"] = linac_state["iso"]
            cpt_dict["gantry_angle"] = linac_state["gantry_angle"]
            cpt_dict["couch_angle"] = linac_state["couch_angle"]
            cpt_dict["col_angle"] = linac_state["col_angle"]
            cpt_dict["cum_weight"] = float(linac_state["cum_weight"])

            local_cpts.append(cpt_dict)
            return local_cpts

        if len(open_leaves) == 1:
            leaf_times = numpy.array([leaves[open_leaves]])
        else:
            leaf_times = leaves[open_leaves]

        # Sort leaves by time spent open
        sorted_leaves = sorted(list(zip(open_leaves, leaf_times)), key=lambda x: x[1])

        time_spent = 0.0

        for leaf_num, time in sorted_leaves:
            aperture = numpy.zeros((linac_state["num_leaves"], 2))
            # Even and odd leaves are closed differently
            aperture[::2] = [4.9, 5.0]
            aperture[1::2] = [-5.0, -4.9]

            for leaf in open_leaves:
                aperture[leaf][0] = -5.0
                aperture[leaf][1] = 5.0

            time_in_cpt = time - time_spent
            linac_state["cum_weight"] += time_in_cpt * linac_state["weight_per_cpt"]

            cpt_dict = {}
            cpt_dict["arc_index"] = linac_state["beam_index"]
            cpt_dict["apertures"] = aperture.tolist()
            cpt_dict["x_jaw"] = linac_state["x_jaw"]
            cpt_dict["y_jaw"] = linac_state["y_jaw"]
            cpt_dict["sad"] = linac_state["sad"]
            cpt_dict["iso"] = linac_state["iso"]
            cpt_dict["gantry_angle"] = linac_state["gantry_angle"]
            cpt_dict["couch_angle"] = linac_state["couch_angle"]
            cpt_dict["col_angle"] = linac_state["col_angle"]
            cpt_dict["cum_weight"] = float(linac_state["cum_weight"])

            local_cpts.append(cpt_dict)

            # Leaf closes after time_in_cpt has passed
            leaf_index = numpy.where(open_leaves == leaf_num)[0][0]
            open_leaves = numpy.delete(open_leaves, leaf_index)
            time_spent = time

        aperture = numpy.zeros((linac_state["num_leaves"], 2))
        aperture[::2] = [4.9, 5.0]
        aperture[1::2] = [-5.0, -4.9]
        linac_state["cum_weight"] += (1.0 - time_spent) * linac_state["weight_per_cpt"]

        cpt_dict = {}
        cpt_dict["arc_index"] = linac_state["beam_index"]
        cpt_dict["apertures"] = aperture.tolist()
        cpt_dict["x_jaw"] = linac_state["x_jaw"]
        cpt_dict["y_jaw"] = linac_state["y_jaw"]
        cpt_dict["sad"] = linac_state["sad"]
        cpt_dict["iso"] = linac_state["iso"]
        cpt_dict["gantry_angle"] = linac_state["gantry_angle"]
        cpt_dict["couch_angle"] = linac_state["couch_angle"]
        cpt_dict["col_angle"] = linac_state["col_angle"]
        cpt_dict["cum_weight"] = float(linac_state["cum_weight"])

        local_cpts.append(cpt_dict)

        return local_cpts

    def _process_local_cpt(self, cpt, linac_state):
        # Private field for sinogram
        local_cpts = []
        try:
            sinogram = cpt[(0x300d, 0x10a7)].value.split("\\")
        except:
            return []

        try:
            leaves = numpy.array([float(x) for x in sinogram])
        except ValueError:
            # Empty control point
            return []

        open_leaves = numpy.nonzero(leaves)[0]
        if len(open_leaves) == 0:
            return local_cpts

        if len(open_leaves) == 1:
            leaf_times = numpy.array([leaves[open_leaves]])
        else:
            leaf_times = leaves[open_leaves]

        # Sort leaves by time spent open
        sorted_leaves = sorted(list(zip(open_leaves, leaf_times)), key=lambda x: x[1])

        time_spent = 0.0
        for leaf_num, time in sorted_leaves:
            aperture = numpy.zeros((linac_state["num_leaves"], 2))
            # Even and odd leaves are closed differently
            aperture[::2] = [4.9, 5.0]
            aperture[1::2] = [-5.0, -4.9]
            for leaf in open_leaves:
                aperture[leaf][0] = -5.0
                aperture[leaf][1] = 5.0

            time_in_cpt = time - time_spent

            cpt_dict = {}
            cpt_dict["arc_index"] = linac_state["beam_index"]
            cpt_dict["apertures"] = aperture.tolist()
            cpt_dict["x_jaw"] = linac_state["x_jaw"]
            cpt_dict["y_jaw"] = linac_state["y_jaw"]
            cpt_dict["sad"] = linac_state["sad"]
            cpt_dict["iso"] = linac_state["iso"]
            cpt_dict["gantry_angle"] = linac_state["gantry_angle"]
            cpt_dict["couch_angle"] = linac_state["couch_angle"]
            cpt_dict["col_angle"] = linac_state["col_angle"]
            cpt_dict["cum_weight"] = float(linac_state["cum_weight"] + time_spent + time_in_cpt)

            local_cpts.append(cpt_dict)

            # Leaf closes after time_in_cpt has passed
            leaf_index = numpy.where(open_leaves == leaf_num)[0][0]
            open_leaves = numpy.delete(open_leaves, leaf_index)
            time_spent = time

        linac_state["cum_weight"] += time_spent

        return local_cpts

    def parse_plan(self):
        plan_dict = {}

        accumulated_meterset = 0.0

        # Tomo RTPlan files are missing the DICM marker, need force=True
        rp = dicom.read_file(self.rtplan_path, force=True)

        if "BeamSequence" not in rp:
            return {}

        cpts = []

        beams = []
        for beam_index, beam in enumerate(rp.BeamSequence):
            # Don't want to simulate setup beams
            if beam.TreatmentDeliveryType == "SETUP":
                continue

            beam_dict = {}
            sad = float(beam.SourceAxisDistance)

            beam_meterset = self._find_meterset(beam, rp.FractionGroupSequence[0])
            accumulated_meterset += beam_meterset
            linac_state = {
                "sad": sad,
                "gantry_angle": 0,
                "couch_angle": 0,
                "col_angle": 0,
                "x_jaw": [],
                "y_jaw": [],
                "cum_weight": 0.0,
                "num_leaves": 64,
                "beam_index": beam_index,
                "weight_per_cpt": 1.0 / len(beam.ControlPointSequence)
            }

            for cpt in beam.ControlPointSequence:
                if "NominalBeamEnergy" in cpt:
                    linac_state["energy"] = float(cpt.NominalBeamEnergy)

                if "BeamLimitingDeviceAngle" in cpt:
                    linac_state["col_angle"] = float(cpt.BeamLimitingDeviceAngle)

                if "GantryAngle" in cpt:
                    linac_state["gantry_angle"] = float(cpt.GantryAngle)

                if "TableTopRollAngle" in cpt:
                    linac_state["couch_angle"] = float(cpt.TableTopRollAngle)

                if "IsocenterPosition" in cpt:
                    linac_state["iso"] = [float(x) for x in cpt.IsocenterPosition]

                if "BeamLimitingDevicePositionSequence" in cpt:
                    for device in cpt.BeamLimitingDevicePositionSequence:
                        if "ASYMX" in device.RTBeamLimitingDeviceType:
                            linac_state["x_jaw"] = [float(x) for x in device.LeafJawPositions]
                        elif "ASYMY" in device.RTBeamLimitingDeviceType:
                            linac_state["y_jaw"] = [float(x) for x in device.LeafJawPositions]
                        elif device.RTBeamLimitingDeviceType == "X":
                            linac_state["x_jaw"] = [float(x) for x in device.LeafJawPositions]
                        elif device.RTBeamLimitingDeviceType == "Y":
                            linac_state["y_jaw"] = [float(x) for x in device.LeafJawPositions]

                local_cpts = self._process_local_cpt(cpt, linac_state)

                cpts += local_cpts

            beam_dict["cpts"] = cpts
            beam_dict["angle_spacing"] = beam.ControlPointSequence[1].GantryAngle - beam.ControlPointSequence[0].GantryAngle
            beam_dict["couch_spacing"] = (numpy.array(beam.ControlPointSequence[1].IsocenterPosition) - numpy.array(beam.ControlPointSequence[0].IsocenterPosition)).tolist()
            beams.append(beam_dict)

        plan_dict["beams"] = beams
        plan_dict["total_meterset"] = float(accumulated_meterset * rp.FractionGroupSequence[0].NumberOfFractionsPlanned)
        # Little hack to get gantry spacing and couch spacing
        return plan_dict
