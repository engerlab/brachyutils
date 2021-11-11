import pydicom as dicom
import numpy


class BrachyPlan(object):
    """
        Attributes:
        rtplan_path (str): path to DICOM RTPlan file.
    """
    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def parse_plan(self):
        plan_dict = {}

        rtplan = dicom.read_file(self.rtplan_path, force=True)

        # Make a list of all unique control points for each channel.
        # Remove duplicate control point information.
        unique_points = []
        for AS in rtplan.ApplicationSetupSequence:
            for channel in AS.ChannelSequence:
                if (0x300a, 0x2c8) in channel:
                    chan_dict = {}
                    chan_dict["total_time"] = channel.ChannelTotalTime
                    chan_dict["final_weight"] = channel.FinalCumulativeTimeWeight
                    chan_dict["cpt"] = []
                    old_weight = 0.0
                    for control_point in channel.BrachyControlPointSequence:
                        current_weight = control_point.CumulativeTimeWeight
                        if current_weight > old_weight:
                            chan_dict["cpt"].append(control_point)
                            old_weight = current_weight
                    unique_points.append(chan_dict)

        # Now we're sure that all control points are unique and valid, so we
        # go through them and compute weights.
        catheters = []
        for channel in unique_points:
            dwell_positions = []
            previous_weight = 0.0
            for index, control_point in enumerate(channel["cpt"]):
                current_cum_weight = control_point.CumulativeTimeWeight
                weight = current_cum_weight - previous_weight
                previous_weight = current_cum_weight
                position = numpy.array(control_point.ControlPoint3DPosition)

                normalized_weight = channel["total_time"] * weight / channel["final_weight"]

                dwell_position = position.tolist()
                dwell_pos_dict = {
                    "x": dwell_position[0],
                    "y": dwell_position[1],
                    "z": dwell_position[2],
                    "time": normalized_weight,
                    "angle": 0
                }

                dwell_positions.append(dwell_pos_dict)

            catheters.append(dwell_positions)

        plan_dict["catheters"] = catheters
        plan_dict["isotope"] = rtplan.SourceSequence[0].SourceIsotopeName
        plan_dict["ref_akr"] = float(rtplan.SourceSequence[0].ReferenceAirKermaRate)

        return plan_dict
