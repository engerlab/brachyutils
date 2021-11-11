import pydicom as dicom
import traceback
import math
from ..utils import dicom_to_spherical_2


class ProtonPlan(object):
    """
        Attributes:
        rtplan_path (str): Path to DICOM RTPlan file.
    """
    def __init__(self, attrs):
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

    def _find_beam_norm(self, beam, fraction_sequence):
        norm_dict = {}
        for ref_beam in fraction_sequence.ReferencedBeamSequence:
            if beam.BeamNumber == ref_beam.ReferencedBeamNumber:
                if "BeamDoseSpecificationPoint" in ref_beam:
                    norm_dict["norm_point"] = [float(x) for x in ref_beam.BeamDoseSpecificationPoint]
                    norm_dict["norm_value"] = float(ref_beam.BeamDose * fraction_sequence.NumberOfFractionsPlanned)
                    return norm_dict
                else:
                    return -1
        return -1

    def parse_plan(self):
        plan_dict = {}

        rp = dicom.read_file(self.rtplan_path, force=True)
        beams = []

        if "IonBeamSequence" not in rp:
            return

        for beam_index, beam in enumerate(rp.IonBeamSequence):
            try:
                # Don't want to simulate setup beams
                if beam.TreatmentDeliveryType == "SETUP":
                    continue

                cpts = []
                beam_dict = {}

                virtual_sad = beam.VirtualSourceAxisDistances

                previous_energy = 6
                previous_couch_angle = 0
                previous_gantry_angle = 0

                beam_norm = self._find_beam_norm(beam, rp.FractionGroupSequence[0])
                if isinstance(beam_norm, dict):
                    beam_dict["norm_point"] = beam_norm["norm_point"]
                    beam_dict["norm_value"] = beam_norm["norm_value"]

                for cpt in beam.IonControlPointSequence:
                    cpt_dict = {}
                    cpt_dict["arc_index"] = int(beam.BeamNumber)
                    cpt_dict["virtual_sad"] = virtual_sad
                    cpt_dict["couch_angle"] = 0
                    cpt_dict["col_angle"] = 0

                    if "NominalBeamEnergy" in cpt:
                        cpt_dict["energy"] = float(cpt.NominalBeamEnergy)
                        previous_energy = float(cpt.NominalBeamEnergy)
                    else:
                        cpt_dict["energy"] = previous_energy

                    if "SnoutPosition" in cpt:
                        cpt_dict["snout_position"] = float(cpt.SnoutPosition)
                        previous_snout_position = float(cpt.SnoutPosition)
                    else:
                        cpt_dict["snout_position"] = previous_snout_position

                    if "ScanningSpotSize" in cpt:
                        cpt_dict["spot_size"] = [float(x) for x in cpt.ScanningSpotSize]
                        previous_spot_size = [float(x) for x in cpt.ScanningSpotSize]
                    else:
                        cpt_dict["spot_size"] = previous_spot_size

                    if "GantryAngle" in cpt:
                        cpt_dict["gantry_angle"] = float(cpt.GantryAngle)
                        previous_gantry_angle = float(cpt.GantryAngle)
                    else:
                        cpt_dict["gantry_angle"] = previous_gantry_angle

                    if "PatientSupportAngle" in cpt:
                        cpt_dict["couch_angle"] = float(cpt.PatientSupportAngle)
                        if "Varian" in rp.Manufacturer:
                            cpt_dict["couch_angle"] = 360.0 - cpt_dict["couch_angle"]

                        previous_couch_angle = cpt_dict["couch_angle"]

                    else:
                        cpt_dict["couch_angle"] = previous_couch_angle

                    if "IsocenterPosition" in cpt:
                        cpt_dict["iso"] = [float(x) for x in cpt.IsocenterPosition]
                        previous_iso = [float(x) for x in cpt.IsocenterPosition]
                    else:
                        cpt_dict["iso"] = previous_iso

                    if "ScanSpotPositionMap" in cpt:
                        cpt_dict["spot_positions"] = [float(x) for x in cpt.ScanSpotPositionMap]

                    if "ScanSpotMetersetWeights" in cpt:
                        # If there's only one position, this field is a simple float instead of a list
                        if isinstance(cpt.ScanSpotMetersetWeights, float):
                            cpt_dict["spot_weights"] = [cpt.ScanSpotMetersetWeights]
                        else:
                            cpt_dict["spot_weights"] = [float(x) for x in cpt.ScanSpotMetersetWeights]

                    spot_sum = sum(cpt_dict["spot_weights"])
                    if spot_sum > 0:
                        cpt_dict["weight"] = spot_sum

                        theta, phi, phicol = dicom_to_spherical_2(cpt_dict["gantry_angle"], cpt_dict["couch_angle"], cpt_dict["col_angle"])
                        cpt_dict["theta"] = theta * 180.0 / math.pi
                        cpt_dict["phi"] = phi * 180.0 / math.pi
                        cpt_dict["phicol"] = phicol * 180.0 / math.pi

                        # Take the average of the two virtual SADs for a rough position.
                        cpt_dict["sad"] = sum(cpt_dict["virtual_sad"]) / len(cpt_dict["virtual_sad"])

                        cpts.append(cpt_dict)

                beam_dict["cpts"] = cpts
                beams.append(beam_dict)

            except BaseException, e:
                traceback.format_exc()
                print e

        plan_dict["beams"] = beams
        return plan_dict
