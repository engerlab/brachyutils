import numpy
import pydicom as dicom
import sys
import copy

def get_leaf_numbers(self, pos, thickness, leaf_boundaries):
	# 1 cm projection at iso for y < -10 and y > 10
	# 0.5 cm projection at iso for -10 <= y <= 10
	neg_limit = pos - 0.5 * thickness
	pos_limit = pos + 0.5 * thickness
	leaves_included = (leaf_boundaries > neg_limit) & (leaf_boundaries <= pos_limit)
	return leaves_included.nonzero()[0] - 1

custom_filename = sys.argv[1]
template_filename = sys.argv[2]

custom_plan = open(custom_filename)
custom_json = json.load(custom_plan)
custom_plan.close()

template = dicom.read_file(template_filename)

new_beam_sequence = dicom.sequence.Sequence()
for beam in custom_json["photon_cpts"]:
    beam_copy = copy.deepcopy(template.BeamSequence[0])
    beam_copy.ControlPointSequence[0].BeamLimitingDevicePositionSequence[0] = [-100.0, 100.0]
    beam_copy.ControlPointSequence[0].BeamLimitingDevicePositionSequence[1] = [-100.0, 100.0]
    beam_copy.ControlPointSequence[0].GantryAngle = beam["gantry_angle"]
    beam_copy.ControlPointSequence[0].CollimatorAngle = beam["col_angle"]
    beam_copy.ControlPointSequence[0].PatientSupportAngle = beam["couch_angle"]
    beam_copy.ControlPointSequence[0].IsocenterPosition = beam["iso"]
    
    a_leaf_values = []
    b_leaf_values = []

	row_positions = numpy.arange(cpt["beamlet_rows"]) * cpt["iso_row_size"] - (cpt["beamlet_rows"] / 2.0 - 0.5) * cpt["iso_row_size"]
	leaf_numbers = [get_leaf_numbers(pos, cpt["iso_row_size"]) for pos in row_positions]




    new_beam_sequence.append(beam_copy)
