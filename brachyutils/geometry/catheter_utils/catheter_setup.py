#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:06:37 2021

@author: sebquet
"""
import copy
import os
import sys
from typing import List
from pathlib import Path
import warnings 
from collections import Counter

import pydicom as dicom
import numpy as np
from scipy.spatial.distance import cdist
import SimpleITK as sitk

from brachyutils.geometry.catheter_utils.utils import distance, min_dist_two_list, min_cost_two_list
from brachyutils.geometry.catheter_utils.digitization.pw_linear_interpolator import PiecewiseLinear3D, Segment, extrapolate_point
from brachyutils.geometry.catheter_utils.utils import create_marker_pts_from_catheter_dict, create_slicer_markup_points

class CatheterSetUp(object):

    def __init__(self, CT_folder: str| Path = None, setup:bool=True):
        """
        We get the digitization points first and then create dwel positions from these digitization points.
        The created dwell positions will serve as catheter index reference. We start from digitizations points
        and not fom dwell positions in the treatment plan (non-0s dwell positions) because some catheters do
        not have any non-0 dwell positions and then it makes it hard to have non-0 dwell positions as catheter
        references to create digitizations point and map them to the correct catheter.
        """
        if CT_folder is not None:
            self.CT_folder = Path(CT_folder)
            plan_file_path = self.get_plan_file()
            self.plan_file = dicom.dcmread(plan_file_path, force=True)
        else:
            self.CT_folder = None
            plan_file_path = None
            self.plan_file = None

        self.digitization_points = None
        self.custom_digitization_points = None
        self.temporary_dwell_positions = None
        self.dwell_positions = None
        self.piece_wise_lines = None
        self.channel_length = None
        self.non_zero_dwell_positions = None
        self.step_dwell_pos = None
        self.tip_coord = None
        self.offset = None
        self.last_digipt_is_first_dwellpos = None
        self.catheter_table = None

        # We create the digitization points first, which do not contains explicitly channel numbers.
        # So we will map them to the channel numbers later.
        self.digi_pts_mapping_to_correct_channel = {}
        
        if self.CT_folder and setup:
            self._setup_patient()

    def _setup_patient(self):
        digitization_points = self._get_digitization_points()
        self.step_size = self.get_step_size()
        # From the digtization points we can derive two sets of dwell positions,
        # one form the first dwell position and one from the last dwell position.
        # We first pick any set (the temporary dwell positions) to assign the 
        # activated dwell positions the correct catheter key. Without the temporary 
        # dwell positions (and with only the digitizaton points), we could mismatch 
        # activated dwell positions and catheters because some catheters have no 
        # activated dwell positoins in the treatment plan and the activated dwell pos
        # might be closer to the digi point of another catheter than the true catheter
        # it belongs to. With temporary dwell positions, such errors are avoided.
        temporary_dwell_positions = self._get_temporary_dwell_positions(digitization_points)
        self.non_zero_dwell_positions, catheter_table_with_only_activated_dp = (
            self._get_non_zero_dwell_positions(temporary_dwell_positions)
        )  
        # The first digitizatin points created did not have the correct channel numbers
        # as keys. We need to map them here to the correct flexi-comfort pair channel 
        # numbers because we use the digi pts keys in get_dwell_positions function.
        self.digitization_points = {}
        for temp_needle_key in digitization_points.keys():
            catheter_channel = self.digi_pts_mapping_to_correct_channel[temp_needle_key]
            self.digitization_points[catheter_channel] = copy.deepcopy(digitization_points[temp_needle_key])

        # The info contained in the catheter table alows us to define the tip side
        # i.e. the side with the smallest RelativePosition to the tip.
        _ = self.get_dwell_positions(catheter_table_with_only_activated_dp)
        # If there are no activated dwell positions for a specific catheter,
        # we need to add those to the catheter table. We also need to add any other 
        # non-activated dwell positions to the catheter table.
        self.catheter_table = self._add_zero_treatment_times(catheter_table_with_only_activated_dp)


    def get_plan_file(self):

        return list(self.CT_folder.glob("RP*.dcm"))[0] 

    def get_reference_absorbed_dose(self):
        file = self.plan_file
        for ROI in file.DoseReferenceSequence:
            print("hi")

    def get_nb_catheters(self):
        return len(self.plan_file.ApplicationSetupSequence[0].ChannelSequence)

    def get_step_size(self):

        step_size = float(
            self.plan_file.ApplicationSetupSequence[0].ChannelSequence[0].SourceApplicatorStepSize
            )
        for channel in self.plan_file.ApplicationSetupSequence[0].ChannelSequence:
            if step_size != float(channel.SourceApplicatorStepSize):
                print("Step size is not the same for all catheters")
                return None
        return step_size
    
    def get_reference_air_kerma_rate(self):
        return float(self.plan_file.SourceSequence[0].ReferenceAirKermaRate)
    
    def get_total_reference_air_kerma(self):
        return float(self.plan_file.ApplicationSetupSequence[0].TotalReferenceAirKerma)
    
    def get_treatment_site(self):
        return self.plan_file.ApplicationSetupSequence[0].ApplicationSetupType
    
    def get_treatment_technique(self):
        return str(self.plan_file.BrachyTreatmentTechnique) + str(self.plan_file.BrachyTreatmentType)
    
    def get_dose_constraints(self):
        # ReferencedROINumber should be linked to the RTSTRUCT file.
        # From StructureSetROISequence -> we can map ROINumber to ROIName
        dose_constraints = {}
        for dose_constraint in self.plan_file.DoseReferenceSequence:

            if "MAX" in dose_constraint.DoseReferenceDescription:
                if "TARGET" in dose_constraint.DoseReferenceType:
                    key = f"ROI_{dose_constraint.ReferencedROINumber}_{dose_constraint.DoseReferenceType}_{dose_constraint.DoseReferenceDescription}"
                    if "TargetMaximumDose" in dose_constraint:
                        dose_constraints[key] = dose_constraint.TargetMaximumDose
                else:
                    assert "ORGAN_AT_RISK" in dose_constraint.DoseReferenceType
                    key = f"ROI_{dose_constraint.ReferencedROINumber}_{dose_constraint.DoseReferenceType}_{dose_constraint.DoseReferenceDescription}"
                    if "OrganAtRiskLimitDose" in dose_constraint:
                        dose_constraints[key] = dose_constraint.OrganAtRiskLimitDose
            elif "MIN" in  dose_constraint.DoseReferenceDescription:
                key = f"ROI_{dose_constraint.ReferencedROINumber}_{dose_constraint.DoseReferenceType}_{dose_constraint.DoseReferenceDescription}"
                if "TargetMinimumDose" in dose_constraint:
                    dose_constraints[key] = dose_constraint.TargetMinimumDose
            else:
                print("We do not include this constraint: ", dose_constraint)
        dose_constraints["Prescribed Dose"] = self.plan_file.FractionGroupSequence[0].ReferencedBrachyApplicationSetupSequence[0].BrachyApplicationSetupDose
        dose_constraints["Number of fractions"] = self.plan_file.FractionGroupSequence[0].NumberOfFractionsPlanned
        return dose_constraints
    
    def get_target_dose(self):
        return self.plan_file.FractionGroupSequence[0].ReferencedBrachyApplicationSetupSequence[0].BrachyApplicationSetupDose
    
    def get_treatment_total_time(self):
        tt = 0
        for channelseq in self.plan_file.ApplicationSetupSequence[0].ChannelSequence:
            tt += channelseq.ChannelTotalTime
        return tt
    
    def get_channel_length(self):
        for channel in self.plan_file.ApplicationSetupSequence[0].ChannelSequence:
            if channel == 0.0:
                continue
            c_length =  channel.ChannelLength
            break
        return c_length
    
    def get_source_name(self):
        return self.plan_file.TreatmentMachineSequence[0].ManufacturerModelName
    
    def get_channel_id(self, channel, verbose:bool=False):
        """
        Inputs:
            - channel: the channel from the DICOM file ApplicationSetupSequence[0].

        Getting the channel id from the channel sequence.
        At the JGH, this ID is placed manually by a physician on each (flexi) catheter 
        via Oncentra TPS to map each catheter to a SourceApplicator (comfort catheter 
        in our JGH breast case). Considering all the catheter are inserted in a grid 
        (the Dr tries to insert more or less in rows and columns), the convention 
        used in our institution is to identify the top right corner catheter as 
        channel 1, then the one on the left as channel 2, etc until the row is done 
        then we go to the row below. The convention tries to 
        have same number for Source Applicator Number (300a,0290) and Channel 
        Number (300a,0282).
        """
        # For Breast Brachytherapy, only:
        # After discussion with a Medical Physicist, the following convention is used 
        # at the JGH: ChannelNumber corresponding to the flexi catheter inserted in 
        # the breast and cut to the correct breast length, and SourceApplicatorNumber
        # corresponding to the comfort catheter (Elekta OncoSmart catheter) that is 
        # inserted during treatment inside the breast are always chosen to be the 
        # same for simplication of the treatment. Comfort catheter 1 goes into flexi 
        # catheter 1 and so on.
        # This is when they use the OncoSmart/comfort catheter, which is what they do
        # now. But if they use the flexi catheter only (not cut, which falls off the 
        # breast for the duration of the treatment ~1 week), I believe the catheter 
        # channel (ChannelNumber) and the SourceApplicator are already the same thing.
        # Now they only use the comfort-flexi combination and not the flexi catheter on
        # its own since it provides better quality of life for patients.  
        assert channel.ChannelNumber == channel.SourceApplicatorNumber, (
            """The channel number and the source applicator number are not the same for patient {}.
            {} VS {}
            """.format(
                self.CT_folder, channel.ChannelNumber, channel.SourceApplicatorNumber)
        )

        if verbose:
            print(
"""Comfort (Elketa OncoSmart) catheter number: {cc} will be inserted in flexi
catheter number: {fc}. The Comfort catheter {cc} is linked to Transfer tube 
number {tt}, itself inserted in the afterloader channel number {ac}. \n""".format(
                      cc=channel.SourceApplicatorNumber, fc=channel.ChannelNumber,
                      tt =channel.TransferTubeNumber, ac=channel.ChannelNumber
                  )
            )

        return channel.ChannelNumber
        
    def _get_non_zero_dwell_positions(self, temporary_dwell_positions:dict):
        catheter_table = []
        dwell_positions = {}
        channel_lengths = []
        total_treatment_time = self.get_treatment_total_time()
        for channel in self.plan_file.ApplicationSetupSequence[0].ChannelSequence:
            channel_total_time = float(channel.ChannelTotalTime)
            if channel_total_time == 0.0:
                print("Channel", channel.ChannelNumber, "has no activated dwell positions")
                continue
            channel_final_cum_time_weight = float(channel.FinalCumulativeTimeWeight)
            channel_lengths.append(channel.ChannelLength)
            previous_cum_time_weight = 0.0
            # Identify the digtization point needle key that matches 
            # the dwell position.
            tmp_non0dp = []
            for dwell_pos in channel.BrachyControlPointSequence:
                if (
                    dwell_pos.ControlPoint3DPosition
                    not in tmp_non0dp
                ): 
                    tmp_non0dp.append(
                        dwell_pos.ControlPoint3DPosition
                    )
            non_zero_dp_needle_key = self.select_best_needle_from_list(
                temporary_dwell_positions, tmp_non0dp
            )
            # Temporary dwell positions have same key as digi pts.
            self.digi_pts_mapping_to_correct_channel[non_zero_dp_needle_key] = "Channel_" + str(self.get_channel_id(channel))

            ctrl_pts = []
            ctrl_pt_counter = 0
            for ctrl_pt in channel.BrachyControlPointSequence:

                if not(non_zero_dp_needle_key in dwell_positions.keys()):
                        dwell_positions[non_zero_dp_needle_key] = []

                # The dwell position coord appear twice in the DICOM
                if (
                    ctrl_pt.ControlPoint3DPosition
                    not in dwell_positions[non_zero_dp_needle_key]
                ): 
                    dwell_positions[non_zero_dp_needle_key].append(
                        ctrl_pt.ControlPoint3DPosition
                    ) 
                    previous_cum_time_weight = float(ctrl_pt.CumulativeTimeWeight)  
                # Second time we can add the cum time weight since the first      
                # cum time weight is the one of the previous dwell pos.        
                else:
                    # Definition taken from:
                    # https://dicom.innolitics.com/ciods/rt-plan/rt-brachy-application-setups/300a0230/300a0280/300a02c8
                    # The treatment time at a given Control Point is equal to the Channel Total Time (300A,0286), 
                    # multiplied by the Cumulative Time Weight (300A,02D6) for the Control Point, 
                    # divided by the Final Cumulative Time Weight (300A,02C8).

                    time_spent_at_dwell = channel_total_time * (
                        (float(ctrl_pt.CumulativeTimeWeight) - previous_cum_time_weight)/
                        channel_final_cum_time_weight)

                    ctrl_pts.append({
                        "index": ctrl_pt_counter,
                        "angle": (
                            ctrl_pt.ControlPointShieldAngle
                            if hasattr(ctrl_pt, "ControlPointShieldAngle")
                            else 0
                        ),
                        "position": np.array(ctrl_pt.ControlPoint3DPosition,dtype=float),
                        "relativePos": ctrl_pt.ControlPointRelativePosition,
                        "rotation": (
                            np.array(
                                ctrl_pt.ControlPointOrientation, dtype=float
                            )
                            if hasattr(ctrl_pt, "ControlPointOrientation")
                            else np.array([0, 0, 0], dtype=float)
                        ),
                        "time": time_spent_at_dwell,
                        # Weight regarind total treatment time since this one will be used 
                        # in the mac file.
                        "weight": time_spent_at_dwell / total_treatment_time
                        })
                    ctrl_pt_counter += 1

            channel_infos = {
                # This channel ID is the exact one used in the treatment plan 
                # to identify the flexi catheter and map it to the comfort catheter,
                # which itself is linked to a transfer tube and an afterloader channel.
                "channel_number": self.get_channel_id(channel),
                "points": [],
                "digi_pts_channel": non_zero_dp_needle_key,
                "channel_total_time": channel_total_time,
                "dwells": ctrl_pts,
                "channel_length": channel.ChannelLength
            }
            catheter_table.append(channel_infos)

            # Check the time weight you gathered are correct 
            assert np.isclose(np.sum([dwp["time"] for dwp in channel_infos["dwells"]]),
                              channel_total_time), (
                """The sum of the dwell positions time weights does not match the channel time weight.
                {} VS {}
                for patient {}
                """.format(
                    np.sum([dwp["time"] for dwp in channel_infos["dwells"]]), channel_total_time,
                    str(self.CT_folder))
            )

            if channel.ChannelTotalTime != 0.0:
                assert int(channel.NumberOfControlPoints/2) == len(dwell_positions[non_zero_dp_needle_key]), (
                    """We did not retrieve the same number of dwell positions as in the treatmentplan. 
                    {} VS {}
                    for patient {}
                    """.format(
                        channel.NumberOfControlPoints, len(dwell_positions[non_zero_dp_needle_key]),
                        str(self.CT_folder))
                    )

        # Adding catheters if ever there were no activated dwell positions in the plan
        for needle_key in temporary_dwell_positions.keys():
            if needle_key not in dwell_positions.keys():
                dwell_positions[needle_key] = []
                print(f"There are no activated dwell positions in the treatment plan for patient {self.CT_folder} for needle ", needle_key)

        # Catheters without any active dwell position are not in the mapping dict.
        # They could not be mapped geomtrically, we will map them based on DICOM 
        # tags. Dwell positions and digitization points are 
        # not stored in the same tag in the DICOM file. We assume that the 
        # ReferencedROINumber of the catheter digitization point
        # corresponds to the ChannelNumber of the ChannelSequence.
        if len(list(temporary_dwell_positions.keys())) != len(
            list(self.digi_pts_mapping_to_correct_channel.keys())):
            unmapped_number = len(list(temporary_dwell_positions.keys())) - len(
                list(self.digi_pts_mapping_to_correct_channel.keys()))
            warnings.warn(
                f"""Mapping digitization points and dwell positions of the DICOM 
                  file for {unmapped_number} catheters based on the assumption that 
                  dcmfile.ApplicationSetupSequence[0].ChannelSequence[i].ChannelNumber
                   can be mapped to 
                  dcmfile[(0x300F, 0x1000)][0].ROIContourSequence[j].ReferencedROINumber""")

            ### Check that the current assignment makes sense: that the keys in digi pts dict
            # either are the same than in the temporary dict or the order is the same.
            # Some digi points channel could have been created without any point by the user so
            # it makes an empty digi point channel which will shift the id of the digi points channel. 
            ## Checking that there are either as many digi points channels as temporary channels or more.
            exact_match = True
            for k, v in self.digi_pts_mapping_to_correct_channel.items():
                idx_k = int(k.split("_")[1])
                idx_v = int(v.split("_")[1])
                assert idx_k >= idx_v, (
                    "Digi points channel index from the DICOM do not match activated dwell positions channel."
                    "Your current mapping looks like:{}".format(self.digi_pts_mapping_to_correct_channel)
                )
                if not (k == v):
                    exact_match = False
            ## Checking the order
            sorted_digi_pts_keys = sorted(
                self.digi_pts_mapping_to_correct_channel.keys(), 
                key=lambda x: int(x.split("_")[1])
                )
            prev_activated_dp_idx = -1
            for k in sorted_digi_pts_keys:
                idx_k = int(k.split("_")[1])
                idx_v = int(self.digi_pts_mapping_to_correct_channel[k].split("_")[1])
                assert idx_k > prev_activated_dp_idx, (
                    "Digi points channel index from the DICOM do not match activated dwell positions channel.",
                    "Your current mapping looks like:{}".format(self.digi_pts_mapping_to_correct_channel),
                    "The ordering is not respected."
                )
                prev_activated_dp_idx = idx_v
            if exact_match:
                for needle_key in temporary_dwell_positions.keys():
                    if needle_key not in self.digi_pts_mapping_to_correct_channel.keys():
                        self.digi_pts_mapping_to_correct_channel[needle_key] = needle_key
            else:
                sorted_temp_dp_keys = sorted(
                    temporary_dwell_positions.keys(), 
                    key=lambda x: int(x.split("_")[1])
                    )
                prev_activated_dp_idx = 1
                for k in sorted_temp_dp_keys:
                    if k not in self.digi_pts_mapping_to_correct_channel.keys():
                        # Manually mapping channel k from the digitization points to 
                        # Channel_{prev_activated_dp_idx} which has no activated dwell position.
                        self.digi_pts_mapping_to_correct_channel[k] = "Channel_" + str(prev_activated_dp_idx)
                    prev_activated_dp_idx += 1

        # Up to the line above, we needed digi pts keys to know which catheter do not have any activated dp 
        # but now we will give keys that are relevant to the treatment plan: channel numbers.
        dwell_positions_mapped_channels = {}
        for needle_key in temporary_dwell_positions.keys():
            catheter_channel = self.digi_pts_mapping_to_correct_channel[needle_key]
            dwell_positions_mapped_channels[catheter_channel] = copy.deepcopy(dwell_positions[needle_key])

        # Getting channel lenght
        if len(set(channel_lengths)) != 1:
            print("***************************************************")
            print("===================================================")
            print("CHANNELS ARE NOT ALL OF THE SAME LENGTH for patient ", self.CT_folder)
            print("===================================================")
            print("***************************************************")
            with open(os.path.join(Path(__file__).parents[0], "digitization/patient_wrong_digitization.txt" ), "a") as myfile:
                myfile.write(f"{self.CT_folder} has channels of different dimensions {set(channel_lengths)}mm. \n")

        self.channel_length = channel_lengths[0]

        return dwell_positions_mapped_channels, catheter_table

    def _get_standalone_non_zero_dwell_positions_list(self):
        """
        This function is here in case the user just wants the non-0s dwell poitions
        from the treatment plan.
        """
        dwell_positions = []
        for channel in self.plan_file.ApplicationSetupSequence[0].ChannelSequence: 
            channel_total_time = channel.ChannelTotalTime
            if channel_total_time == 0.0:
                continue
            for dwell_pos in channel.BrachyControlPointSequence:
                if dwell_pos.CumulativeTimeWeight is not None:
                    if (
                        dwell_pos.ControlPoint3DPosition
                        not in dwell_positions
                    ): 
                        dwell_positions.append(
                            dwell_pos.ControlPoint3DPosition
                        )      
        return dwell_positions
    
    def get_non_zero_dwell_positions(self):
        if self.non_zero_dwell_positions is None:
            temporary_dwell_positions = self._get_temporary_dwell_positions()
            self.non_zero_dwell_positions, _ = (
                self._get_non_zero_dwell_positions(temporary_dwell_positions)
            )
        return self.non_zero_dwell_positions

    def _add_zero_treatment_times(self, catheter_table_to_update:dict):
        """
        Going through the dwell positions not already in the dictionnary and adding 0
        seconds to the dwell positions.
        """

        new_catheter_table = []

        for catheter_key, dwell_positions in self.dwell_positions.items():
            ctrl_pts = []
            if len(self.non_zero_dwell_positions[catheter_key]) > 0:
                existing_catheter_infos_activated_dwells = self.get_catheter_from_table(
                    catheter_table_to_update, catheter_key)
                existing_indexes = [int(x["relativePos"]/self.step_size) 
                                    for x in existing_catheter_infos_activated_dwells["dwells"]]
                used_existing_idx_counter = 0
                for dwell_pos_idx, dwell_pos in enumerate(dwell_positions):
                    if dwell_pos_idx in existing_indexes:
                        # If this dwell position is already in the catheter table, 
                        # it already has a time, weight etc. 
                        min_dist = min_dist_two_list(
                            [dwell_pos], self.non_zero_dwell_positions[catheter_key]
                        ) 
                        assert min_dist < 0.01, (
                            """The dwell position is not the same as the activated dwell position.
                            {} VS {}
                            for patient {}
                            """.format(
                                dwell_pos, self.non_zero_dwell_positions[catheter_key],
                                self.CT_folder)
                            )
                        existing_ctrl_pt = existing_catheter_infos_activated_dwells["dwells"][used_existing_idx_counter]
                        # Updating the index of the dwell position since now we have more dwell positions.
                        existing_ctrl_pt["index"] = dwell_pos_idx
                        ctrl_pts.append(existing_ctrl_pt)
                        used_existing_idx_counter += 1
                    else:
                        # We create the ctrl point with 0 seconds
                        ctrl_pts.append({
                            "index": dwell_pos_idx,
                            "angle": 0,
                            "position": np.array(dwell_pos),
                            "relativePos": dwell_pos_idx*self.step_size,
                            "rotation": np.array([0, 0, 0]),
                            "time": 0.0,
                            "weight": 0.0
                        })

                assert existing_catheter_infos_activated_dwells["channel_number"] == catheter_key.split("_")[1], (
                    f"catheter key: {catheter_key} VS existing_catheter_infos_activated_dwells: {existing_catheter_infos_activated_dwells}"
                    )
                channel_infos = {
                    "channel_number": existing_catheter_infos_activated_dwells["channel_number"],
                    "points": [],
                    "digi_pts_channel": existing_catheter_infos_activated_dwells["digi_pts_channel"],
                    "channel_total_time": existing_catheter_infos_activated_dwells["channel_total_time"],
                    "dwells": ctrl_pts,
                    "channel_length": existing_catheter_infos_activated_dwells["channel_length"]
                }
            else:
                for dwell_pos_idx, dwell_pos in enumerate(dwell_positions):
                        # We create the ctrl point with 0 seconds
                        ctrl_pts.append({
                            "index": dwell_pos_idx,
                            "angle": 0,
                            "position": np.array(dwell_pos),
                            "relativePos": dwell_pos_idx*self.step_size,
                            "rotation": np.array([0, 0, 0]),
                            "time": 0.0,
                            "weight": 0.0
                        })
                treatment_channel_to_digi_pts_channel  = {v :k for k,v in self.digi_pts_mapping_to_correct_channel.items()}
                channel_infos = {
                    "channel_number": catheter_key, 
                    "points": [],
                    "digi_pts_channel": treatment_channel_to_digi_pts_channel[catheter_key],
                    "channel_total_time": 0.0,
                    "dwells": ctrl_pts,
                    "channel_length": None
                }
            
            # Correct for rotation if all are [0,0,0]
            dwells = channel_infos["dwells"]
            if np.all([np.all(dwells[i]["rotation"] == 0)for i in range(len(dwells))]):
                for i in range(len(dwells)):
                    dwells[i]["rotation"] = get_rotation_from_position(i, dwells)

            new_catheter_table.append(channel_infos)
        
        return new_catheter_table

    def get_coord_bounding_box(self):

        list_dwell = self.get_non_zero_dwell_positions()
        dwellpos_list = np.array(list_dwell)
        assert dwellpos_list.shape[1] == 3
        x_coords = dwellpos_list[:, 0]
        y_coords = dwellpos_list[:, 1]
        z_coords = dwellpos_list[:, 2]

        bounding_box = {}
        bounding_box["min"] = np.array([min(x_coords), min(y_coords), min(z_coords)])
        bounding_box["max"] = np.array([max(x_coords), max(y_coords), max(z_coords)])

        return bounding_box

    def _get_digitization_points(self):
        digitization_points = {}
        # There is no keyword for this tag... But this tag is necessary.
        # I had to reexport all the breast patients manually from Oncentra to have 
        # this tag in the DICOM file.
        digi_pts_sequence = self.plan_file[(0x300F, 0x1000)][0].ROIContourSequence
        for digitazation_pts_list in digi_pts_sequence:
            roi_number = digitazation_pts_list.ReferencedROINumber + 1
            # The digi points are a list for which for the full plan (all catheters), 
            # the catheters have been digitized either from tip end or from connector end.
            # There are in the order of digitization normally.
            points = self.split_list(
                digitazation_pts_list.ContourSequence[0].ContourData
            )
            # Checking if the digitization points are in a sequential order
            # Ensuring distance between points 1 and 2 is smaller than distance between 
            # points 1 and 3, etc
            for i in range(1, len(points) - 1):
                dist1 = distance(points[0], points[i])
                dist2 = distance(points[0], points[i + 1])
                # It can happen that if two digi points are super close and for the angle with
                # digipoint[0] make it so that distances are not exactly < between two points.
                # So we take int to remove the edge cases where this condition is not verified.
                assert int(dist1) <= int(dist2), f"""
                Digitization points are not in the correct order for needle {roi_number} and patient {self.CT_folder}.
                Points {i} and {i+1} are not in the correct order, with distances {dist1} and {dist2} respectively.
                """
            # We could maybe also just verify the order of the points projected 
            # on the segment made by first and last digi points.
            # # Create a segment, project the digi points on the segment and see if 
            # # the ts of the project digi points on the segment are ordered.
            # segment = Segment(points,  ref_slice_coord=None,
            # interslice_ax=2, init_line=True)
            # ts = []
            # for point in points:
            #     ts.append(segment._get_t_on_curve(point)) 
            # assert np.all(np.sort(ts) == ts)

            if len(points) > 1:
                digitization_points[f"Channel_{roi_number}"] = points

        return digitization_points


    def save_digitization_points(self, save_dir:str, verbose:bool=True):
        """
        Saving the digitization points in a file for later use.
        """
        if verbose:
            print("Saving digitization points in ", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for k in self.digitization_points.keys():
            create_slicer_markup_points(
                os.path.join(save_dir, f"Clinical_digitizations_points_channel_{k}.mrk.json"), 
                self.digitization_points[k], 
                color=[0.,0.5,0.], # green for ground truth
            )

    def save_dwell_positions(self, save_dir:str, verbose:bool=True):
        """
        Saving the digitization points in a file for later use.
        """
        if verbose:
            print("Saving dwell positions points in ", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for k in self.dwell_positions.keys():
            create_slicer_markup_points(
                os.path.join(save_dir, f"Clinical_dwell_positions_channel_{k}.mrk.json"), 
                self.dwell_positions[k], 
                color=[0.,0.5,0.],# green for ground truth
            )
        
    def get_digitization_points(self):
        """
        Digtization points are points manually placed on the catheters (on the CT with markers)
        to let Oncentra Treatment Planning System know where it can place the dwell positions.
        Dwell positions are linearly interpolated between the different digitization points.
        """
        if self.digitization_points is None:
            self.digitization_points = self._get_digitization_points()
        return self.digitization_points

    @staticmethod
    def split_list(lst, n=3):
        new_list = []
        for i in range(0, len(lst), n):
            new_list.append([float(a) for a in lst[i : i + n]])
        return new_list

    @staticmethod
    def select_best_needle_from_list(
        dp_to_select_from: dict,
        reference_single_needle_dwell_positions: List[float],
        return_min_dist:bool=False
    ):
        """

        Finding the needle that is closest to the dwell positions.

        Args:
            sitk_needles (_type_): _description_
            needle_dwell_positions (_type_): _description_

        Returns:
            _type_: _description_
        """
        min_distance = np.inf
        closest_needle_idx = None
        for (
            needle_idx,
            positions,
        ) in dp_to_select_from.items():
            if len(positions) == 0:
                continue
            min_distance_needle = min_dist_two_list(
                positions,
                reference_single_needle_dwell_positions,
            )
            if min_distance_needle < min_distance:
                min_distance = min_distance_needle
                closest_needle_idx = needle_idx
        if return_min_dist:
            return closest_needle_idx, min_distance
        return closest_needle_idx
    

    def _create_dwell_positions(self, digitization_points:dict, end_first:bool=True, step_size:float=2.5):
        """
        We can get all dwell positions, even 0second dwell positions in the given treatment (retrospectively)
        by lineraly inpterolating between the digitization points and distributing points evenly.
        """
        all_dwell_positions = {}
        piece_wise_lines = {}
        for channel_number, digi_points in digitization_points.items():

            if len(digi_points) <= 1:
                print(
                    f"There does not seem to be a complete digitization points set {channel_number}"
                )
                continue
            dwell_positions, piece_wise_line = self._create_dwell_positions_from_digi_pts(
                digi_points, end_first=end_first, step_size=step_size)
            all_dwell_positions[channel_number] = dwell_positions
            piece_wise_lines[channel_number] = piece_wise_line

        return all_dwell_positions, piece_wise_lines

    def _get_temporary_dwell_positions(self, digitization_points:dict=None):
        """
        We can get all dwell positions, even 0second dwell positions in the given treatment (retrospectively).
        If we computed them already, we just return the results/
        """
        if digitization_points is None:
            digitization_points = self.get_digitization_points()

        # We get any of potential dwell positions, with a small step size to be sure
        # this can help identify the correct corresponding non-0 dwell positions.
        temporary_dwell_positions, _ = self._create_dwell_positions(
            digitization_points, end_first=True, step_size=1.0)

        return temporary_dwell_positions
    

    def _identify_tip_side_based_on_relativepos(self, catheter_table_with_only_activated_dp:dict):
        """
        Going through the relative position of the treatment activated dwell positions to find 
        the tip side and making sure it is the same for every catheter. Normally in Oncentra you
        choose either the last or first digi point as the starting dwell positions so it should 
        always be the case.
        """
        # Identify the best catheter candidate to identify the tip side 
        # i.e. the one with the activated dwell position the closest to 
        # the treatment tip.
        min_dist = np.inf
        for catheter_info in catheter_table_with_only_activated_dp:
            channel_number = catheter_info["channel_number"]
            # We get the dwell position that is the closest to the tip
            for dwell_pos in catheter_info["dwells"]:
                if dwell_pos["relativePos"] < min_dist:
                    min_dist = dwell_pos["relativePos"]
                    dpos_closest_to_catheter_tip = dwell_pos
                    best_channel_number = channel_number
        
        # On Oncentra TPS, the digitizations points are set to be either from 
        # connector end or from tip end. It is the same side for every catheter.
        if (abs(distance(
            dpos_closest_to_catheter_tip["position"], 
            self.digitization_points["Channel_"+str(best_channel_number)][0]
            ) - min_dist) < abs(distance(
                dpos_closest_to_catheter_tip["position"], 
                self.digitization_points["Channel_"+str(best_channel_number)][-1]
                ) - min_dist)):
            all_catheter_idx_digipt_close_to_tips = 0
        else:
            all_catheter_idx_digipt_close_to_tips = -1

        return all_catheter_idx_digipt_close_to_tips == -1

    def _check_for_shift_in_tip(self, catheter_table_with_only_activated_dp:dict, end_first:bool):
        assert self.dwell_positions is not None
        temp_error_shift_tip = None
        error_stored = False
        for catheter_infos in catheter_table_with_only_activated_dp:
            channel_number = catheter_infos["channel_number"]
            our_temp_tip = self.dwell_positions["Channel_"+str(channel_number)][0]
            if end_first:
                digipts = self.digitization_points["Channel_"+str(channel_number)][::-1]
                assert np.allclose(np.array(our_temp_tip),np.array(digipts[0]))
                second_digipt = digipts[1]
            else:
                digipts = self.digitization_points["Channel_"+str(channel_number)]
                assert np.allclose(np.array(our_temp_tip), np.array(digipts[0]))
                second_digipt = digipts[1]
            correct_tip = self._check_tip(catheter_table_with_only_activated_dp, "Channel_"+str(channel_number), our_temp_tip, not error_stored)
            if correct_tip:
                continue
            else:  
                # Creating a new tip based on the relative pos info of the dicom rtplan.
                error_shift_tip = self.get_error_shift_tip(
                    catheter_table_with_only_activated_dp, "Channel_"+str(channel_number), our_temp_tip, second_digipt)
                if not error_stored:
                    print("***************************************************")
                    print("===================================================")
                    print("TIP DOES NOT CORRESPOND TO LAST DIGI POINT FOR PATIENT ", self.CT_folder)
                    print("===================================================")
                    print("***************************************************")
                    with open(os.path.join(Path(__file__).parents[0], "digitization/patient_wrong_digitization.txt"), "a") as myfile:
                        myfile.write(f"{self.CT_folder} has a tip which do not correspond to last digi point, shift: {np.round(error_shift_tip, 2)}mm. \n")
                    error_stored = True
                if temp_error_shift_tip is None:
                    temp_error_shift_tip = error_shift_tip
                else:
                    assert np.allclose(temp_error_shift_tip, error_shift_tip, atol=1e-4), (
                        """Error shift tip is not the same for all catheters for patient {}.
                        We detected {} and {} distance to real treatment tip.
                        """.format(self.CT_folder, temp_error_shift_tip, error_shift_tip)
                    )
        return temp_error_shift_tip                

    def get_dwell_positions(self, catheter_table_with_only_activated_dp:dict):
        """
        We can get all dwell positions, even 0second dwell positions in the given treatment (retrospectively).
        If we computed them already, we just return the results/
        """
        if self.digitization_points is None:
            _ = self.get_digitization_points()

        if self.dwell_positions is None:

            end_first = self._identify_tip_side_based_on_relativepos(catheter_table_with_only_activated_dp)
            self.last_digipt_is_first_dwellpos = end_first
            self.dwell_positions, self.piece_wise_lines = self._create_dwell_positions(
                self.digitization_points, end_first=end_first, step_size=self.step_size)
            
            # At this point we have created dwell positions from the digitization points. 
            # During treatment, does the digi point correspond exactly to the first dwell
            # position? Guidlines to place the tip can be either at the tip marker or at 
            # the end of the catheter core.
            # To know that, we check from the RelativePosition. 
            shift_error_for_tip = self._check_for_shift_in_tip(catheter_table_with_only_activated_dp, end_first)

            if not(shift_error_for_tip is None):
                self.offset = np.round(shift_error_for_tip, 2)
                self.custom_digitization_points = self.modify_digi_pts(
                    self.digitization_points, shift_error_for_tip, end_first, save=True)
                self.dwell_positions, self.piece_wise_lines = self._create_dwell_positions(
                    self.custom_digitization_points, end_first=end_first, step_size=self.step_size)
            else:
                self.offset = 0.0

            # Here we check that all the dwell positions we created are matching 
            # the ones we found in the DICOM treatment plan file i.e. the 
            # activated dwell positions.
            assert self._check_distance_to_activated_dwell_positions()

        return self.dwell_positions

    def _check_distance_to_activated_dwell_positions(self):
        """
        Computing the distance from any activated dwell positions to the closest
        created dwell position. 
        """

        distances_non0_to_created_dp = []
        for channel_number in self.non_zero_dwell_positions.keys():

            # Need to check for distance on the pw line since sometimes the digi points
            # are really close and make the distance between two points very different
            # from the distance between the two points on the pw line.
            for non0_dp in self.non_zero_dwell_positions[channel_number]:
                min_dist = self.min_dist_two_list_on_pw_line(
                    self.dwell_positions[channel_number], 
                    [non0_dp], 
                    channel_number)
                distances_non0_to_created_dp.append(min_dist)
        perfect_match = np.allclose(distances_non0_to_created_dp, 0, atol=1e-2)
        if not perfect_match:
            print("""The created dwell positions do nor match the non-0 dwell positions 
            from the treatment plan file for patient {}, with errors {}.
            """.format(self.CT_folder, distances_non0_to_created_dp))
        return perfect_match
        

    def _get_min_dist_from_dwell_positions(self, reference_points:dict):
        """
        DEPRECATED function: before we were constructing two sets of dwell positions
        and then selecting the best one depending on distance to existing dwell positions. 
        It is preferable to identify the tip side based on the RelativePosition of the 
        dwell position to the tip and only create one set of dwell positions.
        """
        if not (reference_points is None):
            raise ValueError("This function is deprecated. We now only create one set of dwell positions.")

        end_first_dwell_positions, end_first_piece_wise_lines = self._create_dwell_positions(
            self.digitization_points, end_first=True, step_size=self.step_size)
        end_last_dwell_positions, end_last_piece_wise_lines = self._create_dwell_positions(
            self.digitization_points, end_first=False, step_size=self.step_size)
        # Finding if the dwell positions were created from the tip end or from the connector end.
        # In Oncentra, if connector end is chosen, it is chosen for every catheter in the plan.
        # Same thing for tip end. If the person that digitizes wants to change that during the
        # digitization process, then it erases all previous digitization work.
        mins_from_last = []
        mins_from_first = []
        for channel_number in end_first_dwell_positions.keys():
            if len(reference_points[channel_number]) == 0:
                continue
            end_first_index = self.select_best_needle_from_list(
                end_first_dwell_positions, reference_points[channel_number])
            end_last_index = self.select_best_needle_from_list(
                end_last_dwell_positions, reference_points[channel_number])
            # Need to check for distance on the pw line since sometimes the digi points
            # are really close and make the distance between two points very different
            # from the distance between the two points on the pw line.
            end_first_min = self.min_dist_two_list_on_pw_line(
                end_first_dwell_positions[end_first_index], 
                reference_points[channel_number], 
                channel_number)
            end_last_min = self.min_dist_two_list_on_pw_line(
                end_last_dwell_positions[end_last_index], 
                reference_points[channel_number], 
                channel_number)
            mins_from_first.append(end_first_min)
            mins_from_last.append(end_last_min)
        return (mins_from_first, mins_from_last, end_first_dwell_positions, 
                end_last_dwell_positions, end_first_piece_wise_lines, 
                end_last_piece_wise_lines)
    
    def min_dist_two_list_on_pw_line(self, set1:List[List[float]], set2:List[List[float]], channel_number:str, return_points:bool=False):

        linear_interpolator = self.piece_wise_lines[channel_number]
        
        # Compute distance point by point
        min_dist = np.inf
        for point1 in set1:
            for point2 in set2:
                dist = linear_interpolator.distance_on_pw_line(point1, point2)
                if dist < min_dist:
                    min_dist = dist
                    points = [point1, point2]
        if return_points:
            return min_dist, points
        return min_dist

    def get_dwell_positions_list(self):
        assert self.dwell_positions is not None
        dwell_positions = copy.deepcopy(self.dwell_positions)
        dwell_positions_list = []
        for channel_number, dp in dwell_positions.items():
            dwell_positions_list.extend(dp)
        return dwell_positions_list
    
    def get_non_zero_dwell_positions_list(self):
        dwell_positions = self.get_non_zero_dwell_positions()
        dwell_positions_list = []
        for channel_number, dp in dwell_positions.items():
            dwell_positions_list.extend(dp)
        return dwell_positions_list
    
    def _find_step_dwell_pos_manually(self):
        """
        DEPRECATED: now we just get the stepsize from DICOM file. see self.get_step_size() function.
        
        Finding the step used to place dwell positions during the treatment from the distance
        between dwell positions.
        """
        activated_dwell_positions = self.get_non_zero_dwell_positions()
        needle_dwell_pos_step = []
        for needle_idx, needle_dwp in activated_dwell_positions.items():
            min_dist = 1e10
            # Removing first and last dwell positions since they can be not
            # exactly at the correct step size.
            for i in range(len(needle_dwp) - 1):
                dist = distance(needle_dwp[i], needle_dwp[i + 1])
                if dist <= min_dist:
                    min_dist = dist
                    needle_dwell_pos_step.append(round(dist, 1)) # min_dist, 1))
        
        # Using the most common step size here and not just the first one from the list 
        # since sometimes one dwel position is skipped during the treatment.
        # Most common step size should be the correct one.
        most_common_step = Counter(needle_dwell_pos_step).most_common(1)[0][0]
        # Error could happen since we take distance in straigth line but could be a different path.
        # Error should not be big though, so we fix 5% max.
        # Compute the closest multiple of the most common step for each measured step
        needle_dwell_pos_step = np.array(needle_dwell_pos_step)
        multiples = np.round(needle_dwell_pos_step / most_common_step)
        closest_multiple_of_most_common_step = multiples * most_common_step
        percent_error = np.abs(needle_dwell_pos_step - closest_multiple_of_most_common_step) / most_common_step
        assert np.all(
            percent_error < 0.05
        ), f"The dwell positions do not have the same step size accross catheters! {np.array(needle_dwell_pos_step)}"
        return most_common_step

    def get_step_dwell_pos(self):
        if self.step_dwell_pos is None:
            self.step_dwell_pos = self._find_step_dwell_pos_manually()
        return self.step_dwell_pos

    @staticmethod
    def get_catheter_from_table(catheter_table:dict, catheter_key:str):
        catheter_key_number = catheter_key.split("_")[1]
        return catheter_table[np.where(
            [cat["channel_number"]== catheter_key_number for cat in catheter_table ])[0][0]]

    def _check_tip(self, catheter_table_with_only_activated_dp:dict, catheter_key:str, tip_pt:List[float], verbose:bool=False):
        """
        Getting the closest point to the tip from the dwell positions of the catheter
        and checking its distance to our computed tip.
        """
        catheter_info = self.get_catheter_from_table(catheter_table_with_only_activated_dp, catheter_key)
        pw_curve = self.piece_wise_lines[catheter_key]
        dwell_pos_info = catheter_info["dwells"][0]
        assert dwell_pos_info['index'] == 0
        assert dwell_pos_info['relativePos'] == min([x["relativePos"] for x in catheter_info["dwells"]])
        
        dist_between_first_act_dwell_and_our_tip = pw_curve.distance_on_pw_line(dwell_pos_info["position"], tip_pt)

        if np.isclose(dist_between_first_act_dwell_and_our_tip, dwell_pos_info["relativePos"], atol=1e-4):
            pass_check = True
        else:
            if verbose:
                print(
                    f"""Distance between our tip (from digi pts) and first activated dwell position 
                    is not the same as the relative position of the first activated dwell position 
                    to the treatment tip. 
                    {dist_between_first_act_dwell_and_our_tip} != {dwell_pos_info['relativePos']}"""
                )
            pass_check = False

        return pass_check
    
    @staticmethod
    def compare_direction(a, b):
        a = np.array(a)
        b = np.array(b)
        
        # Normalize the vectors
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        
        # Compute dot product
        dot_product = np.dot(a_norm, b_norm)
        
        return dot_product  # Closer to 1: same direction, -1: opposite, 0: perpendicular


    def get_error_shift_tip(self, catheter_table_with_only_activated_dp:dict, catheter_key:str, tip_pt:List[float], second_digipt:List[float]):
        """
        Computing the error that we have between our tip (from digi point)
        and the treatment tip. We don't know the treatment tip coordinate 
        but we know the relative position of the activate dwell positions 
        to the tip. 
        """

        ### Finding which activated dwell position the tip_pt correspond to to get the 
        # relativepos from DICOM.
        catheter_info = self.get_catheter_from_table(catheter_table_with_only_activated_dp, catheter_key)

        activated_dwell_positions = [x["position"] for x in catheter_info["dwells"]]
        distances = cdist([tip_pt], activated_dwell_positions, metric="euclidean")
        min_distance = np.min(distances)
        index_of_dwellpos_corresponding_to_current_tip = np.where(distances == min_distance)[1][0]
        relative_pos = [x["relativePos"] for x in catheter_info["dwells"]]
        # This dwell positions should be at a relativepos 0 of the tip but instead is
        # at a certain relativepos fro the tip that was used in the treatment.
        # We will manually create a new digitization point that is at the 
        # correct relativepos.
        relatpos_activated_dp_closest_to_tip = relative_pos[index_of_dwellpos_corresponding_to_current_tip]
        
        # Is the activated dwell position before or after the last digitizatoin point?
        # It will change the error shift. I need the second digitization point to know.
        pw_line = self.piece_wise_lines[catheter_key]
        # Distance from activated dwell pos to the first digi point in DICOM
        da1 = pw_line.distance_on_pw_line(
            activated_dwell_positions[index_of_dwellpos_corresponding_to_current_tip], 
            tip_pt) 
        # Distance from activated dwell pos to the second digi point in DICOM
        da2 = pw_line.distance_on_pw_line(
            activated_dwell_positions[index_of_dwellpos_corresponding_to_current_tip], 
            second_digipt)
        # Distance between the two first digitization points
        d12 = pw_line.distance_on_pw_line(tip_pt, second_digipt)

        # Vector activated dwell posiitons to our tip
        vector_da1 = np.array(tip_pt) - np.array(activated_dwell_positions[index_of_dwellpos_corresponding_to_current_tip])
        # Vector second digi point to our tip
        vector_d12 = np.array(tip_pt) - np.array(second_digipt)

        if np.isclose(da1, 0.0, atol=1e-5):
            return relatpos_activated_dp_closest_to_tip
        
        cos_theta = self.compare_direction(vector_da1, vector_d12)
        if cos_theta > 0:
            # The two vector are in th same directions: i.e. The activated dwell position 
            # is before the last digitization point.
            # The error shift is the difference between the relative position of the activated 
            # dwell position and the treatment tip and the distance between the activated 
            # dwell position and the last digitization point (our current tip).
            error_in_our_tip = relatpos_activated_dp_closest_to_tip - da1
        else:
            assert da1 + da2 > d12, (
                f""" da1: {da1}, da2: {da2}, d12: {d12} 
                relatpos_activated_dp_closest_to_tip: {relatpos_activated_dp_closest_to_tip}
                The activated dwell position is after the last digitization point.
                The distance between the activated dwell position and the first digitization point
                should be greater than the distance between the activated dp and the first and 
                second digitization points."""
            )
            # The two vectors are in opposite directions: i.e. The activated dwell position
            # is after the last digitization point.
            # The error shift is the difference between the relative position of the activated 
            # dwell position and the treatment tip and the distance between the activated 
            # dwell position and the first digitization point (our current tip).
            error_in_our_tip = relatpos_activated_dp_closest_to_tip + da1
        
        return error_in_our_tip

    def modify_digi_pts(
        self, ref_digi_pts:dict, extrapolation_length:float, end_first:bool, save:bool=False):
        """
        Correcting the tip position by extrapolation from the two last digitization points.
        """
        modified_digit_pts:dict = copy.deepcopy(ref_digi_pts)
        if "Needle key matching" in modified_digit_pts.keys():
            modified_digit_pts.pop("Needle key matching")
        for catheter_key in modified_digit_pts.keys():
            digi_pts:List[List[float]] = ref_digi_pts[catheter_key]
            if end_first:
                digi_pts = digi_pts[::-1]
            digi_closest_to_tip = digi_pts[0]
            second_digi_closest_to_tip = digi_pts[1]
            new_tip = extrapolate_point(second_digi_closest_to_tip, digi_closest_to_tip, extrapolation_length, reverse=False)
            if end_first:
                if extrapolation_length < 0:
                    # We replace the digitization point
                    modified_digit_pts[catheter_key][-1]=new_tip.tolist()
                else:
                    # We add a new digitization point
                    modified_digit_pts[catheter_key].append(new_tip.tolist())
            else:
                if extrapolation_length < 0:
                    # We replace the digitization point
                    modified_digit_pts[catheter_key][0] = new_tip.tolist()
                else:
                    # We add a new digitization point
                    modified_digit_pts[catheter_key].insert(0, new_tip.tolist())
            if save:
                os.makedirs(os.path.join(self.CT_folder, "correct_tip"), exist_ok=True)
                create_slicer_markup_points(
                    os.path.join(self.CT_folder, "correct_tip", f"new_tip_{catheter_key}.mrk.json"), 
                    [new_tip.tolist()], color=[0.,1.,0.])
                create_slicer_markup_points(
                    os.path.join(self.CT_folder, "correct_tip", f"old_tip_{catheter_key}.mrk.json"), 
                    [digi_pts[0]], color=[0.,1.,1.])

        return modified_digit_pts

    def _get_tips(self):
        """
        Return the first dwell positions which is the tip in the treatment plan
        """
        tips_coords = {}

        assert self.dwell_positions is not None
        for catheter_key, dwell_pos in self.dwell_positions.items():
            tips_coords[catheter_key] = dwell_pos[0]

        return tips_coords    

    def get_tips_coords(self):
        if self.tip_coord is None:
            self.tip_coord = self._get_tips()
        return self.tip_coord

    def get_consistent_tip_at_end_of_tip_marker(self, return_list:bool=True):
        """
        Returns the treatment tip if the treatment tip is at the end of the tip marker.
        Compute the tip that is at the end of the tip marker if the treatment tip is
        at the beginning or middle of the tip marker.
        We identify the location of the placed tip based on the channel length, 
        which reflects different guidelines for catheter digitization.

        We are dealing only with comfort catheters here.
        * For MicroSelectron treatment:
        -   When channel legnth was 1240mm, the tip was placed at the very end of the tip marker.
        -   When channel length was 1238mm, the tip was placed at the beginning of the tip marker.
            Originally the idea was to place at the middle and offset except that the TPS cannot
            offset 0.5mm so there is a persistent intrinsic 1mm error in the end...
        -   Channel length is exceptionnally 1228mm for one catheter of patient 208782 since the 
            catheter tip was not in the FOV of the scan so channel length was adapted.

        * For Flexitron source:
        -   WHen the channel length was 1236mm,  the tip was placed at the very end of the tip marker. 
        """
        if self.custom_digitization_points is None:
            consistent_digi_pts = copy.deepcopy(self.digitization_points)
        else:
            consistent_digi_pts = copy.deepcopy(self.custom_digitization_points)
        if "Needle key matching" in consistent_digi_pts.keys():
            consistent_digi_pts.pop("Needle key matching")
        if self.channel_length == 1240.0:
            tip_coords = self.get_tips_coords()
        elif self.channel_length == 1238.0:
            tip_coords = self.extrapolate_tip(3.0)
            if self.last_digipt_is_first_dwellpos:
                for catheter_key in consistent_digi_pts.keys():
                    consistent_digi_pts[catheter_key][-1] = tip_coords[catheter_key]
            else:
                for catheter_key in consistent_digi_pts.keys():
                    consistent_digi_pts[catheter_key][0] = tip_coords[catheter_key]
        elif self.channel_length == 1236.0:
            tip_coords = self.get_tips_coords()
        elif self.channel_length == 1228.0:
            tip_coords = self.extrapolate_tip(3.0)
            if self.last_digipt_is_first_dwellpos:
                for catheter_key in consistent_digi_pts.keys():
                    consistent_digi_pts[catheter_key][-1] = tip_coords[catheter_key]
            else:
                for catheter_key in consistent_digi_pts.keys():
                    consistent_digi_pts[catheter_key][0] = tip_coords[catheter_key]
        else:
            raise ValueError(
                """The channel length has never been seen before, 
                please have a look at the patient to determine the tip point position."""
                )
            
        if return_list:
            for catheter_key in tip_coords.keys():
                tip_coords[catheter_key] = [tip_coords[catheter_key]]

        return tip_coords, consistent_digi_pts

    def extrapolate_tip(self, extrapolate_length:float):
        """
        Modifying the tips by extrapolation from the two last digitization points.
        Retuning only the tip without modifying any digitization points.
        """
        tip_coords = {}
        if self.custom_digitization_points is None:
            ref_digi_pts = self.digitization_points
        else:
            ref_digi_pts = self.custom_digitization_points
        temp_digi_pts = self.modify_digi_pts(ref_digi_pts, extrapolate_length, self.last_digipt_is_first_dwellpos)
        for catheter_key in temp_digi_pts.keys():
            if self.last_digipt_is_first_dwellpos:
                tip_coords[catheter_key] = temp_digi_pts[catheter_key][-1]
            else:
                tip_coords[catheter_key] = temp_digi_pts[catheter_key][0]
        return tip_coords

    def _create_dwell_positions_from_digi_pts(self, digitization_points:dict, end_first:bool=True, step_size:float=2.5):
        """
        We can get all dwell positions, even 0second dwell positions in the given treatment (retrospectively)
        by lineraly inpterolating between the digitization points and distributing points evenly.
        """

        dwell_positions = []
        if end_first:
            digitization_points = digitization_points[::-1]
        dwell_positions.append(digitization_points[0])

        linear_interp = PiecewiseLinear3D(digitization_points)

        previous_pt = digitization_points[0]
        t_used = 0.0
        while t_used < 0.9999:
            point, t, distance_prev_current = linear_interp.step_in_pw_line(
                previous_pt, step_size, bound_min=t_used
            )
            # distance_prev_current = distance(previous_pt, point)
            if distance_prev_current < 0.99 * step_size:
                # Not creating dwell position for the point that hits the 1 bound
                # in the step_in_pw_line function if step size is not respected.
                # ie. Not adding the last dwell position as the last digi point
                # if distance between the two last dwell would be < step size.
                # Giving a 1% error on the step size.
                break
            dwell_positions.append(point)
            previous_pt = point
            t_used = t
        return dwell_positions, linear_interp
    
    def _create_curve(self, digitization_points:List[List[float]], step_size:float = 0.1):
        """
        Creating a curve (many points) from the digitization points.
        """

        linear_interp = PiecewiseLinear3D(digitization_points)
        pts_along_curve = []
        previous_pt = digitization_points[0]
        t_used = 0.0
        while t_used < 0.9999:
            point, t, distance_prev_current = linear_interp.step_in_pw_line(
                previous_pt, step_size, bound_min=t_used
            )
            # distance_prev_current = distance(previous_pt, point)
            if distance_prev_current < 0.99 * step_size:
                # Not creating dwell position for the point that hits the 1 bound
                # in the step_in_pw_line function if step size is not respected.
                # ie. Not adding the last dwell position as the last digi point
                # if distance between the two last dwell would be < step size.
                # Giving a 1% error on the step size.
                break
            pts_along_curve.append(point)
            previous_pt = point
            t_used = t
        return pts_along_curve
    
    def create_curves(self, step_size:float = 0.1):
        """
        Simply creating dwell posiitons with a short step size to plot the curve.
        """
        all_curve_pts = {}
        for channel_number, digitization_points in self.digitization_points.items():

            if len(digitization_points) <= 1:
                print(
                    f"There does not seem to be a complete digitization points set {channel_number}"
                )
                continue
            dwell_positions = self._create_curve(digitization_points, step_size=step_size)
            all_curve_pts[channel_number] = dwell_positions
        return all_curve_pts
    
    def _check_integrity_dwell_positions(self, verbose=False):
        """
        We check if the dwell positions we computed from the digitization points
        are close to the dwell positions that were used to treat the patient.
        """
        if self.dwell_positions is None:
            _ = self.get_dwell_positions()
        if self.non_zero_dwell_positions is None:
            _ = self.get_non_zero_dwell_positions()

        for channel_number, digitization_points in self.digitization_points.items():
            non_zero_dp_channel_number = self.select_best_needle_from_list(
                self.non_zero_dwell_positions, digitization_points
            )

            if len(digitization_points) <= 1:
                # There is a digitization point that is lost somewhere.
                print(
                    f"There does not seem to be a complete digitization points set for  {channel_number}"
                )
                continue
            # Checking distance between each non-zero dwell position and the closest dwell position
            # from the dwell positions computed manually from the digitization points.
            distances = []
            for i in range(
                len(self.non_zero_dwell_positions[non_zero_dp_channel_number]) - 1
            ):

                min_dist, min_pts = min_dist_two_list(
                    [self.non_zero_dwell_positions[non_zero_dp_channel_number][i]],
                    self.dwell_positions[channel_number],
                    return_points=True,
                )
                # min_dist_two_list returns a list of pairs of points that are closest (possibly more than one)
                min_pts = min_pts[0]
                distances.append(min_dist)

            if verbose:
                print(
                    f"Mean distances for needle {non_zero_dp_channel_number}",
                    np.mean(distances),
                )
            # We have to check a distance and cannot check for coordinates equality because of
            # floating precision of our interpolation.
            # Points should be within 5% of the step size
            assert np.all(
                np.array(distances) < 0.05 * self.step_size
            ), f"The dwell positions you created from the digtization points do not match the dwell positions used to treat the patient! Error: {distances}"

    def get_catheter_table(self):
        """
        Getting the catheter table from the class attribute.
        """
        return self.catheter_table
    
    def get_nb_catheters_per_row(self, crop_around_catheters:bool=True, labelling_style:str="zigzag") ->List[int]:
        """
        Getting the number of catheters inserted per row.
        First: creates a grid of insertion points for the catheters.
        Then: Determining the number of catheter per rows based on the labelling style.

        * zigzag style:
        At the Jewish General Hospital, the breast brachy team labels the catheters
        in a zigzag style. The top left catheter is Channel_1, then Channel_2 on
        the right and then when a row is finished they start back from the left. 
        I call this labelling style "zigzag". Here is a schematic of the
        labelling style, "x" is a catheter insertion point, in general the JGH follows
        the Paris system for catheter insertion, which tries to make small triangles:


        Ch1 -----> Ch2 -----> Ch3 -----> Ch4 -----> Ch5 -----> Ch6

         x          x          x          x          x          x
        
         
              x          x          x          x          x          

             Ch7 -----> Ch8 -----> Ch9 -----> Ch10 ----> Ch11   
        
        For this labelling style go from Channel_i to Channel_i+1, when the direction
        is positive in x it is the same row, when it is negative in x (between channel 6 
        and 7 in the above figure) a new row starts.


        * snake style:
        I know from a breast brachytherapy workshop at Curitherapies 2025 that other teams 
        at different institutions can label differently. I have heard of this second style
        which I called "snake" style. In this style, the catheters are labelled from top
        left as well, but when a row is done, we do to the catheter directly below for 
        future channel labelling. Here is a schematic of the "snake" labelling style:



        Ch1 -----> Ch2 -----> Ch3 -----> Ch4 -----> Ch5 -----> Ch6

         x          x          x          x          x          x
        
         
              x          x          x          x          x          

             Ch11 <---- Ch10 <---- Ch9 <----- Ch8 <----- Ch7   

        """

        assert labelling_style in ["zigzag", "snake"], (
            f"The labelling style {labelling_style} is not supported. "
            "Please use 'zigzag' or 'snake'."
        )
        from ai_assisted_brachy.preprocessing.dicom_to_sitk import convert_dicom_images_folder_to_nii
        from ai_assisted_brachy.catheter.contour.creator import CatheterContourCreator
        from ai_assisted_brachy.catheter.digitization.labelling import InsertionGridViewer, get_angle_between_two_vectors

        ct_volume_path = os.path.join(self.CT_folder, "ct.nrrd")
        print(f"Saving CT volume to {ct_volume_path}")
        convert_dicom_images_folder_to_nii(
            (self.CT_folder, ct_volume_path)
        )

        if crop_around_catheters:
            creator = CatheterContourCreator(
                patient_path=None,
                catheter_setup=self,
                patient_volume_path=ct_volume_path,
                processed_folder=None,
                dilation=0, 
                add_tip_marker_contour=True,
                extend_catheters_to_body=False,
                body_contour_mask=None,
                catheter_diameter=1.0,
            )
            print("Creating catheter contour...")
            catheter_contour = creator.create_catheter_contour(
                multiprocess=True,
                use_1_mm_isotropic_spacing=False 
            )
        else:
            catheter_contour = None
        
        grid_viewer = InsertionGridViewer(
            ct_volume_path=ct_volume_path,
            catheters_contour_path=catheter_contour,
            save_details_folder=os.path.dirname(ct_volume_path), 
            dwell_positions=self.dwell_positions,
            crop_around_catheters=crop_around_catheters,
            margin_around_catheters_mm=10.
            )
        viewer_states = grid_viewer.get_insertion_grid_as_rows(
            save_files=True
            )

        if len(viewer_states) == 1:
            insertion_grid_points = viewer_states[0]["grid"]
        else:
            assert len(viewer_states) == 2, (
                f"There should be only one or two viewer states, but there are {len(viewer_states)}."
            )
            grid_pts_view1 = viewer_states[0]["grid"]
            grid_pts_view2 = viewer_states[1]["grid"]
            points_view1 = [grid_pts_view1[f"Channel_{k}"][0] for k in range(1, len(grid_pts_view1.keys()) + 1)]
            points_view2 = [grid_pts_view2[f"Channel_{k}"][0] for k in range(1, len(grid_pts_view2.keys()) + 1)]
            x_axis = np.array([1, 0, 0])
            metrics1 = []
            for pt_idx in range(len(points_view1) - 1):
                direction_between_two_insertions = np.array(points_view1[pt_idx + 1][:3]) - np.array(points_view1[pt_idx][:3])
                if direction_between_two_insertions[0] < 0:
                    direction_between_two_insertions *= -1
                a = get_angle_between_two_vectors(direction_between_two_insertions, x_axis)
                metrics1.append(a)

            metrics2 = []
            for pt_idx in range(len(points_view2) - 1):
                direction_between_two_insertions = np.array(points_view2[pt_idx + 1][:3]) - np.array(points_view2[pt_idx][:3])
                if direction_between_two_insertions[0] < 0:
                    direction_between_two_insertions *= -1
                a = get_angle_between_two_vectors(direction_between_two_insertions, x_axis)
                metrics2.append(a)

            if np.mean(metrics1) < np.mean(metrics2):
                insertion_grid_points = grid_pts_view1
                print("Using the first view with metric", np.mean(metrics1))
            else:
                insertion_grid_points = grid_pts_view2
                print("Using the second view with metric", np.mean(metrics2))

        for k in insertion_grid_points.keys():
            assert "Channel_" in k, (
                f"The key {k} in the insertion grid points does not contain 'Channel_'"
            )

        if labelling_style == "zigzag":
            # Depending on which breast side we are, the channels can be labelled from 
            # left to right or from right to left. So we need to determine the direction.
            determine_direction = True 
            if determine_direction:
                ## Were the channels labelled from left tot rigth (most often the case) or 
                # from right to left?
                xs = [insertion_grid_points[f"Channel_{k}"][0][0] for k in range(1, len(insertion_grid_points.keys()) + 1)]
                diff_xs = np.diff(xs)
                total_positive_directions = np.sum(diff_xs > 0)
                total_negative_directions = np.sum(diff_xs < 0)
                if total_positive_directions > total_negative_directions:
                    direction_checker = lambda x: x > 0
                else:
                    direction_checker = lambda x: x < 0
            else:
                # If we do not determine the direction, we assume that the channels are labelled
                # from left to right.
                direction_checker = lambda x: x > 0

            nb_catheters_per_row = []
            count_ng_catheters_row_i = 1
            for i in range(2, len(self.dwell_positions.keys()) + 1):
                direction_between_two_insertions = np.array(
                    insertion_grid_points[f"Channel_{i}"][0][:3]
                ) - np.array(insertion_grid_points[f"Channel_{i-1}"][0][:3])
                if direction_checker(direction_between_two_insertions[0]):
                    count_ng_catheters_row_i += 1
                else:
                    nb_catheters_per_row.append(count_ng_catheters_row_i)
                    count_ng_catheters_row_i = 1
            nb_catheters_per_row.append(count_ng_catheters_row_i)
            
        else:
            raise NotImplementedError(
                f"The labelling style {labelling_style} is not implemented yet." \
                "What you see below is an attempt to implement it, but not tested" \
                " as I do not have any patient data with this labelling style."
            )
            nb_catheters_per_row = []
            count_ng_catheters_row_i = 1
            for i in range(2, len(self.dwell_positions.keys())):
                direction_between_two_insertions1 = np.array(
                    insertion_grid_points[f"Channel_{i}"][0][:3]
                ) - np.array(insertion_grid_points[f"Channel_{i-1}"][0][:3])
                direction_between_two_insertions2 = np.array(
                    insertion_grid_points[f"Channel_{i+1}"][0][:3]
                ) - np.array(insertion_grid_points[f"Channel_{i}"][0][:3])
                d1_x_positive = direction_between_two_insertions1[0] > 0
                d2_x_positive = direction_between_two_insertions2[0] > 0
                if d1_x_positive == d2_x_positive:
                    # The two directions are the same, we are in the same row.
                    count_ng_catheters_row_i += 1
                elif d1_x_positive and not d2_x_positive:
                    # The first direction is positive and the second is negative,
                    # we are at the end of the row.
                    nb_catheters_per_row.append(count_ng_catheters_row_i)
                    count_ng_catheters_row_i = 1
                "......."
                y_going_down = direction_between_two_insertions1[1] < 0
            nb_catheters_per_row.append(count_ng_catheters_row_i)
        print(f"Number of catheters per row: {nb_catheters_per_row}")
        assert sum(nb_catheters_per_row) == len(self.dwell_positions.keys()), (
            f"The number of catheters per row {nb_catheters_per_row} does not match"
            f" the number of catheters {len(self.dwell_positions.keys())}."
        )
        return nb_catheters_per_row
  
    def remove_inside_mask(self, mask:sitk.Image, margin_mm: float = 0.0) -> None:
        r"""
        ### Purpose:
        - To filter out the dwell positions that are inside a given mask.

        ### Inputs:
        - self := the CatheterTable object.
        - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

        ### Outputs:
        - None
        """
        assert isinstance(mask, sitk.Image), "The mask should be a SimpleITK image."
        warnings.warn("You are altering a retrospective plan by removing dwell positions. Please be careful!", UserWarning)
        if margin_mm > 0.0:
            mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)

        self.keep_dwell_positions(mask, condition="outside")
        
    def remove_outside_mask(self, mask:sitk.Image, margin_mm: float = 0.0) -> None:
        r"""
        ### Purpose:
        - To filter out the dwell positions that are outside a given mask.

        ### Inputs:
        - self := the CatheterTable object.
        - mask:sitk.Image := the mask to filter the dwell positions.

        ### Outputs:
        - None
        """
        assert isinstance(mask, sitk.Image), "The mask should be a SimpleITK image."
        warnings.warn("You are altering a retrospective plan by removing dwell positions. Please be careful!", UserWarning)

        if margin_mm > 0.0:
            mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)
        self.keep_dwell_positions(mask, condition="inside")

    ### TODO: Think about merging the catheter setup into brachyutils. Here we have to 
    # update multiple objects. In brachyutils CatheterTable objects are better organized.
    # We would only need to update one object. 
    def keep_dwell_positions(self, mask:sitk.Image, condition:str="inside") -> None:
        r"""
        ### Purpose:
        - To keep the dwell positions that are inside or outside a given mask.
        ### Inputs:
        - self := the CatheterTable object.
        - mask:sitk.Image := the mask to filter the dwell positions.
        - condition:str := "inside" or "outside" to keep the dwell positions inside or outside the mask.
        ### Outputs:
        - None
        """
        assert self.dwell_positions is not None, "Please compute the dwell positions first."
        assert self.non_zero_dwell_positions is not None
        assert self.catheter_table is not None

        if condition == "inside":
            condition_checker = lambda x: x > 0
        elif condition == "outside":
            condition_checker = lambda x: x == 0
        else:
            raise ValueError(f"Condition {condition} not recognized. Please use 'inside' or 'outside'.")
        
        ### Update the dwell positions
        filtered_dp = {}
        for k, dp in self.dwell_positions.items():
            filtered_dp[k] = []
            for pt in dp:
                pt_index = mask.TransformPhysicalPointToIndex(pt)
                if condition_checker(mask.GetPixel(pt_index)):
                    filtered_dp[k].append(pt)
        self.dwell_positions = filtered_dp

        ### Update the non-zero dwell positions
        filtered_non_zero_dp = {}
        for k, dp in self.non_zero_dwell_positions.items():
            filtered_non_zero_dp[k] = []
            for pt in dp:
                pt_index = mask.TransformPhysicalPointToIndex(pt)
                if condition_checker(mask.GetPixel(pt_index)):
                    filtered_non_zero_dp[k].append(pt)
        self.non_zero_dwell_positions = filtered_non_zero_dp

        ### Update the catheter table
        older_total_treatment_time = sum([ch["channel_total_time"] for ch in self.catheter_table])
        filtered_catheter_table = []
        new_total_treatment_time = 0.0
        for idx, channel in enumerate(self.catheter_table):
            filtered_catheter_table.append(copy.deepcopy(channel))
            filtered_catheter_table[idx]["dwells"] = []
            older_ch_treatment_time = self.catheter_table[idx]["channel_total_time"]
            channel_treatment_time = 0.0
            for dwell in self.catheter_table[idx]["dwells"]:
                pt = dwell["position"]
                pt = [float(x) for x in pt]
                pt_index = mask.TransformPhysicalPointToIndex(pt)
                if condition_checker(mask.GetPixel(pt_index)):
                    filtered_catheter_table[idx]["dwells"].append(dwell)
                    channel_treatment_time += dwell["time"]
            ### Updating the channel treatment time accordingly
            filtered_catheter_table[idx]["channel_total_time"] = channel_treatment_time
            new_total_treatment_time += channel_treatment_time

        # Updating weight
        for idx, channel in enumerate(filtered_catheter_table):
            for dwell in filtered_catheter_table[idx]["dwells"]:
                if new_total_treatment_time > 0:
                    # Original weight is dwell time over total treatment time
                    dwell["weight"] = dwell["weight"] * older_total_treatment_time / new_total_treatment_time
                    assert np.isclose(dwell["weight"], dwell["time"] / new_total_treatment_time)
                else:
                    dwell["weight"] = 0.0
        self.catheter_table = filtered_catheter_table

def dilate_mask_in_mm(mask: sitk.Image, distance_mm: float, voxel_based:bool=False) -> sitk.Image:
    """
    Dilate a binary mask by a specified distance in mm.
    
    Parameters:
        mask (sitk.Image): Binary mask image (1 = structure, 0 = background).
        distance_mm (float): Dilation distance in millimeters.
    
    Returns:
        sitk.Image: Dilated mask.
    """
    if not mask.GetPixelID() == sitk.sitkUInt8:
        mask = sitk.Cast(mask, sitk.sitkUInt8)
    if voxel_based:
        # Get voxel spacing (physical size per voxel)
        spacing = mask.GetSpacing()  # tuple (sx, sy, sz)
        
        # Compute radius in voxels for each axis
        radius_voxels = [int(round(distance_mm / s)) for s in spacing]
        
        # Use binary morphological dilation with anisotropic radius
        dilated = sitk.BinaryDilate(mask, radius_voxels)
        return dilated
    else:
        # Compute signed distance map (inside negative, outside positive)
        distance_map = sitk.SignedMaurerDistanceMap(mask, squaredDistance=False, useImageSpacing=True)
        
        # Threshold: everything within 'distance_mm' of the mask
        dilated = distance_map < distance_mm
        return sitk.Cast(dilated, sitk.sitkUInt8)


def get_rotation_from_position(idx, control_points):
    r"""
    Purpose:
        - To get the rotation of the dwell point from the position of the dwell point.
    Inputs:
        - idx:int := the index of the dwell point.
        - control_point_dcm:pydicom.dataset.Dataset := the control point object.
    Outputs:
        - np.array := the rotation of the dwell point in each axis.
    """
    # TODO: Merge this dicom utils script with my catheter setup class.
    # We need all dwell positions, not only the non 0s ones to be able to 
    # compute correct angles when they are not provided by the DICOM.
    if len(control_points) == 2:
        return angle_betwen_2_points(
            np.array(control_points[1]["position"], dtype=float),
            np.array(control_points[0]["position"], dtype=float),
        )
    
    if idx == 0:
        return get_rotation_from_position(idx+1, control_points)
    elif idx == len(control_points) - 1:
        return get_rotation_from_position(idx-1, control_points)
    else:
        return angle_betwen_2_points(
            np.array(control_points[idx-1]["position"], dtype=float),
            np.array(
                control_points[idx+1]["position"],
                dtype=float,
            ),
        )


def angle_betwen_2_points(a, b):
    r"""
    Purpose:
        - To calculate the angle between two points.
    Inputs:
        - a:np.array := the first point.
        - b:np.array := the second point.      
    Outputs:
        - np.array := the angle between the two points in each axis.
    """
    vec = a - b
    normal = np.sqrt(np.sum(vec ** 2))
    return vec / normal


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import glob
    import tqdm
    ct_folder = "/home/sebquet/EngerLab/Data/export_seb/patients/1325562" #898630" #10423" #182421" 167205
    # ct_folder = "/home/sebquet/EngerLab/Data/original_all_Breast_Patients/871204_Anon"
    all_ct_folder = glob.glob("/home/sebquet/EngerLab/Data/export_seb/patients/*")

    before_tip_marker = ["6515", "10423", "79206", "91249"]
    end_tip_marker = ["36177", "45474", "52193", "66496", "74072", "78110"]
    before_tip_marker = ["6515", "10423", "79206", "91249", "105134", "105416", "106984", "172472", "175042", "178081", "205356"]
    end_tip_marker = ["36177", "45474", "52193", "66496", "74072", "78110", "119127", "136174", "167205", "182421", "203576"]
               
    dates_before = []
    dates_after = []
    for ct_folder in tqdm.tqdm(all_ct_folder, total=len(all_ct_folder), desc="Checking tips..."):
        
        # ct_files = glob.glob(os.path.join(ct_folder, "CT*.dcm"))
        # d = dicom.dcmread(ct_files[0])
        # rp = dicom.dcmread(glob.glob(os.path.join(ct_folder, "RP*.dcm"))[0])
        # year = int(d.StudyDate[:4])
        # if os.path.basename(ct_folder) in ["167205","182421","1708247","1497730","1325562","74072","52193","208782","1684875","434208","649246"]:
        #     continue
        # if os.path.basename(ct_folder) in before_tip_marker:
        #     dates_before.append(year)
        #     print("BEFORE TIP: ", rp.ApplicationSetupSequence[0].ChannelSequence[0].ChannelLength)
        # if os.path.basename(ct_folder) in end_tip_marker:
        #     dates_after.append(year)
        #     print("AFTER TIP :", rp.ApplicationSetupSequence[0].ChannelSequence[0].ChannelLength)
        # else:
        #     continue


        print("PATIENT ", ct_folder)
        if not "91249" in ct_folder:
            continue
        patient_plan = CatheterSetUp(ct_folder) 
        patient_plan.save_digitization_points(os.path.join(ct_folder, "experiment", "digi_pts.mrk.json"), verbose=False)
        tip_end_of_tip_marker, consistent_digi_pts = patient_plan.get_consistent_tip_at_end_of_tip_marker()
        create_marker_pts_from_catheter_dict(os.path.join(ct_folder, "processed", "tip_end_of_tip_marker.mrk.json"), tip_end_of_tip_marker)
        create_marker_pts_from_catheter_dict(os.path.join(ct_folder, "processed", "consistent_digi_pts.mrk.json"), consistent_digi_pts, color=[1/5,1/5,1/5])


        # patient_plan.get_tips_coords()
    plt.hist(dates_before, label="Before tip marker")
    plt.hist(dates_after, label="End tip marker")
    plt.legend()
    plt.show()
    exit()
    patient_plan.create_catheter_table()
    exit(0)
    if "Anon" in os.path.basename(ct_folder):
        non_zero_dwell_positions = patient_plan._get_standalone_non_zero_dwell_positions_list()
        print("there are ", len(non_zero_dwell_positions)  , "non zero dwell positions")
        print(non_zero_dwell_positions[0])
    else:
        print("there are ", len(patient_plan.get_non_zero_dwell_positions_list())  , "non zero dwell positions")
        print(patient_plan.get_non_zero_dwell_positions_list()[0])

    # print(patient_plan.get_treatment_times())
    # print(patient_plan.get_tips_coords())
    # print(patient_plan.get_step_dwell_pos())
    # exit()

    non_zero_dwellpos = patient_plan.get_non_zero_dwell_positions()
    all_dwellpos = patient_plan.get_dwell_positions()

    curves = patient_plan.create_curves()
    tips = patient_plan.get_tips_coords()
    plot_digit_pts = True
    plot_non0_dwellpos = True
    plot_all_created_dwellpos = True
    plot_curves = True
    plot_tips = True
    needle_of_interests = ["Needle_8"] # None



    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if plot_digit_pts:
        # Plot digitization points
        for channel_number, digitization_points in patient_plan.digitization_points.items():
            if needle_of_interests is not None and channel_number not in needle_of_interests:
                continue
            x_coords = [pos[0] for pos in digitization_points]
            y_coords = [pos[1] for pos in digitization_points]
            z_coords = [pos[2] for pos in digitization_points]
            ax.scatter(x_coords, y_coords, z_coords, label="digpt_" + channel_number) #, marker="x")

    if plot_non0_dwellpos:
        # Plot non-zero dwell positions
        for nz_channel_number, nz_dwell_positions in non_zero_dwellpos.items():
            if needle_of_interests is not None and nz_channel_number not in needle_of_interests:
                continue
            x_coords = [pos[0] for pos in nz_dwell_positions]
            y_coords = [pos[1] for pos in nz_dwell_positions]
            z_coords = [pos[2] for pos in nz_dwell_positions]
            ax.scatter(x_coords, y_coords, z_coords, label="non_0_dw_" + nz_channel_number, marker="D")

    if plot_all_created_dwellpos:
        # Plot all dwell positions
        for channel_number, dwell_positions in all_dwellpos.items():
            if needle_of_interests is not None and channel_number not in needle_of_interests:
                continue
            x_coords = [pos[0] for pos in dwell_positions]
            y_coords = [pos[1] for pos in dwell_positions]
            z_coords = [pos[2] for pos in dwell_positions]
            ax.scatter(
                x_coords,
                y_coords,
                z_coords,
                label="all_dw_fromdigpt_" + channel_number,
                marker="x",
            )
    if plot_curves:
        for channel_number, curve in curves.items():
            if needle_of_interests is not None and channel_number not in needle_of_interests:
                continue
            x_coords = [pos[0] for pos in curve]
            y_coords = [pos[1] for pos in curve]
            z_coords = [pos[2] for pos in curve]
            ax.plot(x_coords, y_coords, z_coords, label="curve_" + channel_number)
    if plot_tips:
        for channel_number, tip in tips.items():
            if needle_of_interests is not None and channel_number not in needle_of_interests:
                continue
            x_coords = tip[0]
            y_coords = tip[1]
            z_coords = tip[2]
            ax.scatter(x_coords, y_coords, z_coords, label="tip_" + channel_number, marker="x")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()
