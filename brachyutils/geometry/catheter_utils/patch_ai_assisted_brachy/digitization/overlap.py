"""
This script is intended to check for how many cases we have overlap of catheters in our dataset.
This does not involve any AI tool, we just created catheters based on digitization points and check for overlap.
"""

import glob
import os
import tqdm 

import numpy as np
import SimpleITK as sitk
# nnunet dependent need to be loaded before importing any
# brachyutils dependent module.
from inference.catheters import CatheterDLSegmentor
from catheter.catheter_setup import CatheterSetUp
from catheter.contour.creator import CatheterContourCreator
from catheter.digitization.contour_digitizer import DwellPositionCreator
from catheter.exploration.catheter_types import flexi_catheters_patients
from catheter.contour.postprocess import CatheterPostProcessor
from ai_assisted_brachy.utils.utils import resample_volume

if __name__ == "__main__":
    if "root" in os.getcwd():
        # We are in a docker container
        HOME = "/root"
    else:
        hostname = os.getlogin()
        HOME = os.path.join("/home/", hostname)

    patient_folder = f"{HOME}/EngerLab/Data/patient_seb/"
    patient_paths = glob.glob(os.path.join(patient_folder, "*"))

    model_name = "Dataset004_catheters_and_tip_markers"
    catheter_segmentation_model_path = os.path.join(HOME, f"EngerLab/AI_Assisted_Brachytherapy/nnUNet_results/{model_name}/")
    dose_prediction_model_path = os.path.join(HOME, "EngerLab/RapidBrachyDL/Trainings/ExperimentPaperV4Training85/Training/")
    overwrite = True

    overlapping_catheters_patient_count = 0
    patients_counter = 0
    overlapping_catheters_patients = []
    for patient_path in tqdm.tqdm(patient_paths, desc="Checking for overlapping catheters", total=len(patient_paths)):
        patient_nb = os.path.basename(patient_path)
        if not patient_nb.isdigit() or patient_nb in flexi_catheters_patients:
            continue

        patients_counter += 1
        patient_path =os.path.join(patient_folder, patient_nb)

        # We need the AI contour here and not the analytical one since the analytical one 
        # is piece wise linear and not smooth enough to detect overlapping catheters using
        # our distance-to-spline criteria.
        sitk_needles_contour_path = os.path.join(patient_path, "processed", "ai_generated_catheters_postprocessed.seg.nrrd")
        catheter_creator_dilation = 1
        print(patient_nb)
        if overwrite or not os.path.exists(sitk_needles_contour_path) :

            # Just instanciating this class to create CT nrrd file.
            _ = CatheterContourCreator(
                patient_path,
                dilation=catheter_creator_dilation,
            )
            new_spacing = [1., 1., 1.]
            ct = sitk.ReadImage(os.path.join(patient_path, "ct.nrrd"))
            if not np.allclose(ct.GetSpacing(), new_spacing, atol=1e-5):
                ct = resample_volume(ct, new_spacing=new_spacing, interpolator=sitk.sitkLinear)
                sitk.WriteImage(ct, os.path.join(patient_path, "resampled_ct.nrrd"), True)
            # Creating ai contour
            predictor = CatheterDLSegmentor(
                model_path=catheter_segmentation_model_path,
                fold_num=[0],
                output_folder=os.path.dirname(sitk_needles_contour_path),
                threshold_probabilities=False
            )
            predictor.predict(
                ct_path=os.path.join(patient_path, "resampled_ct.nrrd"),
                output_file_name=os.path.basename(sitk_needles_contour_path))
            post_processor = CatheterPostProcessor(reference_ct=ct)
            post_processed_ai_contours, _, post_processed_infos, separator_infos = post_processor.postprocess_catheters(catheters_contour=sitk_needles_contour_path)
            sitk_needles_contour_path = sitk_needles_contour_path.replace(".seg.nrrd", "_postprocessed.seg.nrrd")
            sitk.WriteImage(post_processed_ai_contours, 
                            sitk_needles_contour_path, 
                            useCompression=True)
        else:
            continue

        patient_plan = CatheterSetUp(patient_path)
        dwellpos_dict = patient_plan.get_dwell_positions()
        step_dwell_pos = patient_plan.get_step_size()
        nb_needles = len(patient_plan.dwell_positions)

        dwell_pos_creator = DwellPositionCreator(
            sitk_needles_contour_path,
            fit_function="spline",
            # CatheterEvaluator takes consistent tip at the most distal part
            # of tip marker now => tip_distal always True.
            tip_distal=True
        )
        contoured_needles, components_img = dwell_pos_creator.preprocess_contour()

        
        if contoured_needles is None:
            overlapping_catheters_patient_count += 1
            overlapping_catheters_patients.append(patient_nb)
    
    print(f"Number of patients with overlapping catheters: {overlapping_catheters_patient_count}/{patients_counter}")
    print(f"Patients with overlapping catheters: {overlapping_catheters_patients}")