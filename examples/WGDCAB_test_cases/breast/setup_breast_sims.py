####################################
from brachyutils import BrachySimulation, BrachyPlan, BrachyEgsphant, BrachyPhantom, BrachyApplicator
from brachyutils.geometry_utils import CatheterTable
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

########################################
VALIDATION_DIR = "/home/jonathan/Documents/Breast_DL_Dose_Prediction_Benchmarking/"
os.chdir(VALIDATION_DIR)
REEXPORT_EGSPHANTS = True
####################################
#Define sim constants
AIR_KERMA_PER_HISTORY = 1.15723e-11
ATOMIC_NUMBER = 77
BEAM_ON = 2e9
CORE_MATERIAL = "G4_Ir"
MASS_NUMBER = 192
NUMBER_HISTORIES = int(2e9)
NUMBER_OF_THREADS = 48
REFERENCE_AIR_KERMA = 40700

####################################
#Create the simulation objects
SIM_DICT = {   
    "number_histories": NUMBER_HISTORIES, "number_of_threads": NUMBER_OF_THREADS, 
    "control_verbose": 0, "run_verbose": 0, "tracking_verbose": 0,
    "print_progress": int(NUMBER_HISTORIES / 100), "world_material": "Water", 
    "source_dict" : {"treatment_type": "HDR", "source_geometry": "GenericHDR", "core_material": CORE_MATERIAL, "mass_number": MASS_NUMBER, "atomic_number": ATOMIC_NUMBER, 
    "air_kerma_per_history": AIR_KERMA_PER_HISTORY, "reference_air_kerma": REFERENCE_AIR_KERMA} }
CONTENT_TO_EXPORT = {"dose": False, "dose_type": "nrrd", "uncertainty": False, "dose_rate_maps": False, "catheter_table": True,
"plan": True, "mac": True, "egsphant": True, "AppliacatorMaterials": True, "applicator_geometry": False, "structure_set": False }
MATERIAL_DICT = {
        "Air": {"encoding": 1, "density": 0.001225,"HU_limit": -1000.0},
        "Lung": {"encoding": 2, "density": 0.260, "HU_limit": -300.0, "structure_name" : ["Left Lung", "Right Lung"]},
        "Adipose": {"encoding": 3, "density": 0.970, "HU_limit": -100.0},
        "Breast_5050": {"encoding": 4, "density": 0.985, "HU_limit": 0.0},
        "SoftTissue": {"encoding": 5, "density": 1.02, "HU_limit": 0.0, "structure_name" : ["PTV", "BODY"]},
        "Bone": {"encoding": 6, "density": 1.920, "HU_limit": 300.0, "structure_name" : ["Left Ribs", "Right Ribs"]},
        }
####################################
STRUCTURE_FILE = "RS.1.2.246.352.71.4.810100034225.661.20150513153609.dcm"
#STRUCTURE_FILE = "RS1.2.826.0.1.3680043.8.274.1.1.2056645051.15934.1741297067.285687.dcm"
####################################
PLAN_FILE = "RP.1.2.246.352.71.5.686590568890.1121.20151125132537.dcm"
####################################
def main():
    os.chdir(VALIDATION_DIR)
    dicom_dir = f"./DICOM_INPUT/"
    #dicom_dir = f"./SlicerDicomExport/"
    export_dir  = f"./RapidBrachy/"
    SIM_DICT["pth_plan"] = "combined.plan"
    SIM_DICT["pth_phantom"] = "ct.egsphant"

    catheter_table = CatheterTable(pth_catheter_table=(dicom_dir + PLAN_FILE))

    phantom = BrachyPhantom(dicom_dir, None, dicom_dir + STRUCTURE_FILE, None)


#These functions with sampling don't work right now due to 
#bad upsampling of DICOM files
    phantom.write_to_egsphant(
    pth_output=export_dir + "breast_phantom.egsphant",
    material_dict=MATERIAL_DICT,
    assign_material_from_ct=False,
    background_material="Air",
    resample_egsphant_to=[1.0, 1.0, 1.0],
    resample_phantom_base=True
    )

    phantom.write_to_egsphant(
    pth_output=export_dir + "breast_phantom_sampled.seq.nrrd",
    material_dict=MATERIAL_DICT,
    assign_material_from_ct=False,
    background_material="Air",
    resample_egsphant_to=[1.0, 1.0, 1.0],
    resample_phantom_base=True
    )

    #phantom.write_to_egsphant(
    #pth_output=export_dir + "delete.seq.nrrd",
    #material_dict=MATERIAL_DICT,
    #assign_material_from_ct=False,
    #background_material="Air",
    #resample_egsphant_to=[1.0, 1.0, 1.0],
    #resample_phantom_base=True
    #)

    phantom.write_to_egsphant(
    pth_output=export_dir + "breast_phantom_unsampled.seq.nrrd",
    material_dict=MATERIAL_DICT,
    assign_material_from_ct=False,
    background_material="Air",
    resample_phantom_base=False
    )
    #outside-b



    SIM_DICT["total_time"] = catheter_table.get_treatment_time()

    plan = BrachyPlan(phantom, None, catheter_table, None, None,
    None, None, ".nrrd", "dose", False, True, SIM_DICT)
    plan.export_brachy_plan( export_dir, CONTENT_TO_EXPORT, "RapidBrachy")
####################################
if __name__ == "__main__":
    main()