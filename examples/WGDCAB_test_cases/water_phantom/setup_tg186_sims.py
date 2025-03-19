####################################
from brachyutils import BrachySimulation, BrachyPlan, BrachyEgsphant, BrachyPhantom, BrachyApplicator
from brachyutils import CatheterTable
import os
from pathlib import Path

########################################
VALIDATION_DIR = "/home/jonathan/Documents/TG186_Validation/"
APPLICATOR_DIR = VALIDATION_DIR + "TestCase4-Elekta/Shield_Design/"
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
REFERENCE_AIR_KERMA = 36260
DWELL_TIME = 10


####################################
#Create the simulation objects
SIM_DICT = {"treatment_type": "HDR", "source_geometry": "GenericHDR", "core_material": CORE_MATERIAL, "mass_number": MASS_NUMBER, "atomic_number": ATOMIC_NUMBER, 
    "air_kerma_per_history": AIR_KERMA_PER_HISTORY, "reference_air_kerma": REFERENCE_AIR_KERMA, 
    "number_histories": NUMBER_HISTORIES, "number_of_threads": NUMBER_OF_THREADS, 
    "control_verbose": 0, "run_verbose": 0, "tracking_verbose": 0, 
    "print_progress": int(NUMBER_HISTORIES / 100), "total_time": DWELL_TIME}
CONTENT_TO_EXPORT = {"dose": False, "dose_type": "nrrd", "uncertainty": False, "dose_rate_maps": False, "catheter_table": False,
"plan": True, "mac": True, "egsphant": False, "AppliacatorMaterials": True, "applicator_geometry": False, "structure_set": False }
####################################
STRUCTURE_FILES = {1: "RS1.3.6.1.4.1.2452.6.180500011.1246813493.4271452319.2554083995.dcm", 
    2: "RS1.3.6.1.4.1.2452.6.4030488904.1122551168.1245550258.529056527.dcm",
    3: "RS1.3.6.1.4.1.2452.6.1886881383.1190312809.852807099.2040955494.dcm",
    4: "RS1.3.6.1.4.1.2452.6.2318640623.1087278024.2522526336.4265498768.dcm" }
####################################
PLAN_FILES = {1: "RP1.3.6.1.4.1.2452.6.1410845409.1298930013.3582811068.2750109044.dcm",
    2: "RP2_new.dcm",
    3: "RP3_new.dcm",
    4: "RP4_new.dcm"
    }
####################################
def main():
    os.chdir(VALIDATION_DIR)
    for i in range(4, 5):
    #for i in range(1, 5):
        dicom_dir = f"./TestCase{i}-Elekta/Case-{i}-OCB/Case-{i}-OCB/"
        export_dir  = f"./TestCase{i}-Elekta/Case-{i}-RapidBrachy/"
        #if REEXPORT_EGSPHANTS:
        SIM_DICT["pth_plan"] = "combined.plan"
        SIM_DICT["pth_phantom"] = "ct.egsphant"
        sim_dict = make_sim_dict_for(i)
        catheter_table = CatheterTable(pth_catheter_table=(dicom_dir + PLAN_FILES[i]))
        fix_source_orientation_for(i, catheter_table)
        phantom = make_egsphant_for(i, dicom_dir,export_dir)
        plan = BrachyPlan(phantom, None, catheter_table, None, None,
        None, None, ".nrrd", "dose", False, True, sim_dict)
        make_applicator_macs_for(i, export_dir)
        #phantom.write_to_egsphant(export_dir + "ct.egsphant", WATER_PHANTOM_MATERIAL_DICT, False)
        plan.export_brachy_plan("RapidBrachy", export_dir, CONTENT_TO_EXPORT)
    ####################################
def make_egsphant_for(i, dicom_dir, export_dir):
    try:
        phantom = BrachyPhantom(dicom_dir, None, dicom_dir + STRUCTURE_FILES[i], None)
        material_dict = create_material_dict_for(i)
        #phantom.egsphant_obj = BrachyEgsphant(phantom=phantom, material_dict=material_dict, assign_material_from_ct=False)
        try:
            phantom = phantom.crop_by_contour("Cube", False)
        except KeyError:
            phantom = phantom.crop_by_contour("BODY", False)
        phantom.write_to_egsphant(
            pth_output=export_dir + "ct.egsphant",
            material_dict=material_dict,
            assign_material_from_ct=False,
        )
        phantom.material_dict = material_dict
    except AssertionError as e:
        print(f"Error in phantom {i} at dicom path {dicom_dir}: {e}")
        exit(1)
    #phantom.write_to_egsphant(export_dir + "phantom.egsphant", WATER_PHANTOM_MATERIAL_DICT, False)
    return phantom
####################################
def fix_source_orientation_for(i, catheter_table):
    rotation = [0, 0, 0]
    if(i == 1):
        rotation = [0, 1, 0]
    elif(i == 2):
        rotation = [0, 1, 0]
    elif(i == 3):
        rotation = [0, 1, 0]
    elif(i == 4):
        rotation = [0, 0, 1]

    catheter_table.catheter_list[0].dwells[0].rotation = rotation
####################################
def create_material_dict_for(i):
    material_dict = {}
    if i == 1:
        material_dict = {
        "Air": {"encoding": 1, "density": 0.001225,"HU_limit": -1000.0},
        "Water": {"encoding": 2, "density": 0.998, "HU_limit": 0, "structure_name": "Cube"},
        "water": {"encoding": 2, "density": 0.998, "HU_limit": 0, "structure_name": "BgBOX"},
        }
    elif i == 2 or i == 3:
        material_dict = {
        "Air": {"encoding": 1, "density": 0.001225,"HU_limit": -1000.0, "structure_name": "BgBOX"},
        "Water": {"encoding": 2, "density": 0.998, "HU_limit": 0, "structure_name": "Cube"},
        }
    elif i == 4:
        material_dict = {
        "Air": {"encoding": 1, "density": 0.001225, "HU_limit": -1000.0, "structure_name": "Applicator_Air"},
        "Water": {"encoding": 2, "density": 0.998, "HU_limit": 0, "structure_name": "BODY"},
        }
    else:
        raise ValueError(f"Invalid case number {i}")
    return material_dict
####################################
def make_sim_dict_for(i):
    sim_dict = SIM_DICT.copy()
    if i == 1:
        sim_dict["world_material"] = "Water"
    elif i == 2 or i == 3 or i == 4:
        sim_dict["world_material"] = "Air"
    return sim_dict

def make_applicator_macs_for(i, export_dir):
    if i != 4:
        return
    applicator_names_and_materials = [
    ["Densimet.stl", "Densimet", 17.6],
    ["Steel.stl", "SS316L", 8.0 ],
    ["PMMA_cyl.stl", "PMMA", 1.19],
    ["PMMA_cap.stl", "PMMA", 1.19]

    #["Air1.stl", "Air", 0.00119],
    #["Air2.stl", "Air", 0.00119]
    ]

    applicator_number = 0
    for [file_name, material, density] in applicator_names_and_materials:
        applicator = BrachyApplicator(APPLICATOR_DIR + file_name, material, density)
        applicator.to_mac(export_dir + f"applicator_{applicator_number}.mac")
        applicator.to_stl(export_dir + f"applicator_{applicator_number}.stl")
        applicator_number += 1

####################################
if __name__ == "__main__":
    main()