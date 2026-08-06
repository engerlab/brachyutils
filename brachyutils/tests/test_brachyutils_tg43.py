####################################
import numpy as np
import os
import json
from brachyutils import (BrachyPlan, get_uniform_phantom,
BrachyEgsphant, BrachyPhantom, BrachyApplicator, CatheterTable, Catheter, DwellPosition, BrachyUtilsTG43)
from pathlib import Path
import pymesh
####################################
# #Define sim constants
ATOMIC_NUMBER = 77
MASS_NUMBER = 192
CORE_MATERIAL = "G4_Ir"
SOURCE_GEOMETRY = "GenericHDR"
AIR_KERMA_STRENGTH = 36260
AIR_KERMA_PER_HISTORY_STRENGTH = 1.158e-11
DWELL_TIME = 1 #second
####################################
#Create the simulation objects
SOURCE_DICT = {"reference_air_kerma_rate": AIR_KERMA_STRENGTH, "air_kerma_per_history": AIR_KERMA_PER_HISTORY_STRENGTH,
                      "source_geometry": SOURCE_GEOMETRY, "atomic_number": ATOMIC_NUMBER, "mass_number": MASS_NUMBER,
                      "core_material": CORE_MATERIAL}
SIM_DICT = {"treatment_type": "HDR", "total_time": DWELL_TIME, "dose_format": "nrrd", "pth_plan": "combined.plan","pth_phantom": "phantom.egsphant",#egsphant", 
     "brachy_source": SOURCE_DICT}
####################################
CONTENT_TO_EXPORT = {"plan": True, "mac": True, "egsphant": True,
                     "materials_table": "phantom_material.json", "assign_material_from_ct": True, "background_material": "Water"}
####################################
CATHETER_TABLE = CatheterTable(
    catheter_list=[
        Catheter(
            index=0,
            dwells=[
                DwellPosition(
                    index=0,
                    position=[0.0, 0.0, 0.0],
                    relativePos=0,
                    rotation=[0.0, 0.0, 1.0],
                    time=DWELL_TIME,
                )]
            )
        ]
)
####################################
#Applicator processing
#z_source is the coordinate describing the source's dwell position relative to the emission window
#it's the source center relative to the applicator emission window start
#Z_SOURCES = np.array([float(i) for i in range(0, 10)] + [10 + 5*float(i) for i in range(0, 12)] + [70 + float(i) for i in range(0, 11)])
#            #0-9 mm in 1 mm steps               + 10-70 mm in 5 mm steps                      #70-80 mm in 1 mm steps
Z_SOURCES = np.array(json.load(open(TG43S_DIR + "zsources.json"))["Z_SOURCES"])
print(f"Number of Z_SOURCES: {len(Z_SOURCES)}")
APPLICATOR_CONSTANTS = json.load(open(APPLICATOR_DIR + "APPLICATOR_CONSTANTS.json"))
APPLICATORS_PATH = TG43S_DIR + "/sim_applicator_models/"
####################################
def write_applicator_models_at_z_source(z_source_mm: float):
    applicator_to_z_source = 10.0 * (APPLICATOR_CONSTANTS["SHIELD"]["ORIGIN_TO_START_LENGTH_CM"] -
                                    APPLICATOR_CONSTANTS["SHIELD"]["START_TO_WINDOW_LENGTH_CM"] ) - \
                                    z_source_mm
    for applicator_component in APPLICATOR_CONSTANTS.keys():
        component_object = BrachyApplicator(APPLICATOR_DIR + APPLICATOR_CONSTANTS[applicator_component]["PATH"],
                                            material=APPLICATOR_CONSTANTS[applicator_component]["MATERIAL"],
                                            density=APPLICATOR_CONSTANTS[applicator_component]["DENSITY_G_CM3"],
                                            origin = [0.0, 0.0, applicator_to_z_source],
                                            rotation = [0.0, 0.0, 1.0, 0.0],
                                            rotation_origin=[0.0, 0.0, 0.0],
                                            coordinates = [0.0, 0.0, 0.0],
                                            normal = [1.0, 0.0, 0.0])
        component_filename_with_transform = f"{applicator_component.split('.')[0].lower()}_{int(z_source_mm)}mm.stl"
        component_object.to_stl(TG43S_DIR + "/sim_applicator_models/" + component_filename_with_transform )

####################################
def make_dir(z_source: float):
    if not os.path.exists(TG43S_DIR + f"/sims/TG43S_{int(z_source)}mm"):
        os.mkdir(TG43S_DIR + f"/sims/TG43S_{int(z_source)}mm")

def copy_plan_files_to_dirs(z_source: float):
    #was gonna copy plan files and egsphants but that's an effing bad idea
    #because each one is 300 MB. try to use relative paths in macs instead
    pass

def copy_mac_to_dirs(z_source: float):
    mac_template_path = TG43S_DIR + "/tg43s_template.mac"
    with open(mac_template_path, 'r', encoding='utf-8') as mac_template:
        mac_lines = mac_template.read()
        #replace the phantom and plan paths
        mac_lines = mac_lines.replace("<PHANTOM_PATH>", "../../phantom.egsphant")
        mac_lines = mac_lines.replace("<PLAN_PATH>", "../../combined.plan")
        for component in APPLICATOR_CONSTANTS.keys():
            mac_lines = mac_lines.replace(f"<{component.upper()}_PATH>", f"../../sim_applicator_models/{component.lower()}_{int(z_source)}mm.stl")
            mac_lines = mac_lines.replace(f"<{component.upper()}_MATERIAL>", APPLICATOR_CONSTANTS[component]["MATERIAL"])
            mac_lines = mac_lines.replace(f"<{component.upper()}_DENSITY>", str(APPLICATOR_CONSTANTS[component]["DENSITY_G_CM3"]))
    mac_output_path = f"{TG43S_DIR}/sims/TG43S_{int(z_source)}mm/tg43s_{int(z_source)}mm.mac"
    with open(mac_output_path, 'w', encoding='utf-8') as mac_output:
        mac_output.write(mac_lines)

####################################
REEXPORT_BRACHY_PLAN_BASE = SO_NOT_TRUE
REEXPORT_APPLICATOR_MODELS_FOR_ZSOURCE = SO_NOT_TRUE
REEXPORT_MACS_AND_PLAN_FOR_ZSOURCE = SO_TRUE
####################################
def main():
    os.chdir(TG43S_DIR)
    if REEXPORT_BRACHY_PLAN_BASE:
        phantom = get_uniform_phantom(0.0, gridSize = [201, 201, 201],
                                    spacing = [1.0, 1.0, 1.0], origin = [-100.0, -100.0, -100.0])
        plan = BrachyPlan(phantom = phantom,
                        catheter_table=CATHETER_TABLE,
                        simulation_setup=SIM_DICT)
        print("Exporting TG43S simulation plan and phantom...")
        plan.export_brachy_plan(TG43S_DIR, CONTENT_TO_EXPORT, "RapidBrachy")
        print("####################################")
    for z_source in Z_SOURCES:
        print("Exporting applicator models:")
        if REEXPORT_APPLICATOR_MODELS_FOR_ZSOURCE:
                write_applicator_models_at_z_source(z_source)
        print("####################################")
        if REEXPORT_MACS_AND_PLAN_FOR_ZSOURCE:
            make_dir(z_source)
            copy_plan_files_to_dirs(z_source)
            copy_mac_to_dirs(z_source)
    print("####################################")

####################################
if __name__ == "__main__":
    main()
####################################
