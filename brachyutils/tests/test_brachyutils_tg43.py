####################################
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from brachyutils import (BrachyPlan, get_uniform_phantom, BrachyPhantom, 
CatheterTable, Catheter, DwellPosition, BrachyUtilsTG43, BrachyDose)
from pathlib import Path
####################################
matplotlib.use('TkAgg')
####################################
# #Define sim constants
ATOMIC_NUMBER = 77
MASS_NUMBER = 192
CORE_MATERIAL = "G4_Ir"
SOURCE_GEOMETRY = "GenericHDR"
AIR_KERMA_STRENGTH = 100 #100 U
DWELL_TIME = 3600 #1 hr
#choosing the dwell time/AKS such that our dose is per cGy/hr/U (=Gy/hr/100 U)
####################################
#Create the simulation objects
SOURCE_DICT = {"reference_air_kerma_rate": AIR_KERMA_STRENGTH,
                      "source_geometry": SOURCE_GEOMETRY, "atomic_number": ATOMIC_NUMBER, "mass_number": MASS_NUMBER,
                      "core_material": CORE_MATERIAL}
SIM_DICT = {"treatment_type": "HDR", "total_time": DWELL_TIME, "dose_format": "nrrd", "pth_plan": "combined.plan","pth_phantom": "phantom.egsphant",#egsphant", 
     "brachy_source": SOURCE_DICT}
####################################
CATHETER_TABLE = CatheterTable(
    catheters_dict=[
        Catheter(
            index=0,
            dwells=[
                DwellPosition(
                    index=0,
                    catheter_index=0,
                    position=[0.0, 0.0, 0.0],
                    relativePos=0,
                    rotation=[0.0, 0.0, 1.0],
                    time=DWELL_TIME,
                )]
            )
    ]
)
####################################
CALC_PARAMETER_QUARGS = { #some arguments for the BrachyUtilsTG43 dose calculator
    "kernel_max_dose_rate": 1000, #Gy/s
    "epsilon": 1e-8, #just a little nudge to certain values :)
    "auto_kernel": False, #if not True, you must set the next two values 
    "kernel_half_width" : 100, #half width to calculate dose rate kernel
    "kernel_res" : 1, #resolution to calculate the dose rate kernel
    "auto_phantom" : False, #crop phantom to all non-body structures
}
####################################
def load_qa_along_away_dose_table(pth_csv = Path(__file__).parent.parent.parent/ "admin/constants/TG43_Parameter_Data/GenericHDR/QA_along_away.csv"):
    table = np.loadtxt(pth_csv, delimiter=",")
    along = table[1:, 0] * 10 #convert cm to mm
    away = table[0, 1:] * 10 #convert cm to mm
    along_away_dose_table = table[1:, 1:]
    return (along, away, along_away_dose_table)
####################################
def test_brachyutils_tg43(examine_values = False):
    #tester to QA the BrachyUtilsTG43 dose calculator by calculating an along-away dose table and comparing to the QA table obtained from the BRAPHYQFS database
    phantom =  get_uniform_phantom(0.0, gridSize = [201, 201, 201], # a uniform 10x10x10 phantom
                                    spacing = [1.0, 1.0, 1.0], origin = [-100.0, -100.0, -100.0])

    test_plan = BrachyPlan(phantom = phantom, catheter_table = CATHETER_TABLE, simulation_setup = SIM_DICT)
    along, away, qa_along_away_dose_table = load_qa_along_away_dose_table() #load the QA table

    #calculate an along away dose table using the BrachyUtilsTG43 dose calculator
    dose_calculator = BrachyUtilsTG43(**CALC_PARAMETER_QUARGS)
    dose_calculator.run_dose_generation(dir_export = None, plan = test_plan, generate_dose_rate_maps = True, export_combined_dose = False)
    combined_dose = test_plan.combined_dose

    brachyutils_along_away_dose_table = np.array([[combined_dose.dose_image.getDataAtPosition([away[j], 0, along[i]]) for j in range(len(away))] for i in range(len(along))])

    #compare the two tables
    percent_difference = np.abs((brachyutils_along_away_dose_table - qa_along_away_dose_table) / qa_along_away_dose_table) * 100
    percent_difference[qa_along_away_dose_table < 0] = 0 #ignore the points where the QA table is zero aka inside the source

    if examine_values:
        examine_values(along, away, qa_along_away_dose_table, brachyutils_along_away_dose_table, percent_difference)
    


    assert np.all(percent_difference < 2), f"Percent difference between BrachyUtilsTG43 and QA table is greater than 2% at some points. Max percent difference: {np.max(percent_difference)}"
####################################
def examine_values(along, away, qa_table, brachyutils_table, percent_difference):
    print("Along values (mm):")
    print(along)
    print("Away values (mm):")
    print(away)
    print("QA Table:")
    print(qa_table)
    print("BrachyUtilsTG43 Table:")
    print(brachyutils_table)
    print("Percent Difference Table:")
    print(percent_difference)
    plt.imshow(percent_difference, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Percent Difference (%)')
    plt.show()
####################################
if __name__ == "__main__":
    test_brachyutils_tg43()
####################################
