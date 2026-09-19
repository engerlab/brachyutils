####################################
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from brachyutils import (BrachyPlan, get_uniform_phantom, BrachyPhantom, 
CatheterTable, Catheter, DwellPosition, BrachyUtilsTG43, RapidBrachyTG43, BrachyDose, ExportConfig_BrachyPlan)
from pathlib import Path
import logging
####################################
matplotlib.use('TkAgg')
logging.basicConfig(level=logging.INFO)
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
    "kernel_half_width" : 120, #half width to calculate dose rate kernel
    "kernel_res" : 0.5, #resolution to calculate the dose rate kernel
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
def make_uniform_phantom():
    #make a uniform phantom for testing
    phantom = get_uniform_phantom(0.0, gridSize = [403, 403, 403], # a uniform 10x10x10 phantom
                                    spacing = [0.5, 0.5, 0.5], origin = [-101.0, -101.0, -101.0])
    return phantom
####################################
def extract_and_compare_along_away_dose_table(combined_dose, along, away, qa_along_away_dose_table, examine = False, calculator = "BrachyUtilsTG43"):
    brachyutils_along_away_dose_table = np.array([[combined_dose.dose_image.getDataAtPosition([away[j], 0, along[i]]) for j in range(len(away))] for i in range(len(along))])

    #compare the two tables
    percent_difference = np.abs((brachyutils_along_away_dose_table - qa_along_away_dose_table) / qa_along_away_dose_table) * 100
    percent_difference[qa_along_away_dose_table < 0] = 0 #ignore the points where the QA table is zero aka inside the source

    if examine:
        examine_values(along, away, qa_along_away_dose_table, brachyutils_along_away_dose_table, percent_difference, calculator)
    assert np.all(percent_difference < 2), f"Percent difference between {calculator} and QA table is greater than 2% at some points. Max percent difference: {np.max(percent_difference)}"

####################################
def test_brachyutilstg43(examine = False):
    #tester to QA the BrachyUtilsTG43 dose calculator by calculating an along-away dose table and comparing to the QA table obtained from the BRAPHYQFS database
    test_plan = BrachyPlan(phantom = make_uniform_phantom(), catheter_table = CATHETER_TABLE, simulation_setup = SIM_DICT)
    along, away, qa_along_away_dose_table = load_qa_along_away_dose_table() #load the QA table

    #calculate an along away dose table using the BrachyUtilsTG43 dose calculator
    dose_calculator = BrachyUtilsTG43(**CALC_PARAMETER_QUARGS)
    dose_calculator.run_dose_generation(dir_export = None, plan = test_plan, generate_dose_rate_maps = True, export_combined_dose = False)
    combined_dose = test_plan.combined_dose

    extract_and_compare_along_away_dose_table(combined_dose, along, away, qa_along_away_dose_table, examine = examine, calculator = "BrachyUtilsTG43")
####################################
def test_rapidbrachytg43(examine = False, dock = True):
    #tester to QA the RapidBrachyTG43 dose calculator by calculating an along-away dose table and comparing to the QA table obtained from the BRAPHYQFS database
    test_plan = BrachyPlan(phantom = make_uniform_phantom(), catheter_table = CATHETER_TABLE, simulation_setup = SIM_DICT)
    along, away, qa_along_away_dose_table = load_qa_along_away_dose_table() #load the QA table

    #calculate an along away dose table using the RapidBrachyTG43 dose calculator
    dir_tmp_data = Path(__file__).parent.parent.parent/ "data_test/test_dose_calculators_tg43/RapidBrachyTG43/"
    if not dock:
        pth_exec = "TG43DoseCalculator" #assumes RapidBrachyTG43 is installed in PATH
        dose_calculator = RapidBrachyTG43(dir_tmp_data, pth_exec)
    else:
        dose_calculator = RapidBrachyTG43(dir_tmp_data)

    dir_tg43_parameters = Path(__file__).parent.parent.parent/ "admin/constants/TG43_Parameter_Data/GenericHDR/"
    export_config = ExportConfig_BrachyPlan(dir_export = dir_tmp_data, export_config_egsphant = True, export_config_plan_and_mac = True)
    dose_calculator.run_dose_generation(plan = test_plan, generate_dose_rate_maps = False, dir_source_parameters=dir_tg43_parameters)
    combined_dose = test_plan.combined_dose

    extract_and_compare_along_away_dose_table(combined_dose, along, away, qa_along_away_dose_table, examine = examine, calculator = "RapidBrachyTG43")
####################################
def examine_values(along, away, qa_table, brachyutils_table, percent_difference, calculator):
    print("Along values (mm):")
    print(along)
    print("Away values (mm):")
    print(away)
    print("QA Table:")
    print(qa_table)
    print(f"{calculator} Table:")
    print(brachyutils_table)
    print("Percent Difference Table:")
    print(percent_difference)
    write_along_away_dose_table_to_csv(along, away, brachyutils_table, calculator)
    plt.title(f"Percent Difference between {calculator} and QA Table")
    plt.imshow(percent_difference, cmap='hot', interpolation='nearest', vmin=0, vmax=2)
    plt.colorbar(label='Percent Difference (%)')
    plt.show()
####################################
def write_along_away_dose_table_to_csv(along, away, table, calculator, pth_csv = None):
    #write the brachyutils along away dose table to a csv file
    if pth_csv is None:
        pth_csv = Path(__file__).parent.parent.parent/ f"data_test/test_dose_calculators_tg43/{calculator}/{calculator}_along_away.csv"
    full_table = np.zeros((len(along) + 1, len(away) + 1))
    full_table[0, 1:] = away / 10 #convert mm to cm
    full_table[1:, 0] = along / 10 #convert mm to cm
    full_table[1:, 1:] = table
    np.savetxt(pth_csv, full_table, delimiter=",", fmt='%1.3e')

####################################
if __name__ == "__main__":
    test_brachyutilstg43(examine = True)
    test_rapidbrachytg43(examine = True, dock = False)
####################################
