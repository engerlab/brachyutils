from operator import rshift
from queue import Empty
import re
from unittest import result
from matplotlib import pyplot as plt
'%matplotlib inline'    # this line only works with jupyter notebooks.
from matplotlib.cm import ScalarMappable
import pickle
import numpy as np
import pandas as pd
# from saveResults_ax_experiment import ax_save_results
from glob import glob
from matplotlib.patches import Rectangle
import copy
import matplotlib.patches as mpatches
import os
import json

def op_with_error_propagate(a:float, err_a:float, b:float, err_b:float, op:str):
    r"""perform the desired operation and propagate the error
        inputs:
            a := the first input
            err_a := the standard deviation on a
            b: := the second input
            err_b := the standard deviation on b
            op := the operation between a and b in the same order a op b

        outputs:
            outcome:float := a op b
            err:float := the error depending on the operation
    """

    if op == "sum":
        outcome = a + b

    if op == "subtract":
        outcome = a - b
    
    if op == "sum" or op == "subtract":
        err = np.sqrt(err_a**2 + err_b**2)

    if op == "divide":
        outcome = a/b
    
    if op == "multiply":
        outcome = a * b

    if op == "divide" or op == "multiply":
        err = outcome * np.sqrt((err_a/a)**2 + (err_b/b)**2)

    if op == "percent_err":
        c, err_c = np.absolute(op_with_error_propagate(a, err_a, b, err_b, "subtract"))
        [outcome, err] = np.array(op_with_error_propagate(c, err_c, b, err_b, "divide")) * 100

    return outcome, err

# a function to count the number of successful plans in a MOBO experiment
def count_success_rate(results:dict):
    r'''a function that counts the percentage of clinically acceptable treatment plans in the results of a MOBO experiment. 
    inputs:
        - results:= output of penalty weight optimization using MOBO (see ax_client_qNEHVI_pwe.py).
            this dictionary must have 'df' key.
    outputs:
        - percent_success:= percentage of clinically acceptable treatment plan in results['df']
    
    dependencies:
        - get_clinical_set()
    '''
    deep_copy_results = copy.deepcopy(results)
    deep_copy_results['df'] = results['df'][results['df']['generation_method'] == 'MOO']
    num_clinical_solutions = get_clinical_set(deep_copy_results).shape[0]

    total_iterations = results['df']['generation_method'].value_counts()['MOO']
    percent_success = 100*num_clinical_solutions/total_iterations
    return percent_success

def get_clinical_set(results:dict):
    r'''a function that filters the clinical DVH metrics from the result of a MOBO experiment. 
    inputs:
        - results:= output of penalty weight optimization using MOBO (see ax_client_qNEHVI_pwe.py).
            this dictionary must have 'df' key.
    outputs:
        - clinical_df:= a subset dataframe of results['df'], where all enteries are clinically acceptable.
    '''
    clinical_df = results['df'].copy(deep=True)
    for col in results['df']:
        if "D" in col:
            # if the column belongs to tumor volume, keep the rows with larger than the clinical DVH goal.
            if 'tv' in col:
                clinical_df = clinical_df[clinical_df[col]>= results['clinic_dvh_goal'][col]]
                continue
            # in case column belongs to an OAR, keep the enteries with smaller than clinical DVH goal
            clinical_df = clinical_df[clinical_df[col] <= results['clinic_dvh_goal'][col]]
 
    return clinical_df

def load_pickled_results(path2file):
    with open(path2file, 'rb') as file:
        return pickle.load(file=file)


def get_meanTime_meanPercentSuccess(iteration_experiment):
    r""" given an iteration experiment obtain the mean and std of performance time and success rate
    input:
    """
    time_mean = np.array([])
    time_err = np.array([])
    success_rate_mean = np.array([])
    success_rate_err = np.array([])

    for iter_num in iteration_experiment:

        time_array = np.array([])
        outcome_array = np.array([])
        percent_success_array = np.array([])

        for rep in iteration_experiment[iter_num]:
            time_array = np.append(time_array, (iteration_experiment[iter_num][rep]['time']))
            outcome_array = np.append(outcome_array, (iteration_experiment[iter_num][rep]['outcomes']))
            percent_success_array, = np.append(percent_success_array, (count_success_rate(iteration_experiment[iter_num][rep])))
            
        time_mean = np.append(time_mean, (time_array.mean()))
        time_err = np.append(time_err, (time_array.std()))

        success_rate_mean = np.append(success_rate_mean, (percent_success_array.mean()))
        success_rate_err = np.append(success_rate_err, (percent_success_array.std()))

    print(f"time mean for iterations {iter_num} is \n {time_mean} +- {time_err}")
    print(f"success rate for iterations {iter_num} is \n {success_rate_mean} +- {success_rate_err}")
    return time_mean, time_err, success_rate_mean, success_rate_err

def check_top_solutions(results):
    return results['df'][results['df']['D0.1cc(ure)']<18].sort_values(by=['D90%(ptv)'], ascending=False)

# def full_FMIO(all_weights, path2mps):

#     return dvh_score_space(all_weights, path2mps, )

def plot_paretoFront(results:dict, patient:str=None, outFigName:str= None):
    r''' a function that plots 2D Pareto surface between DVH metric of the OARs and PTV. 
    inputs:
        - results:= output of penalty weight optimization using MOBO (see ax_client_qNEHVI_pwe.py)
        - patient:= the name of the patient. For example, p1
        - outFigName:= the figures can be automatically saved to a jpg file whose directory is provided here. 
    '''
    algos=f"DVH space of {results['df']['generation_method'].value_counts()['MOO']} MOBO iterations on"
    fontsize = 20

    cm = plt.cm.get_cmap('viridis')

    # find columns that contain the DVH metrics
    dvhMetric_name_list = []
    for col in results['df']:
        if "D" in col:
            if 'tv' in col or 'TV' in col:
                target_dvhMetric_name = col
                continue
            dvhMetric_name_list.append(col)

    # depending on the number of structures, we make different number of subplots in different organizations
    if len(dvhMetric_name_list) <= 3:
        fig1, axis_list = plt.subplots(1,len(dvhMetric_name_list),figsize=(30,7))
    if len(dvhMetric_name_list) == 4:
        fig1, axis_list = plt.subplots(2,2,figsize=(15,15))
    if len(dvhMetric_name_list) >= 5 :
        fig1, axis_list = plt.subplots(2,3,figsize=(30,15))
    if len(dvhMetric_name_list) >= 7 :
        fig1, axis_list = plt.subplots(3,3,figsize=(15,15))
    if len(dvhMetric_name_list) > 3:
        flattened_axis_list = [item for sublist in axis_list for item in sublist]
    else:
        flattened_axis_list = axis_list

    for i, dvhMetric_name in enumerate(dvhMetric_name_list):
        colors = np.where(results['df']['generation_method']=='MOO', 'r', 'b')
        # sc = flattened_axis_list[i].scatter(results['df'][target_dvhMetric_name], results['df'][dvhMetric_name], c=results['df'].trial_index.values, alpha=0.8)
        sc = flattened_axis_list[i].scatter(results['df'][target_dvhMetric_name], results['df'][dvhMetric_name], c=colors, alpha=0.8)

        title = algos + " " + patient if patient is not None else algos[0]
        flattened_axis_list[i].set_title(title, fontsize=fontsize)
        flattened_axis_list[i].set_xlabel(target_dvhMetric_name+' [Gy]', fontsize=fontsize)
        flattened_axis_list[i].set_ylabel(dvhMetric_name+' [Gy]', fontsize=fontsize)
        flattened_axis_list[i].tick_params(axis='x', labelsize=15)
        flattened_axis_list[i].tick_params(axis='y', labelsize=15)
        red_patch = mpatches.Patch(color='red', label='MOBO-qNEHVI')
        blue_patch = mpatches.Patch(color='blue', label='Random')
        legend = flattened_axis_list[i].legend(
            handles=[red_patch, blue_patch], 
            loc='upper left', 
            title='Generation Method', 
            fontsize=fontsize-5)
        flattened_axis_list[i].add_artist(legend)
        
        
        # let's color the region of the pareto surface that is clinically acceptable
        xy_coords = (results['clinic_dvh_goal'][target_dvhMetric_name], results['clinic_dvh_goal'][dvhMetric_name])
        rectangle_size = np.array([results['clinic_dvh_goal'][target_dvhMetric_name], results['clinic_dvh_goal'][dvhMetric_name]])/5
        flattened_axis_list[i].add_patch(Rectangle(xy_coords, rectangle_size[0], -rectangle_size[1]*3, alpha=0.3, color='green'))
        flattened_axis_list[i].text((xy_coords[0]*1.1), (xy_coords[1]*0.6), s="clinically acceptable \n DVH metrics", fontsize=15,
         ha='center', color="black")



    # in case you want to color the dots on the scatter plot with iteration numbers and add a color bar
    # norm = plt.Normalize(results['df'].trial_index.values.min(), results['df'].trial_index.values.max())
    # sm = ScalarMappable(norm=norm, cmap=cm)
    # sm.set_array([])
    # # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig1.add_axes([0.93, 0.15, 0.01, 0.7])
    # cbar = fig1.colorbar(sm, cax=cbar_ax)
    # cbar.ax.set_title("Iteration")
    if outFigName is not None:
        fig1.savefig(outFigName, bbox_inches='tight')


def plot_hv(results):
    if results['hv_list'] != []:
        num_trials = np.max(results['df']['trial_index'])+1
        iters = np.arange(1, num_trials+1)
        log_hv_difference = np.log10(5 - np.asarray(results['hv_list']))[:num_trials]

        fig, ax = plt.subplots(1, 1, figsize=(9,6))
        ax.plot(iters, log_hv_difference, label="qNEHVI-AxClient", linewidth=1.5)
        ax.set(title="Performance of Sequential Multi Objective Optimization on Penalty weight tunning in HDR Brachytherapy", xlabel='Iterations', ylabel='Log Hypervoume Difference')
        ax.legend(loc="upper right")
    else:
        print("Hypervolume was not tracked")

def max_D90_ptv(data_frame:pd.DataFrame):
    if data_frame.empty:
        return None
    for col in data_frame:
        if "D" in col and "tv" in col:
            return data_frame[col].max()

def min_D90_ptv(data_frame:pd.DataFrame):
    if data_frame.empty:
        return None
    for col in data_frame:
        if "D" in col and "tv" in col:
            return data_frame[col].min()

def min_D01cc_ure(data_frame:pd.DataFrame):
    if data_frame.empty:
        return None
    for col in data_frame:
        if "D" in col and "ure" in col:
            return data_frame[col].min()

def max_D01cc_ure(data_frame:pd.DataFrame):
    if data_frame.empty:
        return None
    for col in data_frame:
        if "D" in col and "ure" in col:
            return data_frame[col].max()


def get_weights(results):
    return results['df']['w_ure']['w_rect']['w_blad']

def get_topWeights_perObjective_allPatients(dir_patients:str, pth_json:str):
    r"""
    Purpose: 
        To go through all the patient files inside "dir_patients" and find the best weights
        for each objective along with the value of that objective. The weights and objective values
        are written to pth_json 
    Inputs:
        - dir_patients := Path to the directory where patient files exist. these files contain ???
        - pth_json := Path to the .csv file where the weights and the objective values are written to.
    Outputs: 
        - Void: the info is written to pth_json
    """
    assert os.path.exists(dir_patients), "input patient directory does not exist"
    patient_files = glob(dir_patients+"/*.pkl")
    
    topWeights_perObjective_allPatients_list = []
    
    # loop over the patient files in the dir_patients
    for patient_file in patient_files:
        # all results include plans that were generated randomly and by MOBO
        results = load_pickled_results(patient_file)
        clinical_solutions = get_clinical_set(results)
        mobo_results = clinical_solutions.loc[clinical_solutions['generation_method'] == "MOO"]
        
        # gather the top treatment plan for each objective
        top_treatment_plans_singlePatient = []
        for column in mobo_results:
            if 'D' and "(" and ")" in column:
                if 'tv' in column:
                    top_treatment_plans_singlePatient.append(
                    {
                        'priority_objective': column,
                        'treatment_plan': mobo_results.loc[
                            mobo_results[column].idxmax()].to_dict()
                    })
                else:
                    top_treatment_plans_singlePatient.append(
                    {
                        'priority_objective': column,
                        'treatment_plan': mobo_results.loc[
                            mobo_results[column].idxmin()].to_dict()
                    })
                      
        topWeights_perObjective_allPatients_list.append(
            {
                'patient_id': os.path.splitext(os.path.basename(patient_file))[0],
                'top_treatmentPlans_singlePatient': top_treatment_plans_singlePatient
            })
    
    with open(pth_json, 'w') as json_file:
        json.dump(
            topWeights_perObjective_allPatients_list,
            json_file,
            indent=4)
    
# Testers for the functions above
def _test_get_weights():
    fileName = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_p5.pkl"
    results = load_pickled_results(fileName)
    try:
        weights = get_weights(results)
        print(weights)
    except:
        print("test failed: get_weights")
    return 0

def _test_get_clinical_set():
    fileName = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_p5.pkl"
    results = load_pickled_results(fileName)
    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(PTV)']
    # results['clinic_dvh_goal']['D90%(ptv)'] = 15
    # # }
    get_clinical_set(results)
    return 0


def _test_count_success_rate():
    # fileName = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_p5.pkl"
    # results = load_pickled_results(fileName)
    # # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(PTV)']
    # results['clinic_dvh_goal']['D90%(ptv)'] = 15
    # # # }

    fileName = '../data_files/MOBO/patients/gynAlana_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_gyn-test.pkl'
    results = load_pickled_results(fileName)
    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    del results['clinic_dvh_goal']['D90%(CTV)']
    results['clinic_dvh_goal']['D90%(ctv)'] = 6
    # }
    count_success_rate(results)

def _test_plot_paretoFront():
    file = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_targetDoseIsParam_parallelInitp9.pkl"
    results = load_pickled_results(file)

    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(PTV)']
    # results['clinic_dvh_goal']['D90%(ptv)'] = 15
    # # }
    
    
    # fileName = '../data_files/MOBO/patients/gynAlana_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_gyn-test.pkl'
    # results = load_pickled_results(fileName)
    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(CTV)']
    # results['clinic_dvh_goal']['D90%(ctv)'] = 6
    # # }
    results = load_pickled_results(file)

    plot_paretoFront(results, 'prostate')

def _test_max_D90_ptv():
    # fileName = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_p5.pkl"
    # results = load_pickled_results(fileName)
    # # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(PTV)']
    # results['clinic_dvh_goal']['D90%(ptv)'] = 15
    # # # }

    fileName = '../data_files/MOBO/patients/gynAlana_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_gyn-test.pkl'
    results = load_pickled_results(fileName)
    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    del results['clinic_dvh_goal']['D90%(CTV)']
    results['clinic_dvh_goal']['D90%(ctv)'] = 6
    # }

    print(max_D90_ptv(results['df']))

def _test_min_D01cc_ure():
    fileName = "../data_files/MOBO/patients/finalProstate_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_p5.pkl"
    results = load_pickled_results(fileName)
    # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    del results['clinic_dvh_goal']['D90%(PTV)']
    results['clinic_dvh_goal']['D90%(ptv)'] = 15
    # # }

    # fileName = '../data_files/MOBO/patients/gynAlana_linearFMIO_rtogGoals_absoluteDVHdose/halfRange_gyn-test.pkl'
    # results = load_pickled_results(fileName)
    # # for debugging{ fixing the name of the dvh metric for the target volume. the name of the structures must be lower caps
    # del results['clinic_dvh_goal']['D90%(CTV)']
    # results['clinic_dvh_goal']['D90%(ctv)'] = 6
    # }
    print(min_D01cc_ure(results['df']))

# def change_mobo_results():
    # del results['clinic_dvh_goal']['D90%(PTV)']
    # results['clinic_dvh_goal']['D90%(ptv)'] = 15
    
if __name__ == "__main__":
    dir_patients = "../data_files/MOBO/patients/prostate_Glen_quadFmio_rtogPlus"
    pth_json = "../data_files/MOBO/patients/prostate_Glen_quadFmio_rtogPlus/topWeights_perObjective_allPatients.json"
    get_topWeights_perObjective_allPatients(dir_patients, pth_json)