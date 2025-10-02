r'''
Date 
    2022/9/20

Purpose
    This script runs various experiments with penalty weight optimization using multi objective optimization (MOBO).
        MOBO is implemented by ax api and must be installed, see (https://ax.dev/docs/api.html) for directions
   
Author
    Hossein Jafarzadeh,
        Enger lab
        McGill University 

Functions implemented
    - generate_mobo_parameters()
    - generate_mobo_objectives()
    - ax_save_results()
    - run_mobo_iterations(): the most important function of this package
    - run_experiments_with_variousIterations()
    - run_experiment_on_patients()
'''

from cProfile import run
from importlib.resources import path
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax import OutcomeConstraint
from ax import OptimizationConfig
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.dispatch_utils import choose_generation_strategy

import torch
import numpy as np
import pandas as pd
# Load our sample 2-objective problem
# from penaltyWeightEvaluator import PenaltyWeightEvaluator

from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.registry import Models
from ax.modelbridge.factory import get_MOO_NEHVI
from time import perf_counter

from ax.service.utils.report_utils import exp_to_df
import pickle

# from saveResults_ax_experiment import ax_save_results
import glob
from f_weight_to_DVHmetric import TreatmentPlan
import re

from visualize_mobo_library import count_success_rate

def generate_mobo_parameters(treatmentPlan:TreatmentPlan, param_configs:dict):
    r''' generate a list of paremeters that ax api can interface with
    inputs:
        - treatmentPlan:= an object that has all the structures to be iradiated. the structures must have the type "TPS_structure".
        - param_config:= a dictionary containing the keys: 
            "relative_weights":bool. if this field is True, then  the objective thresholds are normalized to the target dose,
            "weight_range": has the highest and lowest value for the parameter ,
            "target_dose_range": has the highest and lowest value for the target dose parameter.  

    outputs:
        - mobo_params := a list of dictionaries with the following fields:
            {"name":, "type":, "bounds":}
    '''
    mobo_params = []
    for structure in treatmentPlan.structure_list:
        # the weight of the tumor volume (tv) is always 1 when using relative weights
        if 'tv' in structure.name:
            if 'target_dose_range' in list(param_configs.keys()):
                mobo_params.append({"name":'td_'+structure.name, 'type':'range', "bounds":param_configs['target_dose_range']})
            if param_configs['relative_weights']:
                mobo_params.append({"name":'w_'+structure.name, 'type':'fixed', "value":1000.0})
                continue
        mobo_params.append({"name":'w_'+structure.name, 'type':'range', "bounds":param_configs['weight_range']})

    return mobo_params

def generate_mobo_objectives(treatmentPlan:TreatmentPlan, param_configs:dict):
    r'''this function generates the objectives to be optimized by MOBO.
    inputs:
        - treatmentPlan:= an object that has all the structures to be iradiated. the structures must have the type "TPS_structure".
        - param_config:= a dictionary containing the field "relative_dvh_dose":bool. if this field is True, then 
            the objective thresholds are normalized to the target dose. 
    
    outputs:
        mobo_objectives:dict := a dictionary containing {name of DVH metric: ObjectiveProperties(minimize=, threshold=)}
   '''
    mobo_objectives={}
    for structure in treatmentPlan.structure_list:
        full_dvh_metric = structure.dvh_metric_name+'('+structure.name+')'
        threshold  = structure.dvh_metric_clinical_threshold
        # if DVH metric of the structure is to be normalized to the target dose, then threshold (goal) for this objective must be normalized
        if param_configs['relative_dvh_dose']:
            threshold = threshold/treatmentPlan.tumor_target_dose
        # dose to tumor volume ('tv') needs to be maximized
        if 'tv' in structure.name: 
            mobo_objectives[full_dvh_metric] = ObjectiveProperties(minimize=False, threshold=threshold) 
            continue
        # dose to the other structures need to be minimized
        mobo_objectives[full_dvh_metric] = ObjectiveProperties(minimize=True, threshold=threshold)
    return mobo_objectives


def ax_save_results(fileName:str, dictionary:dict):
    r'''This function saves any python dictionary in a .pkl file
    inputs:
        - fileName:= the name of the pickle file where the dictionary is written
        - dictionary:= the dictionary to be saved in fileName
    outputs:
        - a pickle file written to the fileName directory

    dependencies:
        - pickle.dump()
    '''
    if (".pkl" in fileName):
        with open(fileName, 'wb') as file:
            pickle.dump(dictionary, file)
    else:
        print(f"File name should have the .pkl extension. here is the file name at the moment \n {fileName}")

    return 0


def initialize_experiment(axClient:AxClient, treatment_plan:TreatmentPlan, num_fmio_runs:int):
    r''' In this function, we initialize the experiment object with a user-defined number of randomly generated penalty weight vectors
    inputs:
        - axClient := an ax client object that has been already initialized with the right parameters and objectives
        - treatment_plan := a treatment plan object contianing the evaluate_penaltyWeight_space() method
        - num_fmio_runs := number of fmio calls to be run in parallel

    outputs:
        - VOID := this method returns nothing, but it updates the axClient with the result of penalty weight evaluations

    dependencies:
        - AxClient
        - TreatmentPlan
    '''
    # sobol = Models.SOBOL(search_space=experiment.search_space)

    # obtain the parameter configuration from the AxClient.experiment object
    parameters=axClient.experiment.search_space.parameters

    # initialize a dataframe that will hold all the random penalty weight vectors 
    initial_weights_space = np.zeros([num_fmio_runs, len(parameters)])
    random_weight_space = pd.DataFrame(initial_weights_space, columns=list(parameters.keys()))
    # fill the dataframe with the weight vectors in the right range
    for parameter in parameters:
        try:
            random_weight_space[parameter] = np.random.uniform(low=parameters[parameter].lower, high = parameters[parameter].upper, size=num_fmio_runs)
        # in case you have a fixed parameter, set the value accordingly for all rows
        except: 
            random_weight_space[parameter] = np.ones(num_fmio_runs)*parameters[parameter].value

    # run penalty weight evaluation on all the weight vecotrs in parallel
    random_weight_dvh_space = treatment_plan.evaluate_penaltyWeight_space(random_weight_space, multi_thread=True)  
    # random_weight_dvh_space = random_weight_dvh_space.to_dict('records')
    # put the results into the axclient
    for indx in random_weight_dvh_space.index:
        axClient.attach_trial(random_weight_dvh_space.iloc[[indx]][list(parameters.keys())].to_dict('records')[0])
        axClient.complete_trial(indx, random_weight_dvh_space.iloc[[indx]][list(axClient.experiment.metrics.keys())].to_dict('records')[0])
    # for debugging{ in case you want to view the result of added trials uncomment the line below
    # axClient.get_trials_data_frame()
    # }


def run_mobo_iterations(
    num_iterations:int, 
    path2mps:str, 
    clinic_dvh_goal:dict, 
    param_config:dict,
    outputfile:str=None,
    num_parallel_initiation:int=None
    ):
    r'''This function runs multi-objective bayesian optimization (mobo) for as many iterations as the user commands it to. 
        it will save the result of all iterations in a pandas dataframe. 

    inputs:
        - num_iterations:= the number of desired mobo iterations
        - path2mps:= path to an mps file. this file can be generated by the RapidbrachyMCTPS.
            it contains the dose rate from each dwell position and the objective function used in FM
        - clinic_dvh_goal:= a dictionary holding the clinical goals for the dvh metrics below. these goals are obtained from clinical literature
            {"D90%(PTV)":, "D2cc(urethra)":, "D2cc(rectum)":, "D2cc(bladder)":}
        - param_config:= a dictionary containing the configuration of the parameter. it must have the following keys: 
            {
            'relative_weights':bool := if true, the penalty weight of the OARs is normalized to the penalty weight of tumor volume, 
            'weight_range':list := the minimal and maximal value for each weight. for example [0.001, 1],
            'relative_dvh_dose':bool := if true, the dose of the DVH metric is normalized to the target dose. 
                for example, if relative_Dvh_dose is true and the target dose is 15 Gy, then D90%(PTV) = 15/15 = 1,}
        - outputfile:= the path to a pickle file where the outcome of all iterations will be stored. 
        - num_parallel_initiation := if this number is specified, prallel fmio calls are used to initiate axClient.experiment
            by calling initialize_experiment()

    outputs:
        - results:dict := a dictionary containing many outcomes of the mobo iterations. 

    dependencies:
        - TreatmentPlan := an in f_weight_to_DVHmetric.py library, see imports
        - AxClient := a client object in the ax api (https://ax.dev/docs/api.html)
        - generate_mobo_parameters() := helper functio in library TreatmentPlan 
        - generate_mobo_objectives() := helper functio in library TreatmentPlan
        - perf_counter() := external function, see imports
        - get_MOO_NEHVI() := external function, see imports
        - observed_hypervolume() := external function, see imports
        - exp_to_df() := external function, see imports
        - ax_save_results() := external function, see imports
        - initialize_experiment() := runs FMIO calls "out of the loop" in parallel and adds the result to axClient.experiment
    '''
    ###########
    # setup necessary variables{
    # dimension of the problem (# of inputs)
    # d = num_params
    # setup data type and device if GPU is available, if not cpu
    # tkwargs = {
    #     "dtype": torch.double,
    #     "device": torch.device("cpu")
    #     # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # }
    # if fast true, hypervolume is not calculated.
    calc_hv = True
    # a list to keep track of the hypervolumes. 
    hv_list = []
    # }
    ###########
    # Start with caluclations

    # instantiate a penalty weight evaluator, which is the treatment plan object that runs dwell time optimization
    pwe = TreatmentPlan(path2mps, clinic_dvh_goal)
    # pwe = PenaltyWeightEvaluator(path_to_mps=path2mps, dvh_goals=clinic_goal).to(**tkwargs)

    # instantiate an Ax Client object

    generation_stratagey_forMOBO = GenerationStrategy(
        steps=[
            GenerationStep(
            model=Models.SOBOL, 
            # so, please set the second 1 to 8 before i forget... i may forget to put the 8 back :|
            num_trials=1 if num_parallel_initiation!=None else 1),

            GenerationStep(
            model=Models.MOO, 
            num_trials=-1)]
        )
    ax_client = AxClient(generation_strategy=generation_stratagey_forMOBO,) 
    # early_stopping_strategy=)
    # ax_client = AxClient()
    # constrain_PTV = OutcomeConstraint()

    # create parameter lists (these are the weights and their range)
    variable_parameters_forMOBO = generate_mobo_parameters(pwe, param_config)

    objectives_forMOBO = generate_mobo_objectives(pwe, param_config)

    ax_client.create_experiment(
        name="MOBO_penalty_weights",
        parameters= variable_parameters_forMOBO,
        objectives=objectives_forMOBO,
        overwrite_existing_experiment=True,
        is_test=False,
        )
    
    tic = perf_counter()

    if num_parallel_initiation != None:
        # fill out the ax_client experiment with the result of 100 randomly generated penalty weights 
        initialize_experiment(ax_client, pwe, num_parallel_initiation)
    
    # generator_run = generation_stratagey_forMOBO.gen(
    #     experiment=ax_client.experiment
        
    # )

    for i in range(num_iterations+1):
        parameters, trial_index = ax_client.get_next_trial()
        # local evaluation here can be replaced with deployment to external systems
        ax_client.complete_trial(trial_index=trial_index, raw_data=pwe.evaluate_penaltyWeight(parameters, param_config['relative_dvh_dose']))
        if calc_hv:
            try:
                current_model = get_MOO_NEHVI(ax_client.experiment, ax_client.experiment.fetch_data())
                hv = observed_hypervolume(modelbridge=current_model)
            except:
                hv = 0
                print("failed to compute hv")
            hv_list.append(hv)  

    toc = perf_counter()
    execution_time = toc - tic

    # extract the results and save them
    df = exp_to_df(ax_client.experiment).sort_values(by=["trial_index"])
        

    outcomes = df[list(objectives_forMOBO.keys())].values
    result = {'outcomes': outcomes, 'algorithm': 'qNEHVI',
    'df':df, 'pipeline':'AxClient', 'problem':pwe, 'hv_list': hv_list, 'time':execution_time, 'clinic_dvh_goal': clinic_dvh_goal,
     'num_mobo_iterations':len(pwe.structure_list)}

    if outputfile != None:
        ax_save_results(outputfile, result)

    return result

def _test_find_ideal_parameter_configuration():
    mpsdir = "../data_files/mpsFiles/jgh-prostate/lin_target_dose_to_ptv/"
    output_folder = "../data_files/figures and tables/prostate/iteration_experiment/"
    dvh_thresholds = {"D90%(ptv)":15, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25}

    find_ideal_parameter_configuration(mpsDir=mpsdir, clinical_dvh_goal=dvh_thresholds, output_folder=output_folder)

def find_ideal_parameter_configuration(mpsDir, clinical_dvh_goal, output_folder, num_experiment_repetition=1):
    r''' This function shows what parameter configuration allows MOBO to find the pareto surface 
            with as little iterations as possible. The configurations that it's going to test are:
            0. all weights full range
            1. normalized weights full range
            2. normalized weights, half range
            3. normalized weights, half range, target dose range

        Each configuration above is tested 10 times for the same iteration and patient. we keep 
            track of the mean and std time and acceptance rate after 10 itrations are done per patient.

        By default, we do 8 parallel initialization and start counting MOBO iterations after  
            initialization. 
    '''
    # (columns=['iteration', 'patient', 'mean time (s)', 'std time (s)',
    # 'mean acceptance rate (%)', 'std acceptance rate (%)'])

    # let's create the big dataframe for parameter configuration 0 above
    outcome_df_0 = pd.DataFrame()
    # let's create the big dataframe for parameter configuration 1 above
    outcome_df_1 = pd.DataFrame()
    # let's create the big dataframe for parameter configuration 2 above
    outcome_df_2 = pd.DataFrame()
    # let's create the big dataframe for parameter configuration 3 above
    outcome_df_3 = pd.DataFrame()


    mps_files = glob.glob(mpsDir+"*.mps")

    iter_array = np.array([5, 10, 20, 30, 40, 50, 60,])
    for patient in mps_files:
        # # for debugging{ to check if a certain patient is problematic
        # if '5' not in patient:
        #     continue
        # # # }
        for mobo_iter in iter_array:
        # for debugging{ to check if a certain iteration is problematic
            # if mobo_iter != 60:
            #     continue
        # }
            # get patient number
            patient_number = int(re.findall(r'\d+', patient.split("/")[-1].split(".")[0])[0])
            # prepare the numpy arrays that will hold time and acceptance rate for 10 repetittion 
            #   of the same mobo iteration. 1 column per parameter configuration.
            time_array = np.zeros([num_experiment_repetition, 4])
            acceptance_rate_array= np.zeros([num_experiment_repetition, 4])

            # repeat the following experiments 10 times
            try: 
                for i in range(num_experiment_repetition):
                    # break
                    # Testing on the case 0 of the parameter configuration above.
                    results_0 = run_mobo_iterations(
                        num_iterations=mobo_iter, 
                        path2mps=patient, 
                        clinic_dvh_goal=clinical_dvh_goal, 
                        param_config={'relative_weights': False, 
                        'weight_range': [1.0, 1000.0], 
                        'relative_dvh_dose': False})

                    # Testing on the case 1 of the parameter configuration above
                    results_1 = run_mobo_iterations(
                        num_iterations=mobo_iter, 
                        path2mps=patient, 
                        clinic_dvh_goal=clinical_dvh_goal, 
                        param_config={'relative_weights': True, 
                        'weight_range': [1.0, 1000.0], 
                        'relative_dvh_dose': False})

                    # Testing on the case 2 of the parameter configuration above
                    results_2 = run_mobo_iterations(
                        num_iterations=mobo_iter, 
                        path2mps=patient, 
                        clinic_dvh_goal=clinical_dvh_goal, 
                        param_config={'relative_weights': True, 
                        'weight_range': [1.0, 500.0], 
                        'relative_dvh_dose': False})

                    # Testing on the case 3 of the parameter configuration above
                    results_3 = run_mobo_iterations(
                        num_iterations=mobo_iter, 
                        path2mps=patient, 
                        clinic_dvh_goal=clinical_dvh_goal, 
                        param_config={'relative_weights': True, 
                        'weight_range': [1.0, 500.0], 
                        'relative_dvh_dose': False,
                        'target_dose_range':[15.0, 16.5]})

                    # let's isolate the time and success rate from the results_0
                    time_array[i] = [results_0["time"], results_1['time'], results_2['time'], results_3['time']]
                    acceptance_rate_array[i] = [count_success_rate(results=results_0), count_success_rate(results=results_1), count_success_rate(results=results_2), count_success_rate(results=results_3)]
            except:
                with open('failed_cases.txt', 'w') as file:
                    file.write(f"patient {patient_number}, iteration {mobo_iter}")    
                    continue
            # let's record the result each iteration experiment for this patient in the
            outcome_df_0=outcome_df_0.append({
                'number of iterations':mobo_iter,
                'patient':patient_number,
                'mean time (s)': np.mean(time_array[:, 0]),
                'std time (s)': np.std(time_array[:, 0]),
                'mean acceptance rate (%)': np.mean(acceptance_rate_array[:, 0]),
                'std acceptance rate (%)': np.std(acceptance_rate_array[:, 0])
            }, ignore_index=True)

            outcome_df_1=outcome_df_1.append({
                'number of iterations':mobo_iter,
                'patient':patient_number,
                'mean time (s)': np.mean(time_array[:, 1]),
                'std time (s)': np.std(time_array[:, 1]),
                'mean acceptance rate (%)': np.mean(acceptance_rate_array[:, 1]),
                'std acceptance rate (%)': np.std(acceptance_rate_array[:, 1])
            }, ignore_index=True)

            outcome_df_2=outcome_df_2.append({
                'number of iterations':mobo_iter,
                'patient':patient_number,
                'mean time (s)': np.mean(time_array[:, 2]),
                'std time (s)': np.std(time_array[:, 2]),
                'mean acceptance rate (%)': np.mean(acceptance_rate_array[:, 2]),
                'std acceptance rate (%)': np.std(acceptance_rate_array[:, 2])
            }, ignore_index=True)

            outcome_df_3=outcome_df_3.append({
                'number of iterations':mobo_iter,
                'patient':patient_number,
                'mean time (s)': np.mean(time_array[:, 3]),
                'std time (s)': np.std(time_array[:, 3]),
                'mean acceptance rate (%)': np.mean(acceptance_rate_array[:, 3]),
                'std acceptance rate (%)': np.std(acceptance_rate_array[:, 3])
            }, ignore_index=True)

            outcome_df_0.to_csv( output_folder+ "mobo_param_tuning_allWeights_fullRange.csv")
            outcome_df_1.to_csv( output_folder+ "mobo_param_tuning_normalizedWeights_fullRange.csv")
            outcome_df_2.to_csv( output_folder+ "mobo_param_tuning_normalizedWeights_halfRange.csv")
            outcome_df_3.to_csv( output_folder+ "mobo_param_tuning_normalizedWeights_halfRange_variableTargetDose.csv")
    
    return 0

def mobo_constantIterations_on_patients():
    r''' this function runs user defined number of iterations of MOBO on many patients.
    ''' 
    # for prostate patients in JGH
    # input_folder = "../data_files/mpsFiles/jgh-prostate/quad_target_dose_to_ptv/"
    # output_folder = "../data_files/MOBO/patients/finalProstate_quadFMIO_rtogPlusGoals_absoluteDVHdose/"
    # dvh_thresholds = {"D95%(ptv)":15.0, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25}
    # param_config = {'relative_weights':True, 'weight_range':[1.0, 500.0], 'relative_dvh_dose':False, 'target_dose_range':[15.0, 16.5]}

    # for prostate patients in Glen. no bladder, some patients do not have body contours
    input_folder = "../data_files/mpsFiles/prostate-glen/"
    output_folder = "../data_files/MOBO/patients/prostate_Glen_quadFmio_rtogPlus/"
    dvh_thresholds = {"D95%(ctv)":15.0, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25}
    param_config = {'relative_weights':True, 'weight_range':[1.0, 500.0], 'relative_dvh_dose':False, 'target_dose_range':[15.0, 16.5]}

    # for gyn patients
    # input_folder = "../data_files/mpsFiles/gyn-alana/"
    # output_folder = "../data_files/MOBO/patients/gynAlana_linearFMIO_rtogGoals_absoluteDVHdose/"
    # dvh_thresholds = {"D90%(ctv)":8, "D0.1cc(urethra)":4.6, "D2cc(rectum)":6.5, "D2cc(bowel)":7, "D2cc(bladder)":8}
    # param_config = {'relative_weights':True, 'weight_range':[0.001,0.5], 'relative_dvh_dose':False}

    # for breast patients
    # input_folder = "/home/majd/data/Patient_mpsFiles/Sebastien-breast/"
    # output_folder = "../data_files/MOBO/patients/breastSebastien_linearFMIO_rtogGoals_absoluteDVHdose/"
    # dvh_thresholds = {"D95%(ptv)":3., "D1cc(skin)":2.7, "D0.1cc(rib)":2.7, "D0.1cc(heart)":2.7, "D0.1cc(lung)":1.8}
    # param_config = {'relative_weights':True, 'weight_range':[0.001,0.5], 'relative_dvh_dose':False, 'target_dose_range':[3., 3.3]}

    mps_files = glob.glob(input_folder+"*.mps")

    for file in mps_files:
        output_name = "halfRange_targetDoseIsParam_parallelInit" + file.split('/')[-1].split('.')[0] + '.pkl'
        output_name = output_folder+output_name
        # print(output_name)
        run_mobo_iterations(6, file, dvh_thresholds, param_config, output_name, num_parallel_initiation=24)

    return 0


def _test_generate_mobo_parameters():
    # testing on a prostate cases{:
    path2mps = "../data_files/mpsFiles/jgh-prostate/lin_target_dose_to_ptv/p1.mps"
    dvh_thresholds = {"D95%(ptv)":15, "D1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25}
    param_config = {'relative_weights':True, 'weight_range':[0,0.5], 'target_dose_range':[15, 16.5], 'relative_dvh_dose':True}
    # param_config = {'relative_weights':True, 'weight_range':[0,0.5], 'relative_dvh_dose':False}
    # }

    # for gyn patients
    # path2mps = "../data_files/mpsFiles/gyn-alana/gyn-test.mps"
    # dvh_thresholds = {"D90%(CTV)":6, "D2cc(urethra)":4, "D2cc(rectum)":4, "D2cc(bowel)":2, "D2cc(bladder)":4}
    # # param_config = {'relative_weights':False, 'weight_range':[0,1], 'relative_dvh_dose':False}
    # param_config = {'relative_weights':False, 'weight_range':[0.001,0.5], 'relative_dvh_dose':False, 'target_dose_range':[6, 6.6]}


    treatment_plan = TreatmentPlan(path2mps, dvh_thresholds)

    mobo_params = generate_mobo_parameters(treatment_plan, param_config)
    print(mobo_params)

def _test_run_mobo_iterations():
    num_mobo_iter = 30
    num_init = 24
    # testing on a prostate cases:
    path2mps = "../data_files/mpsFiles/jgh-prostate/lin_target_dose_to_ptv/p1.mps"
    dvh_thresholds = {"D90%(PTV)":15, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25}
    param_config = {'relative_weights':True, 'weight_range':[0.001, 0.5], 'relative_dvh_dose':False, 'target_dose_range':[15.0, 16.5]}

    # testing on a gyn case:
    # path2mps = "../data_files/mpsFiles/gyn.mps"
    # dvh_thresholds = {"D90%(CTV)":6, "D2cc(urethra)":4, "D2cc(rectum)":4, "D2cc(bowel)":2, "D2cc(bladder)":4}
    # param_config = {'relative_weights':True, 'weight_range':[0.001,5], 'relative_dvh_dose':False}

    results = run_mobo_iterations(num_mobo_iter, path2mps, dvh_thresholds, param_config,)
    results['df'].to_csv("test_parallel_init.csv")

    # results2 = run_mobo_iterations(num_iter, path2mps, dvh_thresholds, param_config, parallel_initiation=False)
    # results2['df'].to_csv("test.csv")

    print("done testing")

def _test_initialize_experiments():
    num_fmio = 5
    # testing on a prostate cases:
    path2mps = "../data_files/mpsFiles/jgh-prostate/lin_target_dose_to_ptv/p1.mps"
    dvh_thresholds = {"D90%(PTV)":15, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25}
    param_config = {'relative_weights':True, 'weight_range':[0.001, 1.0], 'relative_dvh_dose':False, 'target_dose_range':[15.0, 16.5]}

    pwe = TreatmentPlan(path2mps, dvh_thresholds)
    # pwe = PenaltyWeightEvaluator(path_to_mps=path2mps, dvh_goals=clinic_goal).to(**tkwargs)

    # instantiate an Ax Client object
    ax_client = AxClient()
    # constrain_PTV = OutcomeConstraint()

    # creat parameter lists (these are the weights and their range)
    variable_parameters_forMOBO = generate_mobo_parameters(pwe, param_config)

    objectives_forMOBO = generate_mobo_objectives(pwe, param_config)

    ax_client.create_experiment(
        name="MOBO_penalty_weights",
        parameters= variable_parameters_forMOBO,
        objectives=objectives_forMOBO,
        overwrite_existing_experiment=True,
        is_test=False,
        )

    initialize_experiment(ax_client, pwe, num_fmio)

if __name__=="__main__":
    
    # let's get testing
    # _test_run_mobo_iterations()             # test passed!
    # _test_generate_mobo_parameters()        # test passed!
    # _test_initialize_experiments()          # test passed!

    # let's run real experiments:
    find_ideal_parameter_configuration(
        # this is for prostate JGH 
        # mpsDir = "../data_files/mpsFiles/jgh-prostate/quad_target_dose_to_ptv/",
        # clinical_dvh_goal = {"D90%(ptv)":15, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25, "D1cc(bladder)":11.25},
        # output_folder = "../data_files/figures and tables/prostate/iteration_experiment/quad_fmio/"

        # this is for prostate glen
        mpsDir = "../data_files/mpsFiles/prostate-glen/",
        clinical_dvh_goal = {"D95%(ctv)":15, "D0.1cc(urethra)":18.75, "D1cc(rectum)":11.25},
        output_folder = "../data_files/figures and tables/prostate-glen/iteration_experiment/quadFmio_rtogPlus/"
    )
    mobo_constantIterations_on_patients()

    
    