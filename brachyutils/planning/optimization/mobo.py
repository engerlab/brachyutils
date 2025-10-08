r'''
Date 
    2025/10/08

Purpose
    This script runs various experiments with penalty weight optimization using multi objective optimization (MOBO).
        MOBO is implemented by ax api and must be installed, see (https://ax.dev/docs/api.html) for directions
   
Author
    Sébastien Quetin, work adapted from Hossein Jafarzadeh's previous work on MOBO for prostate brachytherapy,
        Enger lab
        McGill University 
'''
import copy 
import glob
import os 
import re
from time import perf_counter

import numpy as np
import pandas as pd

from ax.adapter.adapter_utils import observed_hypervolume
from ax.adapter.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from brachyutils.planning.optimization.optim_utils import BrachyDwellTimeOptim, Optimization_Config


def generate_mobo_parameters(optimization_config_list:list[Optimization_Config], param_configs:dict):
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
    for optim_config in optimization_config_list:
        # the weight of the tumor volume (tv) is always 1 when using relative weights
        if 'tv' in optim_config.structure_name.lower():
            if 'target_dose_range' in list(param_configs.keys()):
                mobo_params.append({"name":'td_'+optim_config.structure_name, 'type':'range', "bounds":param_configs['target_dose_range']})
            for term in ["uniformity", "hotspot"]:
                pass
                # mobo_params.append({"name":term+ '_w_'+optim_config.structure_name, 'type':'range', "bounds":[0., 100.]})
        for term in ["linear", "quadratic"]:
            if "linear" in term:
                bds = param_configs['weight_range']
                type_ = "range"
                k = "bounds"
            else:
                bds = 1.
                type_ = "fixed"
                k = "value"

            if 'tv' in optim_config.structure_name.lower():
                if param_configs['relative_weights']:
                    if "quadratic" in term:
                        m = 1.
                    else:
                        m = 1000.
                    mobo_params.append({"name":term+ '_w_'+optim_config.structure_name, 'type':'fixed', "value":m}) # 'fixed', "value":m}) # 'range', 'bounds':[0., m]})
                    continue
            mobo_params.append({"name":term+ '_w_'+optim_config.structure_name, 'type':type_, k:bds})

    return mobo_params

def generate_mobo_objectives(dvh_metric_goals:dict, param_configs:dict, target_dose:float):
    r'''this function generates the objectives to be optimized by MOBO.
    inputs:
        - treatmentPlan:= an object that has all the structures to be iradiated. the structures must have the type "BrachyStructure".
        - param_config:= a dictionary containing the field "relative_dvh_dose":bool. if this field is True, then 
            the objective thresholds are normalized to the target dose. 
    
    outputs:
        mobo_objectives:dict := a dictionary containing {name of DVH metric: ObjectiveProperties(minimize=, threshold=)}
   '''
    mobo_objectives={}
    for dvh_metric_name, goal in dvh_metric_goals.items():
        full_dvh_metric = dvh_metric_name
        threshold  = target_dose
        # if DVH metric of the structure is to be normalized to the target dose, then threshold (goal) for this objective must be normalized
        if param_configs['relative_dvh_dose']:
            threshold = threshold/target_dose
        # dose to tumor volume ('tv') needs to be maximized
        if 'tv' in dvh_metric_name.lower(): 
            if "CI" in full_dvh_metric or "HI" in full_dvh_metric:
                pass
            mobo_objectives[full_dvh_metric] = ObjectiveProperties(minimize=False, threshold=50. if "V100" in full_dvh_metric else 1) 
            continue
        # dose to the other structures need to be minimized
        mobo_objectives[full_dvh_metric] = ObjectiveProperties(minimize=True, threshold=threshold)


    # Args for ObjectiveProperties:
    #     - minimize: Boolean indicating whether the objective is to be minimized
    #         or maximized.
    #     - threshold: Optional `float` representing the smallest objective value
    #         (resp. largest if minimize=True) that is considered valuable in the context
    #         of multi-objective optimization. In BoTorch and in the literature, this is
    #         also known as an element of the reference point vector that defines the
    #         hyper-volume of the Pareto front.

    return mobo_objectives
    
def work(brachy_optim, weight):
    local_copy = copy.deepcopy(brachy_optim)
    return local_copy.evaluate_penaltyWeight(weight)

def evaluate_penaltyWeight_space_helper(brachy_optim:BrachyDwellTimeOptim, weight_space:pd.DataFrame, multiprocessing:bool=True, out_folder:str = None):
    r''' This function evaluates a space of penalty weights in parallel using multiprocessing library. 
    inputs:
        - brachy_optim := an optimizer object that can reoptimize and evaluate dvh metrics
        - weight_space := a pandas dataframe containing the penalty weight vectors to be evaluated. 
            each row is a weight vector, and each column is a structure name with the prefix "w_"
        - multi_thread := if true, the evaluation is done in parallel using multiprocessing library
    outputs:

        - a pandas dataframe containing the result of penalty weight evaluations.
    '''
    weight_space_list_of_d = weight_space.to_dict('records')

    if multiprocessing:
        
        # The BrachyOptim object cannot be pickled due to all the internal attributes 
        # like gurobi variables that are not picklable. So, we need to use a workaround here.
        # The BrachyOptim class itself should provide a method to evaluate penalty weights in parallel. 
        weights_and_dvh_space = brachy_optim.evaluate_penaltyWeight_space(
            weight_space_list_of_d, 
            out_folder=out_folder
        )
        return pd.DataFrame(weights_and_dvh_space)
    else:

        results = {}
        for i, w_space in enumerate(weight_space_list_of_d):
            res = brachy_optim.evaluate_penaltyWeight(w_space)
            res.update(w_space)
            results[i] = res
        return pd.DataFrame.from_dict(results, orient='index')

  

def initialize_experiment(axClient:AxClient, brachy_optim:BrachyDwellTimeOptim, num_fmio_runs:int, out_folder:str = None):
    r''' In this function, we initialize the experiment object with a user-defined number of randomly generated penalty weight vectors
    inputs:
        - axClient := an ax client object that has been already initialized with the right parameters and objectives
        - brachy_optim := an optimizer object that can reoptimize and evaluate dvh metrics
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
    random_weight_dvh_space = evaluate_penaltyWeight_space_helper(brachy_optim, random_weight_space, out_folder=out_folder) # , multi_thread=True)  
    # random_weight_dvh_space = random_weight_dvh_space.to_dict('records')
    # put the results into the axclient
    for indx in random_weight_dvh_space.index:
        axClient.attach_trial(random_weight_dvh_space.iloc[[indx]][list(parameters.keys())].to_dict('records')[0])
        axClient.complete_trial(indx, random_weight_dvh_space.iloc[[indx]][list(axClient.experiment.metrics.keys())].to_dict('records')[0])
    # for debugging{ in case you want to view the result of added trials uncomment the line below
    # axClient.get_trials_data_frame()
    # }


def run_mobo_iterationsv2(
    num_iterations:int, 
    brachy_optim:BrachyDwellTimeOptim, 
    target_dose:float,
    optimization_config_list:list[Optimization_Config],
    clinic_dvh_goal:dict, 
    param_config:dict,
    output_filename:str=None,
    num_parallel_initiation:int=None, 
    variable_parameters_forMOBO=None,
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
    # instantiate an Ax Client object

    generation_stratagey_forMOBO = GenerationStrategy(
        steps=[
            GenerationStep(
                ## This is quasi random sampling, good for initial exploration of the space
                # https://ax.dev/docs/0.5.0/tutorials/multiobjective_optimization/
                generator=Generators.SOBOL,
                # so, please set the second 1 to 8 before i forget... i may forget to put the 8 back :|
                num_trials=1 if num_parallel_initiation!= None else min(int(num_iterations/2), 8),
            ),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,  # Use this for multi-objective optimization #Generators.MOO,
                num_trials=-1,
            )
        ]
    )
    ax_client = AxClient(generation_strategy=generation_stratagey_forMOBO,)

    # create parameter lists (these are the weights and their range)
    variable_parameters_forMOBO = generate_mobo_parameters(optimization_config_list, param_config)

    objectives_forMOBO = generate_mobo_objectives(clinic_dvh_goal, param_config, target_dose)

    print("Created experiment for MOBO with parameters:")
    print("variable_parameters_forMOBO", variable_parameters_forMOBO)
    print("objectives_forMOBO", objectives_forMOBO)

    filter_obj = {}
    for k, v in objectives_forMOBO.items():
        if k in ['V100%(PTV)', 'D0.1cc(Skin)', 'D0.1cc(Chestwall)']: # , 'CI(PTV)', 'D0.1cc(Skin)', 'D0.1cc(Chestwall)']: , 'CI(PTV)'
            filter_obj[k] = v
    filtered_var = []
    for p in variable_parameters_forMOBO:
        if p['name'] in [ 'td_PTV','quadratic_w_Skin', 'quadratic_w_Chestwall']: #, 'linear_w_Skin', 'linear_w_Chestwall'
            continue
        else:
            filtered_var.append(p)
    print("filtered_var", filtered_var)
    print("filter_obj", filter_obj)

    ax_client.create_experiment(
        name="MOBO_penalty_weights",
        parameters=filtered_var, # variable_parameters_forMOBO,
        objectives=filter_obj, # objectives_forMOBO,
        overwrite_existing_experiment=True,
        is_test=False,
        )
    
    tic = perf_counter()
    tac = None
    if num_parallel_initiation != None:
        # fill out the ax_client experiment with the result of 100 randomly generated penalty weights 
        initialize_experiment(ax_client, brachy_optim, num_parallel_initiation, out_folder = os.path.dirname(output_filename) if output_filename else None)
        tac = perf_counter()
        print(f"Time taken for {num_parallel_initiation} parallel initialization: {tac - tic:0.4f} seconds")

    optimized_plans = {}
    for i in range(num_iterations+1):
        parameters, trial_index = ax_client.get_next_trial()
        dvh_metrics, optimized_plan = brachy_optim.evaluate_penaltyWeight(parameters, param_config['relative_dvh_dose'], return_plan=True)
        optimized_plans[trial_index] = optimized_plan
        # local evaluation here can be replaced with deployment to external systems
        ax_client.complete_trial(trial_index=trial_index, raw_data=dvh_metrics)
        if calc_hv:
            try:
                current_model = Generators.MOO(
                    experiment=ax_client.experiment,
                    data=ax_client.experiment.fetch_data(),
                    search_space=ax_client.experiment.search_space
                    )
                
                hv = observed_hypervolume(modelbridge=current_model)
            except:
                hv = 0
                print("failed to compute hv")
            hv_list.append(hv)  

    toc = perf_counter()
    if tac is not None:
        print(f"Time taken for {num_parallel_initiation} parallel initialization: {tac - tic:0.4f} seconds")
        print(f"Time taken for {num_iterations} MOBO iterations after initialization: {toc - tac:0.4f} seconds")
    else:
        print(f"Time taken for {num_iterations} MOBO iterations: {toc - tic:0.4f} seconds")

    df = ax_client.get_trials_data_frame().sort_values(by=["trial_index"])
    print(df)
    print("saving the result of mobo iterations to ", output_filename)
    # df.to_excel(output_filename) 
    df.to_csv(output_filename, index=False)
    return df, optimized_plans

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
