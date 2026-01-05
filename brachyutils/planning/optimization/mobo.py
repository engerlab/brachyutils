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
from time import perf_counter
from typing import List 
import copy 

import numpy as np
import pandas as pd
import torch

from ax.adapter.adapter_utils import observed_hypervolume
from ax.adapter.registry import Generators
from ax.core.utils import get_pending_observation_features
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from brachyutils.planning.optimization.optim_utils import BrachyDwellTimeOptim, Optimization_Config

class MOBOOptimizer:
    """
    Class intended to run multi-objective bayesian optimization (MOBO) using ax api.
    it requires a BrachyDwellTimeOptim object to evaluate the penalty weights suggested by
    the ax api.
    
    """
    def __init__(self,
                 brachy_optim:BrachyDwellTimeOptim, 
                 target_dose:float,
                 optimization_config_list:list[Optimization_Config],
                 clinic_dvh_goal:dict, 
                 ):

        self.brachy_optim = brachy_optim
        self.target_dose = target_dose
        self.optimization_config_list = optimization_config_list
        self.clinic_dvh_goal = clinic_dvh_goal
        self.target_structure_name = None  # to be defined in child class
       
    def generate_mobo_parameters(self, **kwargs) -> List[dict]:
        r''' generate a list of paremeters that ax api can interface with
        Please overwrite this function in the child class that should be site specific.
        i.e. BreastMOBOOptimizer, ProstateMOBOOptimizer, etc.

        outputs:
            - a list of parameters that ax api can interface with
        '''
        pass

    def generate_mobo_objectives(self, **kwargs) -> List[ObjectiveProperties]:
        r'''this function generates the objectives to be optimized by MOBO.
        Please overwrite this function in the child class that should be site specific.
        i.e. BreastMOBOOptimizer, ProstateMOBOOptimizer, etc.

        outputs:
            - a list of objectives that ax api can interface with
        '''
        # Args for ObjectiveProperties:
        #     - minimize: Boolean indicating whether the objective is to be minimized
        #         or maximized.
        #     - threshold: Optional `float` representing the smallest objective value
        #         (resp. largest if minimize=True) that is considered valuable in the context
        #         of multi-objective optimization. In BoTorch and in the literature, this is
        #         also known as an element of the reference point vector that defines the
        #         hyper-volume of the Pareto front.
        pass
    
    def get_optim_config_and_target_dose_from_parameters(self, penalty_weights:dict):
        r''' this function updates the optimization configuration list with the new penalty weights
        inputs:
            - penalty_weights := a dictionary containing the penalty weights for each structure.
                the keys are the structure names, and the values are the penalty weights.
        outputs:
            - VOID := this method returns nothing, but it updates the optimization_config_list attribute of the class.
        '''
        new_opt_config_l = copy.deepcopy(self.optimization_config_list)

        for key, val in penalty_weights.items():
            structure_name = key.split("_")[-1]
            attr_name = key.replace(f"_{structure_name}", "")
            for opt_conf in new_opt_config_l:
                if opt_conf.structure_name == structure_name:
                    setattr(opt_conf, attr_name, val)
        
        #Find the target config to copy to hotspot if any
        target_config = None
        for opt_conf in new_opt_config_l:
            if opt_conf.structure_name == self.target_structure_name:
                target_config = opt_conf
                break
        assert target_config is not None, (
            "Target structure optimization config not found in the optimization_config_list."
        )
        # Modifying the hotspot structures if any
        for key, val in penalty_weights.items():
            structure_name = key.split("_")[-1]
            if structure_name.lower() != self.target_structure_name.lower():
                continue
            attr_name = key.replace(f"_{structure_name}", "")
            for opt_conf in new_opt_config_l:
                if "hotspot_estimator" in opt_conf.structure_name.lower():
                    setattr(opt_conf, attr_name, val)

        for opt_conf in new_opt_config_l:
            if opt_conf.structure_name == self.target_structure_name:
                new_target_dose = opt_conf.dose_voxel_goal
        return new_opt_config_l, new_target_dose

    def evaluate_penaltyWeight_space_helper(
            self, list_of_opt_config_lists:List[List[Optimization_Config]], 
            list_of_target_doses:List[float], num_parallel_iterations:int,
            list_of_experiment_indexes:List[int]):
        r''' This function evaluates a space of penalty weights in parallel using multiprocessing library. 
        inputs:
            - list_of_opt_config_lists := a list of optimization configuration lists to be evaluated.
            - list_of_target_doses := a list of target doses corresponding to each optimization configuration list
            - num_parallel_iterations := number of parallel iterations to run
            - list_of_experiment_indexes := a list of experiment indexes corresponding to each optimization configuration list
        outputs:

            - a pandas dataframe containing the result of penalty weight evaluations.
        '''
        assert len(list_of_opt_config_lists) == len(list_of_target_doses), (
            "The length of list_of_opt_config_lists must be equal to the length of list_of_target_doses."
        )
        assert list_of_experiment_indexes is not None, (
            "list_of_experiment_indexes must be provided."
        )
        if num_parallel_iterations > 1:
            
            # The BrachyOptim object cannot be pickled due to all the internal attributes 
            # like gurobi variables that are not picklable. So, we need to use a workaround here.
            # The BrachyOptim class itself should provide a method to evaluate penalty weights in parallel.
            weights_and_dvh_space, cat_tables = self.brachy_optim.evaluate_penaltyWeight_space(
                list_of_opt_config_lists,
                list_of_target_doses,
                list_of_experiment_indexes,
                return_cat_table=True,
                max_parallel_runs=num_parallel_iterations
            )
            return pd.DataFrame(weights_and_dvh_space), cat_tables
        else:

            results = {}
            cat_tables = {}
            for i, (opt_config_list, target_dose, exp_indx) in enumerate(
                zip(list_of_opt_config_lists, list_of_target_doses,list_of_experiment_indexes)):
                optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(opt_config_list)
                res, cat_table = self.brachy_optim.evaluate_penaltyWeight(optim_config_list, target_dose, return_cat_table=True)
                res["trial_index"] = exp_indx
                results[i] = res
                cat_tables[f"trial_{exp_indx}"] = cat_table
            return pd.DataFrame.from_dict(results, orient='index'), cat_tables

  

    def initialize_experiment(self, axClient:AxClient, num_random_initiations:int, num_parallel_iterations:int):
        r''' In this function, we initialize the experiment object with a user-defined number of randomly generated penalty weight vectors
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - num_random_initiations := number of fmio calls to be run in parallel
            - num_parallel_iterations := number of parallel iterations to run

        outputs:
            - VOID := this method returns nothing, but it updates the axClient with the result of penalty weight evaluations

        dependencies:
            - AxClient
            - TreatmentPlan
        '''

        # obtain the parameter configuration from the AxClient.experiment object
        parameters=axClient.experiment.search_space.parameters

        # initialize a dataframe that will hold all the random penalty weight vectors 
        initial_weights_space = np.zeros([num_random_initiations, len(parameters)])
        random_weight_space = pd.DataFrame(initial_weights_space, columns=list(parameters.keys()))

        # fill the dataframe with the weight vectors in the right range
        for parameter in parameters:
            try:
                random_weight_space[parameter] = np.random.uniform(
                    low=parameters[parameter].lower, high = parameters[parameter].upper, size=num_random_initiations)
            # in case you have a fixed parameter, set the value accordingly for all rows
            except: 
                random_weight_space[parameter] = np.ones(num_random_initiations)*parameters[parameter].value

        # run penalty weight evaluation on all the weight vecotrs in parallel
        list_of_opt_config_lists = []
        list_of_target_doses = []
        for index, config_params in random_weight_space.iterrows():
            optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(config_params)
            list_of_opt_config_lists.append(optim_config_list)
            list_of_target_doses.append(target_dose)
        
        list_of_experiment_indexes = range(len(list_of_target_doses))
        random_weight_dvh_space, cat_tables = self.evaluate_penaltyWeight_space_helper(
            list_of_opt_config_lists=list_of_opt_config_lists, 
            list_of_target_doses=list_of_target_doses, 
            list_of_experiment_indexes=list_of_experiment_indexes,
            num_parallel_iterations=num_parallel_iterations)

        # put the results into the axclient
        for indx in list_of_experiment_indexes:
            sub_df = random_weight_dvh_space[random_weight_dvh_space['trial_index'] == indx]
            axClient.attach_trial(sub_df[list(parameters.keys())].to_dict('records')[0])
            axClient.complete_trial(
                trial_index=indx, 
                raw_data=sub_df[list(axClient.experiment.metrics.keys())].to_dict('records')[0]
                )
        # for debugging{ in case you want to view the result of added trials uncomment the line below
        # axClient.get_trials_data_frame()
        # }
        return random_weight_dvh_space, cat_tables


    def run(self,
            num_mobo_iterations:int, 
            num_random_initiation:int=5, 
            num_parallel_iterations:int=1,
            output_filename:str=None,
            calc_hv:bool=False,
            mobo_objective_kwargs:dict=None, 
            mobo_parameter_kwargs:dict=None
            ):
        r'''This function runs multi-objective bayesian optimization (mobo) for as many iterations as the user commands it to. 
            it will save the result of all iterations in a pandas dataframe. 
        
        WARNING: Inplace is True in the different calls to the optimization function, Hence 
        the original plan will be modified!!! By returning the catheter table, one can reconstruct 
        the optimized plan afterwards by manually setting the catheter_table attribute of the 
        BrachyPlan object and then calling the update_plan_from_catheter_table() method.

        inputs:
            - num_mobo_iterations:= the number of desired mobo iterations
            - num_random_initiation := fmio calls are used to initiate axClient.experiment
                If num_parallel_iterations > 1, these runs are completely random and run in parallel with the initialize_experiment()
                function. If not, the runs are initiated one by one inside the mobo loop with the semirandom sobol generator 
                of ax api. Must be greater than 0, otherwise qNoisyExpectedImprovement will not be able to search for next 
                trial.
            - output_filename:= the name of the file to save the result of mobo iterations.
            - calc_hv:= if true, the hypervolume of the pareto front is calculated at each iteration and returned as a list.
            - mobo_objective_kwargs:= a dictionary containing the keyword arguments for the generate_mobo_objectives() method.
            - mobo_parameter_kwargs:= a dictionary containing the keyword arguments for the generate_mobo_parameters
        outputs:
            - results:dict := a dictionary containing many outcomes of the mobo iterations. 
            - optimized_cat_tables:dict := a dictionary containing the optimized catheter tables for each iteration.
                the keys are the iteration number and the values are the catheter tables.
            - hv_list:list := a list containing the hypervolume of the pareto front at each iteration.
            - mobo_time:float := the time taken to run the mobo iterations in seconds.
            - parallel_init_time:float := the time taken to run the parallel initialization in seconds

        '''
        # a list to keep track of the hypervolumes. 
        hv_list = []

        assert num_random_initiation > 0, (
            "num_random_initiation must be greater than 0, otherwise qNoisyExpectedImprovement"\
            " will not be able to search for next trial."
        )

        steps=[]
        
        # Random init runs are parallelized on the CPU while q-noisy search is done on GPU if available
        # Check if CUDA is available and set the device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available for MOBO. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA not available for MOBO. Using CPU.")

        if num_parallel_iterations <= 1:
            steps.append(
                GenerationStep(
                    ## This is quasi random sampling, good for initial exploration of the space
                    # https://ax.dev/docs/0.5.0/tutorials/multiobjective_optimization/
                    generator=Generators.SOBOL,
                    num_trials=num_random_initiation,
                    model_kwargs={
                        "torch_device": device,
                    }
                )
            )
            # The loop though ax client will add num_mobo_iterations to the initial num_random_initiation
            total_ax_iterations = num_mobo_iterations + num_random_initiation
        else:
            total_ax_iterations = num_mobo_iterations

        steps.append(GenerationStep(
            ## Refer to 
            # S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy.
            # Unexpected Improvements to Expected Improvement for Bayesian Optimization. Advances
            # in Neural Information Processing Systems 36, 2023.
            # https://github.com/meta-pytorch/botorch/blob/main/botorch/acquisition/logei.py#L239
            generator=Generators.BOTORCH_MODULAR,  # Use this for multi-objective optimization
            num_trials=-1,
            model_kwargs={
                "botorch_acqf_class": qLogNoisyExpectedImprovement,
                "torch_device": device,
            },
            model_gen_kwargs={
                "pending_observations":get_pending_observation_features
            },
        ))
        
        # Instantiating an Ax Client object
        generation_strategy_forMOBO = GenerationStrategy(
            # https://ax.dev/docs/0.5.0/tutorials/generation_strategy/
            steps=steps
        )
        # torch-device  as argument to the AxClient here should be useless with a custom generation strategy
        ax_client = AxClient(generation_strategy=generation_strategy_forMOBO,torch_device=device)

        variable_parameters_forMOBO = self.generate_mobo_parameters(**mobo_parameter_kwargs)

        objectives_forMOBO = self.generate_mobo_objectives(**mobo_objective_kwargs)

        ax_client.create_experiment(
            name="MOBO_penalty_weights",
            parameters=variable_parameters_forMOBO,
            objectives=objectives_forMOBO,
            overwrite_existing_experiment=True,
            is_test=False,
            )
        
        tic = perf_counter()
        tac = None
        optimized_cat_tables = {}
        if num_parallel_iterations > 1:
            # Completely random experiment initialization in parallel
            results, cat_tables = self.initialize_experiment(ax_client,
                                       num_random_initiations=num_random_initiation,
                                       num_parallel_iterations=num_parallel_iterations)
            optimized_cat_tables.update(cat_tables)
            tac = perf_counter()
            print(f"Time taken for {num_random_initiation} parallel initialization: {tac - tic:0.4f} seconds")
        else:
            results = pd.DataFrame()

        if num_parallel_iterations <= 1:
            for i in range(total_ax_iterations):
                print("RUNNING MOBO ITERATION ", i+1, " OUT OF ", total_ax_iterations)
                
                # Sequential experiment parameter generation and evaluation
                try:
                    parameters, trial_index = ax_client.get_next_trial()
                    ## To print device used
                    # genstep = ax_client.generation_strategy.current_step
                    # generator = genstep.generator_specs[0]
                    # print("Generator is on device:", generator.fitted_adapter.device)

                except Exception as e:
                    print("Failed to get next trial from axClient: ", e)
                    break
                optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(parameters)
                dvh_metrics_and_config, optimized_cat_table = self.brachy_optim.evaluate_penaltyWeight(
                    optim_config_list, target_dose, return_cat_table=True, inplace=False) # inplace Flase since we add the new plan to the dict
                dvh_metrics_and_config["trial_index"] = trial_index
                results = pd.concat([results, pd.DataFrame({**dvh_metrics_and_config})])
                optimized_cat_tables[f"trial_{trial_index}"] = optimized_cat_table
                # local evaluation here can be replaced with deployment to external systems
                ax_client.complete_trial(trial_index=trial_index, raw_data={
                    k:v for k,v in dvh_metrics_and_config.items() if k in list(ax_client.experiment.metrics.keys())})
                
                if calc_hv:
                    genstep = ax_client.generation_strategy.current_step
                    generator = genstep.generator_specs[0]
                    # TO get training data
                    # generator.fitted_adapter.get_training_data()
                    hv = observed_hypervolume(
                            adapter=generator.fitted_adapter,
                            objective_thresholds=ax_client.experiment.optimization_config.objective_thresholds,
                            optimization_config=ax_client.experiment.optimization_config,
                            selected_metrics=list(ax_client.experiment.metrics.keys())
                        )
                    hv_list.append(hv)  
                    print(f"Hypervolume after iteration {i+1}: {hv:0.4f}")
        else:
            # Parallel experiment parameter generation and evaluation
            total_mobo_iter = 0
            for _ in range(int(np.ceil(total_ax_iterations/num_parallel_iterations))):
                if total_mobo_iter + num_parallel_iterations <= total_ax_iterations:
                    max_trials = num_parallel_iterations
                else:
                    max_trials = total_ax_iterations - total_mobo_iter

                total_mobo_iter += max_trials

                parameters, _ = ax_client.get_next_trials(max_trials=max_trials)
                list_of_opt_config_lists = []
                list_of_target_doses = []
                trial_indices = []
                for t_index, param_dict in parameters.items():
                    optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(param_dict)
                    list_of_opt_config_lists.append(optim_config_list)
                    list_of_target_doses.append(target_dose)
                    trial_indices.append(int(t_index))
                    assert int(t_index) not in  optimized_cat_tables.keys(), (
                        "Trial index already exists in optimized_cat_tables. This should not happen."
                    )
                dvh_metrics_and_config_df, cat_tables = self.evaluate_penaltyWeight_space_helper(
                    list_of_opt_config_lists=list_of_opt_config_lists, 
                    list_of_target_doses=list_of_target_doses,
                    num_parallel_iterations=num_parallel_iterations,
                    list_of_experiment_indexes=trial_indices)

                optimized_cat_tables.update(cat_tables)
                results = pd.concat([results, dvh_metrics_and_config_df], ignore_index=True)
                for trial_ind in trial_indices:
                    print("Completing trial index:", trial_ind)
                    sub_df = dvh_metrics_and_config_df[dvh_metrics_and_config_df['trial_index'] == trial_ind]
                    ax_client.complete_trial(
                        trial_index=trial_ind,
                        raw_data={k:v for k,v in sub_df.iloc[[0]][
                            list(ax_client.experiment.metrics.keys())
                            ].to_dict('records')[0].items()}
                    )
                    if calc_hv:
                        genstep = ax_client.generation_strategy.current_step
                        generator = genstep.generator_specs[0]
                        hv = observed_hypervolume(
                                adapter=generator.fitted_adapter,
                                objective_thresholds=ax_client.experiment.optimization_config.objective_thresholds,
                                optimization_config=ax_client.experiment.optimization_config,
                                selected_metrics=list(ax_client.experiment.metrics.keys())
                            )

                        hv_list.append(hv)  
                        print(f"Hypervolume after iteration {trial_ind}: {hv:0.4f}")

        toc = perf_counter()
        if tac is not None:
            parallel_init_time = tac - tic
            mobo_time = toc - tac
            print(f"Time taken for {num_random_initiation} parallel initialization: {tac - tic:0.4f} seconds")
            print(f"Time taken for {num_mobo_iterations} MOBO iterations after initialization: {toc - tac:0.4f} seconds")
        else:
            mobo_time = toc - tic
            parallel_init_time = 0
            print(f"Time taken for {num_mobo_iterations} MOBO iterations: {toc - tic:0.4f} seconds")

        trial_only_results = ax_client.get_trials_data_frame().sort_values(by=["trial_index"])
        trial_only_results.to_csv(output_filename, index=False)
        # Merging both to have the trial indexes and all DVH metrics
        results = trial_only_results.merge(
            results, how='inner'
        )
        results.to_csv(output_filename.replace('.csv', '_with_all_DVH_metrics.csv'), index=False)
        return results, optimized_cat_tables, hv_list, mobo_time, parallel_init_time

    def count_successful_iterations(self, results:pd.DataFrame):
        r''' this function counts the number of successful iterations in the mobo results.
        a successful iteration is defined as an iteration that meets all the clinical goals
        Please overwrite this function in the child class that should be site specific.
        i.e. BreastMOBOOptimizer, ProstateMOBOOptimizer, etc.
        inputs:
            - results:= a pandas dataframe containing the result of mobo iterations. 
                each row is an iteration, and each column is a dvh metric.
        outputs:
            - num_successful_iterations:= the number of successful iterations in the mobo results.
        '''
        pass
      
