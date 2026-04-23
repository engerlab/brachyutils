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

import numpy as np
import pandas as pd

from ax.adapter.adapter_utils import observed_hypervolume
from ax.adapter.registry import Generators
from ax.core.utils import get_pending_observation_features
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from brachyutils.planning.optimization.optim_utils import BrachyDwellTimeOptim
from brachyutils.planning.optimization.optim_configs import Optimization_Config

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
    
    def evaluate_penaltyWeight_space_helper(self, weight_space:pd.DataFrame, multiprocessing:bool=True):
        r''' This function evaluates a space of penalty weights in parallel using multiprocessing library. 
        inputs:
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
            weights_and_dvh_space, cat_tables = self.brachy_optim.evaluate_penaltyWeight_space(
                weight_space_list_of_d,
                return_cat_table=True,
            )
            return pd.DataFrame(weights_and_dvh_space), cat_tables
        else:

            results = {}
            cat_tables = {}
            for i, w_space in enumerate(weight_space_list_of_d):
                res, cat_table = self.brachy_optim.evaluate_penaltyWeight(w_space, return_cat_table=True)
                res.update(w_space)
                results[i] = res
                cat_tables[f"trial_{i}"] = cat_table
            return pd.DataFrame.from_dict(results, orient='index'), cat_tables

  

    def initialize_experiment(self, axClient:AxClient, num_fmio_runs:int):
        r''' In this function, we initialize the experiment object with a user-defined number of randomly generated penalty weight vectors
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - num_fmio_runs := number of fmio calls to be run in parallel

        outputs:
            - None := this method returns nothing, but it updates the axClient with the result of penalty weight evaluations

        dependencies:
            - AxClient
            - TreatmentPlan
        '''

        # obtain the parameter configuration from the AxClient.experiment object
        parameters=axClient.experiment.search_space.parameters

        # initialize a dataframe that will hold all the random penalty weight vectors 
        initial_weights_space = np.zeros([num_fmio_runs, len(parameters)])
        random_weight_space = pd.DataFrame(initial_weights_space, columns=list(parameters.keys()))

        # fill the dataframe with the weight vectors in the right range
        for parameter in parameters:
            try:
                random_weight_space[parameter] = np.random.uniform(
                    low=parameters[parameter].lower, high = parameters[parameter].upper, size=num_fmio_runs)
            # in case you have a fixed parameter, set the value accordingly for all rows
            except: 
                random_weight_space[parameter] = np.ones(num_fmio_runs)*parameters[parameter].value

        # run penalty weight evaluation on all the weight vecotrs in parallel
        random_weight_dvh_space, cat_tables = self.evaluate_penaltyWeight_space_helper(
            random_weight_space, multiprocessing=True)

        # put the results into the axclient
        for indx in random_weight_dvh_space.index:
            axClient.attach_trial(random_weight_dvh_space.iloc[[indx]][list(parameters.keys())].to_dict('records')[0])
            axClient.complete_trial(indx, random_weight_dvh_space.iloc[[indx]][list(axClient.experiment.metrics.keys())].to_dict('records')[0])
        # for debugging{ in case you want to view the result of added trials uncomment the line below
        # axClient.get_trials_data_frame()
        # }
        return random_weight_dvh_space, cat_tables


    def run(self,
            param_config:dict,
            num_iterations:int, 
            num_random_initiation:int=5, 
            parallel_random_init:bool=True,
            output_filename:str=None,
            calc_hv:bool=True,
            ):
        r'''This function runs multi-objective bayesian optimization (mobo) for as many iterations as the user commands it to. 
            it will save the result of all iterations in a pandas dataframe. 
        
        WARNING: Inplace is True in the different calls to the optimization function, Hence 
        the original plan will be modified!!! By returning the catheter table, one can reconstruct 
        the optimized plan afterwards by manually setting the catheter_table attribute of the 
        BrachyPlan object and then calling the update_plan_from_catheter_table() method.

        inputs:
            - num_iterations:= the number of desired mobo iterations
            - num_random_initiation := fmio calls are used to initiate axClient.experiment
                If parallel_random_init, these runs are completely random and run in parallel with the initialize_experiment()
                function. If not, the runs are initiated one by one inside the mobo loop with the semirandom sobol generator 
                of ax api. Must be greater than 0, otherwise qNoisyExpectedImprovement will not be able to search for next 
                trial.
            - param_config:= a dictionary containing the configuration of the parameter. it must have the following keys: 
                {
                'relative_weights':bool := if true, the penalty weight of the OARs is normalized to the penalty weight of tumor volume, 
                'weight_range':list := the minimal and maximal value for each weight. for example [0.001, 1],
                'target_dose_range':list := the minimal and maximal value for the target dose. for example [10, 20],
                }
            - output_filename:= the name of the file to save the result of mobo iterations.
            - calc_hv:= if true, the hypervolume of the pareto front is calculated at each iteration and returned as a list.

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
        if not parallel_random_init:
            steps.append(
                GenerationStep(
                    ## This is quasi random sampling, good for initial exploration of the space
                    # https://ax.dev/docs/0.5.0/tutorials/multiobjective_optimization/
                    generator=Generators.SOBOL,
                    num_trials=num_random_initiation,
                )
            )

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
        ax_client = AxClient(generation_strategy=generation_strategy_forMOBO,)

        variable_parameters_forMOBO = self.generate_mobo_parameters(param_config)

        objectives_forMOBO = self.generate_mobo_objectives()

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
        if parallel_random_init:
            # Completely random experiment initialization in parallel
            results, cat_tables = self.initialize_experiment(ax_client,
                                       num_random_initiation)
            optimized_cat_tables.update(cat_tables)
            tac = perf_counter()
            print(f"Time taken for {num_random_initiation} parallel initialization: {tac - tic:0.4f} seconds")
        else:
            results = pd.DataFrame()

        
        for _ in range(num_iterations):
            parameters, trial_index = ax_client.get_next_trial()

            dvh_metrics, optimized_cat_table = self.brachy_optim.evaluate_penaltyWeight(
                parameters, return_cat_table=True, inplace=False) # inplace Flase since we add the new plan to the dict
            results = pd.concat([results, pd.DataFrame({**dvh_metrics}, index=[trial_index])], ignore_index=True)
            optimized_cat_tables[f"trial_{trial_index}"] = optimized_cat_table
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
            parallel_init_time = tac - tic
            mobo_time = toc - tac
            print(f"Time taken for {num_random_initiation} parallel initialization: {tac - tic:0.4f} seconds")
            print(f"Time taken for {num_iterations} MOBO iterations after initialization: {toc - tac:0.4f} seconds")
        else:
            mobo_time = toc - tic
            parallel_init_time = 0
            print(f"Time taken for {num_iterations} MOBO iterations: {toc - tic:0.4f} seconds")

        trial_only_results = ax_client.get_trials_data_frame().sort_values(by=["trial_index"])
        trial_only_results.to_csv(output_filename, index=False)
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
      
