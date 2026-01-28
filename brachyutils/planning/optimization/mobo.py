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
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


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
            list_of_experiment_indexes:List[int],
            objective_to_scale_to: dict[str, float] = None):
        r''' This function evaluates a space of penalty weights in parallel using multiprocessing library. 
        inputs:
            - list_of_opt_config_lists := a list of optimization configuration lists to be evaluated.
            - list_of_target_doses := a list of target doses corresponding to each optimization configuration list
            - num_parallel_iterations := number of parallel iterations to run
            - list_of_experiment_indexes := a list of experiment indexes corresponding to each optimization configuration list
            - objective_to_scale_to:= a dictionary containing the objective name as key and the value to scale the plan to as value.
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
                list_of_opt_config_lists=list_of_opt_config_lists,
                list_of_target_doses=list_of_target_doses,
                list_of_experiment_indexes=list_of_experiment_indexes,
                return_cat_table=True,
                max_parallel_runs=num_parallel_iterations,
                objective_to_scale_to=objective_to_scale_to
            )
            return pd.DataFrame(weights_and_dvh_space), cat_tables
        else:

            results = {}
            cat_tables = {}
            for i, (opt_config_list, target_dose, exp_indx) in enumerate(
                zip(list_of_opt_config_lists, list_of_target_doses,list_of_experiment_indexes)):
                optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(opt_config_list)
                res, cat_table = self.brachy_optim.evaluate_penaltyWeight(
                    optim_config_list, target_dose, return_cat_table=True,
                    objective_to_scale_to=objective_to_scale_to
                )
                res["trial_index"] = exp_indx
                results[i] = res
                cat_tables[f"trial_{exp_indx}"] = cat_table
            return pd.DataFrame.from_dict(results, orient='index'), cat_tables

    
    def get_next_trial_parameters(self, axClient:AxClient, random:bool=False, trial_index:int=None):
        r''' This function gets the next trial parameters
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - random := if true, get random parameters, else get the next trial parameters from the model
        outputs:
            - a dictionary containing the next trial parameters
            example:
            Trial index:  0
            Parameters:  
            {'dose_voxel_goal_PTV': 3.451593026518822, 
            'penalty_weight_linear_Skin': 362.025554895401, 
            'penalty_weight_linear_Chestwall': 435.0770305991173, 
            'penalty_weight_linear_PTV': 1000.0}
        '''
        
        if random:
            print("GETTING RANDOM PARAMETERS FOR ONE TRIAL")
            assert trial_index is not None, (
                "trial_index must be set when random is True."
            )
            # obtain the parameter configuration from the AxClient.experiment object
            parameters_ax=axClient.experiment.search_space.parameters
            parameters = {}
            for p in parameters_ax:
                # Testing if the parameter is not fixed
                if hasattr(parameters_ax[p], 'lower') and hasattr(parameters_ax[p], 'upper'):
                    parameters[p] = np.random.uniform(
                        low=parameters_ax[p].lower, high=parameters_ax[p].upper)
                
                else:
                    print("Fixed parameter detected:", p)
                    parameters[p] = parameters_ax[p].value
            axClient.attach_trial(parameters)
        else:
            print("GETTING Ax GENERATED PARAMETERS FOR ONE TRIAL")
            assert trial_index is None, (
                "trial_index must be None when random is False since it is given by axclient."
            )
            try:
                parameters, trial_index = axClient.get_next_trial()
                ## To print device used
                # genstep = ax_client.generation_strategy.current_step
                # generator = genstep.generator_specs[0]
                # print("Generator is on device:", generator.fitted_adapter.device)

            except Exception as e:
                print("Failed to get next trial from axClient: ", e)
        return parameters, trial_index

    def get_next_trials_parameters(self, axClient:AxClient, 
                                   num_trials:int, random:bool=False, trial_indexes:List[int]=None):
        r''' This function gets the next trial parameters for multiple trials
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and
            - num_trials := number of trials to get parameters for
            - random := if true, get random parameters, else get the next trial parameters from the model
        outputs:
            - a dictionary containing the next trial parameters
            example:
            Parameters:  
            {5: {'dose_voxel_goal_PTV': 3.451593026518822, 
            'penalty_weight_linear_Skin': 362.025554895401, 
            'penalty_weight_linear_Chestwall': 435.0770305991173, 
            'penalty_weight_linear_PTV': 1000.0},
            6: {'dose_voxel_goal_PTV': 4.123456789012345,
            'penalty_weight_linear_Skin': 123.4567890123456,
            'penalty_weight_linear_Chestwall': 234.5678901234567,
            'penalty_weight_linear_PTV': 1500.0}}

        '''
        if random:
            print(f"GETTING RANDOM PARAMETERS FOR {num_trials} TRIALS")
            assert trial_indexes is not None, (
                "trial_indexes must be provided when random is True."
            )
            # obtain the parameter configuration from the AxClient.experiment object
            parameters_ax=axClient.experiment.search_space.parameters

            # initialize a dataframe that will hold all the random penalty weight vectors 
            initial_weights_space = np.zeros([num_trials, len(parameters_ax)])
            random_weight_space = pd.DataFrame(initial_weights_space, columns=list(parameters_ax.keys()))

            # fill the dataframe with the weight vectors in the right range
            for parameter in parameters_ax:
                try:
                    random_weight_space[parameter] = np.random.uniform(
                        low=parameters_ax[parameter].lower, high=parameters_ax[parameter].upper, size=num_trials)
                # in case you have a fixed parameter, set the value accordingly for all rows
                except:
                    random_weight_space[parameter] = np.ones(num_trials) * parameters_ax[parameter].value
            # Truning the pd dataframe into a dictionary with correct indexes
            parameters = {}
            for i, trial_index in enumerate(trial_indexes):
                param_trial = random_weight_space.iloc[i].to_dict()
                parameters[trial_index] = param_trial
                axClient.attach_trial(param_trial)
        else:
            print(f"GETTING Ax GENERATED PARAMETERS FOR {num_trials} TRIALS")
            parameters, _ = axClient.get_next_trials(max_trials=num_trials)
        return parameters


    def initialize_experiment(
            self, axClient:AxClient, num_random_initiations:int, num_parallel_iterations:int,
            objective_to_scale_to: dict[str, float] = None):
        r''' In this function, we initialize the experiment object with a user-defined number of randomly generated penalty weight vectors
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - num_random_initiations := number of fmio calls to be run in parallel
            - num_parallel_iterations := number of parallel iterations to run
            - objective_to_scale_to:= a dictionary containing the objective name as key and the value to scale the plan to as value.

        outputs:
            - VOID := this method returns nothing, but it updates the axClient with the result of penalty weight evaluations

        dependencies:
            - AxClient
            - TreatmentPlan
        '''

        print(f"INITIALIZING EXPERIMENT WITH {num_random_initiations} RANDOM TRIALS")
        parameters = self.get_next_trials_parameters(
            axClient=axClient,
            num_trials=num_random_initiations,
            random=True,
            trial_indexes=list(range(num_random_initiations))
        )

        return self.complete_multiple_trials(
            axClient=axClient,
            parameters=parameters,
            num_parallel_iterations=num_parallel_iterations,
            calc_hv=False,
            objective_to_scale_to=objective_to_scale_to
        )

    def complete_multiple_trials(
            self, axClient:AxClient, parameters:dict, num_parallel_iterations:int, 
            calc_hv:bool=False, objective_to_scale_to: dict[str, float] = None):
        r''' This function completes multiple trials given their parameters
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - parameters := a dictionary containing the parameters for each trial
            - num_parallel_iterations := number of parallel iterations to run
            - calc_hv := if true, calculate the hypervolume after completing the trials
            - objective_to_scale_to:= a dictionary containing the objective name as key and the value to scale the plan to as value.
        outputs:
            - dvh_metrics_and_config_df := a pandas dataframe containing the result of penalty weight evaluations.
            - cat_tables := a dictionary containing the catheter tables for each trial
            - hv := the hypervolume after completing the trials if calc_hv is true, else None
        '''
        print(f"COMPLETING {len(parameters)} TRIALS")
        list_of_opt_config_lists = []
        list_of_target_doses = []
        trial_indices = []
        for t_index, param_dict in parameters.items():
            optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(param_dict)
            list_of_opt_config_lists.append(optim_config_list)
            list_of_target_doses.append(target_dose)
            trial_indices.append(int(t_index))
            
        dvh_metrics_and_config_df, cat_tables = self.evaluate_penaltyWeight_space_helper(
            list_of_opt_config_lists=list_of_opt_config_lists, 
            list_of_target_doses=list_of_target_doses,
            num_parallel_iterations=num_parallel_iterations,
            list_of_experiment_indexes=trial_indices,
            objective_to_scale_to=objective_to_scale_to
        )
        if calc_hv:
            hvs = []
        else:
            hvs = None
        for trial_ind in trial_indices:
            print("Completing trial index:", trial_ind)
            sub_df = dvh_metrics_and_config_df[dvh_metrics_and_config_df['trial_index'] == trial_ind]
            axClient.complete_trial(
                trial_index=trial_ind,
                raw_data={k:v for k,v in sub_df.iloc[[0]][
                    list(axClient.experiment.metrics.keys())
                    ].to_dict('records')[0].items()}
            )
            if calc_hv:
                genstep = axClient.generation_strategy.current_step
                generator = genstep.generator_specs[0]
                hv = observed_hypervolume(
                        adapter=generator.fitted_adapter,
                        objective_thresholds=axClient.experiment.optimization_config.objective_thresholds,
                        optimization_config=axClient.experiment.optimization_config,
                        selected_metrics=list(axClient.experiment.metrics.keys())
                    )
                print(f"Hypervolume after iteration {trial_ind}: {hv:0.4f}")
                hvs.append(hv)
        return dvh_metrics_and_config_df, cat_tables, hvs

    def complete_single_trial(
            self, axClient:AxClient, parameters:dict, trial_index:int, 
            calc_hv:bool=False, objective_to_scale_to: dict[str, float] = None):
        r''' This function completes a single trial given its parameters
        inputs:
            - axClient := an ax client object that has been already initialized with the right parameters and objectives
            - parameters := a dictionary containing the parameters for the trial
            - calc_hv := if true, calculate the hypervolume after completing the trial
            - objective_to_scale_to:= a dictionary containing the objective name as key and the value to scale the plan to as value.
        outputs:
            - dvh_metrics_and_config := a dictionary containing the result of penalty weight evaluation.
            - cat_table := a catheter table for the trial
            - hv := the hypervolume after completing the trial if calc_hv is true, else None
        '''
        print(f"COMPLETING TRIAL INDEX: {trial_index}")
        optim_config_list, target_dose = self.get_optim_config_and_target_dose_from_parameters(parameters)
                
        dvh_metrics_and_config, cat_table = self.brachy_optim.evaluate_penaltyWeight(
            optim_config_list, target_dose, return_cat_table=True, inplace=False,
            objective_to_scale_to=objective_to_scale_to) # inplace Flase since we add the new plan to the dict
        dvh_metrics_and_config["trial_index"] = trial_index

        
        axClient.complete_trial(
            trial_index=trial_index,
            raw_data={k:v for k,v in dvh_metrics_and_config.items() if k in axClient.experiment.metrics.keys()}
        )
        if calc_hv:
            genstep = axClient.generation_strategy.current_step
            generator = genstep.generator_specs[0]
            # TO get training data
            # generator.fitted_adapter.get_training_data()
            hv = observed_hypervolume(
                    adapter=generator.fitted_adapter,
                    objective_thresholds=axClient.experiment.optimization_config.objective_thresholds,
                    optimization_config=axClient.experiment.optimization_config,
                    selected_metrics=list(axClient.experiment.metrics.keys())
                )
            print(f"Hypervolume after iteration {trial_index}: {hv:0.4f}")
        else:
            hv = None
        return dvh_metrics_and_config, cat_table, hv

    def run(self,
            num_mobo_iterations:int, 
            num_random_initiation:int=5, 
            init_randomization:str="sobol",
            num_parallel_iterations:int=1,
            output_filename:str=None,
            calc_hv:bool=False,
            mobo_objective_kwargs:dict=None, 
            mobo_parameter_kwargs:dict=None,
            objective_to_scale_to: dict[str, float] = None
            ):
        r'''This function runs multi-objective bayesian optimization (mobo) for as many iterations as the user commands it to. 
            it will save the result of all iterations in a pandas dataframe. 
        
        WARNING: Inplace is True in the different calls to the optimization function, Hence 
        the original plan will be modified!!! By returning the catheter table, one can reconstruct 
        the optimized plan afterwards by manually setting the catheter_table attribute of the 
        BrachyPlan object and then calling the update_plan_from_catheter_table() method.

        inputs:
            - num_mobo_iterations:= the number of desired mobo iterations
            - num_random_initiation := number of optimization experiments used to initiate axClient.experiment.
            Must be greater than 0, otherwise qNoisyExpectedImprovement will not be able to search for next 
            trial.
            - num_parallel_iterations:= number of parallel iterations to run. If greater than 1,
            experiments will run in parallel using evaluate_penaltyWeight_space_helper method. else,
            experiments will be run one by one inside the mobo loop.
            - init_randomization:= method to use for initialization of the experiment. sobol (semi random) or full random
            - output_filename:= the name of the file to save the result of mobo iterations.
            - calc_hv:= if true, the hypervolume of the pareto front is calculated at each iteration and returned as a list.
            - mobo_objective_kwargs:= a dictionary containing the keyword arguments for the generate_mobo_objectives() method.
            - mobo_parameter_kwargs:= a dictionary containing the keyword arguments for the generate_mobo_parameters
            - objective_to_scale_to:= a dictionary containing the objective name as key and the value to scale the plan to as value.
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
        assert init_randomization in ["sobol", "full_random"], (
            "init_randomization must be either 'sobol' or 'full_random'"
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

        if init_randomization == "sobol":
            steps.append(
                GenerationStep(
                    ## This is quasi random sampling, good for initial exploration of the space
                    # https://ax.dev/docs/0.5.0/tutorials/multiobjective_optimization/
                    generator=Generators.SOBOL,
                    num_trials=num_random_initiation,
                    max_parallelism=num_parallel_iterations,
                )
            )
        total_ax_iterations = num_mobo_iterations + num_random_initiation

        steps.append(GenerationStep(
            ## Refer to 
            # S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy.
            # Unexpected Improvements to Expected Improvement for Bayesian Optimization. Advances
            # in Neural Information Processing Systems 36, 2023.
            # https://github.com/meta-pytorch/botorch/blob/main/botorch/acquisition/logei.py#L239
            generator=Generators.BOTORCH_MODULAR,  # Use this for multi-objective optimization
            num_trials=-1,
            model_kwargs={
                "botorch_acqf_class": qLogNoisyExpectedHypervolumeImprovement, # qNoisyExpectedHypervolumeImprovement,# qLogNoisyExpectedImprovement,
                # "botorch_acqf_options": {"prune_baseline": True},
                "torch_device": device,

                # TODO: Explore other multi-output GP models available in BoTorch for potentially better performance:
                # See https://botorch.org/docs/models
                # MultiTaskGP: a Hadamard multi-task, multi-output GP using an ICM kernel. Supports both known observation noise levels and inferring a homoskedastic noise level (when noise observations are not provided).
                # KroneckerMultiTaskGP: A multi-task, multi-output GP using an ICM kernel, with Kronecker structure. Useful for multi-fidelity optimization.
                # SaasFullyBayesianMultiTaskGP: a fully Bayesian multi-task GP using an ICM kernel. The data kernel uses the SAAS prior to model high-dimensional parameter spac


            },
            model_gen_kwargs={
                "pending_observations":get_pending_observation_features
            },
            # After BoTorch parameter generation, the process is already using PyTorch, BLAS 
            # (MKL/OpenBLAS), and OpenMP threads, which maintain complex native state in memory. 
            # If multiprocessing is then started (which is what we do when launching 
            # evaluatepenaltyweightspace), especially with fork (default on Linux) or by passing 
            # large Python objects such as plans or model data into worker processes, the child processes 
            # inherit this unstable native state and potentially non-fork-safe objects (e.g., 
            # hidden torch tensors, Gurobi objects, or shared memory). This combination can lead 
            # to race conditions, thread oversubscription, corrupted memory, and improper cleanup 
            # of system resources, which significantly increases the risk of segmentation faults 
            # during the parallel optimization phase. Problem is if we set mp.set_start_method('spawn') globally,
            # it will make the start of process super slow. So the current solution is to limit the number of
            # threads used in the gurobi model see optim_gurobi.py evaluate_penaltyWeight_space() 
            # thread_per_gurobi_model, for more details.
            max_parallelism=num_parallel_iterations,
        ))
        
        # Instantiating an Ax Client object
        generation_strategy_forMOBO = GenerationStrategy(
            # https://ax.dev/docs/0.5.0/tutorials/generation_strategy/
            steps=steps
        )
        
        ax_client = AxClient(
            generation_strategy=generation_strategy_forMOBO,
            # torch-device  as argument to the AxClient here should be useless with a custom generation strategy
            torch_device=device,
            # Putting this to False messes up with the Optimization results
            enforce_sequential_optimization=True
            )

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
        results = pd.DataFrame()

        if num_parallel_iterations <= 1:
            for i in range(total_ax_iterations):
                print("RUNNING MOBO ITERATION ", i+1, " OUT OF ", total_ax_iterations)
                
                full_random_params = (i < num_random_initiation) and (init_randomization == "full_random")

                # Sequential experiment parameter generation and evaluation
                parameters, trial_index = self.get_next_trial_parameters(
                    axClient=ax_client,
                    random=full_random_params,
                    trial_index=i if full_random_params else None
                )

                dvh_metrics_and_config, optimized_cat_table, hv = self.complete_single_trial(
                    axClient=ax_client,
                    parameters=parameters,
                    trial_index=trial_index,
                    calc_hv=calc_hv and num_random_initiation <= i,
                    objective_to_scale_to=objective_to_scale_to
                )
                results = pd.concat([results, pd.DataFrame({**dvh_metrics_and_config}, index=[trial_index])], ignore_index=True)
                optimized_cat_tables[f"trial_{trial_index}"] = optimized_cat_table
                # local evaluation here can be replaced with deployment to external systems

                
                if calc_hv and num_random_initiation <= i:
                    hv_list.append(hv)  
                    print(f"Hypervolume after iteration {i+1}: {hv:0.4f}")
                if i == num_random_initiation-1:
                    tac = perf_counter()
                    print(f"Time taken for {num_random_initiation} sequential initialization: {tac - tic:0.4f} seconds")
        else:
            # Parallel experiment parameter generation and evaluation
            if init_randomization == "full_random":
                results, cat_tables, _ = self.initialize_experiment(
                    ax_client,
                    num_random_initiations=num_random_initiation,
                    num_parallel_iterations=num_parallel_iterations,
                    objective_to_scale_to=objective_to_scale_to
                    )
                optimized_cat_tables.update(cat_tables)
                tac = perf_counter()
                print(f"Time taken for {num_random_initiation} {init_randomization} parallel initialization: {tac - tic:0.4f} seconds")
                total_ax_iterations -= num_random_initiation


            total_mobo_iter = 0
            for _ in range(int(np.ceil(total_ax_iterations/num_parallel_iterations))):
                if total_mobo_iter + num_parallel_iterations <= total_ax_iterations:
                    max_trials = num_parallel_iterations
                else:
                    max_trials = total_ax_iterations - total_mobo_iter

                total_mobo_iter += max_trials

                parameters = self.get_next_trials_parameters(
                    axClient=ax_client,
                    num_trials=max_trials,
                    random=False,
                    trial_indexes=None
                )
                # We can only compute hv with MBOO iterations not sobol initialization
                if init_randomization == "sobol":
                    calc_hv = calc_hv if total_mobo_iter > num_random_initiation else False
                else:
                    calc_hv = calc_hv
                dvh_metrics_and_config_df, cat_tables, hvs = self.complete_multiple_trials(
                    axClient=ax_client,
                    parameters=parameters,
                    num_parallel_iterations=num_parallel_iterations,
                    calc_hv=calc_hv,
                    objective_to_scale_to=objective_to_scale_to
                )

                optimized_cat_tables.update(cat_tables)
                results = pd.concat([results, dvh_metrics_and_config_df], ignore_index=True)
                if calc_hv:
                    hv_list.extend(hvs)
                if total_mobo_iter >= num_random_initiation and tac is None:
                    # This time might not be accurate if num_random_initiation is not a multiple of num_parallel_iterations
                    # Ax client might suggest trials from sobol and MOBO to run in parallel
                    tac = perf_counter()
                    print(f"Time taken for {num_random_initiation} {init_randomization} parallel initialization: {tac - tic:0.4f} seconds")

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
      
