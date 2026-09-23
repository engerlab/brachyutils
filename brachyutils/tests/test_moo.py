from pathlib import Path
from brachyutils.tests.test_optim_catheters import test_catheter_table_optim
from random import randint
import numpy as np
import pandas as pd
from time import time
from brachyutils.planning.optimization.optim_cath.moo import (
    MOO_Optuna, evaluate_parameters, are_acceptable, get_hyper_volume)
from brachyutils.planning.optimization.optim_cath.visualize_moo import plot_dvh_moo_space

def test_update_penalty_weights_and_voxel_goals():
    optim_obj = test_catheter_table_optim(retrun_optim_obj=True)
    prescription_dose = optim_obj.plan.prescription_dose
    optim_configs = list(optim_obj.plan.optimization_config_dict.values())
    hyper_params = [
        "dose_voxel_goal",
        "penalty_weight_linear",
        "penalty_weight_quadratic",
        "penalty_weight_hotspot",
        "penalty_weight_uniformity",
        "penalty_weight_variance_time",
    ]

    for conf in optim_configs:
        for param in hyper_params:
            if param == "dose_voxel_goal":
                if not conf.is_target:
                    continue
                lower_dose_bound = int(np.floor(prescription_dose * 0.8))
                upper_dose_bound = int(np.ceil(prescription_dose * 1.2))
                new_value = randint(lower_dose_bound, upper_dose_bound)
            elif (param == "penalty_weight_variance_time"
                  or param == "penalty_weight_uniformity"
                  or param == "penalty_weight_hotspot"):
                if not conf.is_target:
                    continue
            else:
                new_value = randint(0, 1000)
            setattr(conf, param, new_value)

    print("break point here: check if the new hyper-parameters are updated \
in the config objects inside each structure of the plan")

    t0_update = time()
    optim_obj.update_penalty_weights_and_voxel_goals(
        optimization_configs=optim_configs,
    )
    t1_update = time()
    
    t0_optim = time()
    optimized_plan = optim_obj.get_optimized_plan_from_model()
    t1_optim = time()

    print(f"Time to update the penalty weights and voxel goals: \
{t1_update - t0_update:.4f} seconds")
    print(f"Time to optimize the plan: {t1_optim - t0_optim:.4f} \
seconds")

    print("break point here: Check that the model has the new \
hyper-parameters inside the model")

def test_get_optimization_result_stats():
    from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import get_optimization_result_stats
    optim_obj = test_catheter_table_optim(retrun_optim_obj=True)
    print(get_optimization_result_stats(optim_obj))

def test_init_MOO_Optuna(return_obj=False):
    optim_obj = test_catheter_table_optim(retrun_optim_obj=True)
    dvh_metric_goals = {
        "D95%(CTV)": [">=", 95],
        "D2cc(RECTUM)": ["<=", 66],
        "D10%(URETHRA)": ["<=", 113],
        "D30%(URETHRA)": ["<=", 100],
        "CI(CTV)": None,
        "HI(CTV)": None,
        "V200%(CTV)": None,
        "V150%(CTV)": ["<=", 40],
        "V100%(CTV)": [">=", 100],
    }
    optim_obj.plan.set_dvh_metric_goals(
        dvh_metric_goals=dvh_metric_goals,
        strict_name_match=False,
        )

    # # Build the range of the parameters
    structure_names = ["CTV", "RECTUM", "URETHRA"]
    parameter_space = {}
    for name in structure_names:
        if name == "CTV":
            parameter_space[f"dose_voxel_goal({name})"] = [
                optim_obj.plan.prescription_dose,
                optim_obj.plan.prescription_dose*1.15 
            ]
            parameter_space[f"penalty_weight_hotspot({name})"] = [0, 1000]
            parameter_space[f"hotspot_threshold({name})"] = [1, 2]
            # parameter_space[f"penalty_weight_uniformity({name})"] = [0, 1000]
            # parameter_space[f"penalty_weight_variance_time({name})"] = [0, 1000]

        parameter_space[f"penalty_weight_linear({name})"] = [0, 1000]
        # parameter_space[f"penalty_weight_quadratic({name})"] = [0, 1000]

    Moo_obj = MOO_Optuna(
        catheter_table_optim=optim_obj,
        parameter_space=parameter_space,
    )
    print("break point here: Check that the MOO object is initialized correctly")
    if return_obj:
        return Moo_obj

def test_evaluate_parameters():
    optim_obj = test_catheter_table_optim(retrun_optim_obj=True)
    structure_names = ["CTV", "RECTUM", "URETHRA"]
    parameters_list = []
    for i in range(5):
        parameters = {}
        for name in structure_names:
            if name == "CTV":
                parameters[f"dose_voxel_goal({name})"] = np.random.uniform(
                    optim_obj.plan.prescription_dose,
                    optim_obj.plan.prescription_dose*1.15 
                )
                parameters[f"penalty_weight_hotspot({name})"] = np.random.uniform(0, 1000)
                parameters[f"hotspot_threshold({name})"] = np.random.uniform(1, 2)
                parameters[f"penalty_weight_uniformity({name})"] = np.random.uniform(0, 1000)
                parameters[f"penalty_weight_variance_time({name})"] = np.random.uniform(0, 1000)

            parameters[f"penalty_weight_linear({name})"] = np.random.uniform(0, 1000)
            parameters[f"penalty_weight_quadratic({name})"] = np.random.uniform(0, 1000)
        parameters_list.append(parameters)
    parameters = pd.DataFrame(parameters_list)
    dvh_metrics_data = evaluate_parameters(
        parameters, 
        optim_obj,
        max_workers=16
        )
    print(dvh_metrics_data.mean())

def test_run_warmps():
    Moo_obj = test_init_MOO_Optuna(return_obj=True)
    Moo_obj.run_warmups(batch_size=5)
    print(Moo_obj.trial_data)

def test_run_trials(return_output:bool = False):
    dir_out = Path("data_test/test_export_plan/prostate")
    Moo_obj = test_init_MOO_Optuna(return_obj=True)
    Moo_obj.run_warmups(batch_size=30)
    Moo_obj.run_trials(n_trials=2, batch_size=5)
    print("break point here.")
    print(Moo_obj.trial_data)
    Moo_obj.trial_data.to_csv(dir_out/"test_trial.csv")
    if return_output:
        return Moo_obj

def test_are_acceptable():
    dvh_metric_goals = {
        "D95%(CTV)": [">=", 95],
        "D2cc(RECTUM)": ["<=", 66],
        "D10%(URETHRA)": ["<=", 113],
        "D30%(URETHRA)": ["<=", 100],
        "V150%(CTV)": ["<=", 40],
        "V100%(CTV)": [">=", 100],
    }
    dvh_metrics1 = {
        "D95%(CTV)": 99,
        "D2cc(RECTUM)": 50,
        "D10%(URETHRA)": 98,
        "D30%(URETHRA)": 85,
        "V150%(CTV)": 30,
        "V100%(CTV)": 105,
    }
    dvh_metrics2 = {
        "D95%(CTV)": 87,
        "D2cc(RECTUM)": 50,
        "D10%(URETHRA)": 98,
        "D30%(URETHRA)": 105,
        "V150%(CTV)": 30,
        "V100%(CTV)": 105,
    }
    
    dvh_metrcics = pd.DataFrame([dvh_metrics1,dvh_metrics2])
    print(
        are_acceptable(
            dvh_metrics=dvh_metrcics,
            dvh_metric_goals=dvh_metric_goals)
    )

def test_get_hyper_volume():
    dvh_metric_goals = {
        "D95%(CTV)": [">=", 95],
        "D2cc(RECTUM)": ["<=", 66],
        "D10%(URETHRA)": ["<=", 113],
        "D30%(URETHRA)": ["<=", 100],
        "V150%(CTV)": ["<=", 40],
        "V100%(CTV)": [">=", 100],
    }
    dvh_metrics1 = {
        "D95%(CTV)": 98,
        "D2cc(RECTUM)": 50,
        "D10%(URETHRA)": 98,
        "D30%(URETHRA)": 85,
        "V150%(CTV)": 30,
        "V100%(CTV)": 105,
    }
    dvh_metrics2 = {
        "D95%(CTV)": 87,
        "D2cc(RECTUM)": 50,
        "D10%(URETHRA)": 98,
        "D30%(URETHRA)": 105,
        "V150%(CTV)": 30,
        "V100%(CTV)": 105,
    }
    
    dvh_metrcics = pd.DataFrame([dvh_metrics1,dvh_metrics2])
    print(
        get_hyper_volume(
            dvh_metrics=dvh_metrcics,
            dvh_metric_goals=dvh_metric_goals)
    )

def test_plot_dvh_moo_space():
    dir_out = Path("data_test/test_export_plan/prostate")
    Moo_obj = test_run_trials(True)
    plot_dvh_moo_space(
        trial_df=Moo_obj.trial_data,
        dvh_metric_goals=Moo_obj.dvh_metric_goals,
        path_out_svg=dir_out/"test.svg",
        sampler_col="sampler_name_id",
        title="test trial"
    )

if __name__ == "__main__":
    # test_update_penalty_weights_and_voxel_goals()
    # test_get_optimization_result_stats()
    # test_init_MOO_Optuna()
    # test_evaluate_parameters()
    # test_run_warmps()
    # test_run_trials()
    # test_are_acceptable()
    # test_get_hyper_volume()
    test_plot_dvh_moo_space()