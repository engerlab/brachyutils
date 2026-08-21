from brachyutils.tests.test_optim_catheters import test_catheter_table_optim
from random import randint
import numpy as np
from time import time

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

if __name__ == "__main__":
    # test_update_penalty_weights_and_voxel_goals()
    test_get_optimization_result_stats()