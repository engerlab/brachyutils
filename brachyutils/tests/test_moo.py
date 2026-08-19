from brachyutils.tests.test_optim_catheters import test_catheter_table_optim
from random import randint
import numpy as np

def test_update_penalty_weights_and_targets():
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
                  or param == "penalty_weight_uniformity"):
                if not conf.is_target:
                    continue
            else:
                new_value = randint(0, 1000)
            setattr(conf, param, new_value)

    print("debug here: check if the new hyper-parameters are updated")

if __name__ == "__main__":
    test_update_penalty_weights_and_targets()