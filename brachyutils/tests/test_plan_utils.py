import json
import os
import time

# from plan_utils import BrachyStructure
from brachyutils.plan_utils import (
    BrachyPlan,
    _load_single_dose_or_uncertainty_to_dict,
)


def test_load_catheterTable_json():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"

    with open(pth_cathTable_json, "r") as json_file:
        ground_truth_catheter_table = json.load(json_file)

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)

    # print(plan_obj.catheter_table)
    assert [
        i for i in ground_truth_catheter_table if i not in plan_obj.catheter_table
    ] == [], "loading catheter table did not work as expected"


def test_extract_dwell_numbers_times_coordinates_from_catheterTable():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)

    assert plan_obj.dwell_numbers is not None, "dwell numbers not extracted"
    assert plan_obj.dwell_times is not None, "dwell times not extracted"
    assert plan_obj.dwell_coordinates is not None, "dwell coordinates not extracted"

    print(f"The shape of the dwell_number is {plan_obj.dwell_numbers.shape}")
    print(f"The shape of the dwell_times is {plan_obj.dwell_times.shape}")
    print(f"The shape of the dwell_coordinates is {len(plan_obj.dwell_coordinates)}")


def test_load_dose_rate_or_uncertainty_tensor():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    

    plan_obj.load_dose_rate_or_uncertainty_tensor(
        dir_dose_rate=dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
    )
    print(f"The shape of the dose rate tensor is {plan_obj.dose_rate_tensor.shape}")
    print(f"The shape of the combined dose is {plan_obj.combined_dose.grid.shape}")


def test_set_dvh_metric_goals():
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    plan_obj = BrachyPlan()
    plan_obj.set_dvh_metric_goals(dvh_metric_goals)
    assert (
        plan_obj.dvh_metric_goals == dvh_metric_goals
    ), "dvh metric list not set correctly"
    print(plan_obj.dvh_metric_goals)


def test_create_structures_and_calc_dvh_metrics():
    dir_dicom = "../../data_test/prostate-glen-p1-dcm"
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    
    plan_obj.load_dose_rate_or_uncertainty_tensor(
        dir_dose_rate, load_dose_or_uncertainty="dose", multi_processing=True
    )
    plan_obj.set_dvh_metric_goals(dvh_metric_goals)

    plan_obj.create_structures(
        dir_structures_source=dir_dicom,
        dose_cropped_by_body=True,
    )
    plan_obj.calculate_dvh_metrics()
    # XXX: structure list is empty. fix it tomorrow!
    for structure in plan_obj.structure_list:
        print(f"{structure.name}: {structure.dvh_metric_observed}")


def test_calculate_combined_uncertainty():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    

    plan_obj.load_dose_rate_or_uncertainty_tensor(
        dir_dose_rate, load_dose_or_uncertainty="uncertainty", multi_processing=True
    )
    plan_obj._calculate_combined_uncertainty()
    print(
        f"The shape of the combined uncertainty is {plan_obj.combined_dose.uncertainty.shape}"
    )
    assert (
        plan_obj.combined_dose.uncertainty.shape == plan_obj.combined_dose.grid.shape
    ), "combined uncertainty shape does not match combined dose shape"


def test_calculate_uncertainty_per_structure():
    pth_catheter_table_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }

    plan_obj = BrachyPlan(
        dir_dicom=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
        dose_cropped_by_body=True,
        pth_catheter_table_json=pth_catheter_table_json,
        dir_dose_rate = dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True
    )

    plan_obj.calculate_uncertainty_per_structure()
    for structure in plan_obj.structure_list:
        print(
            f"{structure.name}: mean: {structure.uncertainty_mean},\n \
            std: {structure.uncertainty_std}, \n \
            max: {structure.uncertainty_max}, \n \
            min: {structure.uncertainty_min}"
        )


def test_BrachyPlan():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    t0 = time.time()
    BrachyPlan(
        pth_cathTable_json,
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
    )
    t1 = time.time()
    print(f"loading the plan took {t1-t0} seconds")


def test__load_single_dose_or_uncertainty_to_dict():
    pth_dose_rate = "../../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_rate_dict = _load_single_dose_or_uncertainty_to_dict(pth_dose_rate, "both")
    print(dose_rate_dict[0].shape)  # dose
    print(dose_rate_dict[1].shape)  # uncertainty


def test_export_brachy_plan():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    sim_dict = {
        "treatment_type": "HDR",
        "source_geometry": "MicroSelectronV2",
        "core_material": "G4_Ir",
        "mass_number": "192",
        "atomic_number": "77",
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "air_kerma_per_history": 1.149000e-11,
        "reference_air_kerma": 4.278729e04,
        "number_histories": 1000000,
        "total_time": 5983,
        "number_of_threads": 12,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    plan_obj = BrachyPlan(
        pth_cathTable_json,
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
        dir_egsphant="../../data_test/prostate-glen-p1-planFiles/ct.egsphant",
        combined_simulation_dict=sim_dict,
        # dir_applicator_geometry="../../data_test/prostate-glen-p1-planFiles/applicator_geometry.json",
    )
    dir_export = "../../data_test/test_export_plan/"
    os.makedirs(dir_export, exist_ok=True)
    content_to_export = {
        "dose": True,
        "dose type": ".nrrd",
        "dose rate maps": False,
        "uncertainty": True,
        "catheter_table": True,
        "egsphant": False,
        "structure_set": True,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }
    export_format = "RapidBrachyExport"
    # # This function tests all the exporting functions.
    plan_obj.export_brachy_plan(export_format, dir_export, content_to_export)


def test_load_brachy_plan_from_dicom():
    pth_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    plan_obj = BrachyPlan(pth_dicom, dvh_metric_goals=dvh_metric_goals)
    plan_obj.info()


if __name__ == "__main__":
    # test_load_catheterTable_json()
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable()
    # test_load_dose_rate_or_uncertainty_tensor()
    # test_set_dvh_metric_goals()
    # test_create_structures_and_calc_dvh_metrics()
    # test_calculate_combined_uncertainty()
    test_calculate_uncertainty_per_structure()
    test_BrachyPlan()
    # test__load_single_dose_or_uncertainty_to_dict()
    test_export_brachy_plan()
    # test_load_brachy_plan_from_dicom()
