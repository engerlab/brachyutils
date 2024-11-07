import os
from glob import glob
from pathlib import Path

from brachyutils.dose_generation_utils import DoseTG43
from brachyutils.plan_utils import BrachyPlan


def make_plan_and_export_it(dir_export) -> Path:
    pth_cathTable_json = "../data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "../data_test/prostate-glen-p1-dose"
    dir_dicom = "../data_test/prostate-glen-p1-dcm/"
    pth_combined_dose = glob(dir_dicom + "/RD*.dcm")[0]
    # dir_egsphant = "../data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # assign material based on contours:
    # pth_material = "../data_test/prostate_material_dict.json"
    # assign materials based on CT values:
    pth_material = "../data_test/CTtoDensityProstate.txt"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    sim_dict = {
        "source_dict": {
            "treatment_type": "HDR",
            "source_geometry": "MicroSelectronV2",
            "core_material": "G4_Ir",
            "mass_number": "192",
            "atomic_number": "77",
            "air_kerma_per_history": 1.149000e-11,
            "reference_air_kerma": 4.278729e04,
        },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 1000000,
        "total_time": 5983,
        "number_of_threads": 12,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    # dir_export = "../data_test/test_export_plan"
    export_format = "RapidBrachy"
    os.makedirs(dir_export, exist_ok=True)

    content_to_export = {
        "dose": True,
        "dose_type": ".nrrd",
        "dose_rate_maps": True,
        "uncertainty": True,
        "catheter_table": True,
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": True,
        "structure_set": True,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": True,
        "applicator_geometry": False,
    }

    plan_obj = BrachyPlan(
        phantom=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
        catheter_table=pth_cathTable_json,
        combined_dose=pth_combined_dose,
        combined_simulation_dict=sim_dict,
    )
    # # This function tests all the exporting functions.
    plan_obj.export_brachy_plan(export_format, dir_export, content_to_export)

    return Path(dir_export)


def test_DoseTG43():
    # dir_export = "../temp_data/test_export_plan"
    # dose_setup = make_plan_and_export_it(dir_export)
    dose_setup = Path("../temp_data/test_export_plan")
    dose_generator = DoseTG43(
        dir_dose_setup=dose_setup,
        pth_dose_executable=dose_setup,
    )
    dose_generator.validate_inputs()


if __name__ == "__main__":
    test_DoseTG43()
