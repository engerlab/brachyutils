import os
from glob import glob
from pathlib import Path

from brachyutils.dose.dose_generation_utils import DoseMonteCarlo, DoseTG43
from brachyutils.planning.plan_utils import load_dicom_to_plan


def make_plan_and_export_it(dir_export) -> Path:
    dir_dicom = "data_test/prostate-glen-p1-dcm"
    # dir_egsphant = "data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # # assign material based on contours:
    # pth_material = "data_test/prostate_material_dict.json"
    # # assign materials based on CT values:
    pth_material = "data_test/CTtoDensityProstate.txt"
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
        "number_histories": 10000000,
        "total_time": 5983,
        "number_of_threads": 12,
        "PrintProgress": 10000,
        "beam_on": 10000,
    }
    content_to_export = {
        "dose": False,
        "dose_type": ".nrrd",
        # "dose_rate_maps": True,
        # "uncertainty": True,
        # "catheter_table": True,
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": True,
        # "structure_set": True,
        "plan": True,
        "mac": True,
        # "ApplicatorMaterials": True,
        # "applicator_geometry": False,
    }

    plan_obj = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_dose=False,
    )
    plan_obj.export_brachy_plan(
        dir_export=dir_export,
        content_to_export=content_to_export
        )

    return Path(dir_export)

def test_DoseTG43():
    dir_export = "temp_data/tg43/p1"
    dose_setup = make_plan_and_export_it(dir_export)
    dose_setup = Path("temp_data/tg43/p1")
    pth_exectuable = "http://192.168.1.12:8000/calculate_dose_tg43"
    dose_generator = DoseTG43(
        dir_plan_export=dose_setup,
        pth_dose_executable=pth_exectuable,
    )
    # dose_generator.validate_inputs()
    dose_generator.generate_dose()


def test_DoseMC():
    dir_export = "temp_data/mc/p1"
    dose_setup = make_plan_and_export_it(dir_export)
    dose_setup = Path("temp_data/mc/p1")
    pth_exectuable = "http://192.168.1.11:8000/calculate_dose_mc"
    dose_generator = DoseMonteCarlo(
        dir_plan_export=dose_setup,
        pth_dose_executable=pth_exectuable,
    )
    dose_generator.generate_dose(
        pth_mac=dose_setup.joinpath("combined.mac"),
        random_seed=1,
    )


if __name__ == "__main__":
    test_DoseTG43()
    # test_DoseMC()
    # import requests

    # json_data = {
    #     'pth_mac': 'temp_data/test_export_plan/combined.mac',
    #     'random_seed': 1,
    # }

    # response = requests.post('http://192.168.1.11:8000/calculate_dose_mc', json=json_data, timeout=None)
    # json_data = {
    #     'dir_dose_setup': 'temp_data/test_export_plan',
    # }
    # response = requests.post("http://192.168.1.12:8000/calculate_dose_tg43", json=json_data, timeout=None)

    # print(response.json())