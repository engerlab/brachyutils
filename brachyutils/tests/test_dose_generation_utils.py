import os
from glob import glob
from pathlib import Path

from brachyutils.dose.dose_generation_utils import RapidBrachyMC, RapidBrachyTG43
from brachyutils.planning.plan_utils import load_dicom_to_plan
from brachyutils.tests.test_plan_utils import get_a_plan
from time import time
from brachyutils.dose.dose_utils import BrachyDose
import numpy as np

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

def test_RapidBrachyTG43():
    dir_export = "temp_data/tg43/p1"
    dose_setup = make_plan_and_export_it(dir_export)
    dose_setup = Path("temp_data/tg43/p1")
    pth_exectuable = "http://192.168.1.12:8000/calculate_dose_tg43"
    dose_generator = RapidBrachyTG43(
        dir_plan_export=dose_setup,
        pth_dose_executable=pth_exectuable,
    )
    # dose_generator.validate_inputs()
    dose_generator.generate_dose(output_dose_per_dwell="dose_rate", num_threads=16)


def test_DoseMC():
    dir_export = "temp_data/mc/p1"
    dose_setup = make_plan_and_export_it(dir_export)
    dose_setup = Path("temp_data/mc/p1")
    pth_exectuable = "http://192.168.1.11:8000/calculate_dose_mc"
    dose_generator = RapidBrachyMC(
        dir_plan_export=dose_setup,
        pth_dose_executable=pth_exectuable,
    )
    dose_generator.generate_dose(
        pth_mac=dose_setup.joinpath("combined.mac"),
        random_seed=1,
    )

def test_run_dose_gen_tg43():
    dir_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("temp_data/tg43/RapidBrachyTG43")/dir_dicom.stem
    export_config = {
        "dir_export": dir_export,
        "export_config_egsphant": {
            "strict_name_match": False,
            "crop_by_contour": ["ctv", "urethra", "rectum"]},
        "export_config_plan_and_mac": {
            "name_combined": "cropped_combined"
            },
        }
    plan_obj = get_a_plan(
        dir_dicom=dir_dicom,
        from_delivered_dwellpositions=False,
        dwells_near_ptv=True
    )
    t0=time()
    dose_gen = RapidBrachyTG43(
        dir_plan_export=dir_export,
    )
    dose_gen.run_dose_generation(
        plan=plan_obj,
        export_config_brachyplan=export_config,
    )
    t1=time()
    # test the case with only combined dose
    print(f"time for RapidBrachyTG43: {t1-t0}")
    BrachyDose(dir_export/"cropped_combined.seq.nrrd").write_to_nrrd(dir_export/"cropped_combined.seq.nrrd")

def test_run_brachyutilstg43():
    from brachyutils.dose.tg43_dose_calculator import BrachyUtilsTG43
    dir_tg43_parameters = Path(
        "admin/constants/TG43_Parameter_Data/microSelectron-v2_Consensus")
    dir_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("temp_data/tg43")/"BrachyUtilsTG43"/dir_dicom.stem
    plan_obj = get_a_plan(
        dir_dicom=dir_dicom,
        from_delivered_dwellpositions=False,
        dwells_near_ptv=True
    )

    # crop the phantom
    plan_obj.phantom.crop_by_contour(
        ["ctv", "urethra", "rectum"],
        strict_name_match=False,
        )

    #just for testing
    t0 = time()
    calc_parameter_kwargs = {"kernel_half_width": 100, "kernel_res": 1, "kernel_max_dose_rate" : 100.0}

    tg43_calc = BrachyUtilsTG43(
        dir_tg43_parameters=dir_tg43_parameters,
        dir_output=dir_export,
        **calc_parameter_kwargs)
    tg43_calc.run_dose_generation(
        plan=plan_obj,
        export_combined_dose=True)
    t1 = time()

    print(f"time for BrachyUtilsTG43: {t1-t0}")

def compare_rb_bu():
    from brachyutils.dose.dose_comparison_utils import BrachyDoseComparison
    pth_dose_rb = Path("temp_data/tg43/RapidBrachyTG43/prostate-glen-p1-dcm/combined.seq.nrrd")
    pth_dose_bu = Path("temp_data/tg43/BrachyUtilsTG43/prostate-glen-p1-dcm/combined_TG43.seq.nrrd")
    dir_output = Path(pth_dose_bu.parent)
    plan_id = "p1"
    
    dose_rb = BrachyDose(pth_dose_file=pth_dose_rb)
    dose_bu = BrachyDose(pth_dose_file=pth_dose_bu)

    dose_comp = BrachyDoseComparison(
    dose1=dose_rb,
    dose2=dose_bu,
    compute_percent_difference=True,
    prescription_dose=21.,
    compute_gamma_index=False,
    positive_percent_difference=False,
    )
    vox_centers = dose_rb.get_voxel_centers()
    viz_index_limits = np.array([
        [len(vox_centers[0])*1/4, len(vox_centers[0])*3/4],
        [len(vox_centers[1])*1/4, len(vox_centers[1])*3/4],
        [len(vox_centers[2])//2, 0]
    ]).astype(int)
    dose_comp.plot_local_and_global_differences(
        axis_1_coords=vox_centers[0][viz_index_limits[0][0]:viz_index_limits[0][1]],
        axis_2_coords=vox_centers[1][viz_index_limits[1][0]:viz_index_limits[1][1]],
        plane_coord=vox_centers[2][viz_index_limits[2][0]],
        plane="xy",
        plot_title=(f"MC vs TG43 Percent Error Map for Plan {plan_id}"),
        # pth_fig_save=Path(dir_output)/f"percent_error_map_{plan_id}.svg",
        local_vmax=20.0,
        global_vmax=10.0,
        fig_size_mm=(200, 160)
    )

    
if __name__ == "__main__":
    # test_RapidBrachyTG43()
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

    test_run_dose_gen_tg43()
    # test_run_brachyutilstg43()
    # compare_rb_bu()