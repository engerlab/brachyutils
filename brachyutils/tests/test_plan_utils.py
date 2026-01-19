# import json
import os
import time
from glob import glob
from pathlib import Path
import numpy as np

from brachyutils.planning.plan_utils import BrachyPlan


def testupdate_plan_from_catheter_table():
    pth_cathTable_json = "data_test/prostate-glen-p1-planFiles/catheter_table.json"

    plan_obj = BrachyPlan(catheter_table=pth_cathTable_json)

    assert plan_obj.dwell_numbers is not None, "dwell numbers not extracted"
    assert plan_obj.dwell_times is not None, "dwell times not extracted"
    assert plan_obj.dwell_coordinates is not None, "dwell coordinates not extracted"

    print(f"The shape of the dwell_number is {plan_obj.dwell_numbers.shape}")
    print(f"The shape of the dwell_times is {plan_obj.dwell_times.shape}")
    print(f"The shape of the dwell_coordinates is {len(plan_obj.dwell_coordinates)}")


def test_update_catheter_table_from_plan():
    pth_cathTable_json = "data_test/prostate-glen-p1-planFiles/catheter_table.json"

    plan_obj = BrachyPlan(catheter_table=pth_cathTable_json)
    plan_obj.catheter_table.info()
    plan_obj._update_catheter_table_from_plan()
    plan_obj.catheter_table.info()


def test_load_dose_rate_dict():
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    dir_dose_rate = "data_test/prostate-glen-p1-dose"

    catheter_table = CatheterTable(
        catheter_list=pth_plan,
        from_delivered_dwellpositions=True,
    )
    plan_obj = BrachyPlan(
        catheter_table=catheter_table,
        dir_dose_rate=dir_dose_rate,
        load_dose_or_uncertainty="dose",
        multi_processing=True,
    )
    plan_obj.combined_dose.write_brachydose_to_file(
        "data_test/test_export_plan/prostate/new_combined.seq.nrrd")
    plan_obj.catheter_table.write_to_json(
        "data_test/test_export_plan/prostate/new_cathtabel.json")

def test_create_structures_and_calc_dvh_metrics():
    dir_dicom = "data_test/prostate-glen-p1-dcm"
    pth_cathTable_dcm = list(Path(dir_dicom).glob("RP*.dcm"))[0]
    dir_dose_rate = "data_test/prostate-glen-p1-dose"
    # pth_dose = glob(dir_dicom + "/RD*.dcm")[0]
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
        "CI(ctv)": 100,
        "HI(ctv)": 0.5,
    }
    from time import time
    t0 = time()
    plan_obj = BrachyPlan(
        phantom=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
        catheter_table=pth_cathTable_dcm,
        # combined_dose=pth_dose,
        dir_dose_rate=dir_dose_rate,
        multi_processing=True,
        combined_dose_only=True,
        prescription_dose=21.,
    )
    print(f"Loading the plan took {time()-t0} seconds")
    print(plan_obj.get_dvh_metrics())
    


def test_calculate_combined_uncertainty():
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    dir_dose_rate = "data_test/prostate-glen-p1-dose"

    catheter_table = CatheterTable(
        catheter_list=pth_plan,
        from_delivered_dwellpositions=True,
    )
    plan_obj = BrachyPlan(
        catheter_table=catheter_table,
        dir_dose_rate=dir_dose_rate,
        load_uncertainty=True,
        multi_processing=True,
    )
    print(plan_obj.combined_dose.uncertainty_image.imageArray.mean())

def test_calculate_uncertainty_per_structure():
    pth_catheter_table_json = (
        "data_test/prostate-glen-p1-planFiles/catheter_table.json"
    )
    dir_dose_rate = "data_test/prostate-glen-p1-dose/"
    dir_dicom = "data_test/prostate-glen-p1-dcm/"
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
        dir_dose_rate=dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
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
    pth_catheter_table_json = (
        "data_test/prostate-glen-p1-planFiles/catheter_table.json"
    )
    dir_dose_rate = "data_test/prostate-glen-p1-dose/"
    dir_dicom = "data_test/prostate-glen-p1-dcm"
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    t0 = time.time()
    plan_obj = BrachyPlan(
        catheter_table=pth_catheter_table_json,
        phantom=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
    )
    t1 = time.time()
    print(f"loading the plan took {t1-t0} seconds")


def test__load_single_dose_or_uncertainty_to_dict():
    pth_dose_rate = "data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    dose_rate_dict = _load_single_dose_or_uncertainty_to_dict(pth_dose_rate, "both")
    print(dose_rate_dict[0].shape)  # dose
    print(dose_rate_dict[1].shape)  # uncertainty


def test_export_brachy_plan():
    # pth_cathTable_json = "data_test/prostate-glen-p1-planFiles/catheter_table.json"
    dir_dose_rate = "data_test/prostate-glen-p1-dose"
    dir_dicom = "data_test/prostate-glen-p1-dcm/"
    pth_combined_dose = glob(dir_dicom + "/RD*.dcm")[0]
    pth_cathTable_dcm = glob(dir_dicom + "/RP*.dcm")[0]
    # dir_egsphant = "data_test/prostate-glen-p1-planFiles/ct.egsphant"
    # assign material based on contours:
    pth_material = "admin/constants/structure_materials_prostate.json"
    # assign materials based on CT values:
    # pth_material = "data_test/CTtoDensityProstate.txt"
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
    dir_export = "data_test/test_export_plan/prostate/glen_p1/"
    export_format = "RapidBrachy"
    os.makedirs(dir_export, exist_ok=True)

    content_to_export = {
        "dose": True,
        "dose_type": ".seq.nrrd",
        "dose_rate_maps": True,
        "uncertainty": True,
        "catheter_table": True,
        "egsphant": True,
        "materials_table": pth_material,
        "assign_material_from_ct": False,
        "structure_set": False,
        "plan": False,
        "mac": False,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }

    plan_obj = BrachyPlan(
        phantom=dir_dicom,
        # dvh_metric_goals=dvh_metric_goals,
        catheter_table=pth_cathTable_dcm,
        from_delivered_dwellpositions=True,
        dir_dose_rate=dir_dose_rate,
        multi_processing=True
        # combined_dose=pth_combined_dose,
        # simulation_setup=sim_dict,
    )
    # # This function tests all the exporting functions.
    plan_obj.export_brachy_plan(
        dir_export=dir_export,
        content_to_export=content_to_export,
        multi_processing=True,
        )


def test_load_brachy_plan_from_dicom():
    from brachyutils.geometry.phantom_utils import BrachyPhantom

    pth_dicom = Path("data_test/prostate-glen-p1-dcm")
    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }
    brachy_phant = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=list(pth_dicom.glob("RS*.dcm"))[0],
    )
    plan_obj = BrachyPlan(
        phantom=brachy_phant,
        prescription_dose=15,
        dvh_metric_goals=dvh_metric_goals,
    )
    plan_obj.info()

def test_load_applicator_list():
    dir_dicom = "data_test/rectal-jgh-dcm"
    dir_plan = "data_test/rectal-jgh-planFiles"
    pth_applicator_geometry = os.path.join(dir_plan, "applicator_geometry.json")

    plan_obj = BrachyPlan(
        phantom=dir_dicom, applicator_pth_list=pth_applicator_geometry
    )

    for applicator in plan_obj.applicator_list:
        applicator.info()


def test__export_applicator_geometry():
    dir_dicom = "data_test/rectal-jgh-dcm"
    dir_plan = "data_test/rectal-jgh-planFiles"
    pth_applicator_geometry = os.path.join(dir_plan, "applicator_geometry.json")
    dir_export = "data_test/test_export_plan"
    plan_obj = BrachyPlan(
        phantom=dir_dicom,
        applicator=pth_applicator_geometry,
    )

    plan_obj._export_applicator_geometry(
        dir_export=dir_export,
        export_format="RapidBrachy",
    )


def test_brachy_structure():
    from opentps.core.data import ROIContour

    from brachyutils import BrachyDose, BrachyPhantom, BrachyStructure

    pth_dicom = "data_test/prostate-glen-p1-dcm/"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_dose = glob(pth_dicom + "/RD*.dcm")[0]
    dvh_metric_goals = {
        "D95%(ctv)": 15.,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
        "CI(ctv)": 100
    }
    dose = BrachyDose(pth_dose)
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    mask_dict: dict = phantom_obj.get_structure_mask(phantom_obj.structure_names, ROIContour)
    for structure_name in mask_dict:
        mask_contour = mask_dict[structure_name]
        dvh_metric_goals_per_structure = {}
        for dvh_metric_name in dvh_metric_goals:
            structure_name_in_dvh_metrics = dvh_metric_name.split("(")[1].split(")")[0]
            dvh_metric = dvh_metric_name.split("(")[0]
            if structure_name_in_dvh_metrics.lower() in structure_name.lower():
                dvh_metric_goals_per_structure[f"{dvh_metric}({structure_name})"] = dvh_metric_goals[dvh_metric_name]

        if not any(dvh_metric_goals_per_structure):
            print(f"No DVH metric goals for the structure {structure_name}")
            continue

        structure_obj = BrachyStructure(
            name=structure_name,
            mask=mask_contour,
            is_target=True if "ctv" in structure_name.lower() else False,
            in_dvh=True,
            dvh_metric_goals=dvh_metric_goals_per_structure
        )
        structure_obj.info()
        print(structure_obj.get_dvh_metric(dose, 21, True))

def test_load_phantom():
    from pathlib import Path

    pth_dicom = Path("data_test/prostate-glen-p1-dcm/")

    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }

    plan_obj = BrachyPlan(phantom=pth_dicom, dvh_metric_goals=dvh_metric_goals)
    plan_obj.info()


if __name__ == "__main__":
    # testupdate_plan_from_catheter_table()
    # test_update_catheter_table_from_plan()
    # test_load_dose_rate_dict()
    # test_create_structures_and_calc_dvh_metrics()
    # test_calculate_combined_uncertainty()
    # test_calculate_uncertainty_per_structure()
    # test_BrachyPlan()
    # test__load_single_dose_or_uncertainty_to_dict()
    test_export_brachy_plan()
    # test_load_brachy_plan_from_dicom()
    # test_load_applicator_list()
    # test__export_applicator_geometry()
    # test_brachy_structure()
    # test_load_phantom()
