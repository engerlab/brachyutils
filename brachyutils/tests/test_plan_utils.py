# import json
import os
import time
from glob import glob
from pathlib import Path
import numpy as np

from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.planning.plan_utils import load_dicom_to_plan
def get_a_plan(
    dir_dicom:str | Path,
    **kwargs)->BrachyPlan:
    prescription_dose = kwargs.get("prescription_dose", 21)
    dvh_metric_goals = {
        "D90%(CTV)": prescription_dose,
        "D2cc(RECTUM)": prescription_dose * 0.75,
        "D10%(URETHRA)": prescription_dose * 1.133,
        "D30%(URETHRA)": prescription_dose,
        "CI(CTV)": 1.0,
        "HI(CTV)": 0.5,
        "V200%(CTV)": prescription_dose * 0.2,
        "V150%(CTV)": prescription_dose * 0.4,
        "V100%(CTV)": 100.0,
    }

    plan_obj = load_dicom_to_plan(
        dir_dicom=dir_dicom,
        load_dicom_dose=kwargs.get("load_dicom_dose", False),
        load_dicom_catheter_table=kwargs.get("load_dicom_catheter_table", True),
        load_dicom_prescription_dose=kwargs.get("load_dicom_prescription_dose", True),
        strict_name_match=kwargs.get("strict_name_match", False),
        dir_dose_rate=kwargs.get("dir_dose_rate", None),
        from_delivered_dwellpositions=kwargs.get("from_delivered_dwellpositions", False),
        multi_processing=True,
        # prescription_dose=prescription_dose,
        dvh_metric_goals=kwargs.get("dvh_metric_goals", None),
        optimization_config_list=kwargs.get("optimization_config_list", None),
        dwells_near_ptv=kwargs.get("dwells_near_ptv", True),
        add_hotspots_to_phantom=kwargs.get("add_hotspots_to_phantom", False),
        one_hotspot_structure=kwargs.get("one_hotspot_structure", True),
        )
    return plan_obj

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
    dir_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(dir_dicom + "/RP*.dcm")[0]
    dir_dose_rate = "data_test/prostate-glen-p1-dose"

    catheter_table = CatheterTable(
        catheters_dict=pth_plan,
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
    dir_dose_rate = "data_test/prostate-glen-p1-dose"
    prescription_dose=21
    dvh_metric_goals = {
        "D90%(CTV)": prescription_dose,
        "D2cc(RECTUM)": prescription_dose * 0.75,
        "D10%(URETHRA)": prescription_dose * 1.133,
        "D30%(URETHRA)": prescription_dose,
        "CI(CTV)": 1.0,
        "HI(CTV)": 0.5,
        "V200%(CTV)": prescription_dose * 0.2,
        "V150%(CTV)": prescription_dose * 0.4,
        "V100%(CTV)": 100.0,
    }
    # load plan without dvh or dose
    plan = get_a_plan(
        dir_dicom=dir_dicom,
        prescription_dose=prescription_dose,
    )
    # test with DVH names and dicom dose
    plan = get_a_plan(
        dir_dicom=dir_dicom,
        prescription_dose=prescription_dose,
        load_dicom_dose=True,
        dvh_metric_goals=list(dvh_metric_goals.keys()),
        strict_name_match=False,
    )
    print("This is the loaded DVH metric goals")
    print(plan.dvh_metric_goals)
    print(plan.get_dvh_metrics())

    # test with DVH dict and dicom dose
    plan = get_a_plan(
        dir_dicom=dir_dicom,
        prescription_dose=prescription_dose,
        load_dicom_dose=True,
        dvh_metric_goals=dvh_metric_goals,
        strict_name_match=False,
    )
    print("This is the loaded DVH metric goals")
    print(plan.dvh_metric_goals)
    print(plan.get_dvh_metrics())

    # test with DVH names and dose rates
    plan = get_a_plan(
        dir_dicom=dir_dicom,
        prescription_dose=prescription_dose,
        load_dicom_dose=False,
        dvh_metric_goals=dvh_metric_goals,
        strict_name_match=False,
        dir_dose_rate=dir_dose_rate,
        from_delivered_dwellpositions=False,
    )
    print("This is the loaded DVH metric goals")
    print(plan.dvh_metric_goals)
    print(plan.get_dvh_metrics())

    print("This is a test for structure dict")
    print(plan.structure_dict.keys())

def test_calculate_combined_uncertainty():
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    dir_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(dir_dicom + "/RP*.dcm")[0]
    dir_dose_rate = "data_test/prostate-glen-p1-dose"

    catheter_table = CatheterTable(
        catheters_dict=pth_plan,
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
        prescription_dose=21,
        catheter_table=pth_catheter_table_json,
        phantom=dir_dicom,
        dvh_metric_goals=dvh_metric_goals,
    )
    t1 = time.time()
    print(f"loading the plan took {t1-t0} seconds")
    plan_obj.combined_dose.write_brachydose_to_file(
        "data_test/test_export_plan/prostate/combined.seq.nrrd"
    )

def test_export_brachy_plan():
    dir_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("data_test/test_export_plan/prostate").resolve()
    target_dose = 21
    # # for loading the delivered dose rates. 
    dir_dose_rate = Path("data_test/prostate-glen-p1-dose").resolve()
    gen_dose_rates = False
    from_delivered_dwellpositions=True

    export_config = {
        "dir_export": dir_export,
        "export_config_dose": True,
        "export_config_cathetertable": True,
        "export_config_egsphant": True,
        "export_config_plan_and_mac": {"combined_only": False},
        "applicator_geometry": False,
        "structure_set": False
    }

    plan:BrachyPlan = get_a_plan(
        dir_dicom=dir_dicom,
        dir_dose_rate=dir_dose_rate,
        from_delivered_dwellpositions=from_delivered_dwellpositions,
        generate_dose_rates=gen_dose_rates,
        load_dicom_dose=True,
        )
    plan.export_brachy_plan(
        content_to_export=export_config
    )

def test_load_brachy_plan_from_dicom():
    from brachyutils.geometry.phantom_utils import BrachyPhantom

    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    pth_dose_rates = Path("data_test/prostate-glen-p1-dose")
    dvh_metric_goals = {
        "D95%(ctv)": 21,
        "D1cc(rectum)": 21*0.75,
        "D0.1cc(urethra)": 21*1.25,
    }
    brachy_phant = BrachyPhantom(
        dir_dicom=dir_dicom,
        pth_structures_file=list(dir_dicom.glob("RS*.dcm"))[0],
    )
    plan_obj = BrachyPlan(
        phantom=brachy_phant,
        prescription_dose=21,
        dvh_metric_goals=dvh_metric_goals,
        dir_dose_rate=pth_dose_rates,
        combined_dose_only=True,
        catheter_table=list(dir_dicom.glob("RP*.dcm"))[0],
        from_delivered_dwellpositions=True,
    )
    plan_obj.info()
    plan_obj.combined_dose.write_brachydose_to_file(
        "data_test/test_export_plan/prostate/combined.seq.nrrd"
    )
    plan_obj.catheter_table.write_to_json(
        "data_test/test_export_plan/prostate/combined_cathtable.json"
    )

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

    dir_dicom = "data_test/prostate-glen-p1-dcm/"
    pth_structure = glob(dir_dicom + "/RS*.dcm")[0]
    pth_dose = glob(dir_dicom + "/RD*.dcm")[0]
    dvh_metric_goals = {
        "D95%(ctv)": 15.,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
        "CI(ctv)": 100
    }
    dose = BrachyDose(pth_dose)
    phantom_obj = BrachyPhantom(dir_dicom=dir_dicom, pth_structures_file=pth_structure)
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

    dir_dicom = Path("data_test/prostate-glen-p1-dcm/")

    dvh_metric_goals = {
        "D95%(ctv)": 15,
        "D1cc(rectum)": 11.25,
        "D0.1cc(urethra)": 18.75,
    }

    plan_obj = BrachyPlan(phantom=dir_dicom, dvh_metric_goals=dvh_metric_goals)
    plan_obj.info()

if __name__ == "__main__":
    # testupdate_plan_from_catheter_table()
    # test_update_catheter_table_from_plan()
    # test_load_dose_rate_dict()
    test_create_structures_and_calc_dvh_metrics()
    # test_calculate_combined_uncertainty()
    # test_calculate_uncertainty_per_structure()
    # test_BrachyPlan()
    # test__load_single_dose_or_uncertainty_to_dict()
    # test_export_brachy_plan()
    # test_load_brachy_plan_from_dicom()
    # test_load_applicator_list()
    # test__export_applicator_geometry()
    # test_brachy_structure()
    # test_load_phantom()
