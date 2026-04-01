from pathlib import Path
from time import time
from brachyutils import BrachyPlan, load_dicom_to_plan
from brachyutils import RapidBrachyTG43

def gen_combined_doserates():
    r"""
    """

    dir_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    dir_export = Path("temp_data/tg43/cat-optim")/dir_dicom.stem
    export_config = {
        "dir_export": dir_export,
        "export_config_egsphant": {
            "strict_name_match": False,
            "crop_by_contour": ["ctv", "urethra", "rectum"]},
        "export_config_macfile": {
            "name_combined": "cropped_combined"
            },
        "export_config_planfile": {
            "name_combined": "cropped_combined"
        }
        }
    plan_obj = load_dicom_to_plan(
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


    dir_dicom = ""
    plan = load_dicom_to_plan()


if __name__ == "__main__":
    gen_combined_doserates()