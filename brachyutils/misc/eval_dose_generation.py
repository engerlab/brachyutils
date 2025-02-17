from typing import List, Dict, Union
from pathlib import Path
from brachyutils.plan_utils import BrachyPlan, load_dicom_to_plan
from brachyutils.dose_comparison_utils import DoseComparison
from brachyutils.dose_generation_utils import DoseGenerator, DoseMonteCarlo, DoseTG43

def run_multi_proc(function, input_list, **kwargs):
    pass

def generate_single_dose(dir_plan_export, dose_generator_class, **kwargs):
    r"""
    Purpose:
        - To run a single dose calculation for an exported plan.
    Inputs:
        - dir_plan_export: Union[Path, str]: The path to the exported plan.
        - dose_generator_class: DoseGenerator: The dose generator class to use.
        - dir_export: Union[Path, str]: The path to export the dose calculation.
        - **kw_args: dict: Additional keyword arguments to pass to the dose generator.
    Outputs:
        - None
    """
    dose_gen_obj = dose_generator_class(
        dir_plan_export=dir_plan_export,
        pth_dose_executable=kwargs.get(
            "pth_dose_executable",
            "http://192.168.1.11:8000/calculate_dose_mc",
            ),
    )
    dose_gen_obj.generate_dose(
        pth_mac = dir_plan_export.joinpath("combined.mac"),
        all_dwells=kwargs.get("all_dwells", False),
    )

def export_single_dicom_to_plan(
    dir_dicom:Dict,
    dir_export: Path | str,
    pth_material: Path | str,
    ) -> Path:
    r"""
    Purpose:
        - Export the plans to the given directory.
    Inputs:
        - dir_dicom: Path | str: The path to the dicom directory for one plan. it should have images,
        and a plan file. Structure file is optional.
        - dir_export: Path | str: The directory to export the plans to. Each plan will have its own subdir.
        - **kwargs: dict: Additional keyword arguments to pass to the plan creation function.
    Outputs:
        - dir_export_plan: Path: The path to the exported plan.
    """
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
    content_to_export = {
        "egsphant": True,
        "materials_table": pth_material,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }
    plan_obj = load_dicom_to_plan(dir_dicom) #simulation_dict=sim_dict)

    dir_export = Path(dir_export)
    dir_export.mkdir(parents=True, exist_ok=True)

    dir_export_plan = dir_export.joinpath(dir_dicom.stem)

    plan_obj.export_brachy_plan(
        dir_export=dir_export_plan,
        content_to_export=content_to_export,
    )
    return dir_export_plan

def run_export():
    pth_single_dicom = Path("/root/YourLocalHome/Data/prostate-glen-2023/p1")
    dir_export = Path("../temp_data/mc/prostate-glen-2023")
    pth_material = Path("/root/YourLocalHome/Data/CTtoDensityProstate.txt")
    
    if not pth_material.exists():
        raise FileNotFoundError(f"The material file {pth_material} does not exist.")
    export_single_dicom_to_plan(pth_single_dicom, dir_export, pth_material)

def run_dose_calc():
    dir_plan_export = Path("../temp_data/mc/prostate-glen-2023/p1")
    dose_generator_class = DoseMonteCarlo
    generate_single_dose(dir_plan_export, dose_generator_class)

if __name__ == "__main__":
    run_export()
    # run_dose_calc()