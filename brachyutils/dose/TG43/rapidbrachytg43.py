from abc import ABC, abstractmethod
import subprocess
import sys
import os
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union
from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.planning.plan_export_configs import ExportConfig_BrachyPlan, ExportConfig_Egsphant
from brachyutils.dose.dose_utils import BrachyDose
from brachyutils.dose.dose_generation_utils import BrachyDoseGenerator

class RapidBrachyTG43(BrachyDoseGenerator):
    def __init__(
        self,
        dir_plan_export: Union[Path, str],
        pth_dose_executable: Union[Path, str]="http://192.168.1.12:8000/calculate_dose_tg43",
    ) -> None:
        r"""
        ### Purpose:
        - A class to generate dose distribution using the TG43 formalism.
        This class uses RapidBrachyTG43 to calculate the dose distribution.
        """
        super().__init__(dir_plan_export, pth_dose_executable)

    def generate_dose(
        self,
        dir_output: Optional[str] = None,
        dose_output_extension: Optional[Literal[".3ddose", ".seq.nrrd"]] = ".seq.nrrd",
        pth_egsphant: Optional[Path] = None,
        pth_plan: Optional[Path] = None,
        pth_mac: Optional[Path] = None,
        num_threads: Optional[int] = 12,
        output_dose_per_dwell: Optional[Literal[True, False, "dose_rate"]] = False,
        dir_source_parameters: Optional[str] = "SourceParameters/GenericHDR",
        using_imbt_plan: Optional[bool] = False,
        shield_model: Optional[Literal["step", "tanh"]] = None,
        critical_angle: Optional[float] = None,
        correction_angle: Optional[float] = None,
        rotation_angle_config: Optional[str] = None,
    ):
        r"""
        ### Purpose:
        - To define the input parameters for the calculate_dose function.

        ### Inputs:
        - dir_dose_setup: str := The directory where the dose setup files are stored. This directory
            should containe the egsphant file, plan files, and mac files. it can also contain the
            the optional applicator_geometry.json file.
        - dir_output: Optional[str] := The directory where the dose files will be written.
            If None, the dose files will be written to the dir_dose_setup directory.    
        - dose_output_extension: Literal[".3ddose", ".nrrd"] := The extension of the
            dose files that are written by the executable. The default is ".nrrd".
        - pth_egsphant: Optional[Path] := The path to the egsphant file (.egsphant or .seq.nrrd).
            If None, the function will search for a single .egsphant file in the dir_dose_setup directory.
        - pth_plan: Optional[Path] := The path to the plan file (.plan).
        - pth_mac: Optional[Path] := The path to the mac file (.mac).
        - num_threads: Optional[int] := The number of threads to use for the calculation. The default is 4.
        - output_dose_per_dwell: Literal[bool, str] := A flag to indicate if the dose per dwell position should be output.
            The default is False. Other options are True and "dose_rate". for optimization, select "dose_rate".
        - dir_source_parameters: Optional[str] := The directory where the source parameters are stored.
            The default is "./SourceParameters/GenericHDR".
        - using_imbt_plan: Optional[bool] := a binary flag to indicate if the plan is an IMBT plan.
        - shield_model: Optional[Literal["step", "tanh"]] := The model to use for the shield. The default is None.
        - critical_angle: Optional[float] := The critical angle for the phi dependence function, if necessary.
        - correction_angle: Optional[float] := The correction angle for the phi dependence function, if necessary.
        - rotation_angle_config- [optional] either a nine-character string representing the start, end, and increment
        angles (e.g. 000220015 for IMBT delievered from 0-220 degree increments) or a path to the
        catheter_table.json file where this information can be extracted.

        ### Outputs:
            - response: The response from the dose executable. God know what it is.
        """
        if output_dose_per_dwell == True:
            output_dose_per_dwell = "true"
        elif output_dose_per_dwell == False:
            output_dose_per_dwell = "false"
        elif output_dose_per_dwell == "dose_rate":
            output_dose_per_dwell = "dose_rate"
        else:
            raise ValueError("Invalid value for output_dose_per_dwell.")

        if pth_egsphant is None:
            pth_egsphant = list(self.dir_plan_export.glob("*egsphant.seq.nrrd")).pop()
        if pth_mac is None:
            pth_mac = list(self.dir_plan_export.glob("combined.mac")).pop()

        if dir_output is None:
            dir_output = Path(".")

        if "http" in str(self.pth_dose_executable):
            # use fast api post to request the dose calculation
            import requests

            response = requests.post(
                self.pth_dose_executable,
                json={
                    "dir_dose_setup": str(self.dir_plan_export),
                    "dir_output": str(dir_output) if dir_output is not None else None,
                    "dose_output_extension": str(dose_output_extension) if dose_output_extension is not None else None,
                    "pth_egsphant": str(pth_egsphant) if pth_egsphant is not None else None,
                    "pth_plan": str(pth_plan) if pth_plan is not None else None,
                    "pth_mac": str(pth_mac) if pth_mac is not None else None,
                    "num_threads": str(num_threads) if num_threads is not None else None,
                    "dir_source_parameters": str(dir_source_parameters) if dir_source_parameters is not None else None,
                    "output_dose_per_dwell": output_dose_per_dwell,
                    "using_imbt_plan": str(using_imbt_plan) if using_imbt_plan is not None else None,
                    "shield_model": str(shield_model) if shield_model is not None else None,
                    "critical_angle": str(critical_angle) if critical_angle is not None else None,
                    "correction_angle": (correction_angle) if correction_angle is not None else None,
                    "rotation_angle_config": str(rotation_angle_config) if rotation_angle_config is not None else None,
                },
                timeout=None,
            )
            # let's handle the response here
            if response.status_code == 200:
                print("Dose calculation completed successfully.")
            else:
                raise RuntimeError(
                    f"Dose calculation failed with status code {response.status_code}: {response.text}"
                )
        elif ".py" in str(self.pth_dose_executable):
            # use subprocess to run the python script
            raise NotImplementedError("This feature is not implemented yet.")
        else:
            cmd = [
                str(self.pth_dose_executable),
                str(pth_egsphant),
                str(pth_plan),
                str(pth_mac),
                str(dir_source_parameters),
                str(dir_output),
                str(num_threads),
                str(output_dose_per_dwell),
                str(using_imbt_plan) if using_imbt_plan else "",
                str(shield_model) if shield_model is not None else "",
                str(critical_angle) if critical_angle is not None else "",
                str(correction_angle) if correction_angle is not None else "",
                str(rotation_angle_config) if rotation_angle_config is not None else "",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.dir_plan_export,
            )
            # iterate over stdout lines as they become available
            if proc.stdout is not None:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            proc.wait()
            response = subprocess.CompletedProcess(args=cmd, returncode=proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(f"Dose calculation process exited with code {proc.returncode}")

        return response

    def validate_inputs(self):
        r"""
        ### Purpose:
        - Validate the inputs o f the TG43 dose generator.
        """
        assert self.dir_plan_export.exists(), "The dose setup directory does not exist."
        # assert self.pth_dose_executable.exists(), "The dose executable does not exist."

        # look through the files in dose setup directory
        all_files: list = glob(str(self.dir_plan_export / "*"))
        assert len(all_files) > 0, "The dose setup directory is empty."
        assert any(".plan" in file for file in all_files), "The plan file is missing."
        assert any(
            ".egsphant" in file for file in all_files
        ), "The egsphant file is missing."
        assert any(".mac" in file for file in all_files), "The mac file is missing."

    def run_dose_generation(
        self,
        plan: BrachyPlan,
        generate_dose_rate_maps: bool = False,
        export_config_brachyplan: ExportConfig_BrachyPlan | bool | dict = None,
        dir_source_parameters: Path = "SourceParameters/GenericHDR"
        ) -> BrachyPlan:
        r"""
        ### Purpose:
        - to run the dose generation for the plan and return a plan with combined dose filled as well
        as the dose rate dictionary if desired.

        ### Inputs:
        - plan:= The treatment plan for which we want to generate the dose. 
        - generate_dose_rate_maps := whether to generate dose rate maps for each dwell position.
        If True, the dose_rate_dict will be populated with the dose rate maps for each dwell position.
        - export_config_brachyplan := If false, we assume the egsphant, mac files, and plan files have been
        exported before. If True or None, we create a default ExportConfig_BrachyPlan to export the setup files
        needed for RapidBrachyTG43. You can also provide your own custom export config.

        ### Output:
        - plan: BrachyPlan := The brachy plan with the combined dose and optionally the dose rate dict filled.
        """
        if export_config_brachyplan is None or export_config_brachyplan == True:
            export_config_brachyplan = ExportConfig_BrachyPlan(
                dir_export=self.dir_plan_export,
                export_config_egsphant=True,
                export_config_plan_and_mac=True,
            )
        elif isinstance(export_config_brachyplan, dict):
            export_config_brachyplan = ExportConfig_BrachyPlan(**export_config_brachyplan)

        if export_config_brachyplan:
            plan.export_brachy_plan(export_config_brachyplan)

        if not export_config_brachyplan.export_config_egsphant:
            export_config_brachyplan.export_config_egsphant = ExportConfig_Egsphant(
                dir_export=export_config_brachyplan.export_config_plan_and_mac.dir_export,
                name=export_config_brachyplan.export_config_plan_and_mac.pth_phantom.stem.split(".")[0],
                file_extension="".join(export_config_brachyplan.export_config_plan_and_mac.pth_phantom.suffixes)
            )
        # call the dose generator to generate the dose maps
        self.generate_dose(
            # dir_output=self.dir_plan_export,
            pth_mac=export_config_brachyplan.export_config_plan_and_mac.pth_mac_combined,
            pth_plan=export_config_brachyplan.export_config_plan_and_mac.pth_plan_combined,
            pth_egsphant=export_config_brachyplan.export_config_egsphant.pth_egsphant,
            output_dose_per_dwell= "dose_rate" if generate_dose_rate_maps else False,
            num_threads=plan.simulation_setup.number_of_threads,
            dir_source_parameters=dir_source_parameters,
        )

        pth_combined_dose = export_config_brachyplan.export_config_plan_and_mac.pth_mac_combined.with_suffix(".seq.nrrd")

        # load the generated dose maps and update the plan
        if generate_dose_rate_maps:
            plan.catheter_table.load_dose_rates(
                dir_dose_rate=self.dir_plan_export,
            )
        else:
            # Export the combined dose file, overwriting the tg43 automatically generated file 
            combined_dose = BrachyDose(pth_combined_dose)
            #combined_dose.write_to_nrrd(pth_combined_dose)
            plan.catheter_table.set_combined_dose(combined_dose)
        return plan
