from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union

# from brachyutils.dose_utils import BrachyDose


class BrachyDoseGenerator(ABC):
    def __init__(
        self,
        dir_plan_export: Union[Path, str],
        pth_dose_executable: Union[Path, str],
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of dose generators. Each generator should
            support the attributes of this class and implements its abstract methods.
        Attributes:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Inputs:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Functions:
            - generate_dose(): generates the dose distribution as well as its uncertaity per voxel.
            - validate_inputs(): validates the dose setup directory.
        """
        self.dir_plan_export: Path = Path(dir_plan_export)
        self.pth_dose_executable: Path = pth_dose_executable

    @abstractmethod
    def validate_inputs(self):
        r"""
        Purpose:
            - Abstract method to validate the inputs of the dose generator.
            Each dose generator should implement this method.
        """
        pass

    @abstractmethod
    def generate_dose(self, pth_output: Optional[Path] = None):
        r"""
        Purpose:
            - Abstract method to generate the dose distribution.
            Each dose generator should implement this method.
        Inputs:
            - pth_output: Optional[Path]: If provided, the dose distribution will be saved to this path.
        """
        pass


class DoseTG43(BrachyDoseGenerator):
    def __init__(
        self,
        dir_plan_export: Union[Path, str],
        pth_dose_executable: Union[Path, str],
    ) -> None:
        r"""
        Purpose:
            - A class to generate dose distribution using the TG43 formalism.
            This class uses RapidBrachyTG43 to calculate the dose distribution.
        """
        super().__init__(dir_plan_export, pth_dose_executable)

    def generate_dose(
        self,
        dir_output: Optional[str] = None,
        dose_output_extension: Optional[Literal[".3ddose", ".nrrd"]] = ".nrrd",
        pth_egsphant: Optional[Path] = None,
        pth_plan: Optional[Path] = None,
        pth_mac: Optional[Path] = None,
        num_threads: Optional[int] = 12,
        output_dose_per_dwell: Optional[Literal[True, False, "dose_rate"]] = False,
        dir_source_parameters: Optional[str] = "SourceParameters/microSelectron-v2",
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
        - output_dose_per_dwell[str] := A flag to indicate if the dose per dwell position should be output.
            The default is "false". Other options are "true" and "dose_rate". for optimization, select "dose_rate".
        - dir_source_parameters: Optional[str] := The directory where the source parameters are stored.
            The default is "./SourceParameters/microSelectron-v2".
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

        if "http" in self.pth_dose_executable:
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
        elif ".py" in self.pth_dose_executable:
            # use subprocess to run the python script
            raise NotImplementedError("This feature is not implemented yet.")
        else:
            raise ValueError(
                "The dose executable is not supported. It should be a URL or a python script."
            )

        return response

    def validate_inputs(self):
        r"""
        Purpose:
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


class DoseMonteCarlo(BrachyDoseGenerator):
    def __init__(
        self, dir_plan_export: Path | str, pth_dose_executable: Path | str
    ) -> None:
        r"""
        Purpose:
            - A class to generate dose distribution using Monte Carlo simulations.
            This class uses RapidBrachyMC to calculate the dose distribution.
        """
        super().__init__(dir_plan_export, pth_dose_executable)

    def validate_inputs(self):
        r"""
        Purpose:
            - Validate the inputs of the Monte Carlo dose generator.
        """
        pass
    
    def generate_batch_plans():
        r"""
        Purpose:
            - Generate the batch plans for the Monte Carlo simulation to achieve the 
            desired uncertainty with much less time.
        """
        raise NotImplementedError("This feature is not implemented yet.")
    
    def generate_dose(
        self,
        pth_mac: Path = None,
        random_seed: int = 1,
    ):
        r""""""

        if pth_mac is None:
            print("No mac file is provided. Will use all mac files in the directory. except the combined.mac")
            pth_all_mac = list(self.dir_plan_export.glob("*.mac")) #glob(str(self.dir_plan_export / "*.mac"))
            if len(pth_all_mac) == 0:
                raise ValueError(
                    f"No mac file is found at {self.dir_plan_export}."
                )
            for pth_mac in pth_all_mac:
                if str(pth_mac.name) == "combined.mac":
                    continue
                self.generate_dose(
                    pth_mac=pth_mac,
                    random_seed=random_seed,
                )
        else:
            if "http" in self.pth_dose_executable:
                pth_mac = str(pth_mac.resolve())
                # use fast api post to request the dose calculation
                import requests
                response = requests.post(
                    self.pth_dose_executable,
                    json={
                        "pth_mac": str(pth_mac),
                        "random_seed": str(random_seed),
                    },
                    timeout=None,
                )
            elif ".py" in self.pth_dose_executable:
                # use subprocess to run the python script
                raise NotImplementedError("This feature is not implemented yet.")
            else:
                raise ValueError(
                    "The dose executable is not supported. It should be a URL or a python script."
                )
            return response
