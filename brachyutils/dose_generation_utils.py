from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union

# from brachyutils.dose_utils import BrachyDose


class DoseGenerator(ABC):
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


class DoseTG43(DoseGenerator):
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
        num_threads: Optional[int] = 4,
        dir_source_parameters: Optional[str] = "./SourceParameters/microSelectron-v2",
        using_imbt_plan: Optional[bool] = False,
        shield_model: Optional[Literal["step", "tanh"]] = None,
        critical_angle: Optional[float] = None,
        correction_angle: Optional[float] = None,
        rotation_angle_config: Optional[str] = None,
    ):
        r"""
        Purpose:
            - Generate the dose distribution using the TG43 formalism.
        Inputs:
            - dir_output: str := The directory to save the dose distribution. If left empty,
            the dose will be saved to dir_plan_export.
            - dose_output_extension: Literal[".3ddose", ".nrrd"] := The format of the dose files.
            - num_threads: int := The number of threads to use for the dose calculation. leave it at 4.
            - dir_source_parameters: str:= The directory to the source parameters. Leave it at the default.
            - using_imbt_plan: bool := Whether the plan is using the IMBT technique.
            - shield_model: Optional[Literal["step", "tanh"]] := The model to use for the shield. The default is None.
            - critical_angle: Optional[float] := The critical angle for the phi dependence function, if necessary.
            - correction_angle: Optional[float] := The correction angle for the phi dependence function, if necessary.
            - rotation_angle_config- [optional] either a nine-character string representing the start, end, and increment
            angles (e.g. 000220015 for IMBT delievered from 0-220 degree increments) or a path to the
            catheter_table.json file where this information can be extracted.
        Outputs:
            - response: The response from the dose executable. God know what it is.
        """
        if "http" in self.pth_dose_executable:
            # use fast api post to request the dose calculation
            import requests

            response = requests.post(
                self.pth_dose_executable,
                json={
                    "dir_dose_setup": str(self.dir_plan_export),
                    "dir_output": str(dir_output),
                    "dose_output_extension": str(dose_output_extension),
                    "num_threads": str(num_threads),
                    "dir_source_parameters": str(dir_source_parameters),
                    "using_imbt_plan": str(using_imbt_plan),
                    "shield_model": str(shield_model),
                    "critical_angle": str(critical_angle),
                    "correction_angle": (correction_angle),
                    "rotation_angle_config": str(rotation_angle_config),
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
            - Validate the inputs of the TG43 dose generator.
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


class DoseMonteCarlo(DoseGenerator):
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

    def generate_dose(
        self,
        pth_mac: Path = None,
        random_seed: int = 1,
        all_dwells: bool = False,
    ):
        r""""""

        if pth_mac is None:
            assert all_dwells, "If pth_mac is not provided, all_dwells must be True."
            pth_all_mac = glob(str(self.dir_plan_export / "*.mac"))
            assert (
                len(pth_all_mac) > 0
            ), f"no mac file is found at {self.dir_plan_export}."
            for pth_mac in pth_all_mac:
                self.generate_dose(
                    pth_mac=pth_mac,
                    random_seed=random_seed,
                    all_dwells=False,
                )
        else:
            if "http" in self.pth_dose_executable:
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
