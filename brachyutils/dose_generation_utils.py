from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Optional, Union

from brachyutils.dose_utils import BrachyDose


class DoseGenerator(ABC):
    def __init__(
        self,
        dir_dose_setup: Union[Path, str],
        pth_dose_executable: Union[Path, str],
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of dose generators. Each generator should
            support the attributes of this class and implements its abstract methods.
        Attributes:
            - dir_dose_setup: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Inputs:
            - dir_dose_setup: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Functions:
            - generate_dose(): generates the dose distribution as well as its uncertaity per voxel.
            - validate_dose_setup(): validates the dose setup directory.
        """
        self.dir_dose_setup: Path = Path(dir_dose_setup)
        self.pth_dose_executable: Path = Path(pth_dose_executable)
        # this will be set by the generate_dose() method
        self.dose: BrachyDose = None

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
        if pth_output is not None:
            self.dose.write_brachydose_to_file(pth_output)


class DoseTG43(DoseGenerator):
    def __init__(
        self,
        dir_dose_setup: Union[Path, str],
        pth_dose_executable: Union[Path, str],
    ) -> None:
        super().__init__(dir_dose_setup, pth_dose_executable)
        r"""
        Purpose:
            - A class to generate dose distribution using the TG43 formalism.
            This class uses RapidBrachyTG43 to calculate the dose distribution.
        """
        super().__init__(dir_dose_setup, pth_dose_executable)

    def generate_dose(self, filename: Optional[Path] = None):
        r"""
        Purpose:
            - Generate the dose distribution using the TG43 formalism.
        Inputs:
            - filename: Optional[Path] = None
        """
        # do the dose calculation here
        # self.dose = BrachyDose(...)
        super().generate_dose(filename)

    def validate_inputs(self):
        r"""
        Purpose:
            - Validate the inputs of the TG43 dose generator.
        """
        assert self.dir_dose_setup.exists(), "The dose setup directory does not exist."
        assert self.pth_dose_executable.exists(), "The dose executable does not exist."

        # look through the files in dose setup directory
        all_files: list = glob(str(self.dir_dose_setup / "*"))
        assert len(all_files) > 0, "The dose setup directory is empty."
