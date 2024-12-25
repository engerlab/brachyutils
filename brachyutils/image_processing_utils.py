
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union

class ImageRegistration(ABC):
    def __init__(
        self,
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of image registration methods and algorithms. 
            Each registration method should support the attributes of this class and implements its abstract methods.
        Attributes:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - type_algorithm: Literal["static", "deformable-demons", ""]: The type of registration algorithm.
            - registration_backend: Literal["elastix", "plastimatch", "opentps"] = "opentps
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Inputs:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Functions:
            - generate_dose(): generates the dose distribution as well as its uncertaity per voxel.
            - validate_inputs(): validates the dose setup directory.
        """
        pass
