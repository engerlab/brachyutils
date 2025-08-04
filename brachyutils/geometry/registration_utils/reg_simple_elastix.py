from brachyutils.geometry.registration_utils.reg_utils import BrachyPhantomRegistration
from brachyutils.geometry.phantom_utils import BrachyPhantom
from pathlib import Path
from typing import Literal, Optional, Union

class Registration_SimpleElastix(BrachyPhantomRegistration):
    def __init__(
        self,
        pth_simple_elastix: Path | str,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        backend = "simple_elastix",
        tryGPU: bool = False,
        **kwargs
        ):
        r"""
        Purpose:
            - A class to wrap around the simple_elastix image registration method.
        Inputs:
            - pth_simple_elastix: Path | str: The path to the simple_elastix executable or the URL where
            the simple_elastix server is running.
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
            if this input is provided, contour based registration is used. If "common" is provided, registeration is done
            based on all the common structures.
            - deformable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["simple_elastix"] = "opentps" The backend package used to handle 
            the registration process.
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        Dependencies:
            - simple_elastix
        """

        super().__init__(
            static_phantom,
            moving_phantom,
            register_on_contour,
            deformable,
            # algorithm,
            backend,
            tryGPU
            )
        self.pth_simple_elastix = pth_simple_elastix if pth_simple_elastix else kwargs.popitem("pth_simple_elastix", None)

    def register(self):
        r"""
        
        """