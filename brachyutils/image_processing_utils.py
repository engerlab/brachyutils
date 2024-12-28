from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union, List

from brachyutils.geometry_utils import BrachyPhantom, phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D

class PhantomRegistration(ABC):
    def __init__(
        self,
        static_phantom: Union[BrachyPhantom, str],
        moving_phantom: Union[BrachyPhantom, str],
        deformable: bool = False,
        algorithm: Literal["demons", "morphons"] = None,
        backend: Literal["elastix", "plastimatch", "opentps"] = None,
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of image registration methods and algorithms. 
            Each registration method should support the attributes of this class and implements its abstract methods.
        Attributes:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["elastix", "plastimatch", "opentps"] = "opentps" The backend package used to handle 
            the registration process.
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Inputs:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        """
        self.static_phantom = static_phantom
        self.moving_phantom = moving_phantom
        self.deformable = deformable
        self.algorithm = algorithm
        self.backend = backend
        # the following attributes will be computed during the registration process
        self.deformed_phantom: BrachyPhantom = None
        self.deformation: Transform3D = None

    @abstractmethod
    def register(self) -> List[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        pass

class RegistrationWithOpenTPS(PhantomRegistration):
    def __init__(
        self,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons", "quick"] = None,
        backend = "opentps",
        ):
        r"""
        Purpose:
            - A class to wrap around the OpenTPS image registration method.
        Inputs:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["elastix", "plastimatch", "opentps"] = "opentps" The backend package used to handle 
            the registration process.
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        Dependencies:
            - OpenTPS
        """

        super().__init__(
            static_phantom,
            moving_phantom,
            deformable,
            algorithm,
            backend,
            )


    def register(
        self,
        baseResolution:float = 2.0,
        tryGPU: bool = False,
        multimodal: bool = False,
        ) -> List[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom using the OpenTPS package.
        Inputs:
            - baseResolution: float = 2.0: The base resolution of the registration algorithm in mm.
            - tryGPU: bool = False: A flag to indicate whether to use the GPU for the registration process.
            - multimodal: bool = False: A flag to indicate whether the registration is multimodal or not.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        assert self.static_phantom is not None, "The static phantom is not defined."
        assert self.moving_phantom is not None, "The moving phantom is not defined."
        if self.deforemable:
            assert self.algorithm is not None, "The registration algorithm is not defined."
        
        if self.deformable:

            if self.algorithm == "demons":
                from opentps.core.processing.registration.registrationDemons import RegistrationDemons
                
                reg = RegistrationDemons(
                    fixed=self.static_phantom.image_obj,
                    moving=self.moving_phantom.image_obj,
                    baseResolution=baseResolution,
                    tryGPU=tryGPU
                )
                self.deformation = reg.compute()
                self.deformed_phantom = phantom_with_empty_image_like(self.static_phantom)
                self.deformed_phantom.image_obj = reg.deformed
                
                reg.deformed

            elif self.algorithm == "morphons":
                from opentps.core.processing.registration.registrationMorphons import RegistrationMorphons
    
                reg = RegistrationMorphons(
                    fixed=self.static_phantom.image_obj,
                    moving=self.moving_phantom.image_obj,
                    baseResolution=baseResolution,
                    tryGPU=tryGPU
                )
                self.deformation = reg.compute()
                self.deformed_phantom = phantom_with_empty_image_like(self.static_phantom)
                self.deformed_phantom.image_obj = reg.deformed

            elif self.algorithm == "quick":
                from opentps.core.processing.registration.registrationQuick import RegistrationQuick

                reg = RegistrationQuick(
                    fixed=self.static_phantom.image_obj,
                    moving=self.moving_phantom.image_obj,
                )
                self.deformation = reg.compute(tryGPU=tryGPU)
                self.deformed_phantom = phantom_with_empty_image_like(self.static_phantom)
                self.deformed_phantom.image_obj = reg.deformed

            else:
                raise ValueError("The registration algorithm is not supported. Please choose between 'demons' and 'morphons'.")

        else:
            from opentps.core.processing.registration.registrationRigid import RegistrationRigid

            reg = RegistrationRigid(
                fixed=self.static_phantom.image_obj,
                moving=self.moving_phantom.image_obj,
                multimodal=multimodal
            )
            self.deformation = reg.compute()
            self.deformed_phantom = phantom_with_empty_image_like(self.static_phantom)
            self.deformed_phantom.image_obj = reg.deformed

        return self.deformed_phantom, self.deformation