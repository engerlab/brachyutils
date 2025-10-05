from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from typing import Literal, Optional, Union, List, Dict
from collections import defaultdict
import numpy as np
from pathlib import Path

from brachyutils.geometry.phantom_utils import BrachyPhantom
# from brachyutils.geometry.phantom_utils import phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D
# from opentps.core.data.images import Deformation3D, VectorField3D
# from opentps.core.data.images import ROIMask

from brachyutils.geometry.registration_utils.reg_utils import BrachyPhantomRegistration

class Registration_OpenTPS(BrachyPhantomRegistration):
    def __init__(
        self,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons", "quick"] = None,
        backend = "opentps",
        tryGPU: bool = False,
        **kwargs
        ):
        r"""
        Purpose:
            - A class to wrap around the OpenTPS image registration method.
        Inputs:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
            if this input is provided, contour based registration is used. If "common" is provided, registeration is done
            based on all the common structures.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["opentps"] = "opentps" The backend package used to handle 
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
            register_on_contour if register_on_contour else kwargs.get("register_on_contour", None),
            deformable if deformable else kwargs.get("deformable", False),
            algorithm if algorithm else kwargs.get("algorithm", None),
            backend,
            tryGPU if tryGPU else kwargs.get("tryGPU", False)
            )
        
    def register(
        self,
        dir_phantom_export: Path | str = None,
        **kwargs
        ) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom using the OpenTPS package.
        Inputs:
            - dir_phantom_export := directory where the registered phantom is exported to.
            
            kwargs entries could include:
            - baseResolution: float = 2.0: The base resolution of the registration algorithm in mm.
            - tryGPU: bool = False: A flag to indicate whether to use the GPU for the registration process.
            - multimodal: bool = False: A flag to indicate whether the registration is multimodal or not.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        if self.deformable:
            assert self.algorithm is not None, "The registration algorithm is not defined."
        
        if self.deformable:

            if self.algorithm == "demons":
                from opentps.core.processing.registration.registrationDemons import RegistrationDemons

                reg = RegistrationDemons(
                    fixed=self._static_data,
                    moving=self._moving_data,
                    baseResolution=kwargs.get("baseResolution", np.max(self._static_data.spacing)),
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "morphons":
                from opentps.core.processing.registration.registrationMorphons import RegistrationMorphons
    
                reg = RegistrationMorphons(
                    fixed=self._static_data,
                    moving=self._moving_data,
                    baseResolution=kwargs.get("baseResolution", np.max(self._static_data.spacing)),
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "quick":
                from opentps.core.processing.registration.registrationQuick import RegistrationQuick

                reg = RegistrationQuick(
                    fixed=self._static_data,
                    moving=self._moving_data,
                )
                self.deformation = reg.compute(tryGPU=self.tryGPU)
            else:
                raise ValueError("The registration algorithm is not supported. Please choose between 'demons' and 'morphons'.")

        else:
            from opentps.core.processing.registration.registrationRigid import RegistrationRigid
            reg = RegistrationRigid(
                fixed=self._static_data,
                moving=self._moving_data,
                multimodal=kwargs.get("multimodal", False)
            )
            self.deformation = reg.compute()
        # self._registered_data = reg.deformed
        self._registered_data = resampleImage3DOnImage3D(
                reg.deformed,
                self._static_data,
            )

        self.synch_registered_phantom_with_data()
        if dir_phantom_export is not None:
            self.export_to(dir_phantom_export)

        return self.registered_phantom, self.deformation

    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        super().export_to(dir_registered_phantom, output_type)

    def synch_registered_phantom_with_data(self) -> None:
        super().synch_registered_phantom_with_data()
        
    def evaluate_on_contours(self):
        return super().evaluate_on_contours()
