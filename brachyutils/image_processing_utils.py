from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union, List

from brachyutils.geometry_utils import BrachyPhantom, phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D
from opentps.core.data.images import ROIMask
class PhantomRegistration(ABC):
    def __init__(
        self,
        static_phantom: Union[BrachyPhantom, str],
        moving_phantom: Union[BrachyPhantom, str],
        register_based_on: Literal["image", "contour"] = "image",
        contour_name: Optional[str] = None,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons"] = None,
        backend: Literal["elastix", "plastimatch", "opentps"] = None,
        tryGPU: bool = False,
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of image registration methods and algorithms. 
            Each registration method should support the attributes of this class and implements its abstract methods.
        Attributes:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_based_on: Literal["image", "contour"] = "image": The registration target. Could be image or a specific contour.
            - contour_name: Optional[str] = None: The name of the contour to be used in the registration process.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["elastix", "plastimatch", "opentps"] = "opentps" The backend package used to handle 
            the registration process.

        Inputs:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        """
        if register_based_on not in ["image", "contour"]:
            raise ValueError("register_based_on must be either 'image' or 'contour'")
        if register_based_on == "contour" and contour_name is None:
            raise ValueError("contour_name must be provided when register_based_on is set to 'contour'")
        if register_based_on == "image" and contour_name is not None:
            raise ValueError("contour_name should not be provided when register_based_on is set to 'image'")

        self.static_phantom = static_phantom
        self.moving_phantom = moving_phantom
        self.register_based_on = register_based_on
        self.contour_name = contour_name if self.register_based_on == "contour" else None
        self.deformable = deformable
        self.algorithm = algorithm
        self.backend = backend
        self.tryGPU = tryGPU
        # the following attributes will be computed during the registration process
        self.registered_phantom: BrachyPhantom = None
        self.deformation: Transform3D = None
        self.static_data = None
        self.moving_data = None

        # depending on the registration target we will set the static and moving data
        if self.register_based_on == "image":
            self.static_data = self.static_phantom.image_obj
            self.moving_data = self.moving_phantom.image_obj
        else:
            self.static_data = self.static_phantom.get_structure_mask(
                self.contour_name,
                mask_type=ROIMask
            )
            self.moving_data = self.moving_phantom.get_structure_mask(
                self.contour_name,
                mask_type=ROIMask
            )

    @abstractmethod
    def register(self) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        pass

    @abstractmethod
    def export_to(self, pth_phantom_export) -> None:
        """
        Purpose:
            - To export the obtained registered image to a given path file.
        Inputs:
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Output:
            - None     
        """
        pass
    
    @abstractmethod
    def synch_image_and_contours(self) -> None:
        """
        Purpose:
            - To match the image and the contours of the registered phantom. If the registration
            was based on the image, the same deformation will be applied to the contours.
            If the registration was based on the contours, the deformation will be applied to the image
            and the contours will be resampled on the deformed image.  
        Inputs:
            - None
        Output:
            - None
        """
        pass

from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
class OpenTPS(PhantomRegistration):
    def __init__(
        self,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_based_on: Literal["image", "contour"] = "image",
        contour_name: Optional[str] = None,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons", "quick"] = None,
        backend = "opentps",
        tryGPU: bool = False,
        ):
        r"""
        Purpose:
            - A class to wrap around the OpenTPS image registration method.
        Inputs:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_based_on: Literal["image", "contour"] = "image": The registration target. Could be image or a specific contour.
            - contour_name: Optional[str] = None: The name of the contour to be used in the registration process.
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
            register_based_on,
            contour_name,
            deformable,
            algorithm,
            backend,
            tryGPU
            )
        # resample the moving image on the static image.
        # new_moving_phantom = phantom_with_empty_image_like(
        #     self.moving_phantom,
        #     self.moving_phantom.pth_image.stem
        # )
        # new_moving_phantom.image_obj = resampleImage3DOnImage3D(
        #         image=self.moving_phantom.image_obj,
        #         fixedImage=self.static_phantom.image_obj,
        #         inPlace=False)
        # self.moving_phantom = new_moving_phantom

    def register(
        self,
        baseResolution:float = 2.0,
        multimodal: bool = False,
        ) -> tuple[BrachyPhantom, Transform3D]:
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
        assert self.static_data is not None, "The static phantom is not defined."
        assert self.moving_data is not None, "The moving phantom is not defined."
        if self.deformable:
            assert self.algorithm is not None, "The registration algorithm is not defined."
        
        if self.deformable:

            if self.algorithm == "demons":
                from opentps.core.processing.registration.registrationDemons import RegistrationDemons

                reg = RegistrationDemons(
                    fixed=self.static_data,
                    moving=self.moving_data,
                    baseResolution=baseResolution,
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "morphons":
                from opentps.core.processing.registration.registrationMorphons import RegistrationMorphons
    
                reg = RegistrationMorphons(
                    fixed=self.static_data,
                    moving=self.moving_data,
                    baseResolution=baseResolution,
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "quick":
                from opentps.core.processing.registration.registrationQuick import RegistrationQuick

                reg = RegistrationQuick(
                    fixed=self.static_data,
                    moving=self.moving_data,
                )
                self.deformation = reg.compute(tryGPU=self.tryGPU)
            else:
                raise ValueError("The registration algorithm is not supported. Please choose between 'demons' and 'morphons'.")

        else:
            from opentps.core.processing.registration.registrationRigid import RegistrationRigid
            reg = RegistrationRigid(
                fixed=self.static_data,
                moving=self.moving_data,
                multimodal=multimodal
            )
            self.deformation = reg.compute()

        self.registered_phantom = phantom_with_empty_image_like(
            self.static_phantom,
            new_pth_image=f"reg_{self.moving_phantom.pth_image.stem}")

        self.registered_phantom.image_obj = reg.deformed
        # self.registered_phantom.image_obj.origin = self.static_phantom.image_obj.origin
        # self.registered_phantom.image_obj = resampleImage3DOnImage3D(
        #     reg.deformed,
        #     self.static_phantom.image_obj
        # )
        self.synch_image_and_contours()
        return self.registered_phantom, self.deformation
    
    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        """
        Purpose:
            - To export the obtained registered imag
        Inputs:
            - dir_phantom_export: Union[Path, str]: 
        Output:
            - None     
        """
        assert self.registered_phantom is not None
        if output_type == ".nrrd":
            self.registered_phantom.write_to_file(
                dir_nrrd_out=dir_registered_phantom
                )
        elif output_type == ".dcm":
            self.registered_phantom.write_to_file(
                dir_dicom_out=dir_registered_phantom
            )
        else:
            raise ValueError(f"The output type {output_type} is not supported. please specify .nrrd or .dcm")

    def synch_image_and_contours(self) -> None:
        """
        Purpose:
            - To match the image and the contours of the registered phantom. If the registration
            was based on the image, the same deformation will be applied to the contours.
            If the registration was based on the contours, the deformation will be applied to the image
            and the contours will be resampled on the deformed image.  
        Inputs:
            - None
        Output:
            - None
        """
        if self.register_based_on == "image":
            # apply the deformation to the contours
            contour_mask_dict = self.registered_phantom.get_structure_mask(
                self.registered_phantom.structure_names,
                mask_type=ROIMask
            )
            for contour_name in contour_mask_dict:
                new_mask = self.deformation.deformImage(contour_mask_dict[contour_name])
                self.registered_phantom.structure_set.removeContour(contour_mask_dict[contour_name])
                self.registered_phantom.structure_set.appendContour(new_mask.getROIContour())
        else:
            # apply the deformation to the image and the rest of the contours.
            self.registered_phantom.image_obj = self.deformation.deformImage(self.registered_phantom.image_obj)
            contour_mask_dict = self.registered_phantom.get_structure_mask(
                self.registered_phantom.structure_names,
                mask_type=ROIMask
            )
            for contour_name in contour_mask_dict:
                # skip the contour that was transformed
                if contour_name == self.contour_name:
                    continue
                new_mask = self.deformation.deformImage(contour_mask_dict[contour_name])
                self.registered_phantom.structure_set.removeContour(contour_mask_dict[contour_name])
                self.registered_phantom.structure_set.appendContour(new_mask.getROIContour())