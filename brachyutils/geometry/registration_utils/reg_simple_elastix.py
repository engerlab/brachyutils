from brachyutils.geometry.registration_utils.reg_utils import (
    BrachyPhantomRegistration, phantom_with_empty_image_like
)
from brachyutils.geometry.phantom_utils import BrachyPhantom
from pathlib import Path
from typing import Literal, Optional, Union
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.data.images import ROIMask

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

    def register(
        self,
        parameter_map: str = "translation",
        dir_phantom_export: str | Path = None,
        ):
        r"""
        ### Purpose:
        - Register the moving phantom to the static phantom using the simple_elastix package.
        ### Inputs:
        - parameter_map: str: not sure what this is, but is a string that is passed to the simple_elastix executable.
        - dir_phantom_export: str | Path: The path to the directory where the registered phantom is exported to.
        ### Outputs:
        - BrachyPhantom: The registered phantom object.
        """
        # leave some space to figure out the rigidness and options for the registration.

        # need to write out the images for simple elastix to read them.
        # first sort out the paths to the images
        if dir_phantom_export is None:
            dir_temp_data = Path(__file__).resolve().parent.parent.parent.parent.joinpath("temp_data/registration/temp")

        elif "temp_data/registration" not in str(dir_phantom_export.resolve()):
            dir_temp_data = Path(__file__).resolve().parent.parent.parent.parent.joinpath("temp_data/registration")
        else:
            dir_temp_data = dir_phantom_export.resolve().joinpath("temp/"+self.moving_phantom.pth_image.stem)

        pth_static = dir_temp_data.joinpath("static.nrrd")
        pth_moving = dir_temp_data.joinpath("moving.nrrd")
        pth_output = dir_temp_data.joinpath("registered.nrrd")

        # create phantoms with empty image data. remember, the image data could be structure masks
        # or actual image data.
        for data, pth in zip([self._static_data, self._moving_data], [pth_static, pth_moving]):
            empty_phant = phantom_with_empty_image_like(
                self.moving_phantom,
                new_pth_image=pth
            )
            empty_phant.image_obj = data
            empty_phant.write_image_to_nrrd(
                pth_output=pth
            )
        # now we have the paths to the images, we can register them.
        if "http" in self.pth_simple_elastix:
            import requests
            http_pth_static = str(pth_static).split("temp_data/registration/")[-1]
            http_pth_moving = str(pth_moving).split("temp_data/registration/")[-1]
            http_pth_output = str(pth_output).split("temp_data/registration/")[-1]
            response = requests.post(
                self.pth_simple_elastix+"/elastix_register",
                json={
                    "pth_fixed_image": http_pth_static,
                    "pth_moving_image": http_pth_moving,
                    "parameter_map": parameter_map,
                    "pth_output_image": http_pth_output
                }
            )
        else:
            raise NotImplementedError("The local simple_elastix registration is not implemented yet.")
        if response.status_code != 200:
            raise RuntimeError(f"Registration failed with status code {response.status_code}: {response.text}")
        pth_transform_maps = list(dir_temp_data.glob("transform_parameter_*.txt"))
        if not pth_transform_maps:
            raise FileNotFoundError(f"Registration failed, no parameter map files were created.")
        # now we load the registered image and create a new phantom object.
                # load the registered image
        self._registered_data = BrachyPhantom(
            pth_phantom_file=pth_output
        ).image_obj
        # do not resample the registered data on static data. the registration is already done that.
        # self._registered_data = resampleImage3DOnImage3D(
            # self._registered_data,
            # self._static_data,
            # )

        self.synch_registered_phantom_with_data(
            transform_params=pth_transform_maps
            )
        if dir_phantom_export is not None:
            self.export_to(dir_phantom_export)
        return self.registered_phantom, self.deformation

    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        r"""
        See `BrachyPhantomRegistration.export_to` for more details.
        """
        super().export_to(dir_registered_phantom, output_type)

    def synch_registered_phantom_with_data(self, transform_params:list[str | Path]):
        r"""
        Synchronize the registered phantom with the original data. 
        """
        # we have the path to the parameter map. we make an api call to simple_elastix to 
        # to apply this transformation to the registered phantom (image or contours).
                # load the registered data into registered phantom
        self.registered_phantom = phantom_with_empty_image_like(
            self.moving_phantom,
            new_pth_image=f"reg_{self.moving_phantom.pth_image.stem}"
            )

        # if registration based on image:
        # load the image into the registered phantom
        if self.register_on_contour is None:
            self.registered_phantom.image_obj = self._registered_data
        # registration based on contour
        # create a new contour based on the registered mask.
        else:
            # pass the moving image to the registered phantom image
            self.registered_phantom.image_obj = self.moving_phantom.image_obj
            self._registered_data = extract_mask_from_image(self._registered_data)
            # create a new contour based on the registered mask.
            new_contour = ROIMask(
                name=self.register_on_contour,
                imageArray=self._registered_data.imageArray.astype(bool),
                origin=self._registered_data.origin,
                spacing=self._registered_data.spacing,
            )
            self.registered_phantom.set_structure_set({self.register_on_contour: new_contour})

        structure_mask_dict = self.registered_phantom.get_structure_mask(
            self.registered_phantom.structure_names,
            mask_type=ROIMask
        )
        all_data = structure_mask_dict | {"image": self.registered_phantom.image_obj}
        for data_name in all_data:
            # write out the data to be warped by simple elastix
            pth_in = transform_params[0].parent.joinpath(f"{data_name}.nrrd")
            pth_warped = transform_params[0].parent.joinpath(f"{data_name}_warped.nrrd")

            empty_phant = phantom_with_empty_image_like(
                self.moving_phantom,
                new_pth_image=pth_in
            )
            empty_phant.image_obj = all_data.get(data_name)
            empty_phant.write_image_to_nrrd(
                pth_output=pth_in
            )
            if data_name == self.register_on_contour:
                # skip the contour that is used for registration
                continue
            # call simple elastix warp to deform the image and the contours.
            if "http" in self.pth_simple_elastix:
                import requests
                pth_in_http = str(pth_in).split("temp_data/registration/")[-1]
                pth_output_http = str(pth_warped).split("temp_data/registration/")[-1]
                pth_transform_http = [str(pth).split("temp_data/registration/")[-1] for pth in transform_params]

                response = requests.post(
                    url=self.pth_simple_elastix+"/elastix_warp",
                    json={
                        "pth_input": pth_in_http,
                        "pth_output": pth_output_http,
                        "pth_transform_maps": pth_transform_http,
                        },
                    timeout=None
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Registration failed with status code {response.status_code}: {response.text}")
            else:
                raise NotImplementedError("The local simple elastix registration is not implemented yet.")

            # load the deformed image and contours back into the registered phantom.
            if data_name == self.register_on_contour:
                continue
            deformed_data = BrachyPhantom(
                pth_phantom_file=pth_warped,
            ).image_obj
            # deformed_data = resampleImage3DOnImage3D(
                # deformed_data,
                # self._static_data
            # )
            if data_name == "image":
                self.registered_phantom.image_obj = deformed_data
            else:
                structure_mask_dict[data_name] = ROIMask(
                    extract_mask_from_image(deformed_data).imageArray,
                    name=data_name,
                    spacing=self.registered_phantom.image_obj.spacing,
                    origin=self.registered_phantom.image_obj.origin
                    )
        self.registered_phantom.set_structure_set(structure_mask_dict)

    def evaluate_on_contours(self):
        r"""
        See `BrachyPhantomRegistration.evaluate_on_contours` for more details.
        """
        return super().evaluate_on_contours()
    
from opentps.core.data.images import Image3D
def extract_mask_from_image(image_obj: Image3D) -> Image3D:
    r"""
    Simple Elastix generates a floating point image as output, even if the input is a binary mask.
    This function extracts a binary mask from the floating point image by applying a threshold.
    We only take the top 15% of the intensity values to create the mask.
    """
    if not hasattr(image_obj, 'imageArray'):
        raise ValueError("The image object does not have an 'imageArray' attribute.")
    # Assuming the imageArray is a numpy array, we can create a mask based on a threshold.
    from opentps.core.processing.segmentation.segmentation3D import applyThreshold
    import numpy as np
    threshold_min = np.percentile(image_obj.imageArray[image_obj.imageArray!=0], [85])
    threshold_max = np.percentile(image_obj.imageArray[image_obj.imageArray!=0], [100])
    mask = applyThreshold(image_obj, thresholdMin=threshold_min, thresholdMax=threshold_max)
    image_obj.imageArray = mask.imageArray.astype(bool)
    return image_obj